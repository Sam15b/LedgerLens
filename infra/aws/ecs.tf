# The app on ECS Fargate: the four long-running services of docker-compose (api, worker,
# mcp, web) and its one-off `init` task (migrations, Pinecone index, Neo4j schema).
#
# Services find each other by name through ECS Service Connect, as on the compose
# network: web calls http://api:8000, the api calls http://mcp:8765/mcp. Only web (and the
# api's /health) are reachable from outside, through the load balancer (alb.tf).
#
# Rollouts: Terraform registers task definitions for var.image_tag, but the services
# ignore task definition changes after they are created. The deploy workflow applies
# Terraform, runs `init` with the new revision, and only then moves the services to it,
# so migrations always run before new code serves traffic.

locals {
  image = { for s in local.services : s => "${aws_ecr_repository.service[s].repository_url}:${var.image_tag}" }

  app_secret = aws_secretsmanager_secret.app.arn

  # What every Python service gets (docker-compose's x-app-env, for the cloud).
  python_env = {
    APP_ENV    = var.environment
    LOG_LEVEL  = "INFO"
    AWS_REGION = var.region # Bedrock; boto3 takes credentials from the task role
    REDIS_URL  = local.redis_url
    MCP_URL    = "http://mcp:8765/mcp"
  }
  python_secrets = {
    DATABASE_URL      = aws_secretsmanager_secret.database_url.arn
    API_KEYS          = "${local.app_secret}:API_KEYS::"
    OPENAI_API_KEY    = "${local.app_secret}:OPENAI_API_KEY::"
    GOOGLE_API_KEY    = "${local.app_secret}:GOOGLE_API_KEY::"
    MISTRAL_API_KEY   = "${local.app_secret}:MISTRAL_API_KEY::"
    PINECONE_API_KEY  = "${local.app_secret}:PINECONE_API_KEY::"
    NEO4J_PASSWORD    = "${local.app_secret}:NEO4J_PASSWORD::"
    LANGSMITH_API_KEY = "${local.app_secret}:LANGSMITH_API_KEY::"
  }

  python_healthcheck = ["CMD", "python", "-c"]

  tasks = {
    api = {
      image   = "api"
      port    = 8000
      command = null
      volumes = ["uploads"]
      env     = local.python_env
      secrets = local.python_secrets
      health  = concat(local.python_healthcheck, ["import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"])
    }
    worker = {
      image   = "worker"
      port    = null
      command = null
      volumes = ["uploads", "processed", "hf_cache", "models"]
      env     = local.python_env
      secrets = local.python_secrets
      health  = null
    }
    mcp = {
      image   = "mcp"
      port    = 8765
      command = null
      volumes = ["processed", "hf_cache", "models"]
      env     = local.python_env
      secrets = local.python_secrets
      health  = concat(local.python_healthcheck, ["import socket; socket.create_connection(('127.0.0.1', 8765), timeout=4).close()"])
    }
    web = {
      image   = "web"
      port    = 3000
      command = null
      volumes = []
      env     = { LEDGERLENS_API_URL = "http://api:8000" }
      secrets = {
        LEDGERLENS_API_KEY       = "${local.app_secret}:LEDGERLENS_API_KEY::"
        DASHBOARD_PASSWORD       = "${local.app_secret}:DASHBOARD_PASSWORD::"
        DASHBOARD_SESSION_SECRET = "${local.app_secret}:DASHBOARD_SESSION_SECRET::"
      }
      health = ["CMD", "node", "-e", "fetch('http://127.0.0.1:3000/healthz').then(r => process.exit(r.ok ? 0 : 1), () => process.exit(1))"]
    }
    # One-off, before every rollout (all three steps are idempotent).
    init = {
      image = "api"
      port  = null
      command = [
        "sh", "-c",
        "set -e; alembic upgrade head; python scripts/create_pinecone_index.py; python scripts/init_neo4j_schema.py",
      ]
      volumes = []
      env     = local.python_env
      secrets = local.python_secrets
      health  = null
    }
  }

  # Long-running services; which of them other services call by name (Service Connect).
  services_running   = { for name, task in local.tasks : name => task if name != "init" }
  service_connect_to = toset(["api", "mcp"])
}

resource "aws_ecs_cluster" "main" {
  name = local.name

  setting {
    name  = "containerInsights"
    value = var.container_insights ? "enabled" : "disabled"
  }

  service_connect_defaults {
    namespace = aws_service_discovery_http_namespace.main.arn
  }
}

resource "aws_service_discovery_http_namespace" "main" {
  name        = local.name
  description = "LedgerLens service-to-service names (Service Connect)"
}

resource "aws_cloudwatch_log_group" "task" {
  for_each = local.tasks

  name = "/ecs/${local.name}/${each.key}"
}

resource "aws_ecs_task_definition" "app" {
  for_each = local.tasks

  family                   = "${local.name}-${each.key}"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.task_sizes[each.key].cpu
  memory                   = var.task_sizes[each.key].memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64" # build.yml builds linux/amd64
  }

  dynamic "volume" {
    for_each = toset(each.value.volumes)
    content {
      name = volume.key
      efs_volume_configuration {
        file_system_id     = aws_efs_file_system.shared.id
        transit_encryption = "ENABLED"
        authorization_config {
          access_point_id = aws_efs_access_point.volume[volume.key].id
          iam             = "ENABLED"
        }
      }
    }
  }

  container_definitions = jsonencode([merge(
    {
      name      = each.key
      image     = local.image[each.value.image]
      essential = true
      portMappings = each.value.port == null ? [] : [{
        # Service Connect needs a named port. No appProtocol: the proxy then passes TCP
        # through, with none of its HTTP request timeouts (chat streams for minutes).
        name          = "${each.key}-port"
        containerPort = each.value.port
        protocol      = "tcp"
      }]
      environment = [for k in sort(keys(each.value.env)) : { name = k, value = each.value.env[k] }]
      secrets     = [for k in sort(keys(each.value.secrets)) : { name = k, valueFrom = each.value.secrets[k] }]
      mountPoints = [for v in each.value.volumes : {
        sourceVolume  = v
        containerPath = local.efs_volumes[v].path
        readOnly      = local.efs_volumes[v].read_only
      }]
      linuxParameters = { initProcessEnabled = true }   # ECS Exec
      stopTimeout     = each.key == "worker" ? 120 : 30 # let the worker finish its job
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.task[each.key].name
          awslogs-region        = var.region
          awslogs-stream-prefix = each.key
        }
      }
    },
    each.value.command == null ? {} : { command = each.value.command },
    each.value.health == null ? {} : {
      healthCheck = {
        command     = each.value.health
        interval    = 15
        timeout     = 5
        retries     = 4
        startPeriod = 60 # the embedding model loads on the first start
      }
    },
  )])
}

resource "aws_ecs_service" "app" {
  for_each = local.services_running

  name                   = each.key
  cluster                = aws_ecs_cluster.main.id
  task_definition        = aws_ecs_task_definition.app[each.key].arn
  desired_count          = var.service_desired_count[each.key]
  launch_type            = "FARGATE"
  platform_version       = "LATEST"
  enable_execute_command = true
  propagate_tags         = "SERVICE"

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.tasks.id]
    assign_public_ip = true # outbound through the internet gateway; see network.tf
  }

  dynamic "load_balancer" {
    for_each = contains(keys(aws_lb_target_group.service), each.key) ? [each.key] : []
    content {
      target_group_arn = aws_lb_target_group.service[each.key].arn
      container_name   = each.key
      container_port   = each.value.port
    }
  }

  health_check_grace_period_seconds = contains(keys(aws_lb_target_group.service), each.key) ? 120 : null

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  dynamic "service_connect_configuration" {
    for_each = each.value.port == null ? [] : [each.key]
    content {
      enabled   = true
      namespace = aws_service_discovery_http_namespace.main.arn

      dynamic "service" {
        for_each = contains(local.service_connect_to, each.key) ? [each.key] : []
        content {
          port_name      = "${each.key}-port"
          discovery_name = each.key
          client_alias {
            dns_name = each.key
            port     = each.value.port
          }
        }
      }
    }
  }

  lifecycle {
    # Set by the deploy workflow after `init` has run (see the top of this file).
    ignore_changes = [task_definition]
  }

  depends_on = [
    aws_lb_listener.http,
    aws_iam_role_policy.execution_secrets,
    aws_iam_role_policy.task,
    aws_efs_mount_target.shared,
    aws_secretsmanager_secret_version.app_initial,
    aws_secretsmanager_secret_version.database_url,
  ]
}
