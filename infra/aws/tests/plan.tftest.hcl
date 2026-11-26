# `terraform test`: plans the whole configuration against mocked providers, so it needs no
# AWS account or credentials. CI runs it on every change under infra/ (ci.yml, terraform).
# It catches what `validate` cannot: bad counts and indexes, wrong references between
# resources, invalid CIDR maths, and the guarantees asserted below.

variables {
  image_tag           = "0123456789abcdef0123456789abcdef01234567"
  budget_alert_emails = ["alerts@example.com"]
  neo4j_uri           = "neo4j+s://0123abcd.databases.neo4j.io"
}

mock_provider "aws" {
  # Generate the mocked computed values at plan time too, so the assertions can look
  # inside the rendered task definitions.
  override_during = plan

  mock_resource "aws_ecr_repository" {
    defaults = { repository_url = "123456789012.dkr.ecr.us-east-1.amazonaws.com/ledgerlens/mock" }
  }
  mock_resource "aws_secretsmanager_secret" {
    defaults = { arn = "arn:aws:secretsmanager:us-east-1:123456789012:secret:mock" }
  }
  mock_resource "aws_elasticache_cluster" {
    defaults = { cache_nodes = [{ address = "queue.mock.cache.amazonaws.com", port = 6379 }] }
  }
  mock_resource "aws_cloudfront_distribution" {
    defaults = { domain_name = "d111111abcdef8.cloudfront.net" }
  }

  mock_data "aws_availability_zones" {
    defaults = { names = ["us-east-1a", "us-east-1b", "us-east-1c", "us-east-1d"] }
  }
  mock_data "aws_caller_identity" {
    defaults = { account_id = "123456789012" }
  }
  mock_data "aws_partition" {
    defaults = { partition = "aws" }
  }
  mock_data "aws_iam_policy_document" {
    defaults = { json = "{\"Version\":\"2012-10-17\",\"Statement\":[]}" }
  }
}

mock_provider "random" {}

run "plan_defaults" {
  command = plan

  assert {
    condition     = length(aws_subnet.public) == 2 && length(aws_subnet.private) == 2
    error_message = "expected two public and two private subnets"
  }

  assert {
    condition     = aws_subnet.public[0].availability_zone != aws_subnet.public[1].availability_zone
    error_message = "public subnets must be in different availability zones"
  }

  assert {
    condition     = alltrue([for s in aws_subnet.private : !s.map_public_ip_on_launch])
    error_message = "private subnets must not hand out public IPs"
  }

  assert {
    condition     = alltrue([for r in aws_ecr_repository.service : r.image_tag_mutability == "IMMUTABLE"])
    error_message = "image tags must be immutable (one tag per commit)"
  }

  assert {
    condition     = toset(keys(aws_ecr_repository.service)) == toset(["api", "worker", "mcp", "web"])
    error_message = "one repository per service image"
  }

  assert {
    condition = anytrue([
      for c in data.aws_iam_policy_document.build_trust.statement[0].condition :
      c.variable == "token.actions.githubusercontent.com:sub"
      && c.values == tolist(["repo:Sam15b/LedgerLens:ref:refs/heads/main"])
    ])
    error_message = "the build role must only trust main of this repository"
  }

  assert {
    condition = anytrue([
      for c in data.aws_iam_policy_document.deploy_trust.statement[0].condition :
      c.variable == "token.actions.githubusercontent.com:sub"
      && c.values == tolist(["repo:Sam15b/LedgerLens:environment:production"])
    ])
    error_message = "the deploy role must only trust jobs in the production environment"
  }

  assert {
    condition = alltrue([
      for st in data.aws_iam_policy_document.deploy_iam.statement :
      alltrue([for r in st.resources : r == "*" || strcontains(r, ":123456789012:")])
      if st.sid != "ReadManagedPolicies"
    ])
    error_message = "the deploy role's IAM rights stay inside this account"
  }
}

run "database_is_private" {
  command = plan

  assert {
    condition     = !aws_db_instance.main.publicly_accessible && aws_db_instance.main.storage_encrypted
    error_message = "RDS must be private and encrypted"
  }

  assert {
    condition     = aws_db_instance.main.password == null
    error_message = "the database password must only be set write-only (password_wo), never in state"
  }

  assert {
    condition     = one([for p in aws_db_parameter_group.postgres.parameter : p.value if p.name == "rds.force_ssl"]) == "1"
    error_message = "Postgres must refuse unencrypted connections"
  }

  assert {
    condition     = aws_db_instance.main.deletion_protection && !aws_db_instance.main.skip_final_snapshot
    error_message = "by default the database is protected and leaves a final snapshot"
  }
}

run "compute" {
  command = plan

  assert {
    condition     = toset(keys(aws_ecs_service.app)) == toset(["api", "worker", "mcp", "web"])
    error_message = "four long-running services; init is a one-off task, not a service"
  }

  assert {
    condition = alltrue([
      for td in aws_ecs_task_definition.app : alltrue([
        for e in jsondecode(td.container_definitions)[0].environment :
        length(regexall("KEY|PASSWORD|SECRET|DATABASE_URL", e.name)) == 0
      ])
    ])
    error_message = "credentials must come from Secrets Manager (secrets), never plain environment"
  }

  assert {
    condition = (
      join(" ", jsondecode(aws_ecs_task_definition.app["seed"].container_definitions)[0].command)
      == "python scripts/seed_demo_data.py --bundle s3://ledgerlens-prod-demo-data-123456789012/demo/demo-bundle.tar.gz"
    )
    error_message = "the seed task loads the bundle from the demo bucket"
  }

  assert {
    condition     = strcontains(join(" ", jsondecode(aws_ecs_task_definition.app["init"].container_definitions)[0].command), "alembic upgrade head")
    error_message = "the init task runs the migrations"
  }

  assert {
    condition = alltrue([
      for name, td in aws_ecs_task_definition.app :
      endswith(jsondecode(td.container_definitions)[0].image, ":0123456789abcdef0123456789abcdef01234567")
    ])
    error_message = "every task runs the image_tag being deployed"
  }

  assert {
    condition     = length(aws_ecs_service.app["worker"].load_balancer) == 0 && length(aws_ecs_service.app["web"].load_balancer) == 1
    error_message = "only web (and the api's /health) sit behind the load balancer"
  }

  assert {
    condition     = aws_cloudfront_distribution.main.default_cache_behavior[0].viewer_protocol_policy == "redirect-to-https"
    error_message = "the dashboard must be served over HTTPS (Secure session cookie)"
  }

  assert {
    condition     = output.url == "https://d111111abcdef8.cloudfront.net"
    error_message = "without a custom domain the demo lives at its CloudFront address"
  }
}

run "managed_stores" {
  command = plan

  assert {
    condition = alltrue([
      for name in ["api", "worker", "mcp", "init"] : contains(
        jsondecode(aws_ecs_task_definition.app[name].container_definitions)[0].environment,
        { name = "PINECONE_NAMESPACE", value = "prod" }
      )
    ])
    error_message = "prod tasks use the prod Pinecone namespace"
  }

  assert {
    condition = contains(
      jsondecode(aws_ecs_task_definition.app["api"].container_definitions)[0].environment,
      { name = "PINECONE_REGION", value = "us-east-1" }
    )
    error_message = "the Pinecone index defaults to the app's region"
  }
}

run "load_balancer_keep_alive" {
  command = plan

  assert {
    condition = alltrue([
      for e in jsondecode(aws_ecs_task_definition.app["web"].container_definitions)[0].environment :
      tonumber(e.value) > aws_lb.main.idle_timeout * 1000 if e.name == "KEEP_ALIVE_TIMEOUT"
      ]) && contains(
      [for e in jsondecode(aws_ecs_task_definition.app["web"].container_definitions)[0].environment : e.name],
      "KEEP_ALIVE_TIMEOUT"
    )
    error_message = "web must keep idle connections open longer than the load balancer (else 502s)"
  }

  assert {
    condition = contains(
      jsondecode(aws_ecs_task_definition.app["api"].container_definitions)[0].environment,
      { name = "UVICORN_TIMEOUT_KEEP_ALIVE", value = "310" }
    )
    error_message = "the api must keep idle connections open longer than the load balancer"
  }

  assert {
    condition     = aws_lb_target_group.service["api"].health_check[0].path == "/livez"
    error_message = "the load balancer probes /livez, not /health (which queries every store)"
  }
}

run "sampled_tracing" {
  command = plan

  assert {
    condition = alltrue([
      for pair in [["TRACING_ENABLED", "true"], ["TRACING_SAMPLE_RATE", "0.1"], ["LANGSMITH_PROJECT", "ledgerlens-prod"]] :
      contains(jsondecode(aws_ecs_task_definition.app["api"].container_definitions)[0].environment, { name = pair[0], value = pair[1] })
    ])
    error_message = "prod traces a 10 % sample of chats to its own LangSmith project"
  }
}

run "neo4j_uri_must_be_auradb_style" {
  command = plan

  variables {
    neo4j_uri = "bolt://localhost:7687/neo4j"
  }

  expect_failures = [var.neo4j_uri]
}

run "cost_and_logs" {
  command = plan

  assert {
    condition = (
      length(aws_budgets_budget.monthly.notification) == 4
      && alltrue([for n in aws_budgets_budget.monthly.notification : n.subscriber_email_addresses == toset(["alerts@example.com"])])
    )
    error_message = "the budget emails at 50/80/100 % actual and 100 % forecast"
  }

  assert {
    condition     = alltrue([for g in aws_cloudwatch_log_group.task : g.retention_in_days == 14])
    error_message = "task logs expire (log_retention_days)"
  }
}

run "budget_needs_a_recipient" {
  command = plan

  variables {
    budget_alert_emails = []
  }

  expect_failures = [var.budget_alert_emails]
}

run "existing_github_oidc_provider" {
  command = plan

  variables {
    create_github_oidc_provider = false
  }

  assert {
    condition     = length(aws_iam_openid_connect_provider.github) == 0
    error_message = "must reuse the account's provider instead of creating one"
  }
}
