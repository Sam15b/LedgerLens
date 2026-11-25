# `terraform test`: plans the whole configuration against mocked providers, so it needs no
# AWS account or credentials. CI runs it on every change under infra/ (ci.yml, terraform).
# It catches what `validate` cannot: bad counts and indexes, wrong references between
# resources, invalid CIDR maths, and the guarantees asserted below.

variables {
  image_tag = "0123456789abcdef0123456789abcdef01234567"
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
