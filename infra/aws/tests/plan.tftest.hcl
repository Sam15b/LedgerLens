# `terraform test`: plans the whole configuration against mocked providers, so it needs no
# AWS account or credentials. CI runs it on every change under infra/ (ci.yml, terraform).
# It catches what `validate` cannot: bad counts and indexes, wrong references between
# resources, invalid CIDR maths, and the guarantees asserted below.

mock_provider "aws" {
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
