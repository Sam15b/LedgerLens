# Application secrets: one Secrets Manager secret holding a JSON object. ECS injects each
# key as an environment variable (valueFrom "<arn>:<KEY>::"), so nothing secret is in a
# task definition, an image or this repository.
#
# Terraform writes the first version only: keys the app generates for itself get random
# values, the third-party keys start empty. Fill those in afterwards (README.md, "Secrets");
# later versions are never overwritten by Terraform (ignore_changes below). Note that the
# generated values are in the Terraform state, which is why the state bucket is private
# and encrypted.

locals {
  # Every key the task definitions reference. ECS refuses to start a task whose secret
  # lacks a referenced key, so the first version has all of them.
  app_secret_keys = [
    "API_KEYS",                 # keys the API accepts (X-API-Key)
    "LEDGERLENS_API_KEY",       # the one the dashboard sends: must be in API_KEYS
    "DASHBOARD_PASSWORD",       # dashboard login
    "DASHBOARD_SESSION_SECRET", # signs dashboard session cookies
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "MISTRAL_API_KEY",
    "PINECONE_API_KEY",
    "NEO4J_PASSWORD",
    "LANGSMITH_API_KEY",
  ]
}

resource "random_password" "api_key" {
  length  = 40
  special = false
}

resource "random_password" "dashboard_password" {
  length  = 20
  special = false
}

resource "random_password" "session_secret" {
  length  = 48
  special = false
}

resource "aws_secretsmanager_secret" "app" {
  name                    = "${local.name}/app"
  description             = "LedgerLens ${var.environment}: API keys, dashboard login, third-party credentials (JSON)"
  recovery_window_in_days = var.secret_recovery_days
}

resource "aws_secretsmanager_secret_version" "app_initial" {
  secret_id = aws_secretsmanager_secret.app.id
  secret_string = jsonencode(merge(
    { for key in local.app_secret_keys : key => "" },
    {
      API_KEYS                 = random_password.api_key.result
      LEDGERLENS_API_KEY       = random_password.api_key.result
      DASHBOARD_PASSWORD       = random_password.dashboard_password.result
      DASHBOARD_SESSION_SECRET = random_password.session_secret.result
    },
  ))

  lifecycle {
    ignore_changes = [secret_string] # the live values are managed outside Terraform
  }
}
