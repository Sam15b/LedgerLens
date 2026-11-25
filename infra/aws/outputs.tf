output "region" {
  value = var.region
}

output "vpc_id" {
  value = aws_vpc.main.id
}

output "ecr_registry" {
  description = "Registry host the images are pushed to."
  value       = "${local.account_id}.dkr.ecr.${var.region}.amazonaws.com"
}

output "ecr_repositories" {
  description = "Repository URL per service."
  value       = { for service, repo in aws_ecr_repository.service : service => repo.repository_url }
}

output "build_role_arn" {
  description = "Set as the AWS_BUILD_ROLE_ARN repository variable (build.yml)."
  value       = aws_iam_role.build.arn
}

output "app_secret_arn" {
  description = "Secrets Manager secret holding the app's keys (JSON); fill in the third-party ones."
  value       = aws_secretsmanager_secret.app.arn
}

output "db_address" {
  description = "RDS endpoint host (private subnets: reachable from the tasks only)."
  value       = aws_db_instance.main.address
}

output "database_url_secret_arn" {
  description = "Secrets Manager secret holding DATABASE_URL."
  value       = aws_secretsmanager_secret.database_url.arn
}
