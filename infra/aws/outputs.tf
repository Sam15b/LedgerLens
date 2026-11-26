output "region" {
  value = var.region
}

output "environment" {
  description = "The app's APP_ENV, which its /health reports."
  value       = var.environment
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

output "deploy_role_arn" {
  description = "Set as the AWS_DEPLOY_ROLE_ARN repository variable (deploy-aws.yml)."
  value       = aws_iam_role.deploy.arn
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

output "url" {
  description = "Public address of the dashboard."
  value       = length(var.domain_aliases) > 0 ? "https://${var.domain_aliases[0]}" : "https://${aws_cloudfront_distribution.main.domain_name}"
}

output "cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "service_names" {
  value = { for name, service in aws_ecs_service.app : name => service.name }
}

output "task_definition_arns" {
  description = "Revision per task for var.image_tag; the deploy workflow moves the services to these after running init."
  value       = { for name, td in aws_ecs_task_definition.app : name => td.arn }
}

output "task_network" {
  description = "Network settings for one-off tasks (aws ecs run-task --network-configuration)."
  value = {
    subnets         = aws_subnet.public[*].id
    security_groups = [aws_security_group.tasks.id]
  }
}

output "log_groups" {
  description = "CloudWatch log group per task (streams are <task>/<task>/<task id>)."
  value       = { for name, group in aws_cloudwatch_log_group.task : name => group.name }
}

output "demo_bundle_uri" {
  description = "Where the seed task reads the demo bundle: upload it here (aws s3 cp)."
  value       = local.demo_bundle_uri
}
