variable "project" {
  description = "Prefix for every resource name."
  type        = string
  default     = "ledgerlens"
}

variable "environment" {
  description = "Deployment environment; also the app's APP_ENV (picks the Pinecone namespace)."
  type        = string
  default     = "prod"

  validation {
    condition     = contains(["dev", "eval", "prod"], var.environment)
    error_message = "environment must be one of dev, eval, prod (the namespaces in configs/pinecone.yaml)."
  }
}

variable "region" {
  description = "AWS region. Keep it the Pinecone index's region (PINECONE_REGION) to avoid cross-region latency."
  type        = string
  default     = "us-east-1"
}

variable "github_repository" {
  description = "owner/name of the GitHub repository whose workflows may assume the CI roles."
  type        = string
  default     = "Sam15b/LedgerLens"
}

variable "create_github_oidc_provider" {
  description = "Create the account's GitHub OIDC identity provider. Set false if the account already has one (there can be only one per URL)."
  type        = bool
  default     = true
}

variable "github_deploy_environment" {
  description = "GitHub environment whose jobs may assume the deploy role (deploy-aws.yml runs in it)."
  type        = string
  default     = "production"
}

# --- network ---

variable "vpc_cidr" {
  description = "CIDR block of the VPC; split into two public and two private /20 subnets."
  type        = string
  default     = "10.40.0.0/16"
}

variable "az_count" {
  description = "Availability zones to spread subnets over (RDS and the ALB need at least two)."
  type        = number
  default     = 2

  validation {
    condition     = var.az_count >= 2 && var.az_count <= 4
    error_message = "az_count must be between 2 and 4."
  }
}

# --- secrets ---

variable "secret_recovery_days" {
  description = "Days a deleted secret can be restored. 0 deletes at once (handy when tearing a demo down)."
  type        = number
  default     = 7
}

# --- database ---

variable "db_engine_version" {
  description = "PostgreSQL major version (docker-compose runs postgres:16)."
  type        = string
  default     = "16"
}

variable "db_instance_class" {
  description = "RDS instance class. db.t4g.micro is the cheapest Graviton class (free tier eligible)."
  type        = string
  default     = "db.t4g.micro"
}

variable "db_allocated_storage_gb" {
  type    = number
  default = 20
}

variable "db_max_allocated_storage_gb" {
  description = "Storage autoscaling ceiling."
  type        = number
  default     = 100
}

variable "db_multi_az" {
  description = "Standby in a second AZ (doubles the cost). Off for the demo."
  type        = bool
  default     = false
}

variable "db_backup_retention_days" {
  type    = number
  default = 7
}

variable "db_deletion_protection" {
  description = "Refuse to delete the database, and take a final snapshot when it is deleted. Set false (and apply) before destroying a demo."
  type        = bool
  default     = true
}

variable "db_password_version" {
  description = "Bump to generate a new database password and write it to RDS and the DATABASE_URL secret."
  type        = number
  default     = 1
}

# --- compute ---

variable "image_tag" {
  description = "Image tag to run: the git SHA build.yml pushed (the deploy workflow passes it)."
  type        = string

  validation {
    condition     = length(var.image_tag) > 0
    error_message = "image_tag must name an image in ECR (a commit SHA)."
  }
}

variable "task_sizes" {
  description = "Fargate CPU units and memory (MiB) per task. The worker and mcp load PyTorch and the embedding model."
  type        = map(object({ cpu = number, memory = number }))
  default = {
    api    = { cpu = 512, memory = 1024 }
    worker = { cpu = 1024, memory = 4096 }
    mcp    = { cpu = 512, memory = 2048 }
    web    = { cpu = 256, memory = 512 }
    init   = { cpu = 512, memory = 1024 }
    seed   = { cpu = 1024, memory = 4096 }
  }

  validation {
    condition     = alltrue([for t in ["api", "worker", "mcp", "web", "init", "seed"] : contains(keys(var.task_sizes), t)])
    error_message = "task_sizes needs an entry for api, worker, mcp, web, init and seed."
  }
}

variable "service_desired_count" {
  description = "Running tasks per service. Set a service to 0 to stop paying for it (e.g. the worker between ingestion runs)."
  type        = map(number)
  default     = { api = 1, worker = 1, mcp = 1, web = 1 }

  validation {
    condition     = alltrue([for s in ["api", "worker", "mcp", "web"] : contains(keys(var.service_desired_count), s)])
    error_message = "service_desired_count needs an entry for api, worker, mcp and web."
  }
}

variable "container_insights" {
  description = "ECS Container Insights metrics (billed as custom CloudWatch metrics)."
  type        = bool
  default     = false
}

variable "redis_node_type" {
  description = "ElastiCache node type for the ingestion queue."
  type        = string
  default     = "cache.t4g.micro"
}

# --- public entry point ---

variable "domain_aliases" {
  description = "Optional custom domain names for the CloudFront distribution. Empty: the demo is served at its *.cloudfront.net address."
  type        = list(string)
  default     = []
}

variable "acm_certificate_arn" {
  description = "ACM certificate (in us-east-1, as CloudFront requires) covering domain_aliases."
  type        = string
  default     = ""

  validation {
    condition     = length(var.domain_aliases) == 0 || startswith(var.acm_certificate_arn, "arn:")
    error_message = "domain_aliases needs an acm_certificate_arn in us-east-1."
  }
}

# --- cost and logs ---

variable "budget_alert_emails" {
  description = "Who gets the monthly budget's alert emails. Required: the budget exists before anything else is created."
  type        = list(string)

  validation {
    condition     = length(var.budget_alert_emails) > 0 && alltrue([for e in var.budget_alert_emails : can(regex("^[^@ ]+@[^@ ]+$", e))])
    error_message = "budget_alert_emails needs at least one email address."
  }
}

variable "monthly_budget_usd" {
  description = "Monthly AWS spend (whole account) the alerts are measured against. Everything running 24/7 is about $160 (ADR-0002)."
  type        = number
  default     = 200
}

variable "log_retention_days" {
  description = "Days CloudWatch keeps the tasks' logs."
  type        = number
  default     = 14

  validation {
    condition     = contains([1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365], var.log_retention_days)
    error_message = "log_retention_days must be a CloudWatch retention value (1, 3, 5, 7, 14, 30, 60, 90, ...)."
  }
}

# --- managed stores outside AWS ---

variable "neo4j_uri" {
  description = "Neo4j AuraDB connection URI (neo4j+s://<instance>.databases.neo4j.io). Its password goes in the app secret as NEO4J_PASSWORD."
  type        = string

  validation {
    condition     = can(regex("^neo4j([+]s)?://[^/]+$", var.neo4j_uri))
    error_message = "neo4j_uri must look like neo4j+s://<instance>.databases.neo4j.io (AuraDB uses TLS: neo4j+s)."
  }
}

variable "neo4j_user" {
  description = "Neo4j user (AuraDB: the instance's generated user, usually `neo4j`)."
  type        = string
  default     = "neo4j"
}

variable "pinecone_index" {
  description = "Pinecone serverless index (configs/pinecone.yaml `index`); init creates it if missing."
  type        = string
  default     = "ledgerlens-chunks"
}

variable "pinecone_region" {
  description = "Region of the Pinecone index (cloud `aws`). Empty: the same region as everything else."
  type        = string
  default     = ""
}
