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
  }

  validation {
    condition     = alltrue([for t in ["api", "worker", "mcp", "web", "init"] : contains(keys(var.task_sizes), t)])
    error_message = "task_sizes needs an entry for api, worker, mcp, web and init."
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
