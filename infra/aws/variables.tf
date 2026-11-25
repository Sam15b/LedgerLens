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
