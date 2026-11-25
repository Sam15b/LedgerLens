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
