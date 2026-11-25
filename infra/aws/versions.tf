terraform {
  required_version = ">= 1.10" # S3 backend native locking (use_lockfile)

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  # State lives in S3, locked with a lock file next to it. The bucket is created once by
  # hand (see README.md); pass its name with `terraform init -backend-config=backend.hcl`.
  backend "s3" {}
}
