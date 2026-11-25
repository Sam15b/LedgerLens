# LedgerLens on AWS: see README.md for the layout and docs/adr/0002-cloud-target.md for why.

locals {
  name = "${var.project}-${var.environment}"

  # The four images built by .github/workflows/build.yml (one ECR repository each).
  services = ["api", "worker", "mcp", "web"]

  account_id = data.aws_caller_identity.current.account_id
  partition  = data.aws_partition.current.partition
}
