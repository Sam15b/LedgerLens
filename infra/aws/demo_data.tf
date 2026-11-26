# Demo data: a private bucket for the demo bundle that the one-off `seed` task loads
# (scripts/seed_demo_data.py --export makes it where the synthetic ledger is built).
# Only the app's task role can read it.

resource "aws_s3_bucket" "demo_data" {
  bucket        = "${local.name}-demo-data-${local.account_id}" # bucket names are global
  force_destroy = true                                          # holds a rebuildable bundle only
}

resource "aws_s3_bucket_public_access_block" "demo_data" {
  bucket                  = aws_s3_bucket.demo_data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "demo_data" {
  bucket = aws_s3_bucket.demo_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

data "aws_iam_policy_document" "task_demo_data" {
  statement {
    sid       = "ReadDemoBundle"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.demo_data.arn}/demo/*"]
  }
}

resource "aws_iam_role_policy" "task_demo_data" {
  name   = "read-demo-bundle"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task_demo_data.json
}

locals {
  demo_bundle_uri = "s3://${aws_s3_bucket.demo_data.bucket}/demo/demo-bundle.tar.gz"
}
