# Shared files (EFS): what docker-compose keeps in named volumes and bind mounts.
#
#   uploads    /app/data/uploads    files the API stores and the worker ingests
#   processed  /app/data/processed  BM25 parameters (bm25.json) for sparse vectors
#   hf_cache   /cache/huggingface   embedding model, downloaded on first use
#   models     /app/models          anomaly model (trained outside the images), read-only
#
# Each is an access point owned by the images' unprivileged user (uid/gid 10001), mounted
# over TLS with the task role's IAM identity.

locals {
  efs_volumes = {
    uploads   = { path = "/app/data/uploads", read_only = false }
    processed = { path = "/app/data/processed", read_only = false }
    hf_cache  = { path = "/cache/huggingface", read_only = false }
    models    = { path = "/app/models", read_only = true }
  }
  app_uid = 10001
}

resource "aws_efs_file_system" "shared" {
  creation_token   = local.name
  encrypted        = true
  performance_mode = "generalPurpose"
  throughput_mode  = "elastic"

  lifecycle_policy {
    transition_to_ia = "AFTER_30_DAYS"
  }

  tags = { Name = local.name }
}

resource "aws_security_group" "efs" {
  name        = "${local.name}-efs"
  description = "LedgerLens EFS: NFS from the app's tasks only"
  vpc_id      = aws_vpc.main.id

  tags = { Name = "${local.name}-efs" }
}

resource "aws_vpc_security_group_ingress_rule" "efs_from_tasks" {
  security_group_id            = aws_security_group.efs.id
  referenced_security_group_id = aws_security_group.tasks.id
  ip_protocol                  = "tcp"
  from_port                    = 2049
  to_port                      = 2049
  description                  = "NFS from the app's tasks"
}

resource "aws_efs_mount_target" "shared" {
  count = var.az_count

  file_system_id  = aws_efs_file_system.shared.id
  subnet_id       = aws_subnet.private[count.index].id
  security_groups = [aws_security_group.efs.id]
}

resource "aws_efs_access_point" "volume" {
  for_each = local.efs_volumes

  file_system_id = aws_efs_file_system.shared.id

  posix_user {
    uid = local.app_uid
    gid = local.app_uid
  }

  root_directory {
    path = "/${each.key}"
    creation_info {
      owner_uid   = local.app_uid
      owner_gid   = local.app_uid
      permissions = "0755"
    }
  }

  tags = { Name = "${local.name}-${each.key}" }
}
