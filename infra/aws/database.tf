# Managed PostgreSQL 16 (RDS) in the private subnets, reachable only from the app's tasks.
#
# The master password never enters the Terraform state: it is an ephemeral value, sent to
# RDS and to Secrets Manager through write-only arguments. Both are written again only when
# db_password_version changes; bump it to rotate the password (one apply sets both).
# The app reads the whole connection string from the `<name>/database-url` secret as
# DATABASE_URL, the same variable as in docker-compose.

locals {
  db_name = "ledgerlens"
  db_user = "ledger"
}

ephemeral "random_password" "db" {
  length  = 32
  special = false # the password goes into a URL
}

resource "aws_db_subnet_group" "main" {
  name        = local.name
  description = "LedgerLens private subnets"
  subnet_ids  = aws_subnet.private[*].id
}

resource "aws_security_group" "db" {
  name        = "${local.name}-db"
  description = "LedgerLens Postgres: from the app's tasks only"
  vpc_id      = aws_vpc.main.id

  tags = { Name = "${local.name}-db" }
}

resource "aws_vpc_security_group_ingress_rule" "db_from_tasks" {
  security_group_id            = aws_security_group.db.id
  referenced_security_group_id = aws_security_group.tasks.id
  ip_protocol                  = "tcp"
  from_port                    = 5432
  to_port                      = 5432
  description                  = "Postgres from the app's tasks"
}

resource "aws_db_parameter_group" "postgres" {
  name        = "${local.name}-postgres16"
  family      = "postgres16"
  description = "LedgerLens: TLS only, slow-query log"

  parameter {
    name  = "rds.force_ssl"
    value = "1"
  }

  parameter {
    name  = "log_min_duration_statement"
    value = "1000" # ms: log queries slower than a second
  }
}

resource "aws_db_instance" "main" {
  identifier     = local.name
  engine         = "postgres"
  engine_version = var.db_engine_version
  instance_class = var.db_instance_class

  allocated_storage     = var.db_allocated_storage_gb
  max_allocated_storage = var.db_max_allocated_storage_gb # storage autoscaling ceiling
  storage_type          = "gp3"
  storage_encrypted     = true

  db_name             = local.db_name
  username            = local.db_user
  password_wo         = ephemeral.random_password.db.result
  password_wo_version = var.db_password_version

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.db.id]
  parameter_group_name   = aws_db_parameter_group.postgres.name
  publicly_accessible    = false
  multi_az               = var.db_multi_az

  backup_retention_period    = var.db_backup_retention_days
  backup_window              = "03:00-04:00"         # UTC
  maintenance_window         = "sun:04:30-sun:05:30" # UTC, after the backup
  auto_minor_version_upgrade = true
  copy_tags_to_snapshot      = true

  deletion_protection       = var.db_deletion_protection
  skip_final_snapshot       = !var.db_deletion_protection
  final_snapshot_identifier = var.db_deletion_protection ? "${local.name}-final" : null

  tags = { Name = local.name }
}

resource "aws_secretsmanager_secret" "database_url" {
  name                    = "${local.name}/database-url"
  description             = "LedgerLens ${var.environment}: DATABASE_URL (SQLAlchemy URL with the RDS password)"
  recovery_window_in_days = var.secret_recovery_days
}

resource "aws_secretsmanager_secret_version" "database_url" {
  secret_id = aws_secretsmanager_secret.database_url.id
  secret_string_wo = format(
    "postgresql+psycopg://%s:%s@%s:%d/%s?sslmode=require",
    local.db_user,
    ephemeral.random_password.db.result,
    aws_db_instance.main.address,
    aws_db_instance.main.port,
    local.db_name,
  )
  secret_string_wo_version = var.db_password_version
}
