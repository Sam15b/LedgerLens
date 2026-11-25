# Redis for the ingestion queue (arq): ElastiCache, one small node in the private subnets.
# The queue is transient (a lost job is re-sent from the dashboard), so no replica and no
# snapshots. Only the app's tasks can connect.

resource "aws_elasticache_subnet_group" "main" {
  name        = local.name
  description = "LedgerLens private subnets"
  subnet_ids  = aws_subnet.private[*].id
}

resource "aws_security_group" "redis" {
  name        = "${local.name}-redis"
  description = "LedgerLens Redis: from the app's tasks only"
  vpc_id      = aws_vpc.main.id

  tags = { Name = "${local.name}-redis" }
}

resource "aws_vpc_security_group_ingress_rule" "redis_from_tasks" {
  security_group_id            = aws_security_group.redis.id
  referenced_security_group_id = aws_security_group.tasks.id
  ip_protocol                  = "tcp"
  from_port                    = 6379
  to_port                      = 6379
  description                  = "Redis from the app's tasks"
}

resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "${local.name}-queue"
  engine               = "redis"
  engine_version       = "7.1"
  node_type            = var.redis_node_type
  num_cache_nodes      = 1
  port                 = 6379
  parameter_group_name = "default.redis7"

  subnet_group_name  = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]

  snapshot_retention_limit = 0
  maintenance_window       = "sun:05:30-sun:06:30" # UTC, after the database's
}

locals {
  redis_url = "redis://${aws_elasticache_cluster.redis.cache_nodes[0].address}:6379/0"
}
