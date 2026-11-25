# Public entry point: CloudFront -> Application Load Balancer -> web (and the api's /health).
#
# CloudFront gives the demo an HTTPS address (https://<id>.cloudfront.net) without a
# domain or certificate; the dashboard needs HTTPS because its session cookie is Secure in
# production. The load balancer accepts traffic from CloudFront only (AWS-managed prefix
# list), so the app cannot be reached around it. Nothing is cached except Next.js's
# content-hashed static files.

data "aws_ec2_managed_prefix_list" "cloudfront" {
  name = "com.amazonaws.global.cloudfront.origin-facing"
}

locals {
  # Services the load balancer forwards to, and their health checks.
  public_targets = {
    web = { port = 3000, health_path = "/healthz" }
    # Only /health is routed to the api. The probe itself uses /livez: /health queries all
    # four stores, and the load balancer probes every 15 s from each of its nodes.
    api = { port = 8000, health_path = "/livez" }
  }
}

resource "aws_security_group" "alb" {
  name        = "${local.name}-alb"
  description = "LedgerLens load balancer: HTTP from CloudFront only"
  vpc_id      = aws_vpc.main.id

  tags = { Name = "${local.name}-alb" }
}

resource "aws_vpc_security_group_ingress_rule" "alb_from_cloudfront" {
  security_group_id = aws_security_group.alb.id
  prefix_list_id    = data.aws_ec2_managed_prefix_list.cloudfront.id
  ip_protocol       = "tcp"
  from_port         = 80
  to_port           = 80
  description       = "CloudFront origin-facing addresses"
}

resource "aws_vpc_security_group_egress_rule" "alb_to_tasks" {
  for_each = local.public_targets

  security_group_id            = aws_security_group.alb.id
  referenced_security_group_id = aws_security_group.tasks.id
  ip_protocol                  = "tcp"
  from_port                    = each.value.port
  to_port                      = each.value.port
  description                  = "to ${each.key}"
}

resource "aws_vpc_security_group_ingress_rule" "tasks_from_alb" {
  for_each = local.public_targets

  security_group_id            = aws_security_group.tasks.id
  referenced_security_group_id = aws_security_group.alb.id
  ip_protocol                  = "tcp"
  from_port                    = each.value.port
  to_port                      = each.value.port
  description                  = "${each.key} from the load balancer"
}

resource "aws_lb" "main" {
  name               = local.name
  load_balancer_type = "application"
  internal           = false
  subnets            = aws_subnet.public[*].id
  security_groups    = [aws_security_group.alb.id]

  idle_timeout               = 300 # chat answers stream for up to CHAT_TIMEOUT_S
  drop_invalid_header_fields = true
}

resource "aws_lb_target_group" "service" {
  for_each = local.public_targets

  name                 = "${local.name}-${each.key}"
  port                 = each.value.port
  protocol             = "HTTP"
  target_type          = "ip"
  vpc_id               = aws_vpc.main.id
  deregistration_delay = 30

  health_check {
    path                = each.value.health_path
    matcher             = "200"
    interval            = 15
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 4
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.service["web"].arn
  }
}

resource "aws_lb_listener_rule" "api_health" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 10

  condition {
    path_pattern {
      values = ["/health"] # the deploy smoke test; every other API call goes through web
    }
  }

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.service["api"].arn
  }
}

# --- CloudFront ---

data "aws_cloudfront_cache_policy" "disabled" {
  name = "Managed-CachingDisabled"
}

data "aws_cloudfront_cache_policy" "optimized" {
  name = "Managed-CachingOptimized"
}

data "aws_cloudfront_origin_request_policy" "all_viewer" {
  # Host, cookies, query strings and headers as the browser sent them (Next.js server
  # actions, the login form, compare Origin with Host), plus CloudFront-Forwarded-Proto:
  # the load balancer sees plain HTTP, and the login redirect has to stay on HTTPS.
  name = "Managed-AllViewerAndCloudFrontHeaders-2022-06"
}

resource "aws_cloudfront_distribution" "main" {
  enabled         = true
  comment         = "LedgerLens ${var.environment}"
  price_class     = "PriceClass_100" # North America and Europe edge locations
  http_version    = "http2and3"
  is_ipv6_enabled = true
  aliases         = var.domain_aliases

  origin {
    origin_id   = "alb"
    domain_name = aws_lb.main.dns_name

    custom_origin_config {
      http_port                = 80
      https_port               = 443
      origin_protocol_policy   = "http-only" # the ALB has no certificate; it only admits CloudFront
      origin_ssl_protocols     = ["TLSv1.2"]
      origin_read_timeout      = 60 # the longest gap between streamed chat events
      origin_keepalive_timeout = 60
    }
  }

  default_cache_behavior {
    target_origin_id         = "alb"
    viewer_protocol_policy   = "redirect-to-https"
    allowed_methods          = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods           = ["GET", "HEAD"]
    cache_policy_id          = data.aws_cloudfront_cache_policy.disabled.id
    origin_request_policy_id = data.aws_cloudfront_origin_request_policy.all_viewer.id
    compress                 = true
  }

  ordered_cache_behavior {
    path_pattern           = "/_next/static/*" # content-hashed file names: safe to cache
    target_origin_id       = "alb"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD"]
    cached_methods         = ["GET", "HEAD"]
    cache_policy_id        = data.aws_cloudfront_cache_policy.optimized.id
    compress               = true
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = length(var.domain_aliases) == 0
    acm_certificate_arn            = length(var.domain_aliases) == 0 ? null : var.acm_certificate_arn
    ssl_support_method             = length(var.domain_aliases) == 0 ? null : "sni-only"
    minimum_protocol_version       = length(var.domain_aliases) == 0 ? "TLSv1" : "TLSv1.2_2021"
  }
}
