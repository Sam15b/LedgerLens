# Cost guard rails. The budget covers the whole account (Bedrock included), not only the
# resources tagged here: a forgotten resource elsewhere costs the same. Emails go out at
# 50 %, 80 % and 100 % of actual spend, and as soon as the month's forecast passes 100 %.
# LLM providers billed outside AWS (OpenAI, Gemini, Mistral) need their own limits in
# their consoles.

locals {
  budget_alerts = [
    { type = "ACTUAL", threshold = 50 },
    { type = "ACTUAL", threshold = 80 },
    { type = "ACTUAL", threshold = 100 },
    { type = "FORECASTED", threshold = 100 },
  ]
}

resource "aws_budgets_budget" "monthly" {
  name         = "${local.name}-monthly"
  budget_type  = "COST"
  limit_amount = format("%.2f", var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  dynamic "notification" {
    for_each = local.budget_alerts
    content {
      notification_type          = notification.value.type
      comparison_operator        = "GREATER_THAN"
      threshold                  = notification.value.threshold
      threshold_type             = "PERCENTAGE"
      subscriber_email_addresses = var.budget_alert_emails
    }
  }
}
