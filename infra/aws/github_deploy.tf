# The role .github/workflows/deploy-aws.yml assumes (GitHub OIDC, no stored keys).
#
# The deploy runs `terraform apply`, so the role can change everything this configuration
# manages: PowerUserAccess for the services, plus IAM limited to this project's roles
# (`<project>-*`), the GitHub OIDC provider and service-linked roles. Only jobs that run in
# the repository's GitHub environment var.github_deploy_environment can assume it; give that
# environment required reviewers or a tag rule in the repository settings to gate deploys.

data "aws_iam_policy_document" "deploy_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [local.github_oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repository}:environment:${var.github_deploy_environment}"]
    }
  }
}

resource "aws_iam_role" "deploy" {
  name                 = "${var.project}-github-deploy"
  description          = "GitHub Actions deploy-aws.yml: terraform apply, run init, roll out services"
  assume_role_policy   = data.aws_iam_policy_document.deploy_trust.json
  max_session_duration = 7200 # a deploy waits for image pulls and service stability
}

resource "aws_iam_role_policy_attachment" "deploy_power_user" {
  role       = aws_iam_role.deploy.name
  policy_arn = "arn:${local.partition}:iam::aws:policy/PowerUserAccess"
}

data "aws_iam_policy_document" "deploy_iam" {
  statement {
    sid = "ProjectRoles"
    actions = [
      "iam:AttachRolePolicy",
      "iam:CreateRole",
      "iam:DeleteRole",
      "iam:DeleteRolePolicy",
      "iam:DetachRolePolicy",
      "iam:GetRole",
      "iam:GetRolePolicy",
      "iam:ListAttachedRolePolicies",
      "iam:ListInstanceProfilesForRole",
      "iam:ListRolePolicies",
      "iam:PassRole",
      "iam:PutRolePolicy",
      "iam:TagRole",
      "iam:UntagRole",
      "iam:UpdateAssumeRolePolicy",
      "iam:UpdateRole",
    ]
    resources = ["arn:${local.partition}:iam::${local.account_id}:role/${var.project}-*"]
  }

  statement {
    sid = "GithubOidcProvider"
    actions = [
      "iam:AddClientIDToOpenIDConnectProvider",
      "iam:CreateOpenIDConnectProvider",
      "iam:DeleteOpenIDConnectProvider",
      "iam:GetOpenIDConnectProvider",
      "iam:RemoveClientIDFromOpenIDConnectProvider",
      "iam:TagOpenIDConnectProvider",
      "iam:UntagOpenIDConnectProvider",
      "iam:UpdateOpenIDConnectProviderThumbprint",
    ]
    resources = ["arn:${local.partition}:iam::${local.account_id}:oidc-provider/token.actions.githubusercontent.com"]
  }

  statement {
    sid       = "ReadManagedPolicies"
    actions   = ["iam:GetPolicy", "iam:GetPolicyVersion", "iam:ListOpenIDConnectProviders"]
    resources = ["*"]
  }

  statement {
    sid       = "ServiceLinkedRoles" # ECS, ELB, RDS, ElastiCache create theirs on first use
    actions   = ["iam:CreateServiceLinkedRole"]
    resources = ["arn:${local.partition}:iam::${local.account_id}:role/aws-service-role/*"]
  }
}

resource "aws_iam_role_policy" "deploy_iam" {
  name   = "project-iam"
  role   = aws_iam_role.deploy.id
  policy = data.aws_iam_policy_document.deploy_iam.json
}
