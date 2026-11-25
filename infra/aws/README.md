# infra/aws: LedgerLens on AWS (Terraform)

The cloud deployment of LedgerLens. [ADR 0002](../../docs/adr/0002-cloud-target.md) records
why it is AWS.

| File | What it creates |
|------|-----------------|
| `network.tf` | VPC, two public and two private subnets, internet gateway, the tasks' security group |
| `registry.tf` | ECR repositories `ledgerlens/{api,worker,mcp,web}`, the GitHub OIDC provider, the `build.yml` role |
| `secrets.tf` | Secrets Manager secret `ledgerlens-<env>/app` (JSON) with the app's keys |
| `database.tf` | RDS PostgreSQL 16 (private, encrypted, TLS only) and the `ledgerlens-<env>/database-url` secret |

## Before the first apply

1. **Set a budget alert** in the account (Billing → Budgets) before anything is created.
2. Create the state bucket, once per account:

   ```sh
   aws s3api create-bucket --bucket <your-tf-state-bucket> --region us-east-1
   aws s3api put-bucket-versioning --bucket <your-tf-state-bucket> \
     --versioning-configuration Status=Enabled
   aws s3api put-public-access-block --bucket <your-tf-state-bucket> \
     --public-access-block-configuration \
     BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
   ```

   Buckets are encrypted by default (SSE-S3). The state holds generated secrets (the API key,
   the dashboard password): keep the bucket private.
3. `cp backend.hcl.example backend.hcl` and `cp terraform.tfvars.example terraform.tfvars`,
   then edit both. Both are git-ignored.

## Apply

```sh
cd infra/aws
terraform init -backend-config=backend.hcl
terraform plan -out=tfplan
terraform apply tfplan
```

The first apply runs from a laptop with administrator credentials (it creates the roles CI
uses afterwards). Then set the repository variables `AWS_REGION` and `AWS_BUILD_ROLE_ARN`
(`terraform output build_role_arn`), and `build.yml` starts pushing images on every push to
main.

## Secrets

Terraform writes the secret's first version: generated values for `API_KEYS` /
`LEDGERLENS_API_KEY`, `DASHBOARD_PASSWORD` and `DASHBOARD_SESSION_SECRET`, and empty
strings for the third-party keys. Fill those in without overwriting the generated ones:

```sh
SECRET=$(terraform output -raw app_secret_arn)
aws secretsmanager get-secret-value --secret-id "$SECRET" --query SecretString --output text \
  | jq '.OPENAI_API_KEY = "sk-..." | .PINECONE_API_KEY = "pcsk_..." | .NEO4J_PASSWORD = "..."' \
  > /tmp/app-secret.json
aws secretsmanager put-secret-value --secret-id "$SECRET" --secret-string file:///tmp/app-secret.json
rm /tmp/app-secret.json
```

Terraform ignores later versions, so `apply` never resets them.

`DATABASE_URL` has a secret of its own, written by Terraform. The database password is never
in the state (write-only arguments); to rotate it, bump `db_password_version` and apply. Running tasks read secrets
at start: restart the services after a change (`aws ecs update-service --force-new-deployment`).

## Checks without an AWS account

```sh
terraform fmt -check -recursive
terraform init -backend=false
terraform validate
terraform test        # plans everything against mocked providers (tests/plan.tftest.hcl)
```
