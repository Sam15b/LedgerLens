# infra/aws: LedgerLens on AWS (Terraform)

The cloud deployment of LedgerLens. [ADR 0002](../../docs/adr/0002-cloud-target.md) records
why it is AWS.

| File | What it creates |
|------|-----------------|
| `network.tf` | VPC, two public and two private subnets, internet gateway, the tasks' security group |
| `registry.tf` | ECR repositories `ledgerlens/{api,worker,mcp,web}`, the GitHub OIDC provider, the `build.yml` role |
| `secrets.tf` | Secrets Manager secret `ledgerlens-<env>/app` (JSON) with the app's keys |
| `database.tf` | RDS PostgreSQL 16 (private, encrypted, TLS only) and the `ledgerlens-<env>/database-url` secret |
| `queue.tf` | ElastiCache Redis, one node: the ingestion queue |
| `storage.tf` | EFS with four access points: uploads, BM25 parameters, the Hugging Face cache, the anomaly model |
| `iam.tf` | Task execution role (pull, secrets, logs) and task role (Bedrock, EFS, ECS Exec) |
| `ecs.tf` | ECS cluster, Service Connect names, task definitions for api, worker, mcp, web and the one-off `init`, the four services |
| `github_deploy.tf` | The role `deploy-aws.yml` assumes (GitHub OIDC, `production` environment only) |
| `budget.tf` | Monthly AWS budget (whole account) with email alerts at 50/80/100 % and on a forecast overrun |
| `alb.tf` | Load balancer (admits CloudFront only) and the CloudFront distribution: the public HTTPS address |

```
browser --HTTPS--> CloudFront --> ALB --> web :3000 --> api :8000 --> mcp :8765
                                   \--> api /health      (Service Connect names)
api, worker, mcp --> RDS Postgres, ElastiCache Redis, EFS     (private subnets)
                 --> Pinecone, Neo4j AuraDB, Bedrock / LLM APIs (internet)
```

## Before the first apply

1. Decide who gets the budget alerts (`budget_alert_emails`, required). The first apply
   creates the budget before anything that costs money.
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

The first apply runs from a laptop with administrator credentials (it creates the roles CI
uses). The images have to exist before the services can start, so it goes in three steps:

1. The budget, the registry and the build role only (`image_tag` is not used yet):

   ```sh
   cd infra/aws
   terraform init -backend-config=backend.hcl
   terraform apply -var image_tag=none -target=aws_budgets_budget.monthly \
     -target=aws_ecr_lifecycle_policy.service -target=aws_iam_role_policy.build
   ```

2. Set the repository variables `AWS_REGION` and `AWS_BUILD_ROLE_ARN`
   (`terraform output build_role_arn`). Then run **Build images** (`build.yml`) on main. It
   pushes `ledgerlens/<service>:<sha>`, and from now on it runs on every push to main.
3. Everything else, for that commit:

   ```sh
   terraform plan -var image_tag=<sha> -out=tfplan
   terraform apply tfplan
   terraform output url
   ```

   Then fill in the third-party keys (next section), run `init` once, and restart the
   services:

   ```sh
   CLUSTER=$(terraform output -raw cluster_name)
   NET=$(terraform output -json task_network)
   aws ecs run-task --cluster "$CLUSTER" --launch-type FARGATE \
     --task-definition "$(terraform output -json task_definition_arns | jq -r .init)" \
     --network-configuration "awsvpcConfiguration={subnets=[$(jq -r '.subnets | join(",")' <<< "$NET")],securityGroups=[$(jq -r '.security_groups[0]' <<< "$NET")],assignPublicIp=ENABLED}"
   for s in api worker mcp web; do
     aws ecs update-service --cluster "$CLUSTER" --service "$s" --force-new-deployment
   done
   ```

Services ignore task-definition changes once they exist. A new `image_tag` (or a changed
environment variable) registers new revisions; a deployment runs `init` with them, then
moves the services over, so migrations always come first.

## Deployments

After the first apply, every deployment goes through **Deploy (AWS)**
(`.github/workflows/deploy-aws.yml`): push a tag `v*`, or run it by hand on main. It
deploys that commit's images; nothing is rebuilt. Set up once:

- a GitHub environment `production` (Settings → Environments). Add required reviewers or a
  `v*` tag rule there to gate deploys; the deploy role trusts only jobs in it.
- repository variables `AWS_DEPLOY_ROLE_ARN` (`terraform output deploy_role_arn`) and
  `TF_STATE_BUCKET`, and `TERRAFORM_TFVARS`: the contents of a `terraform.tfvars` with the
  non-secret settings, at least `budget_alert_emails`.

The deploy role can change everything this configuration manages: PowerUserAccess plus IAM
on `ledgerlens-*` roles. Treat the `production` environment as the gate.

## Neo4j AuraDB and Pinecone

Two stores are managed by their vendors rather than AWS. The tasks reach them over TLS
through the internet gateway.

- **Neo4j AuraDB**: create an instance at console.neo4j.io (AuraDB Free is enough for the
  demo) and set `neo4j_uri` (`neo4j+s://<id>.databases.neo4j.io`, required) and, if it
  differs, `neo4j_user`. Put its password in the app secret as `NEO4J_PASSWORD`. Free
  instances pause after a few days without queries: resume them in the console, or the
  api's `/health` reports `neo4j` down.
- **Pinecone serverless**: put the API key in the app secret as `PINECONE_API_KEY`. `init`
  creates the index (`pinecone_index`) on `aws` in `pinecone_region`, which defaults to
  `region`. One index serves every environment: each writes and reads its own namespace,
  so prod tasks use `prod` and an eval run never touches it.

## Costs and logs

`monthly_budget_usd` sets an account-wide AWS Budget. The default is $200, against an
estimated $160 a month with everything running (ADR-0002). Task logs go to CloudWatch, one
group per task, and are kept for `log_retention_days` (default 14).

To pause the demo, set `service_desired_count` to 0 for the services you do not need, then
deploy. The database, Redis and the load balancer (with its two public IPs) keep costing about
$50 a month.

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

Terraform ignores later versions, so `apply` never resets them. Running tasks read secrets
at start: restart the services after a change (`aws ecs update-service --force-new-deployment`).

`DATABASE_URL` has a secret of its own, written by Terraform. The database password is never
in the state (write-only arguments); to rotate it, bump `db_password_version` and apply.

## Checks without an AWS account

```sh
terraform fmt -check -recursive
terraform init -backend=false
terraform validate
terraform test        # plans everything against mocked providers (tests/plan.tftest.hcl)
```

CI's `terraform` job runs the same four commands when anything under `infra/aws/` changes.
