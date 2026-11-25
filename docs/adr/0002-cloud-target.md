# ADR-0002 · Deploy to AWS (ECS Fargate + RDS + Bedrock)

- **Status:** Accepted
- **Date:** 2026-11-25
- **Context links:** [Layer 10 · Deployment](../layers/10-deployment.md), [`infra/aws/`](../../infra/aws/README.md)

## Context
The roadmap leaves two deployment paths open: AWS (ECS Fargate + RDS + Bedrock) or GCP
(Cloud Run + Cloud SQL + Gemini). The live demo runs on one of them, to keep the cost to a
single bill. The other stays documentation.

What the deployment has to carry:

- **Four long-running services**: `web`, `api`, `mcp`, and the ingestion `worker`, an arq
  consumer that holds a Redis connection all the time. Plus a one-off `init` task
  (migrations, Pinecone index, Neo4j schema) that must run before each rollout.
- **Shared files**: the API stores uploads and the worker reads them, and the worker and
  mcp share the embedding model cache, BM25 parameters and the anomaly model. In compose
  these are volumes.
- **Long responses**: `/chat` streams for up to `CHAT_TIMEOUT_S` (180 s), and the agents
  keep an MCP session open to the mcp service.
- **Stores**: PostgreSQL (managed), Redis (the queue). Pinecone serverless and Neo4j
  AuraDB are managed by their vendors and work from either cloud.
- **LLMs**: the router already has a Bedrock adapter (`bedrock/...` over the Converse API).
  It authenticates through the standard AWS credential chain, so in the cloud it needs no
  stored key.
- **No stored cloud credentials in CI**, and a budget small enough for a portfolio demo.

## Decision
Deploy to **AWS** with Terraform in [`infra/aws/`](../../infra/aws/README.md):

| Concern | Choice |
|---------|--------|
| Images | ECR, one repository per service, immutable tags = commit SHA (`build.yml`) |
| Compute | ECS Fargate: services for web, api, mcp, worker, and an `init` task definition run before rollouts |
| Service names | ECS Service Connect (`http://api:8000`, `http://mcp:8765/mcp`, as in compose) in TCP mode, so no proxy request timeouts |
| Public entry | CloudFront (HTTPS on `*.cloudfront.net`, no domain needed) → ALB that admits only CloudFront's origin-facing addresses |
| Postgres | RDS PostgreSQL 16, `db.t4g.micro`, private subnets, TLS enforced, password never in Terraform state (write-only arguments) |
| Queue | ElastiCache Redis, one `cache.t4g.micro` node |
| Shared files | EFS with one access point per compose volume |
| Secrets | Secrets Manager, injected by ECS as environment variables |
| LLM in cloud | Bedrock via the task role (no key); API-key providers still work from the secret |
| CI auth | GitHub OIDC → IAM roles; the build role only trusts `main` |
| Network | Tasks in public subnets with public IPs and a closed security group; stores in private subnets; **no NAT gateway** |

## Consequences

**Positive**
- Bedrock gives the `smart` alias a model with no API key to manage or leak: the task's IAM
  role is the credential.
- The worker stays a plain long-running consumer, unchanged. On a scale-to-zero platform it
  would have to become a push-triggered job.
- EFS keeps the code path from compose: the same paths, the same uid. The app needs no
  object-storage code for the demo.
- One public address with HTTPS from day one, which the dashboard's Secure session cookie
  needs, and no certificate or domain to buy.
- Everything is reproducible from `terraform apply`. CI checks the configuration on every
  PR with `terraform test` against mocked providers, so it needs no account.

**Negative / mitigations**
| Trade-off | Mitigation |
|-----------|-----------|
| Fargate does not scale to zero: about **$160/month** with every service up (estimate below) | `service_desired_count`: stop the worker (or everything) between demos; budget alerts set before the first apply |
| Public IPs on tasks instead of a NAT gateway | Security group admits only the ALB and other tasks; a NAT gateway would cost more than all the compute |
| One Redis node, no snapshots: a node failure drops queued jobs | The queue is transient; uploads are kept on EFS and can be re-sent |
| EFS is slower than local disk for the model cache | Read once at start; the embedding model stays in memory afterwards |
| CloudFront waits at most 60 s between bytes from the origin | Chat streams events as the agents work; long silent tool calls are bounded by the API's own timeouts |
| Two IaC paths in the docs, one implemented | `infra/gcp/` stays empty; the mapping in Layer 10 shows what it would take |

### Estimated monthly cost (us-east-1, on-demand list prices, everything on 24/7)

| Item | Size | ≈ USD/month |
|------|------|------:|
| Fargate api | 0.5 vCPU, 1 GB | 18 |
| Fargate worker | 1 vCPU, 4 GB | 43 |
| Fargate mcp | 0.5 vCPU, 2 GB | 21 |
| Fargate web | 0.25 vCPU, 0.5 GB | 9 |
| Application Load Balancer | 1, low traffic | 18 |
| Public IPv4 addresses | 4 tasks + 2 ALB | 22 |
| RDS PostgreSQL | db.t4g.micro + 20 GB gp3 | 14 |
| ElastiCache Redis | cache.t4g.micro | 12 |
| EFS, Secrets Manager, CloudWatch Logs, CloudFront | demo volumes | 3–5 |
| **Total** | | **≈ 160** |

Pinecone (serverless starter), Neo4j AuraDB Free and the LLM calls are billed by their
vendors, per use.

## Alternatives considered
- **GCP: Cloud Run + Cloud SQL + Gemini.** Scales to zero, so it is cheaper when idle. But
  the worker would have to become a push-triggered job, and shared files would need Cloud
  Storage code or Filestore (which costs more than EFS at small sizes). Gemini already works
  from any cloud with an API key.
- **A single EC2 instance running `docker compose`.** Cheapest (one t3.large is about $60)
  and the closest to local. But it is a pet server: no rolling deploys or health-checked
  replacement, patching is manual, and the database lives on the instance.
- **AWS App Runner.** Simple for `web` and `api`, but it has no long-running worker or
  shared file system, so it would still need ECS next to it.
- **A NAT gateway with private-subnet tasks.** The textbook layout. It adds about $33/month
  per AZ for no gain at demo scale, since the security group already closes the tasks.
