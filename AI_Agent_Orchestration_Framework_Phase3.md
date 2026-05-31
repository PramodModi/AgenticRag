# AI Agent Orchestration Framework
## Evolutionary High-Level System Design
### Phase 3 of 3 — Steps 14 to 20: Control Plane, Multi-Tenancy, and Unified View

> **Continuing from Phase 2** (Steps 8–13: Agent Abstraction → Orchestrator)
> This phase adds the control plane — the layer that governs, configures,
> and manages everything the data plane does.
> Data plane = execution. Control plane = governance.

---

## Step 14 — Prompt Registry and Lifecycle

### The problem with Step 13

Prompts are hardcoded inside agent classes. To change a prompt you
deploy code. To test two prompt variants you run two deployments.
Different enterprise tenants need domain-specific language — a legal
firm needs different tone than a retail company. There is no rollback
if a new prompt degrades output quality. There is no audit trail
connecting a bad output to the prompt version that produced it.

**Prompts are configuration, not code. They need versioning, A/B testing, per-tenant override, and instant rollback.**

### System at this step

```
┌───────────────────────────────────────────────────────────────────┐
│                       Prompt Registry                             │
│                                                                   │
│  Key: (agent_name, tenant_id, version)                            │
│                                                                   │
│  planner  · global  · v1  → template_A  · traffic=0%  · RETIRED  │
│  planner  · global  · v2  → template_B  · traffic=90% · STABLE   │
│  planner  · global  · v3  → template_C  · traffic=10% · CANARY   │
│                                                                   │
│  planner  · acme    · v1  → template_X  · traffic=100%· STABLE   │
│  retrieval· acme    · v2  → template_Y  · traffic=100%· STABLE   │
│                                                                   │
│  Operations:                                                      │
│    upsert(agent, tenant, template, version, traffic_split)        │
│    resolve(agent, tenant)  → weighted random over active versions │
│    rollback(agent, tenant, to_version)  → flip traffic instantly  │
│    audit_log(agent, tenant, version, execution_id, result)        │
└────────────────────────────┬──────────────────────────────────────┘
                             │ resolve() called before every LLM call
                             ▼
                    ┌─────────────────┐
                    │   LLM Router    │
                    │  caches result  │
                    │  for 60 seconds │
                    └─────────────────┘
```

### Prompt resolution algorithm

```
llm_router.complete(agent_name="planner", tenant_id="acme"):

  Step 1: Check in-process cache (60s TTL)
    → hit: return cached template
    → miss: proceed

  Step 2: registry.resolve("planner", "acme")
    → tenant-specific prompt exists? YES → return acme · v1 (100%)
    → NO: fall back to global
      weighted random selection over active versions:
        rand < 0.10 → return v3 (canary)
        rand >= 0.10 → return v2 (stable)

  Step 3: Log resolution
    audit_log("planner", "acme", "v1", execution_id, ...)
    → Every output is traceable to exact prompt version

  Step 4: Inject resolved template as system message
    return LLM response
```

### A/B testing lifecycle

```
Phase 1 — author new prompt
  Prompt engineer writes v3 template
  upsert(planner, global, v3, traffic=0.00)  ← DRAFT, no traffic

Phase 2 — canary at 10%
  upsert(planner, global, v3, traffic=0.10)
  upsert(planner, global, v2, traffic=0.90)
  Monitor for 48 hours:
    Compare eval_scores(v2) vs eval_scores(v3)
    Metrics: task_completion_rate, groundedness, latency

Phase 3a — promote (v3 wins)
  upsert(planner, global, v3, traffic=1.00)
  upsert(planner, global, v2, traffic=0.00)
  → Takes effect within 60s (cache TTL)
  → Zero deployment

Phase 3b — rollback (v3 regresses)
  upsert(planner, global, v2, traffic=1.00)
  upsert(planner, global, v3, traffic=0.00)
  → Takes effect within 60s
  → Zero deployment
```

### Version drift problem and fix

```
Problem:
  PlannerAgent uses v3 which changed output from:
    {"plan": [...]}
  to:
    {"steps": [...]}
  RetrievalAgent still reads ctx.outputs["planner"]["plan"]
  → KeyError at runtime — silent breakage

Fix: Pin prompt versions at workflow definition time
  WorkflowNode:
    agent:          "planner"
    pinned_prompt:  null   ← null = always use current active version
    OR
    pinned_prompt:  "v2"   ← always use v2 regardless of registry state

  Default (null): workflow automatically benefits from prompt improvements
  Pinned: stability-critical workflows are insulated from version changes
  Deprecated prompts retained in registry until all pinned workflows complete
```

### Design decisions

| Decision | Rationale |
|---|---|
| Weighted random for A/B, not deterministic split | Hash-based split (e.g. by tenant_id) makes one tenant always see canary — random gives true statistical sampling across all tenants |
| Resolve at call time, not at startup | Prompt changes take effect within 60s without restart |
| 60s in-process cache | Avoids registry lookup on every LLM call; stale window is acceptable for prompt resolution |
| Audit log every resolution | Bad output → trace execution_id → find prompt version → reproduce exact conditions |
| Per-tenant override wins over global | Enterprise customization without code forks or separate deployments |

### Pros

- Prompt changes without deployment — engineers ship faster, prompt engineers work independently
- Rollback in seconds — flip traffic_split, effective within 60s
- A/B testing makes prompt improvement data-driven, not intuition-driven
- Per-tenant customization at the registry layer — zero code forks
- Every bad output is traceable to an exact prompt version

### Cons

- Registry lookup adds a network hop per LLM call — mitigated by 60s cache
- Version drift if prompt schema changes without coordinating downstream agents
- Governance overhead — who approves a prompt before it reaches production?
  Regulated industries need an approval gate built on top of the registry

### Talking points

> "The audit log is what makes this production-grade. When a tenant
> files a complaint about a wrong answer, you pull the execution trace,
> find the prompt version that was resolved, and reproduce the exact
> conditions that produced the bad output. Without version tracking,
> that investigation is impossible — you have no idea what prompt ran."

> "For regulated industries, the prompt registry needs an approval gate.
> A new prompt version cannot be set above 0% traffic without a
> compliance review sign-off. That approval gate is a workflow built
> on top of the registry — the registry itself stays simple."

---

## Step 15 — Centralized Policy Engine

### The problem with Step 14

Policy and security rules are scattered:
- Tool RBAC lives in `ToolExecutor`
- Content filtering lives in the output guardrail
- Cost caps live in `ObservabilityService`
- Prompt access lives in the `PromptRegistry`
- Compliance requirements live nowhere consistently

When a GDPR requirement arrives, you find every place it touches and
update each one. That is fragile, auditably incomplete, and slow.
When a tenant is suspended, you need to reject their requests everywhere
simultaneously — but "everywhere" has no single address.

**Every layer should ask one place: "is this action allowed?"**

### System at this step

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Policy Engine                                │
│                                                                      │
│  ┌──────────────────┐  ┌─────────────────┐  ┌──────────────────┐   │
│  │  Access policy   │  │  Content policy │  │  Cost policy     │   │
│  │  agent whitelist │  │  PII detection  │  │  token budgets   │   │
│  │  tool whitelist  │  │  toxicity score │  │  cost per call   │   │
│  │  prompt access   │  │  topic blocklist│  │  monthly cap     │   │
│  └──────────────────┘  └─────────────────┘  └──────────────────┘   │
│                                                                      │
│  ┌──────────────────┐  ┌─────────────────────────────────────────┐  │
│  │  Rate policy     │  │  Compliance policy                      │  │
│  │  calls/sec       │  │  GDPR · HIPAA · SOX                     │  │
│  │  burst limits    │  │  data residency · audit logging         │  │
│  │  quota windows   │  │  right-to-erasure · retention rules     │  │
│  └──────────────────┘  └─────────────────────────────────────────┘  │
│                                                                      │
│  evaluate(PolicyRequest) → PolicyDecision                            │
│  { allow | deny | redact }                                           │
│  Every decision logged with audit_ref                                │
└──────────────────────────────────────────────────────────────────────┘
         ▲               ▲               ▲               ▲
         │               │               │               │
   API Gateway     Tool Executor    Agent Guardrail   LLM Router
```

### PolicyRequest and PolicyDecision

```
PolicyRequest:
  caller_id:    str    # tenant_id or agent_name
  action:       str    # "invoke_agent" | "call_tool" | "llm_complete"
                       # "read_memory" | "write_memory" | "emit_output"
  resource:     str    # agent_name | tool_name | memory_key
  context:      Dict   # payload sample, token count, content snippet
  execution_id: str    # for audit correlation

PolicyDecision:
  verdict:   "allow" | "deny" | "redact"
  reason:    str          # human-readable, always logged
  redaction: List[str]    # keys to strip if verdict == "redact"
  audit_ref: str          # unique ID for compliance audit trail

Evaluation order (stops at first deny):
  1. Access policy    — is this action on this resource allowed for this caller?
  2. Rate policy      — is this caller within rate limits right now?
  3. Content policy   — does the payload contain blocked content?
  4. Cost policy      — would this action exceed cost/token budget?
  5. Compliance       — does this action meet regulatory requirements?
```

### Suspension gate — why centralized policy is the right place

```
When tenant is suspended (non-payment, policy violation):

  WRONG — update every service individually:
    API Gateway: add tenant_id to blocklist
    Orchestrator: add tenant_id to reject list
    Tool Executor: add tenant_id to RBAC deny
    → 3 code changes, 3 deployments, race condition window

  CORRECT — update one field in Policy Engine:
    policy_engine.set_tenant_status("acme", SUSPENDED)
    → All layers call evaluate() → access policy → deny
    → Takes effect immediately, everywhere, atomically
    → Reactivation: set_tenant_status("acme", ACTIVE)
    → Takes effect within cache TTL (60s)
```

### Fail-open vs fail-closed

```
If Policy Engine is unavailable:

  Fail-closed (safety wins):
    All actions denied until engine recovers
    Use for: HIPAA, SOX, government — no output is better than a bad one

  Fail-open (availability wins):
    Actions proceed, skip policy evaluation
    Use for: consumer products — a brief unguarded window is acceptable

  Recommended hybrid:
    Cache last-known PolicyDecisions in-process (5 min TTL)
    Serve from cache during outage
    Alert immediately — bounded stale window, platform stays up
    Log all cache-served decisions for post-incident audit
```

### Design decisions

| Decision | Rationale |
|---|---|
| Single evaluate() call | Policy logic never leaks into callers — each layer makes one call, does not interpret rules |
| "Redact" as a verdict | Some outputs are legal but need PII stripped — deny is too blunt for these cases |
| Rules stored in DB, not code | Compliance teams update rules without deployments |
| Audit ref on every decision | Compliance audits reconstruct every policy evaluation for any execution |
| Policy Engine governs suspension | One field change propagates everywhere instantly — no multi-service updates |

### Pros

- Single enforcement point — add a compliance rule once, effective everywhere
- Suspension is instant via one field change — no multi-service coordination
- Redact verdict enables PII stripping without hard rejection
- Policy-as-data — compliance teams own rules without engineering involvement
- Full audit trail by construction

### Cons

- Policy Engine is now a critical dependency — must be multi-instance, highly available
- Latency on every action — target < 5ms with in-process caching
- Fail-open/fail-closed is a product decision — must be explicit and per-tenant

### Talking points

> "Policy-as-data is the architectural principle here. When GDPR
> article 17 requires a specific user's data be excluded from future
> responses, a compliance officer updates a database record.
> No code change, no deployment, effective within one cache TTL.
> That is what it means for governance to be a first-class citizen
> in the architecture — not an afterthought bolted onto services."

---

## Step 16 — Single Tenant to Multi-Tenant

### The problem with Step 15

The system was designed for one tenant. A second tenant arrives.
Now config, memory, compute, tools, prompts, policies, and cost
attribution must all be isolated. Multi-tenancy is not one feature.
It is a property that must be enforced simultaneously at every layer.

### Tenant Config Service

```
┌──────────────────────────────────────────────────────────────────┐
│                    Tenant Config Service                         │
│                                                                  │
│  TenantConfig {                                                  │
│    tenant_id:          "acme-corp"                               │
│    display_name:       "Acme Corporation"                        │
│    tier:               "enterprise"  # free | pro | enterprise   │
│    status:             "active"      # active | suspended | ...  │
│                                                                  │
│    enabled_agents:     ["planner","retrieval","summarizer"]      │
│    enabled_tools:      ["jira","slack","sql-readonly"]           │
│                                                                  │
│    llm_overrides:      {primary: "gpt-4o-mini"}                  │
│    prompt_overrides:   {planner: "acme-planner-v1"}              │
│                                                                  │
│    max_concurrent_wf:  5                                         │
│    monthly_token_budget:   5_000_000                             │
│    monthly_cost_usd_budget: 500.00                               │
│    memory_retention_days:  90                                    │
│    data_residency:     "eu-west-1"                               │
│                                                                  │
│    compliance_flags:   ["gdpr", "sox"]                           │
│    contact_email:      "admin@acme.com"                          │
│  }                                                               │
└──────────────────────────────────────────────────────────────────┘
```

### Isolation enforcement at every layer

```
Layer                  │ Isolation mechanism
───────────────────────┼────────────────────────────────────────────────────
API Gateway            │ JWT contains tenant_id — every request carries identity
Orchestrator           │ asyncio.Semaphore(max_concurrent_wf) per tenant
Tenant Config Service  │ validate_agent() and validate_tool() before every node
Policy Engine          │ All rules scoped by tenant_id in PolicyRequest
Tool Executor          │ RBAC: enabled_tools whitelist from TenantConfig
Tool Secrets           │ Vault paths: vault/tenants/{tenant_id}/{tool}_token
Memory — Redis         │ Key prefix: "{tenant_id}:{execution_id}:{key}"
Memory — Postgres      │ tenant_id column + WHERE clause + Row-Level Security
Memory — Vector DB     │ namespace = tenant_id (physical index partition)
LLM Router             │ Resolves LLM config from tenant llm_overrides first
Prompt Registry        │ resolve(agent, tenant_id) — tenant override wins
Observability          │ All spans tagged tenant_id — dashboards filterable
Billing                │ Token + cost tracking per tenant_id per window
```

### Defense in depth at Postgres

```
Three independent layers at the database:

  Layer 1 — Application query filter:
    SELECT * FROM memory
    WHERE tenant_id = %(tenant_id)s AND user_id = %(user_id)s
    → Developer adds this correctly in 99% of cases

  Layer 2 — Postgres Row-Level Security (RLS):
    CREATE POLICY tenant_isolation ON memory
    USING (tenant_id = current_setting('app.tenant_id'))
    → Even if Layer 1 is missing, RLS blocks cross-tenant reads
    → A SQL injection cannot cross tenant boundary

  Layer 3 — Vault credential scoping:
    Tenant A's DB credentials only have access to tenant_a schema
    Even if Layers 1 and 2 fail, credentials limit blast radius
```

### Deployment models — one codebase, three configurations

```
Model A — Shared infrastructure (default for most tenants)
  All tenants share:
    Orchestrator pool, Agent worker pool, Tool Execution Service
    Kafka cluster, Memory backends (isolated by key/namespace)
  Isolated by: tenant_id in every request, query, key
  Cost: lowest   Isolation: logical

Model B — Dedicated worker pool (for enterprise SLA)
  Tenant gets:
    Dedicated Kubernetes namespace
    Dedicated agent worker pods (guaranteed compute)
    Dedicated Redis namespace (or separate Redis cluster)
  Shared: Orchestrator, Kafka, Postgres (with RLS)
  Cost: medium   Isolation: compute + memory

Model C — Fully dedicated VPC (for regulated industries)
  Tenant gets their own:
    VPC / network boundary — no shared network plane
    All services — orchestrator, workers, Kafka, DBs
    Data residency guarantee — data never leaves specified region
  Cost: highest   Isolation: physical
  Use for: healthcare (HIPAA), government, data sovereignty requirements
```

### Design decisions

| Decision | Rationale |
|---|---|
| Whitelist for agents and tools, not blacklist | A new tool registered globally is blocked from all tenants until explicitly enabled — safe by default |
| Tenant override wins at every config layer | Enterprise customization without code forks |
| Data residency as a TenantConfig flag | Orchestrator routes memory writes to correct regional backend — no special-case code |
| RLS as second line of defense | Application-level tenant_id filtering is human; RLS is mechanical — defense in depth |
| Three deployment models, one codebase | Same code serves free-tier and fully-dedicated enterprise tenants |

### Pros

- Defense in depth — isolation enforced at 12+ independent layers
- Config-driven customization — no code forks per tenant
- Data residency compliance — tenant data stays in the right region by configuration

### Cons

- Tenant Config Service is now a critical dependency — must be cached and highly available
- Testing multi-tenancy requires automated cross-tenant isolation tests for every new feature
- Data residency adds routing complexity to the Memory Service

### Talking points

> "The two-layer defense at Postgres is the answer to 'what happens
> if a developer forgets to add WHERE tenant_id'.
> Layer 1 is the application query filter — human-written.
> Layer 2 is Postgres Row-Level Security — mechanical, always on.
> Even if both fail, Vault credential scoping means the DB connection
> itself only has access to that tenant's schema.
> Three independent layers. One can fail. Two can fail. Three failing
> simultaneously is a coordinated attack, not an accident."

---

## Step 17 — Tenant Lifecycle Management

### The problem with Step 16

Tenants are static objects. In reality they have a lifecycle:
created, trialing, active, suspended, offboarding, deleted.
Each transition has operational consequences.

- Suspend: in-flight workflows must be drained or cancelled. New requests rejected.
- Offboard: all tenant data must be purged — this is a GDPR compliance requirement.
- Upgrade: new agents, tools, and higher quotas available immediately.
- Downgrade: excess capacity revoked without breaking active workflows.

### Tenant lifecycle state machine

```
┌──────────────────────────────────────────────────────────────────┐
│                    Tenant Lifecycle States                       │
│                                                                  │
│   PENDING ──────────────────────────────► ACTIVE                │
│      │           (provision complete)        │                   │
│      │                                       │                   │
│   (failed)                         ┌─────────┤                   │
│      │                             │         │                   │
│      ▼                         (suspend)  (upgrade/downgrade)    │
│   FAILED                           │         │                   │
│                                    ▼         │                   │
│                              SUSPENDED ◄─────┘                   │
│                                    │                             │
│                         (reactivate│or offboard)                 │
│                                    │                             │
│                          ┌─────────┴──────────┐                  │
│                          ▼                    ▼                  │
│                       ACTIVE           OFFBOARDING               │
│                                              │                   │
│                                        (purge complete)          │
│                                              │                   │
│                                           DELETED                │
└──────────────────────────────────────────────────────────────────┘
```

### State transition operations

```
PENDING → ACTIVE (provisioning pipeline):
  1. Create TenantConfig in Tenant Config Service
  2. Provision Vault secret paths: vault/tenants/{tenant_id}/
  3. Initialize memory namespaces (Redis prefix, Postgres schema, VectorDB namespace)
  4. Set default prompt versions in Prompt Registry
  5. Set default policy rules in Policy Engine
  6. Emit tenant.activated → Kafka → billing starts
  7. Send welcome email to contact_email

ACTIVE → SUSPENDED:
  1. policy_engine.set_tenant_status(SUSPENDED)
     → All evaluate() calls return deny immediately
     → API Gateway, Orchestrator, Tool Executor all reject in one step
  2. Orchestrator: drain in-flight workflows
     Option A (non-payment): graceful drain — complete current tier, stop before next
     Option B (policy breach): immediate cancel — all in-flight → CANCELLED
  3. Emit tenant.suspended → Kafka → billing paused → alert to tenant admin

SUSPENDED → ACTIVE (reactivation):
  1. policy_engine.set_tenant_status(ACTIVE)
     → All evaluate() calls resume normal evaluation
  2. Orchestrator resumes SUSPENDED_PENDING workflows from checkpoint
  3. Emit tenant.reactivated → billing resumes

ACTIVE/SUSPENDED → OFFBOARDING:
  1. API Gateway: reject all requests
  2. Orchestrator: cancel all in-flight workflows
  3. Schedule async data purge job:
       Redis:    DELETE keys matching "{tenant_id}:*"
       Postgres: DELETE FROM * WHERE tenant_id = ?  (or DROP SCHEMA)
       VectorDB: delete_namespace(tenant_id)
       Vault:    revoke all secrets at vault/tenants/{tenant_id}/
       Logs:     archive to cold storage, purge after retention period
  4. Emit tenant.offboarded → compliance audit record created
  5. Status → DELETED
     TenantConfig retained (with purge timestamps) for audit
     All data purged

Why retain config after DELETED:
  Compliance (SOX, GDPR) requires proof that data was purged.
  The TenantConfig row with status=DELETED and purge_completed_at
  is the compliance artifact. The data is gone; the proof of deletion stays.
```

### In-flight workflow handling during suspension

```
Graceful drain (default for non-payment):
  Current tier completes → checkpoint saved
  Workflow status = SUSPENDED_PENDING
  Next tier does not start — orchestrator checks tenant status before dispatch
  On reactivation: resume from checkpoint — no work lost
  Pro: no work lost, good UX   Con: workflows hang until reactivated

Immediate cancel (for policy violations):
  Orchestrator receives tenant.suspended event from Kafka
  Sets all running executions to status = CANCELLED
  Emits execution.cancelled events
  Pro: clean state immediately   Con: work lost, tenant must resubmit
  Use when: reactivation is unlikely (policy breach, fraud)

Configuration per suspension_reason:
  non_payment     → graceful_drain
  policy_violation → immediate_cancel
  tenant_request  → graceful_drain
  security_breach  → immediate_cancel
```

### Design decisions

| Decision | Rationale |
|---|---|
| Policy Engine as suspension gate | One field change propagates everywhere — no per-service suspension logic |
| Data purge as async job | Purging terabytes synchronously blocks the offboarding API — schedule durable job, track progress |
| Retain config after data purge | Audit requirement — proof of deletion must outlive the data itself |
| Separate suspension from purge | Suspended tenant may reactivate; purge only on explicit offboarding |

### Pros

- GDPR right-to-erasure compliance — purge is automated, auditable, and complete across all backends
- Suspension is instant — Policy Engine gate requires no per-service code
- Graceful drain preserves work for recoverable suspensions
- Provisioning pipeline ensures tenant isolation is configured before first request

### Cons

- Data purge across Redis, Postgres, VectorDB, Vault, and logs must be resumable if any step fails
- In-flight workflow handling adds orchestrator complexity — must subscribe to tenant lifecycle events
- DELETED tenants accumulate config records — needs periodic archival to cold storage

### Talking points

> "The GDPR right-to-erasure workflow is the most legally consequential
> piece of this design. The purge job runs across five systems.
> Each step is logged. The TenantConfig row with status=DELETED,
> purge_started_at, purge_completed_at, and a list of systems purged
> is what you hand to a regulator. The actual data is gone;
> the proof of deletion stays."

> "Using the Policy Engine as the suspension gate is the elegant part.
> I do not add suspension checks to the API Gateway, the Orchestrator,
> the Tool Executor, and the LLM Router independently.
> I set one field in the Policy Engine and it propagates everywhere
> those services call evaluate(). That is the value of centralized policy."

---

## Step 18 — Rate Limiting and Quota Management

### The problem with Step 17

One tenant submitting thousands of concurrent workflows can:
- Exhaust the orchestrator queue, delaying all other tenants (infrastructure threat)
- Consume their entire monthly token budget in an hour (commercial threat)

These are two different problems with different time windows and different responses.
Rate limiting protects infrastructure per second.
Quota management enforces commercial limits per month.
Conflating them makes both worse.

### Two-layer model

```
Layer 1 — Rate Limiter (infrastructure protection)
  What:    How many requests per second/minute is this tenant making?
  Where:   API Gateway — before anything else runs
  Window:  1 second, 1 minute
  Response: HTTP 429 with Retry-After header
  Storage: Redis atomic counters (shared across API Gateway instances)
  Goal:    Protect infrastructure from burst; noisy-neighbour isolation

Layer 2 — Quota Manager (commercial enforcement)
  What:    How much of their monthly allocation has this tenant consumed?
  Where:   Policy Engine — before every billable action
  Window:  Daily, monthly
  Tracked: API calls, LLM tokens, tool invocations, workflow executions
  Response: PolicyDecision{deny, reason=quota_exceeded}
  Storage: Redis (fast counters) + Postgres (billing source of truth)
  Goal:    Enforce commercial limits; protect cost; drive upgrade conversations
```

### Rate limiter — token bucket at API Gateway

```
Token bucket per tenant, rate R tokens/sec, capacity B tokens:

  On each request:
    MULTI (Redis atomic transaction)
      tokens = GET "rl:{tenant_id}:tokens"
      if tokens >= 1:
        DECR "rl:{tenant_id}:tokens"
        EXPIRE "rl:{tenant_id}:tokens" {window_sec}
        → allow
      else:
        ttl = TTL "rl:{tenant_id}:tokens"
        → deny, return Retry-After: {ttl}
    EXEC

Bucket config per tier:
  free:       capacity=10,  rate=10 req/sec,  burst=1.5x for 5s
  pro:        capacity=50,  rate=50 req/sec,  burst=2x   for 10s
  enterprise: capacity=200, rate=200 req/sec, burst=3x   for 30s

Why token bucket over fixed window:
  Fixed window: 100 req allowed at 11:59:59, another 100 at 12:00:00
  → 200 req in 2 seconds — looks fine to fixed window, spikes infrastructure
  Token bucket: bucket fills at rate R; burst limited to capacity B
  → Natural smoothing — burst is absorbed up to bucket capacity
  → No boundary spike

Why Redis (not in-memory):
  Multiple API Gateway instances must share rate limit state
  Redis atomic operations prevent race conditions
  TTL-based keys auto-expire — no cleanup jobs
```

### Quota manager — monthly allocation tracking

```
┌────────────────────────────────────────────────────────────────┐
│                      Quota Manager                             │
│                                                                │
│  Tracks per tenant per time window:                            │
│    workflow_executions_today:       current / daily_limit      │
│    llm_tokens_this_month:           current / monthly_limit    │
│    tool_invocations_this_month:     current / monthly_limit    │
│    cost_usd_this_month:             current / monthly_limit    │
│                                                                │
│  consume(tenant_id, metric, amount) → ok | quota_exceeded     │
│  get_usage(tenant_id) → UsageSummary                          │
│  set_limit(tenant_id, metric, limit) → admin operation        │
│  reset(tenant_id, window) → called at window boundary         │
│                                                                │
│  Alert thresholds (configurable per tenant):                   │
│    70% of monthly token budget  → in-app warning              │
│    80%                          → email to tenant admin        │
│    95%                          → email + urgent banner        │
│    100%                         → enforce deny                 │
└───────────────────────────────────┬────────────────────────────┘
                                    │ stores in
                    ┌───────────────┼──────────────┐
                    ▼               │              ▼
             Redis counters         │         Postgres
             (enforcement path)     │         (billing source of truth)
             INCRBY, TTL            │         usage_events table
             sub-ms                 │         INSERT on every consume()
                                    │         queryable for billing
                                    │         reconciliation
```

### Enforcement points for quota

```
Before workflow submission:
  quota.consume(tenant_id, "workflow_executions", 1)
  rate_limiter.check(tenant_id)
  → Reject at the gate before any work starts

Before LLM call:
  estimated_tokens = len(prompt_tokens) + max_output_tokens
  quota.consume(tenant_id, "llm_tokens", estimated_tokens)
  → Deny if would exceed monthly token budget
  On completion (actual token count known):
  quota.adjust(tenant_id, "llm_tokens", actual - estimated)
  → Correct for estimation error

Before tool invocation:
  quota.consume(tenant_id, "tool_invocations", 1)
  → Deny if over monthly tool call limit
```

### Graceful degradation on quota breach mid-workflow

```
Soft stop (for pro and enterprise tiers):
  Current LLM call completes
  Workflow status → QUOTA_PAUSED
  Next tier does not start
  On quota reset (midnight / next billing cycle):
    orchestrator resumes from checkpoint
  Pro: no work lost, good UX for paying customers
  Con: workflows stall until reset

Hard stop (for free tier):
  Current agent fails with QuotaBudgetExceeded
  Workflow transitions to FAILED
  Tenant receives notification with usage summary
  Pro: strict enforcement   Con: partial work lost

Enterprise burst allowance:
  Negotiate overage pricing: X cents per 1000 tokens over limit
  quota.consume() returns ok with overage_flag=true
  Overage billed at end of cycle
  Pro: no workflow interruption   Con: unexpected bill if not monitored
```

### Quota reset and billing reconciliation

```
Monthly reset (00:00 UTC on the 1st):
  1. Snapshot final usage from Redis → Postgres (billing record row)
  2. Reset Redis counters for new month (RESET keys or SET to 0)
  3. Resume all QUOTA_PAUSED workflows from checkpoint
  4. Generate invoice line items from Postgres usage_events
  5. Send usage summary email to tenant admins
  6. Emit tenant.monthly_reset → billing service

Staggered reset — prevent queue spike:
  All tenants resetting simultaneously = all QUOTA_PAUSED workflows
  resuming simultaneously = orchestrator queue spike
  Stagger resets over first 5 minutes of the month (hash tenant_id to minute)

Two-store rationale:
  Redis: enforcement in hot path (< 1ms per consume() call)
  Postgres: billing reconciliation, usage reports, audit trail
  If Redis fails: fail-open on quota (consume() returns ok)
                  Redis and Postgres reconcile on next job run
```

### Design decisions

| Decision | Rationale |
|---|---|
| Rate limit at API Gateway, quota at Policy Engine | Different purposes, different time windows, different responses — do not conflate |
| Token bucket over fixed window | Smooths traffic; handles burst correctly at window boundaries |
| Two-store quota tracking | Redis for enforcement speed; Postgres for billing accuracy — serve different masters |
| Soft stop for paid tiers | Hard stops on paying customers cause churn; soft stop + resume is better UX |
| Alert at 70/80/95% | Turn quota exhaustion into a proactive upgrade conversation, not a sudden failure |

### Pros

- Token bucket prevents boundary spikes — better infrastructure protection
- Two-layer separation — infrastructure and commercial concerns stay clean
- Soft stop prevents work loss for paying tenants
- Early alerts convert a negative event into a sales opportunity
- Redis atomicity prevents race conditions in high-concurrency environments

### Cons

- Redis is now in the critical path for every request — if unavailable, rate limiting must fail-open or fail-closed
- Pre-consumption token estimate is inaccurate — post-call adjustment can push tenant slightly over limit
- Staggered reset complexity — simple but requires correct hash distribution

### Talking points

> "The rate limiter and quota manager solve fundamentally different
> problems. Rate limiting asks 'is this tenant being a noisy neighbour
> right now' — it protects infrastructure on a per-second basis.
> Quota asks 'has this tenant consumed their monthly allocation'
> — it enforces commercial limits. Conflating them into one system
> makes both worse: rate limiting becomes too slow and quota enforcement
> becomes too blunt."

> "The Redis fail-open/fail-closed decision is a product decision
> disguised as a technical one. If the rate limiter is down and you
> fail-closed, every user gets 429 and your platform looks broken.
> If you fail-open, you have no burst protection during the outage.
> For most platforms, fail-open with aggressive alerting is right —
> a rate limiter outage is rare and short; a full-platform outage
> is visible and costly."

---

## Step 19 — Agent and Tool Lifecycle (CI/CD for AI Components)

### The problem with Step 18

New agent versions are deployed by updating the registry — 100% traffic
immediately. A broken v2 serves all traffic before anyone notices.
There is no canary. Workflows that were authored against AgentV1 have
no guarantee V1 still exists when they execute weeks later.
Tool deprecation has no warning system — a sunset tool silently breaks
workflows that depend on it.

**Agents and tools are software. They need the same lifecycle management as software — canary deployments, eval gates, versioned rollouts, and deprecation policies.**

### Agent version lifecycle

```
DRAFT ─────────► CANARY ────────► STABLE ────────► DEPRECATED ─────► RETIRED
   │                │                │                   │               │
Register       1–10% traffic    100% traffic        0% new traffic   Removed from
No traffic     A/B evaluation   Primary version     In-flight safe   registry
Dev/staging    Compare vs       All new workflows   No new workflow  After all
only           STABLE           use this version    definitions may  pinned wf
                                                    reference this   complete

Key rules:
  DRAFT:      Only accessible in staging — safe to break
  CANARY:     Small % of production traffic — monitored closely
  STABLE:     Default for new workflow executions
  DEPRECATED: Traffic=0 but kept for in-flight workflows with pinned_version
  RETIRED:    Removed only after zero pinned-version references remain
```

### Canary traffic splitting in the agent registry

```
AgentRegistry resolves "retrieval-agent":
  versions:
    v1: status=STABLE,  traffic_split=0.90
    v2: status=CANARY,  traffic_split=0.10
    v3: status=DRAFT,   traffic_split=0.00

Resolution at execution time:
  random() < 0.10  →  assign v2 (canary)
  else             →  assign v1 (stable)

ExecutionContext records assigned version:
  ctx.metadata["retrieval_agent_version"] = "v2"

Observability segments by version:
  v1: avg_latency=420ms, error_rate=0.1%, eval_score=0.82
  v2: avg_latency=380ms, error_rate=0.2%, eval_score=0.86
  → v2 shows better latency and quality, slight error uptick
  → Investigate errors → decide promote or rollback

Promote: v2.traffic_split=1.0, v1.traffic_split=0.0  → no restart
Rollback: v1.traffic_split=1.0, v2.traffic_split=0.0 → no restart
Both take effect on next resolution — within seconds
```

### Workflow pinning — protecting in-flight executions

```
WorkflowDefinition.node:
  agent:          "retrieval-agent"
  pinned_version: null    ← null = always resolve current STABLE
  OR
  pinned_version: "v1"    ← always use v1, regardless of registry state

Pinned_version=null (default):
  → Workflow automatically benefits from agent improvements
  → Gets v2 when v2 is promoted to STABLE
  → Correct for most workflows

Pinned_version="v1":
  → Workflow always uses v1 even if v1 is DEPRECATED
  → Correct for long-running or stability-critical workflows
  → v1 stays in registry (DEPRECATED, not RETIRED) until all
    pinned workflows complete or migrate

Retirement safety check before RETIRED:
  SELECT COUNT(*) FROM workflow_executions
  WHERE status IN ('RUNNING', 'PENDING', 'QUOTA_PAUSED', 'SUSPENDED_PENDING')
  AND definition->nodes @> '[{"pinned_version": "v1"}]'
  → If count > 0: cannot retire v1 yet
  → Alert workflow owners to update definitions or complete executions
```

### Deployment pipeline for agent versions

```
Developer pushes new agent version:

  ┌─────────────────────────────────────────────┐
  │  1. Unit tests                              │
  │     agent logic, schema validation          │
  │     tool integration tests                 │
  └───────────────────┬─────────────────────────┘
                      │ pass
  ┌───────────────────▼─────────────────────────┐
  │  2. Automated eval gate                     │
  │     Run against golden dataset              │
  │     Metrics: task_completion, groundedness  │
  │     Gate: must not regress vs STABLE        │
  │     Gate: latency ≤ STABLE_latency × 1.2   │
  │     Fail → block promotion, alert engineer  │
  └───────────────────┬─────────────────────────┘
                      │ pass eval gate
  ┌───────────────────▼─────────────────────────┐
  │  3. Register as DRAFT                       │
  │     Staging environment validation          │
  │     Synthetic tenant data                   │
  │     Human review of sample outputs          │
  └───────────────────┬─────────────────────────┘
                      │ approved
  ┌───────────────────▼─────────────────────────┐
  │  4. Set status=CANARY, traffic_split=0.05   │
  │     Monitor 24 hours:                       │
  │       error_rate vs STABLE                  │
  │       latency_p99 vs STABLE                 │
  │       eval_scores vs STABLE                 │
  └───────────────────┬─────────────────────────┘
                      │ metrics pass thresholds
  ┌───────────────────▼─────────────────────────┐
  │  5. Progressive promotion                   │
  │     0.05 → 0.20 (wait 6h, check metrics)   │
  │     0.20 → 0.50 (wait 6h, check metrics)   │
  │     0.50 → 1.00 (wait 6h, check metrics)   │
  └───────────────────┬─────────────────────────┘
                      │
  ┌───────────────────▼─────────────────────────┐
  │  6. Set status=STABLE, traffic_split=1.0    │
  │     Previous STABLE → DEPRECATED            │
  │     Wait for all pinned workflows to clear  │
  │     Previous STABLE → RETIRED               │
  └─────────────────────────────────────────────┘
```

### Tool lifecycle — same model with shadow mode

```
Tool version lifecycle mirrors agent lifecycle:
  DRAFT → CANARY → STABLE → DEPRECATED → RETIRED

Shadow mode (unique to tools):
  New tool version runs alongside current version with live traffic
  traffic_split=0.10 for real requests (v2 handles real calls)
  shadow_mode=true for remaining 90% (v2 called, results discarded)
  → v2 gets full production load for performance testing — zero risk
  → Compare v1 vs v2 outputs on identical inputs in observability

Tool deprecation notice:
  tool_registry.deprecate(
      name="jira-v1",
      sunset_date="2025-12-31",
      migration_note="Use jira-v2 with OAuth 2.0"
  )
  All agents calling jira-v1 receive ToolDeprecationWarning in logs
  30 days before sunset: escalate to ERROR level
  On sunset date: tool status → RETIRED, calls fail with ToolRetiredError
```

### Design decisions

| Decision | Rationale |
|---|---|
| Eval gate before any production traffic | A version that regresses on task completion never reaches users |
| Progressive canary (5% → 20% → 50% → 100%) | Incremental exposure — catch problems while blast radius is small |
| Workflow pinning at definition time | Long-running workflows must not be broken by version changes mid-execution |
| Shadow mode for tools | Real production load testing with zero customer impact |
| Retirement blocked by pinned references | Cannot delete a version that in-flight executions depend on |

### Pros

- Canary rollout catches regressions before full exposure
- Eval gate is the quality enforcement — not hope, but measurement
- Workflow pinning provides stability guarantees for long-running workflows
- Shadow mode gives real load testing with no risk
- Progressive promotion gives multiple checkpoints to catch delayed failures

### Cons

- Registry resolution is now a weighted random selection — slightly more complex than a direct lookup
- Canary requires segmented observability — must tag all metrics with agent version
- RETIRED status check requires querying in-flight executions — potentially expensive query at scale

### Talking points

> "The eval gate before canary is what separates this from just
> adding traffic splitting. A Senior engineer adds canary.
> A Principal engineer adds the automated quality gate that prevents
> a version from ever reaching canary if it regresses on task
> completion rate. The gate is the discipline that makes the pipeline
> trustworthy — you cannot ship a degraded version by accident."

> "Shadow mode for tools is borrowed directly from ML model deployment.
> You run the new version in parallel on real production traffic,
> compare its outputs to the current version on identical inputs,
> and only promote when outputs are consistently better or equivalent.
> Zero risk, full production load. The same principle applies whether
> you are deploying a new SQL tool or a new LLM."

---

## Step 20 — Unified View: Data Plane and Control Plane

### The complete picture

Every step so far has been motivated by a real problem.
The final step is to see all of it together.

The **data plane** is everything in the execution path — the route a
workflow takes when it runs.

The **control plane** is everything that governs, configures, and
manages the data plane — without being in the hot path of execution.

```
╔══════════════════════════════════════════════════════════════════════════╗
║                           CONTROL PLANE                                  ║
║                                                                          ║
║  ┌────────────────────┐  ┌──────────────────┐  ┌──────────────────────┐ ║
║  │  Tenant Config     │  │  Prompt Registry  │  │   Policy Engine      │ ║
║  │  Service           │  │                  │  │                      │ ║
║  │  TenantConfig      │  │  versioned        │  │  access policy       │ ║
║  │  lifecycle SM      │  │  per-tenant       │  │  content policy      │ ║
║  │  provisioning      │  │  A/B traffic      │  │  cost policy         │ ║
║  │  data purge        │  │  instant rollback │  │  compliance          │ ║
║  │  PENDING→DELETED   │  │  audit log        │  │  suspension gate     │ ║
║  └────────────────────┘  └──────────────────┘  └──────────────────────┘ ║
║                                                                          ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │              Rate Limiter + Quota Manager                        │   ║
║  │  Token bucket (API Gateway) · quota counters · alert thresholds  │   ║
║  │  Redis (enforcement) · Postgres (billing) · monthly reset        │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
║                                                                          ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │              Agent + Tool Lifecycle Registry                     │   ║
║  │  DRAFT→CANARY→STABLE→DEPRECATED→RETIRED                         │   ║
║  │  Eval gate · traffic splitting · shadow mode · workflow pinning  │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════╝
              │ governs                          │ configures
              │ configures                       │ version routing
              ▼                                  ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                            DATA PLANE                                    ║
║                                                                          ║
║  ┌───────────────────────────────────────────────────────────────────┐  ║
║  │  API Gateway                                                      │  ║
║  │  auth · JWT tenant_id · rate limit check · input validation       │  ║
║  └─────────────────────────────┬─────────────────────────────────────┘  ║
║                                │                                         ║
║  ┌─────────────────────────────▼─────────────────────────────────────┐  ║
║  │  Orchestrator (stateless, multi-instance)                         │  ║
║  │  DAG compiler · tier scheduler · state machine · checkpoint       │  ║
║  │  Per-tenant semaphore · event bus publisher                       │  ║
║  └──────────┬──────────────────────────────────┬───────────────────┘   ║
║             │                                  │                         ║
║    ┌────────▼────────┐                ┌────────▼────────┐               ║
║    │   Agent A       │     ···        │    Agent N      │               ║
║    │  input guardrail│                │  input guardrail│               ║
║    │  execute()      │                │  execute()      │               ║
║    │  output guardrail               │  output guardrail               ║
║    │  → all call     │                │  → all call     │               ║
║    │  Policy Engine  │                │  Policy Engine  │               ║
║    └────────┬────────┘                └────────┬────────┘               ║
║             └──────────────┬───────────────────┘                         ║
║                            │                                             ║
║  ┌─────────────────────────▼─────────────────────────────────────────┐  ║
║  │  Tool Execution Service                                           │  ║
║  │  RBAC · sandbox · timeout · retry · cache · sanitize             │  ║
║  └──────────┬──────────────────────────────────┬───────────────────┘   ║
║             │                                  │                         ║
║    SQL/DB · Search · REST · Slack · Jira · MCP · Lambda · gRPC          ║
║                                                                          ║
║  ┌───────────────────────────────────────────────────────────────────┐  ║
║  │  Memory Service (facade, all tenant-namespaced)                   │  ║
║  │  Redis (short-term) · Postgres (long-term) · Vector DB (semantic) │  ║
║  └───────────────────────────────────────────────────────────────────┘  ║
║                                                                          ║
║  ┌───────────────────────────────────────────────────────────────────┐  ║
║  │  LLM Router + Prompt Registry resolution                          │  ║
║  │  Primary → fallback chain · circuit breaker · cost routing        │  ║
║  └───────────────────────────────────────────────────────────────────┘  ║
║                                                                          ║
║  ┌───────────────────────────────────────────────────────────────────┐  ║
║  │  Kafka Event Bus                                                  │  ║
║  │  All events fan-out to: observability · billing · alerting · audit│  ║
║  └───────────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### Control plane → data plane governance map

```
Control Plane Component      │ What it governs in the Data Plane
─────────────────────────────┼──────────────────────────────────────────────
Tenant Config Service        │ Agent/tool whitelist at Orchestrator
                             │ Concurrency semaphore value per tenant
                             │ Memory retention and residency routing
                             │ LLM and prompt overrides in LLM Router
─────────────────────────────┼──────────────────────────────────────────────
Prompt Registry              │ System message injected at LLM Router
                             │ A/B traffic routing between prompt versions
                             │ prompt_version field in Observability spans
─────────────────────────────┼──────────────────────────────────────────────
Policy Engine                │ Every evaluate() call in every layer
                             │ Suspension gate — one field blocks all access
                             │ PII redaction before output reaches user
                             │ Compliance audit_ref in every decision log
─────────────────────────────┼──────────────────────────────────────────────
Rate Limiter                 │ Token bucket at API Gateway (per-second)
                             │ Quota gate at Policy Engine (per-month)
                             │ QUOTA_PAUSED workflow state in Orchestrator
─────────────────────────────┼──────────────────────────────────────────────
Agent + Tool Lifecycle       │ Version resolved by Agent Registry
                             │ Traffic split between canary and stable
                             │ Deprecation warnings in Tool Executor logs
                             │ Pinned version resolution for in-flight wf
```

### 5-minute whiteboard summary

```
Opening (30 seconds):
  "I will design this evolutionarily — starting from user + LLM,
  adding components only when a real problem demands it.
  The architecture has two planes: a data plane that executes
  workflows, and a control plane that governs everything the data
  plane does. I will build both together."

Evolution arc (4 minutes):
  User + LLM                → need real data
  + RAG / single tool       → need multiple tools
  + Multiple tools           → need tool RBAC, sanitization
  + Tool security            → need retry, observability, cache
  + Tool reliability         → need uniform abstraction
  + Tool base class/registry → need vendor independence
  + LLM router + prompts     → LLM + tools + memory = agent unit
  + Agent base class/registry → need agent security
  + Agent guardrails + policy → need typed contracts, no direct calls
  + Agent messages/orchestration → parallel is faster
  + Parallel execution        → agents need persistent memory
  + Memory service (3 tiers) → need coordination + crash recovery
  + Orchestrator + Kafka      → prompts are config not code
  + Prompt registry           → need one governance point
  + Policy engine             → second tenant needs isolation everywhere
  + Multi-tenancy             → tenants have lifecycle states
  + Tenant lifecycle          → need to protect infrastructure and enforce commercial limits
  + Rate limiting + quota     → agents and tools need CI/CD
  + Agent/tool lifecycle

Closing (30 seconds):
  "The data plane executes workflows — agents call tools, read memory,
  invoke the LLM, produce typed messages, orchestrated across
  DAG tiers with crash recovery and full observability.
  The control plane governs all of it — tenant config, prompt registry,
  policy engine, rate limits, quota management, and component lifecycle.
  Every data plane action consults the control plane before proceeding.
  The result is a system that is safe by construction, configurable
  without deployments, auditable end-to-end, and scalable from
  one tenant to thousands."
```

---

## Complete Evolution Summary — All 20 Steps

| Step | Added | Forced by |
|---|---|---|
| 1 | User + LLM | Starting point |
| 2 | Single tool / RAG | LLM has no real-time or private data |
| 3 | Multiple tools | One tool cannot serve all query types |
| 4 | Tool security — RBAC, sanitization, Vault | No control = data breach + prompt injection |
| 5 | Tool observability + retry + cache | No visibility; transient failures break workflows |
| 6 | Tool base class + registry | 20 tools need a uniform contract and plugin architecture |
| 7 | LLM Router + fallback chain + Prompt Registry | Single provider = single point of failure; prompts need versioning |
| 8 | Agent base class + registry + ExecutionContext | LLM + tools + memory = reusable unit needing same treatment as tools |
| 9 | Agent guardrails + centralized policy engine | Agent output is an attack surface; compliance rules scattered |
| 10 | Typed AgentMessage + orchestrator mediation | Direct agent calls = tight coupling, no retry, no observability |
| 11 | Parallel execution — DAG tiers + within-agent | Sequential execution wastes time on independent work |
| 12 | Memory service — three tiers | Stateless agents need three kinds of persistent state |
| 13 | Stateless orchestrator + Kafka + observability | Coordination, crash recovery, decoupled consumers, full audit trail |
| 14 | Prompt registry + lifecycle + A/B testing | Prompts are config not code; need versioning, rollback, per-tenant override |
| 15 | Centralized policy engine | Rules scattered everywhere; need single evaluate() for all governance |
| 16 | Single → multi-tenant + TenantConfig Service | Second tenant requires isolation at every layer simultaneously |
| 17 | Tenant lifecycle management | Tenants onboard, suspend, offboard; data must be purged on GDPR request |
| 18 | Rate limiting (token bucket) + quota management | Burst protection per-second + commercial enforcement per-month |
| 19 | Agent + tool lifecycle — CI/CD for AI | Canary, eval gate, shadow mode, workflow pinning, deprecation policy |
| 20 | Unified data + control plane | Full picture — governance mapped to execution at every layer |

---

*AI Agent Orchestration Framework — Complete Evolutionary HLSD*
*Principal AI Engineer Interview Reference — All 20 Steps*
*Phase 1: Steps 1–7 | Phase 2: Steps 8–13 | Phase 3: Steps 14–20*
