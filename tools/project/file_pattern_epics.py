#!/usr/bin/env python3
"""File the CSP pattern-coverage epic tree from csp_pattern_epics.json.

Creates (idempotently, keyed by issue title):
  - labels and per-wave milestones
  - one epic issue per manifest entry
  - the umbrella epic, with all epics linked as native sub-issues
  - sub-issues (T1..T5 + extras) for epics whose wave <= --max-sub-wave,
    linked as native sub-issues of their epic

Usage:
  python3 tools/project/file_pattern_epics.py [--repo owner/name]
      [--max-sub-wave 1] [--dry-run]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

MANIFEST = Path(__file__).with_name("csp_pattern_epics.json")


def run(args, dry=False, capture=True):
    if dry:
        print("DRY:", " ".join(args))
        return ""
    r = subprocess.run(args, capture_output=capture, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(args)}\n{r.stderr}")
    return r.stdout.strip()


def gh(repo, *args, dry=False):
    return run(["gh"] + list(args) + ["--repo", repo], dry=dry)


def ensure_labels(repo, labels, dry):
    for lb in labels:
        run(["gh", "label", "create", lb["name"], "--repo", repo,
             "--color", lb["color"], "--description", lb["description"],
             "--force"], dry=dry)
        print(f"label: {lb['name']}")


def ensure_milestones(repo, waves, dry):
    existing = json.loads(run(["gh", "api", f"repos/{repo}/milestones?state=all&per_page=100"]))
    have = {m["title"] for m in existing}
    for w in waves:
        title = f"csp-patterns-wave-{w}"
        if title in have:
            print(f"milestone exists: {title}")
            continue
        run(["gh", "api", f"repos/{repo}/milestones", "-f", f"title={title}",
             "-f", f"description=CSP pattern coverage wave {w}"], dry=dry)
        print(f"milestone: {title}")


def existing_issues(repo):
    out = run(["gh", "issue", "list", "--repo", repo, "--state", "all",
               "--limit", "500", "--json", "number,title"])
    return {i["title"]: i["number"] for i in json.loads(out)}


def create_issue(repo, title, body, labels, milestone, existing, dry):
    if title in existing:
        print(f"exists  #{existing[title]}: {title}")
        return existing[title]
    args = ["gh", "issue", "create", "--repo", repo, "--title", title,
            "--body", body, "--milestone", milestone]
    for lb in labels:
        args += ["--label", lb]
    url = run(args, dry=dry)
    num = 0 if dry else int(url.rstrip("/").rsplit("/", 1)[-1])
    print(f"created #{num}: {title}")
    return num


def node_id(repo, number):
    return run(["gh", "issue", "view", str(number), "--repo", repo,
                "--json", "id", "-q", ".id"])


def link_sub(repo, parent_num, child_num, dry):
    if dry:
        print(f"DRY: link #{child_num} under #{parent_num}")
        return
    p, c = node_id(repo, parent_num), node_id(repo, child_num)
    run(["gh", "api", "graphql",
         "-f", "query=mutation($p: ID!, $c: ID!) {"
               " addSubIssue(input: {issueId: $p, subIssueId: $c})"
               " { issue { number } } }",
         "-f", f"p={p}", "-f", f"c={c}"])
    print(f"linked #{child_num} under #{parent_num}")


def epic_body(m, e):
    lines = [
        f"Part of the CSP pattern-coverage program - see `{m['plan_doc']}`.",
        "",
        f"**Pattern classes:** {e['patterns']}",
        f"**Wave:** {e['wave']}  |  **Depends on:** {e['deps']}",
    ]
    if e.get("absorbs"):
        lines.append(f"**Absorbs/relates:** {e['absorbs']}")
    lines += [
        "",
        "## Scope",
        "",
        e["scope"],
        "",
        "## Decomposition",
        "",
        "Five medium (estimate 5) sub-issues: T1 pattern design & envelope",
        "analysis, T2 ISA/executor capability closure, T3 envelope-aware",
        "schedule generator, T4 functional integration & oracle, T5",
        "regression & characterization. Sub-issues are filed when this",
        "epic's wave opens.",
        "",
        "## Definition of done",
        "",
        "All sub-issues closed; the operator executes functionally on the",
        "CSP executor with oracle-verified values under default and",
        "constrained envelopes; regression in CI; characterization numbers",
        "recorded here.",
    ]
    return "\n".join(lines)


def sub_body(m, e, sub):
    desc = sub["d"].replace("OPERATOR", e.get("operator", e["title"]))
    return "\n".join([
        f"Sub-issue of the **{e['title']}** epic "
        f"(CSP pattern-coverage program, `{m['plan_doc']}`).",
        "",
        "**Estimate: 5 (medium)**",
        "",
        desc,
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="stillwater-sc/kpu-sim")
    ap.add_argument("--max-sub-wave", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    m = json.loads(MANIFEST.read_text())
    waves = sorted({e["wave"] for e in m["epics"]})

    ensure_labels(args.repo, m["labels"], args.dry_run)
    ensure_milestones(args.repo, waves, args.dry_run)
    existing = {} if args.dry_run else existing_issues(args.repo)

    # Epics
    epic_nums = {}
    for e in m["epics"]:
        num = create_issue(
            args.repo, e["title"], epic_body(m, e),
            ["epic", "csp-patterns", f"wave-{e['wave']}"],
            f"csp-patterns-wave-{e['wave']}", existing, args.dry_run)
        epic_nums[e["id"]] = num

    # Umbrella with epic checklist
    u = m["umbrella"]
    checklist = "\n".join(
        f"- [ ] #{epic_nums[e['id']]} {e['id']} (wave {e['wave']}): {e['title']}"
        for e in m["epics"])
    u_body = f"{u['scope']}\n\nPlan: `{m['plan_doc']}`\n\n## Epics\n\n{checklist}\n"
    u_num = create_issue(args.repo, u["title"], u_body,
                         ["epic", "csp-patterns"],
                         "csp-patterns-wave-0", existing, args.dry_run)
    for e in m["epics"]:
        link_sub(args.repo, u_num, epic_nums[e["id"]], args.dry_run)

    # Sub-issues for open waves
    for e in m["epics"]:
        if e["wave"] > args.max_sub_wave:
            continue
        subs = e.get("custom_subs") or [dict(s) for s in m["template_subs"]]
        subs = subs + m.get("extra_subs", {}).get(e["id"], [])
        for i, sub in enumerate(subs, 1):
            title = f"{e['short']}-T{i}: {sub['t']}"
            num = create_issue(
                args.repo, title, sub_body(m, e, sub),
                ["csp-patterns", "estimate:5", f"wave-{e['wave']}"],
                f"csp-patterns-wave-{e['wave']}", existing, args.dry_run)
            link_sub(args.repo, epic_nums[e["id"]], num, args.dry_run)

    print("\nDone. Umbrella:", f"#{u_num}")
    print("Epics:", ", ".join(f"{k}=#{v}" for k, v in epic_nums.items()))


if __name__ == "__main__":
    sys.exit(main())
