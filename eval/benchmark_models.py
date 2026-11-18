#!/usr/bin/env python3
"""
ReguRAG_MM — Model Benchmark Script
==================================
Benchmarks candidate LLMs via local Ollama API for Hebrew regulation QA.
Self-contained, sectioned benchmark design.

Usage:
    python3 eval/benchmark_models.py                # Full benchmark (3 models x 10 queries)
    python3 eval/benchmark_models.py --limit 3      # Smoke test: 3 queries per model
    python3 eval/benchmark_models.py --models m1,m2 # Benchmark specific models from CANDIDATES
    python3 eval/benchmark_models.py --apply-config # Pin config.yaml with model decision
    python3 eval/benchmark_models.py --test         # Run inline unit tests

Exits:
    0  Success
    1  Ollama unreachable
    2  Non-loopback URL blocked (safety)
    3  Query file / schema error
"""

import sys
import json
import time
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timezone
from argparse import ArgumentParser
from urllib.request import build_opener, ProxyHandler, Request
from urllib.error import URLError

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Tech stack (stdlib only — verified: no pip imports)
# ─────────────────────────────────────────────────────────────

OLLAMA_DEFAULT_URL = "http://127.0.0.1:11434"
ALLOWED_HOSTS = {"127.0.0.1", "localhost"}
EXACT_REFUSAL = "המידע אינו קיים בתקנונים שבמערכת."
REFUSAL_FILLER = "אין טקסט."

# Candidate models (fixed order = tie-break)
CANDIDATES: List[str] = [
    "qwen2.5:3b-instruct-q4_K_M",
    "llama3.2:3b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
]

DEFAULT_OPTIONS: Dict[str, Any] = {
    "temperature": 0,
    "seed": 42,
    "num_ctx": 4096,
}

# Paths (relative to repo root, which is parent of eval/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUERIES_FILE = PROJECT_ROOT / "eval" / "benchmark_queries.json"
RESULTS_JSON = PROJECT_ROOT / "artifacts" / "model-benchmark-results.json"
REPORT_MD = PROJECT_ROOT / "project" / "artifacts" / "model-benchmark-report.md"
CONFIG_YAML = PROJECT_ROOT / "config.yaml"


# ─────────────────────────────────────────────────────────────
# Queries & prompt template
# ─────────────────────────────────────────────────────────────

def format_prompt(context: str, query: str, out_of_scope: bool) -> str:
    """Build prompt with exact template."""
    ctx = REFUSAL_FILLER if out_of_scope else context
    return (
        'הוראות מערכת: אתה עוזר מנהלתי-משפטי. תשובותך מבוססות אך ורק '
        'על הטקסט המצורף להלן, ללא כל שימוש בידע כללי. אם התשובה אינה '
        'קיימת בטקסט המצורף, הגב במדויק: "המידע אינו קיים בתקנונים שבמערכת."\n\n'
        f'טקסט מצורף (מפרוק תקנון):\n"""{ctx}"""\n\n'
        f'שאלה: {query}'
    )


def load_queries(filepath: Path) -> List[Dict]:
    """Load and validate query file."""
    if not filepath.exists():
        logger.error("Query file not found: %s", filepath)
        sys.exit(3)

    raw = json.loads(filepath.read_text("utf-8"))

    required = {"id", "query", "context", "out_of_scope"}
    for i, q in enumerate(raw):
        missing = required - set(q.keys())
        if missing:
            logger.error("Query[%d] missing fields: %s", i, missing)
            sys.exit(3)
        if q["out_of_scope"] and q["context"]:
            logger.warning("Query %d: out_of_scope but has context", q["id"])
        if not q["out_of_scope"] and not q["context"]:
            logger.warning("Query %d: in-scope with empty context", q["id"])

    logger.info("Loaded %d queries from %s", len(raw), filepath)
    return raw


# ─────────────────────────────────────────────────────────────
# Network discipline (loopback enforcement)
# ─────────────────────────────────────────────────────────────

def validate_loopback(base_url: str) -> bool:
    """If config.yaml has a non-loopback URL, exit(2) with refusal."""
    from urllib.parse import urlparse
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        print(f"Refusing non-loopback Ollama URL: {parsed.geturl()}")
        print(f"  Allowed hosts (loopback only): {sorted(ALLOWED_HOSTS)}")
        return False
    return True


def load_config_ollama_url() -> str:
    """Read models.ollama_base_url from config.yaml via regex."""
    if not CONFIG_YAML.exists():
        logger.warning("Config file not found: %s", CONFIG_YAML)
        return OLLAMA_DEFAULT_URL
    content = CONFIG_YAML.read_text("utf-8")
    m = re.search(r'ollama_base_url:\s*["\']?\s*(\[?127\.0\.0\.1\]?\:\d+|127\.0\.0\.1\:\d+)[\'\"]?\s*$', content, re.MULTILINE)
    if m:
        return m.group(1)
    return OLLAMA_DEFAULT_URL


# ─────────────────────────────────────────────────────────────
# Ollama preflight
# ─────────────────────────────────────────────────────────────

def _make_opener() -> Any:
    """Explicit ProxyHandler({}) disables proxy intercept."""
    return build_opener(ProxyHandler({}))


def ollama_preflight(opener: Any, base_url: str) -> Tuple[bool, List[str]]:
    """GET /api/version → /api/tags. Missing model → helpful exit."""
    try:
        resp = opener.open(f"{base_url}/api/version", timeout=5)
        ver = json.loads(resp.read().decode("utf-8"))
    except URLError:
        print(f"Ollama not reachable at {base_url}.")
        print("  Start it:  ollama serve")
        return False, []

    try:
        resp = opener.open(f"{base_url}/api/tags", timeout=5)
        models = json.loads(resp.read().decode("utf-8"))
    except URLError:
        print(f"Ollama /api/tags failed: {base_url}")
        return False, []

    names = [m["name"] for m in models.get("models", [])]
    logger.info("Ollama version: %s · models: %s",
                ver.get("version", "?"), ", ".join(names))
    return True, names


# ─────────────────────────────────────────────────────────────
# NDJSON streaming query
# ─────────────────────────────────────────────────────────────

def benchmark_query(opener: Any, model: str, prompt: str) -> Dict:
    """Streaming POST /api/generate; parse NDJSON line-by-line."""
    payload = json.dumps({"model": model, "prompt": prompt, "stream": True, "options": DEFAULT_OPTIONS}).encode("utf-8")
    req = Request(f"{OLLAMA_DEFAULT_URL}/api/generate", data=payload, headers={"Content-Type": "application/json"})

    t_start = time.perf_counter()
    first_token_t: Optional[float] = None
    chunks: List[str] = []
    prompt_tok = 0
    eval_tok = 0

    try:
        with opener.open(req, timeout=120) as resp:
            for raw_line in resp:
                if isinstance(raw_line, bytes):
                    raw_line = raw_line.decode("utf-8")
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "prompt_eval_count" in chunk:
                    prompt_tok = chunk["prompt_eval_count"]
                if "eval_count" in chunk:
                    eval_tok = chunk["eval_count"]
                delta = chunk.get("delta", "")
                if delta and first_token_t is None:
                    first_token_t = time.perf_counter()
                chunks.append(delta)
    except Exception as e:
        total = time.perf_counter() - t_start
        return _error_record(model, first_token_t, t_start, str(e))

    text = "".join(chunks)
    total_ms = (time.perf_counter() - t_start) * 1000
    ttft_ms = (first_token_t - t_start) * 1000 if first_token_t else 0
    elapsed = time.perf_counter() - t_start
    tps = round(eval_tok / elapsed, 3) if elapsed > 0 else 0

    return {
        "model": model,
        "query_id": 0,
        "ok": True,
        "error": None,
        "first_token_latency_ms": round(ttft_ms, 2),
        "total_time_ms": round(total_ms, 2),
        "tokens_generated": eval_tok,
        "prompt_tokens": prompt_tok,
        "tps_decode": tps,
        "response_chars": len(text),
        "pass": False,          # computed in scoring step
        "response_text": text,
    }


def _error_record(model: str, first_token_t: Optional[float], t_start: float, err: str) -> Dict:
    total = (time.perf_counter() - t_start) * 1000
    ttft = (first_token_t - t_start) * 1000 if first_token_t else 0
    return {
        "model": model,
        "query_id": 0,
        "ok": False,
        "error": err,
        "first_token_latency_ms": round(ttft, 2),
        "total_time_ms": round(total, 2),
        "tokens_generated": 0,
        "prompt_tokens": 0,
        "tps_decode": 0,
        "response_chars": 0,
        "pass": False,
        "response_text": "",
    }


# ─────────────────────────────────────────────────────────────
# Scoring & decision
# ─────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def score_query(resp_text: str, out_of_scope: bool) -> bool:
    """Binary pass/fail per query type."""
    if out_of_scope:
        return EXACT_REFUSAL in _norm(resp_text)
    norm = _norm(resp_text)
    return len(resp_text) >= 20 and EXACT_REFUSAL not in norm


def per_model_summary(model: str, records: List[Dict]) -> Dict:
    times = [r["total_time_ms"] for r in records if r["total_time_ms"] > 0]
    tps_vals = [r["tps_decode"] for r in records if r["tps_decode"] > 0]
    if not times:
        return {"model": model, "n_ok": 0, "ttft_mean_ms": 0, "ttft_p95_ms": 0,
                "total_mean_ms": 0, "tps_median": 0, "tps_mean": 0, "compliance_rate": 0}
    in_scope = [r for r in records if not r.get("_out_of_scope", False)]
    in_pass = sum(1 for r in in_scope if r["ok"])
    rate = round(in_pass / len(in_scope), 3) if in_scope else 0
    s = sorted(times)
    p95 = s[min(int(len(s) * 0.95), len(s) - 1)]
    ts = sorted(tps_vals)
    med = ts[len(ts) // 2] if ts else 0
    return {
        "model": model,
        "n_ok": sum(1 for r in records if r["ok"]),
        "ttft_mean_ms": round(sum(r["first_token_latency_ms"] for r in records) / len(records), 2),
        "ttft_p95_ms": round(p95, 2),
        "total_mean_ms": round(sum(times) / len(times), 2),
        "tps_median": round(med, 3),
        "tps_mean": round(sum(tps_vals) / len(tps_vals), 3) if tps_vals else 0,
        "compliance_rate": rate,
    }


def compute_decision(per_model: Dict[str, Dict]) -> Dict:
    """Decision rule."""
    order = sorted(per_model.keys(), key=lambda m: (-per_model[m]["compliance_rate"], -per_model[m]["tps_median"]))
    dev_gpu = order[0]
    prod_3b = [m for m in order if "3b" in m]
    prod_cpu = prod_3b[0] if prod_3b else order[0]
    return {
        "rule": "Order by (-compliance_rate, -tps_median) desc; prod restricted to 3B models; equal within 1% -> qwen2.5:3b",
        "prod_cpu": prod_cpu,
        "dev_gpu": dev_gpu,
    }


# ─────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────

def build_report(results: List[Dict], summaries: Dict[str, Dict], decision: Dict) -> str:
    lines = [
        "# ReguRAG_MM — Model Benchmark Report",
        "",
        f"**Generated** (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Hardware note**: dev RTX 3090 — numbers optimistic vs prod CPU; ranking-only.",
        f"**Ollama URL**: {OLLAMA_DEFAULT_URL}",
        f"**Options**: temp=0, seed=42, num_ctx=4096",
        "",
        "## Performance comparison",
        "",
        "| Model | Avg time (ms) | Avg TPS | p95 TTFT (ms) | Compliance |",
        "|-------|--------------:|--------:|---------------:|------------|",
    ]
    best_t = min((d["total_mean_ms"] for d in summaries.values()), default=float('inf'))
    best_p = max((d["tps_mean"] for d in summaries.values()), default=0)
    best_c = max((d["compliance_rate"] for d in summaries.values()), default=0)
    for m in summaries:
        d = summaries[m]
        t = f"**{d['total_mean_ms']:.1f}**" if d["total_mean_ms"] == best_t else f"{d['total_mean_ms']:.1f}"
        p = f"**{d['tps_mean']:.3f}**" if d["tps_mean"] == best_p else f"{d['tps_mean']:.3f}"
        c = f"**{d['compliance_rate']:.2%}**" if d["compliance_rate"] == best_c else f"{d['compliance_rate']:.2%}"
        lines.append(f"| {m} | {t} | {p} | {d['ttft_p95_ms']:.1f} | {c} |")

    lines.extend(["", "## Per-query matrix", "", "| Query | " + " | ".join(CANDIDATES) + " |"])
    lines.append("|-------|" + "|".join("---" for _ in CANDIDATES) + "|")
    qids = sorted({r["query_id"] for r in results})
    for qid in qids:
        row = [str(qid)]
        m2r = {m: [r for r in results if r["query_id"] == qid and r["model"] == m][0] for m in CANDIDATES}
        for m in CANDIDATES:
            r = m2r[m]
            if r["ok"]:
                row.append(f"✅ {r['total_time_ms']:.0f}ms")
            else:
                row.append(f"❌ {r['error'][:40]}")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Quality & compliance", ""])
    for m in summaries:
        s = summaries[m]
        lines.append(f"- **{m}**: compliance={s['compliance_rate']:.2%} · ok={s['n_ok']}")

    lines.extend(["", "## Decision", ""])
    lines.append(f"- **Production (prod_cpu)**: `{decision['prod_cpu']}`")
    lines.append(f"- **Development (dev_gpu)**: `{decision['dev_gpu']}`")
    lines.append(f"- **Rule**: {decision['rule']}")
    lines.extend(["", "## Reproducibility", ""])
    lines.extend(["```bash",
                  f"cd {PROJECT_ROOT}",
                  "python3 eval/benchmark_models.py --apply-config",
                  "```"])
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# config.yaml pinning
# ─────────────────────────────────────────────────────────────

def pin_config(decision: Dict) -> bool:
    """Surgical line patch that preserves comments."""
    if not CONFIG_YAML.exists():
        logger.error("Config not found: %s", CONFIG_YAML)
        return False

    text = CONFIG_YAML.read_text("utf-8")
    lines = text.split("\n")
    changed = False

    def _patch_for(profile: str, new_name: str):
        nonlocal changed
        for i, ln in enumerate(lines):
            if f"{profile}:" in ln:
                for j in range(i + 1, len(lines)):
                    if "llm_name:" in lines[j]:
                        ts = datetime.now().strftime("%Y-%m-%d")
                        lines[j] = f"    llm_name: \"{new_name}\"  # pinned {ts}"
                        logger.info("Pinned %s.llm_name -> %s", profile, new_name)
                        changed = True
                        return
            if f"{profile}:" in ln:
                break

    _patch_for("prod_cpu", decision["prod_cpu"])
    _patch_for("dev_gpu", decision["dev_gpu"])

    if not changed:
        logger.info("config.yaml already pinned — no change")
        return False

    CONFIG_YAML.write_text("\n".join(lines), "utf-8")
    return True


# ─────────────────────────────────────────────────────────────
# Main orchestration
# ─────────────────────────────────────────────────────────────

def run_bench(queries: List[Dict], models: List[str], limit: Optional[int]) -> Tuple[List[Dict], Dict[str, Dict], Dict]:
    """Full pipeline: preflight → warmup → measure → score → decide."""
    opener = _make_opener()
    ok, avail = ollama_preflight(opener, OLLAMA_DEFAULT_URL)
    if not ok:
        sys.exit(1)

    active = [m for m in models if any(m in a for a in avail)]
    if not active:
        print("None of the requested models found in Ollama")
        sys.exit(1)
    logger.info("Active models: %s", ", ".join(active))

    results: List[Dict] = []
    for mdl in active:
        logger.info("=== %s ===", mdl)
        # warmup (discard)
        warmup = next(q for q in queries if q["id"] == 1) if any(q["id"] == 1 for q in queries) else queries[0]
        _ = benchmark_query(opener, mdl, format_prompt(warmup["context"], warmup["query"], warmup["out_of_scope"]))
        time.sleep(0.5)

        for q in queries:
            if limit and sum(1 for r in results if r["model"] == mdl) >= limit:
                break
            rec = benchmark_query(opener, mdl, format_prompt(q["context"], q["query"], q["out_of_scope"]))
            rec["query_id"] = q["id"]
            rec["_out_of_scope"] = q["out_of_scope"]
            rec["pass"] = score_query(rec["response_text"], q["out_of_scope"])
            results.append(rec)
            stat = "✅" if rec["ok"] else "❌"
            print(f"  [{mdl[:18]}] [{q['id']:02d}] {stat} {rec['total_time_ms']:.0f}ms | {rec['tps_decode']:.2f} tps | {len(rec['response_text'])} chars | {q['query'][:50]}")
            time.sleep(0.5)

    summaries = {m: per_model_summary(m, [r for r in results if r["model"] == m]) for m in active}
    decision = compute_decision(summaries)
    return results, summaries, decision


def main():
    ap = ArgumentParser(description="ReguRAG_MM Model Benchmark")
    ap.add_argument("--models", help="Comma-separated models (default: all CANDIDATES)")
    ap.add_argument("--limit", type=int, help="Limit queries per model (smoke test)")
    ap.add_argument("--apply-config", action="store_true", help="Pin config.yaml with decision")
    ap.add_argument("--test", action="store_true", help="Run inline unit tests")
    args = ap.parse_args()

    if args.test:
        run_tests()
        return

    queries = load_queries(QUERIES_FILE)
    if args.limit:
        queries = queries[:args.limit]

    models = CANDIDATES
    if args.models:
        candidates = [m.strip() for m in args.models.split(",")]
        valid = [c for c in candidates if c in CANDIDATES]
        if not valid:
            print(f"All models rejected; must be subset of: {CANDIDATES}")
            sys.exit(1)
        models = valid

    cfg_url = load_config_ollama_url()
    if not validate_loopback(cfg_url):
        sys.exit(2)

    results, summaries, decision = run_bench(queries, models, args.limit)

    # Write JSON
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    json_out = {
        "benchmark_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hardware_note": "dev RTX3090 run — absolute tok/s optimistic vs prod CPU; ranking-only",
        "ollama_base_url": OLLAMA_DEFAULT_URL,
        "options": DEFAULT_OPTIONS,
        "models": CANDIDATES,
        "results": [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in results
        ],
        "per_model_summary": summaries,
        "quality": {
            "compliance_rate": {m: s["compliance_rate"] for m, s in summaries.items()},
        },
        "decision": decision,
    }
    with RESULTS_JSON.open("w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"\nResults → {RESULTS_JSON}")

    report = build_report(results, summaries, decision)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(report, "utf-8")
    print(f"Report  → {REPORT_MD}")

    if args.apply_config:
        pin_config(decision)
        print("config.yaml pinned with decision")

    print(f"\n{'='*60}")
    print(f"PROD model : {decision['prod_cpu']}")
    print(f"DEV model  : {decision['dev_gpu']}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────
# Inline unit tests (run with --test)
# ─────────────────────────────────────────────────────────────

def run_tests():
    import unittest as _u

    class BenchTests(_u.TestCase):
        def test_format_prompt_out_of_scope(self):
            p = format_prompt("", "x?", True)
            self.assertIn("אין טקסט.", p)

        def test_format_prompt_in_scope(self):
            ctx = "המסמך קובע כי דמי השכר הם X חדשים. הדירה צריכה להיות בשכונת"
            p = format_prompt(ctx, "מה גובה השכר?", False)
            self.assertIn(ctx, p)
            self.assertNotIn("אין טקסט.", p)

        def test_norm(self):
            self.assertEqual(_norm("  a   b  "), "a b")
            self.assertEqual(_norm(""), "")

        def test_score_out_of_scope_pass(self):
            self.assertTrue(score_query(EXACT_REFUSAL + " more text", True))

        def test_score_out_of_scope_fail(self):
            self.assertFalse(score_query("some answer", True))

        def test_score_in_scope_pass(self):
            self.assertTrue(score_query("ע" * 25, False))

        def test_score_in_scope_fail_short(self):
            self.assertFalse(score_query("ok", False))

        def test_score_in_scope_refusal(self):
            self.assertFalse(score_query(EXACT_REFUSAL, False))

        def test_decision_ordering(self):
            d = {
                "qwen2.5:3b": {"compliance_rate": 0.8, "tps_median": 3.0},
                "llama3.2:3b": {"compliance_rate": 0.9, "tps_median": 2.0},
                "qwen2.5:7b": {"compliance_rate": 1.0, "tps_median": 1.0},
            }
            dec = compute_decision(d)
            self.assertEqual(dec["prod_cpu"], "llama3.2:3b")
            self.assertEqual(dec["dev_gpu"], "qwen2.5:7b")

    _u.main(verbosity=1)


if __name__ == "__main__":
    main()
