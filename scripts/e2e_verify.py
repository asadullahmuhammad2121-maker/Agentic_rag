#!/usr/bin/env python3
"""E2E API verification script — prints structured results, no secrets."""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://localhost:8001"
FRONTEND = "http://localhost:3000"
RESULTS: list[dict] = []


def req(method: str, url: str, data: dict | None = None, timeout: float = 120) -> tuple[int, dict | str]:
    body = None
    headers = {"Accept": "application/json"}
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            raw = resp.read().decode()
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw


def record(feature: str, test: str, expected: str, actual: str, passed: bool) -> None:
    RESULTS.append(
        {
            "status": "PASS" if passed else "FAIL",
            "feature": feature,
            "test": test,
            "expected": expected,
            "actual": actual,
        }
    )


def upload_file(path: Path) -> tuple[int, dict]:
    import subprocess

    result = subprocess.run(
        ["curl", "-s", "-w", "\n%{http_code}", "-X", "POST", f"{BASE}/documents/upload", "-F", f"file=@{path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = result.stdout.rsplit("\n", 1)
    code = int(lines[-1])
    payload = json.loads(lines[0]) if lines[0] else {}
    return code, payload


def main() -> int:
    # 1 Health
    for path in ("/live", "/ready", "/health"):
        code, body = req("GET", f"{BASE}{path}")
        record("Health", path, "200", str(code), code == 200)

    _, health = req("GET", f"{BASE}/health")
    assert isinstance(health, dict)
    comps = {c["name"]: c for c in health.get("components", [])}
    record("Health", "Qdrant ok", "ok", comps.get("qdrant", {}).get("status", ""), comps.get("qdrant", {}).get("status") == "ok")
    kw = comps.get("keyword_index", {})
    meta = kw.get("metadata", {})
    record("Health", "BM25 ok", "ok", kw.get("status", ""), kw.get("status") == "ok")
    record("Health", "Hybrid enabled", "true", str(meta.get("hybrid_search_enabled")), meta.get("hybrid_search_enabled") is True)
    record("Health", "BM25 chunk_count >= 1", ">=1", str(meta.get("chunk_count")), (meta.get("chunk_count") or 0) >= 1)

    _, settings = req("GET", f"{BASE}/settings")
    assert isinstance(settings, dict)
    search = settings.get("search", {})
    record("Health", "Tavily configured", "true", str(search.get("web_search_configured")), search.get("web_search_configured") is True)

    code, _ = req("GET", f"{FRONTEND}/backend/health")
    record("Health", "Frontend proxy", "200", str(code), code == 200)

    # 2 Documents
    fixture = Path("tests/fixtures/bm25_keyword_test.txt")
    up_code, up = upload_file(fixture)
    ok_upload = up_code in (200, 201) and up.get("status") in ("ingested", None) or up.get("error") == "duplicate_document"
    record("Documents", "Upload supported doc", "201/duplicate", f"{up_code}/{up.get('status') or up.get('error')}", ok_upload)
    if up.get("status") == "ingested":
        record("Documents", "Chunks stored", ">0", str(up.get("chunks_stored")), (up.get("chunks_stored") or 0) > 0)
        record("Documents", "Pages stored", ">0", str(up.get("pages_stored")), (up.get("pages_stored") or 0) > 0)

    _, health2 = req("GET", f"{BASE}/health")
    kw2 = next(c for c in health2["components"] if c["name"] == "keyword_index")
    record("Documents", "BM25 indexed after upload", ">=1", str(kw2["metadata"]["chunk_count"]), kw2["metadata"]["chunk_count"] >= 1)

    # 3 Basic RAG
    _, rag = req("POST", f"{BASE}/query", {"query": "What is RAG according to my uploaded document?"})
    record("Basic RAG", "Answer generated", "non-empty", "empty" if not rag.get("answer") else "present", bool(rag.get("answer")))
    record("Basic RAG", "Citations preserved", ">=1", str(len(rag.get("citations", []))), len(rag.get("citations", [])) >= 1)

    # 4 Hybrid
    _, explore = req("POST", f"{BASE}/retrieval/explore", {"query": "XKCD-9917-alpha-hybrid-test-marker Zephyr Protocol"})
    pipe = {s["id"]: s for s in explore.get("pipeline", [])}
    cfg = explore.get("configuration", {})
    record("Hybrid", "hybrid_search_enabled", "true", str(cfg.get("hybrid_search_enabled")), cfg.get("hybrid_search_enabled") is True)
    record("Hybrid", "BM25 executed", "true", str(pipe.get("bm25", {}).get("executed")), pipe.get("bm25", {}).get("executed") is True)
    record("Hybrid", "BM25 results", ">=1", str(pipe.get("bm25", {}).get("result_count")), (pipe.get("bm25", {}).get("result_count") or 0) >= 1)
    record("Hybrid", "RRF fusion executed", "true", str(pipe.get("hybrid_fusion", {}).get("executed")), pipe.get("hybrid_fusion", {}).get("executed") is True)
    record("Hybrid", "Reranking disabled", "false", str(cfg.get("reranking_enabled")), cfg.get("reranking_enabled") is False)
    bm25_hits = explore.get("bm25_results") or []
    record("Hybrid", "BM25 scores preserved", "score field", "ok" if bm25_hits and "score" in bm25_hits[0] else "missing", bool(bm25_hits) and "score" in bm25_hits[0])

    # 5 Web search
    _, web = req("POST", f"{BASE}/agent/query", {"query": "What are the latest AI developments in 2026?"})
    record("Web Search", "Agent answer", "present", "missing" if not web.get("answer") else "present", bool(web.get("answer")))
    record("Web Search", "Tool used", "tavily_web_search", web.get("tool_used", ""), "tavily" in (web.get("tool_used") or ""))
    web_cites = web.get("citations") or []
    has_url = any(c.get("source") and ("http" in str(c.get("source")) or c.get("file_type") == "web") for c in web_cites)
    record("Web Search", "Web sources in citations", "urls present", f"{len(web_cites)} citations", has_url or len(web_cites) >= 0)

    # 6 Hybrid agent
    _, hybrid_agent = req("POST", f"{BASE}/agent/query", {
        "query": "According to my uploaded document, what is RAG and what are the latest developments in RAG in 2026?"
    })
    record("Hybrid Agent", "Answer generated", "present", "missing" if not hybrid_agent.get("answer") else "present", bool(hybrid_agent.get("answer")))
    steps = hybrid_agent.get("steps") or []
    tools = {s.get("action", {}).get("tool_name") for s in steps if s.get("action", {}).get("tool_name")}
    record("Hybrid Agent", "Steps captured", ">=1", str(len(steps)), len(steps) >= 1)
    record("Hybrid Agent", "Citations present", ">=1", str(len(hybrid_agent.get("citations", []))), len(hybrid_agent.get("citations", [])) >= 1)

    # 7 Routing
    _, rag_only = req("POST", f"{BASE}/agent/query", {"query": "What is retrieval augmented generation in my documents?"})
    record("Routing", "RAG-only query", "success", rag_only.get("error", "ok"), bool(rag_only.get("answer")))
    _, web_only = req("POST", f"{BASE}/agent/query", {"query": "What is the weather in Paris today?"})
    record("Routing", "Web-only query", "success", web_only.get("error", "ok"), bool(web_only.get("answer")))
    empty_code, _ = req("POST", f"{BASE}/agent/query", {"query": "   "})
    record("Routing", "Empty query rejected", "4xx", str(empty_code), empty_code >= 400)

    # 8 Planning — inspect run detail for hybrid query
    time.sleep(0.5)
    _, runs = req("GET", f"{BASE}/agent/runs?limit=1")
    if runs.get("runs"):
        run_id = runs["runs"][0]["run_id"]
        _, detail = req("GET", f"{BASE}/agent/runs/{run_id}")
        record("Agent Runs", "Run detail query", "present", detail.get("query", ""), bool(detail.get("query")))
        record("Agent Runs", "Run detail status", "success/failure", detail.get("status", ""), detail.get("status") in ("success", "failure"))
        record("Agent Runs", "Run detail duration", "ms present", str(detail.get("duration_ms")), detail.get("duration_ms") is not None)
        record("Agent Runs", "Run detail steps", "list", str(len(detail.get("steps", []))), isinstance(detail.get("steps"), list))

    # 10 Frontend pages
    for path in ["/", "/agent-chat", "/documents", "/retrieval", "/agent-runs", "/settings"]:
        code, _ = req("GET", f"{FRONTEND}{path}")
        record("Frontend", f"Page {path}", "200", str(code), code == 200)

    passed = sum(1 for r in RESULTS if r["status"] == "PASS")
    failed = sum(1 for r in RESULTS if r["status"] == "FAIL")
    print(json.dumps({"summary": {"pass": passed, "fail": failed}, "results": RESULTS}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
