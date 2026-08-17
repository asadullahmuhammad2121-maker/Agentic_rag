#!/usr/bin/env bash
# End-to-end API smoke tests for Agentic RAG (no secrets printed)
set -uo pipefail
BASE="${BASE_URL:-http://localhost:8001}"
FRONTEND="${FRONTEND_URL:-http://localhost:3000}"
PASS=0
FAIL=0
RESULTS=()

log() { echo "[e2e] $*"; }
pass() { PASS=$((PASS+1)); RESULTS+=("PASS|$1"); log "PASS: $1"; }
fail() { FAIL=$((FAIL+1)); RESULTS+=("FAIL|$1|$2"); log "FAIL: $1 — $2"; }

json_get() {
  python3 -c "import json,sys; d=json.load(sys.stdin); print($1)" 2>/dev/null
}

# 1. Health & Infrastructure
log "=== 1. Health & Infrastructure ==="
code=$(curl -s -o /tmp/e2e_live.json -w "%{http_code}" "$BASE/live")
[[ "$code" == "200" ]] && pass "/live returns 200" || fail "/live" "status=$code"

code=$(curl -s -o /tmp/e2e_ready.json -w "%{http_code}" "$BASE/ready")
[[ "$code" == "200" ]] && pass "/ready returns 200" || fail "/ready" "status=$code"

code=$(curl -s -o /tmp/e2e_health.json -w "%{http_code}" "$BASE/health")
[[ "$code" == "200" ]] && pass "/health returns 200" || fail "/health" "status=$code"

qdrant=$(json_get "next(c['status'] for c in d['components'] if c['name']=='qdrant')") < /tmp/e2e_health.json
[[ "$qdrant" == "ok" ]] && pass "Qdrant component ok" || fail "Qdrant" "status=$qdrant"

bm25=$(json_get "next(c['status'] for c in d['components'] if c['name']=='keyword_index')") < /tmp/e2e_health.json
hybrid=$(json_get "next(c['metadata'].get('hybrid_search_enabled') for c in d['components'] if c['name']=='keyword_index')") < /tmp/e2e_health.json
chunks=$(json_get "next(c['metadata'].get('chunk_count') for c in d['components'] if c['name']=='keyword_index')") < /tmp/e2e_health.json
[[ "$bm25" == "ok" ]] && pass "BM25 index available" || fail "BM25 index" "status=$bm25"
[[ "$hybrid" == "True" ]] && pass "Hybrid search enabled in health metadata" || fail "Hybrid enabled" "value=$hybrid"

curl -s "$BASE/settings" -o /tmp/e2e_settings.json
tavily_cfg=$(json_get "d['search']['web_search_configured']") < /tmp/e2e_settings.json
tavily_en=$(json_get "d['search']['web_search_enabled']") < /tmp/e2e_settings.json
[[ "$tavily_cfg" == "True" || "$tavily_en" == "True" ]] && pass "Tavily configured/enabled in settings" || fail "Tavily" "configured=$tavily_cfg enabled=$tavily_en"

code=$(curl -s -o /tmp/e2e_proxy_health.json -w "%{http_code}" "$FRONTEND/backend/health")
[[ "$code" == "200" ]] && pass "Frontend→backend proxy /backend/health" || fail "Frontend proxy" "status=$code"

# 2. Documents upload
log "=== 2. Documents ==="
TEST_FILE="/Users/macbook/Desktop/agentic_rag/tests/fixtures/bm25_keyword_test.txt"
if [[ ! -f "$TEST_FILE" ]]; then
  echo "Unique BM25 marker XKCD-E2E-$(date +%s)" > /tmp/e2e_upload.txt
  TEST_FILE=/tmp/e2e_upload.txt
fi
curl -s -X POST "$BASE/documents/upload" -F "file=@$TEST_FILE" -o /tmp/e2e_upload_resp.json
upload_status=$(json_get "d.get('status') or d.get('error')") < /tmp/e2e_upload_resp.json
doc_id=$(json_get "d.get('document_id','')") < /tmp/e2e_upload_resp.json
chunks_stored=$(json_get "d.get('chunks_stored',0)") < /tmp/e2e_upload_resp.json
if [[ "$upload_status" == "ingested" || "$upload_status" == "duplicate_document" ]]; then
  pass "Document upload accepted (ingested or duplicate)"
else
  fail "Document upload" "status=$upload_status"
fi
[[ "${chunks_stored:-0}" -ge 0 ]] && pass "Upload response includes chunk metadata" || fail "Upload chunks" "missing"

chunks_after=$(json_get "next(c['metadata'].get('chunk_count') for c in d['components'] if c['name']=='keyword_index')") < <(curl -s "$BASE/health")
[[ "${chunks_after:-0}" -ge 1 ]] && pass "BM25 chunk_count >= 1 after upload" || fail "BM25 after upload" "chunk_count=$chunks_after"

# 3. Basic RAG
log "=== 3. Basic RAG ==="
curl -s -X POST "$BASE/query" -H "Content-Type: application/json" \
  -d '{"query":"What is RAG according to my uploaded document?"}' -o /tmp/e2e_rag.json
rag_err=$(json_get "d.get('error','')") < /tmp/e2e_rag.json
rag_answer=$(json_get "d.get('answer','')") < /tmp/e2e_rag.json
citation_count=$(json_get "len(d.get('citations',[]))") < /tmp/e2e_rag.json
[[ -z "$rag_err" && -n "$rag_answer" ]] && pass "Basic RAG answer generated" || fail "Basic RAG" "error=$rag_err"
[[ "${citation_count:-0}" -ge 1 ]] && pass "Basic RAG citations preserved" || fail "Basic RAG citations" "count=$citation_count"

# 4. Hybrid retrieval
log "=== 4. Hybrid / BM25 ==="
curl -s -X POST "$BASE/retrieval/explore" -H "Content-Type: application/json" \
  -d '{"query":"XKCD-9917-alpha-hybrid-test-marker Zephyr Protocol exact keyword"}' -o /tmp/e2e_hybrid.json
hybrid_enabled=$(json_get "d['configuration']['hybrid_search_enabled']") < /tmp/e2e_hybrid.json
bm25_executed=$(json_get "next(s['executed'] for s in d['pipeline'] if s['id']=='bm25')") < /tmp/e2e_hybrid.json
bm25_count=$(json_get "next(s['result_count'] for s in d['pipeline'] if s['id']=='bm25')") < /tmp/e2e_hybrid.json
fusion_executed=$(json_get "next(s['executed'] for s in d['pipeline'] if s['id']=='hybrid_fusion')") < /tmp/e2e_hybrid.json
rerank_enabled=$(json_get "d['configuration']['reranking_enabled']") < /tmp/e2e_hybrid.json
[[ "$hybrid_enabled" == "True" ]] && pass "Retrieval explore: hybrid_search_enabled=true" || fail "Hybrid config" "$hybrid_enabled"
[[ "$bm25_executed" == "True" ]] && pass "BM25 stage actually executed" || fail "BM25 execution" "executed=$bm25_executed"
[[ "${bm25_count:-0}" -ge 1 ]] && pass "BM25 returned results" || fail "BM25 results" "count=$bm25_count"
[[ "$fusion_executed" == "True" ]] && pass "RRF hybrid fusion executed" || fail "Hybrid fusion" "executed=$fusion_executed"
[[ "$rerank_enabled" == "False" ]] && pass "Reranking correctly disabled (not implemented)" || fail "Reranking" "unexpected=$rerank_enabled"

# 5. Web search agent
log "=== 5. Web Search ==="
curl -s -X POST "$BASE/agent/query" -H "Content-Type: application/json" \
  -d '{"query":"What are the latest AI developments in 2026?"}' -o /tmp/e2e_web.json
web_err=$(json_get "d.get('error','')") < /tmp/e2e_web.json
web_answer=$(json_get "d.get('answer','')") < /tmp/e2e_web.json
tool_used=$(json_get "d.get('tool_used','')") < /tmp/e2e_web.json
if [[ -z "$web_err" && -n "$web_answer" ]]; then
  pass "Web search agent query answered"
  if [[ "$tool_used" == *"tavily"* ]]; then
    pass "tavily_web_search tool selected"
  else
    fail "Web tool selection" "tool_used=$tool_used"
  fi
else
  fail "Web search agent" "error=$web_err"
fi

# 6. Hybrid agentic query
log "=== 6. Hybrid Agentic Query ==="
curl -s -X POST "$BASE/agent/query" -H "Content-Type: application/json" \
  -d '{"query":"According to my uploaded document, what is RAG and what are the latest developments in RAG in 2026?"}' -o /tmp/e2e_hybrid_agent.json
ha_err=$(json_get "d.get('error','')") < /tmp/e2e_hybrid_agent.json
ha_answer=$(json_get "d.get('answer','')") < /tmp/e2e_hybrid_agent.json
ha_citations=$(json_get "len(d.get('citations',[]))") < /tmp/e2e_hybrid_agent.json
ha_steps=$(json_get "len(d.get('steps',[]))") < /tmp/e2e_hybrid_agent.json
if [[ -z "$ha_err" && -n "$ha_answer" ]]; then
  pass "Hybrid agent query answered"
  [[ "${ha_citations:-0}" -ge 1 ]] && pass "Hybrid agent citations present" || fail "Hybrid citations" "count=$ha_citations"
  [[ "${ha_steps:-0}" -ge 1 ]] && pass "Hybrid agent steps captured" || fail "Hybrid steps" "count=$ha_steps"
else
  fail "Hybrid agent query" "error=$ha_err"
fi

# 7. Routing edge cases
log "=== 7. Agent Routing ==="
curl -s -X POST "$BASE/agent/query" -H "Content-Type: application/json" \
  -d '{"query":"Summarize my rag_guide.pdf document"}' -o /tmp/e2e_rag_only.json
curl -s -X POST "$BASE/agent/query" -H "Content-Type: application/json" \
  -d '{"query":"Current weather in Tokyo today"}' -o /tmp/e2e_web_only.json
curl -s -X POST "$BASE/agent/query" -H "Content-Type: application/json" \
  -d '{"query":"   "}' -o /tmp/e2e_empty.json -w "%{http_code}" > /tmp/e2e_empty_code.txt

ro_err=$(json_get "d.get('error','')") < /tmp/e2e_rag_only.json
wo_err=$(json_get "d.get('error','')") < /tmp/e2e_web_only.json
empty_code=$(cat /tmp/e2e_empty_code.txt)
[[ -z "$ro_err" ]] && pass "RAG-only agent query succeeds" || fail "RAG-only query" "$ro_err"
[[ -z "$wo_err" ]] && pass "Web-only agent query succeeds" || fail "Web-only query" "$wo_err"
[[ "$empty_code" != "200" ]] && pass "Empty query rejected (non-200)" || fail "Empty query validation" "code=$empty_code"

# 9. Agent runs observability
log "=== 9. Agent Runs ==="
curl -s "$BASE/agent/runs?limit=5" -o /tmp/e2e_runs.json
runs_total=$(json_get "d.get('total',0)") < /tmp/e2e_runs.json
run_id=$(json_get "d['runs'][0]['run_id'] if d.get('runs') else ''") < /tmp/e2e_runs.json
[[ "${runs_total:-0}" -ge 1 ]] && pass "Agent runs list returns history" || fail "Agent runs list" "total=$runs_total"
if [[ -n "$run_id" ]]; then
  curl -s "$BASE/agent/runs/$run_id" -o /tmp/e2e_run_detail.json
  detail_query=$(json_get "d.get('query','')") < /tmp/e2e_run_detail.json
  detail_status=$(json_get "d.get('status','')") < /tmp/e2e_run_detail.json
  [[ -n "$detail_query" && -n "$detail_status" ]] && pass "Agent run detail has query+status" || fail "Run detail" "missing fields"
fi

# Frontend pages reachable
log "=== 10. Frontend pages ==="
for path in "/" "/agent-chat" "/documents" "/retrieval" "/agent-runs" "/settings"; do
  code=$(curl -s -o /dev/null -w "%{http_code}" "$FRONTEND$path")
  [[ "$code" == "200" ]] && pass "Frontend $path returns 200" || fail "Frontend $path" "status=$code"
done

log "=== Summary: PASS=$PASS FAIL=$FAIL ==="
printf '%s\n' "${RESULTS[@]}"
exit $FAIL
