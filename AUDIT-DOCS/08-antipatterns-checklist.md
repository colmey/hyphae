# 08 — Anti-Patterns & the Include/Avoid Checklist

**Research basis:** Doc 09 (Anti-Patterns & Checklist).
**Verdict:** ✅ **Mostly clean** — the harness avoids every *architectural* and *engineering* anti-pattern the research warns about. The live exceptions are concentrated in the two gap areas already covered: **no evals** (Doc 08) and **unmanaged context** (Doc 05), plus the **no approval gates** safety gap (Doc 07).

Doc 09's headline insight: *the majority of agent failures are context/data-layer, not architecture.* The harness is architecturally sound; its residual risk is exactly where the research predicts — the context layer and the absence of measurement.

---

## The 13 harness anti-patterns

### Architectural (~20%)
| # | Anti-pattern | Status | Evidence / note |
|---|---|---|---|
| 1 | Monolithic mega-prompt | ✅ Avoided | Orchestrator generates 2–6 sentence prompts (`config/orchestrator_prompt.md`); no giant system prompt |
| 2 | Invisible state (LLM-as-memory) | 🟡 Partial risk | History *is* persisted in `Session`, but it's unbounded and in-memory only; no compaction → relies on the window to "remember" on long runs (`04`) |
| 3 | All-or-nothing autonomy | 🔴 Present | No approval gates for write/destructive tools (`06`) |
| 4 | Compounding error cascade | 🟡 Partial | Per-step reliability is strong (retries, errors-as-feedback); cascade-stopping is weak — nudge never aborts, no spend cap (`06`) |

### Execution / tool-layer (~25%)
| # | Anti-pattern | Status | Evidence / note |
|---|---|---|---|
| 5 | Tool bloat (>20 tools) | ✅ Avoided | Orchestrator selects smallest subset; 2 servers configured (`03`, `05`) |
| 6 | Hallucinated tool arguments | 🔴 Present | No arg validation before execution ([agent/loop.py:412-417](../agent/loop.py#L412-L417)); OpenAI provider coerces bad args to `{}` ([llm/providers/openai.py:315-319](../llm/providers/openai.py#L315-L319)) (`03`) |
| 7 | Schema drift in tool calls | 🟡 Partial | Tools fetched live from MCP at startup (no stale baked-in schemas — good), but no validation/alert when a server's schema changes mid-run |
| 8 | Chronic tool-call failure rate | ✅ Mitigated | Errors-as-feedback + per-tool timeout + stall detection ([agent/loop.py:418-460](../agent/loop.py#L418-L460)) |

### Data / context-layer (~55% — the silent majority)
| # | Anti-pattern | Status | Evidence / note |
|---|---|---|---|
| 9 | Dumb RAG / context flooding | ✅ Avoided | No pre-loaded RAG; agent-driven web search instead (`04`) |
| 10 | Stale context / context drift | ✅ Mostly | Tools/data fetched at use-time via MCP, not pre-loaded |
| 11 | Schema-drift blindness | 🟡 Partial | No alert on upstream tool-schema change (same as #7) |
| 12 | Uncertified source selection | ⬜ N/A | No data-source registry; out of scope for current tools |
| 13 | Missing business context | ⬜ N/A | Delegated to MCP tool descriptions |

> Doc 09 stresses these context/data failures are **silent** — they produce no exceptions, just wrong answers, and are caught by **evals**, not crashes. The harness's lack of evals (`07`) is what leaves #9–#13 unmonitored even where currently fine.

---

## Cross-cutting failure modes

| Mode | Status | Evidence |
|---|---|---|
| Context rot | 🔴 Present | No budget/compaction; unbounded history (`04`) — acute on the local-model default |
| Goal drift | 🟡 Partial | System prompt re-sent each turn; no pinned task header (`04`) |
| Retry loops / in-context locking | ✅ Mitigated | Stall detection on identical calls ([agent/loop.py:405-409](../agent/loop.py#L405-L409)) |
| Unverified progress | ✅ Mitigated | Tool results fed back; `truncated` distinguished from `end_turn` ([agent/loop.py:382-394](../agent/loop.py#L382-L394)) |
| Context fragmentation | ✅ Avoided | Single-agent, single thread (`05`) |

---

## Engineering ("over-engineering") anti-patterns

| Anti-pattern | Status | Evidence |
|---|---|---|
| Framework lock-in | ✅ Avoided | Owns its own loop/prompts/control flow; no LangGraph/CrewAI ([agent/loop.py](../agent/loop.py)) |
| Premature multi-agent | ✅ Avoided | Single agent + routing workflow (`05`) |
| Over-abstraction | ✅ Avoided | ~5,200 LOC core; seams added only where a second use exists |
| Hidden control flow | ✅ Avoided | Explicit, inspectable route → orchestrator → loop |
| Speculative generality | ✅ Avoided | `SessionStore` ABC etc. map to real, documented future needs, not imagined ones |
| Vendor coupling | ✅ Avoided | Provider registry + OpenAI-compatible client (`02`) |
| No evals | 🔴 Present | Smoke tests only; no outcome evals (`07`) |

---

## The Include / Avoid checklist (Doc 09), scored

### Include — the lean core (MUST-HAVE)
- [x] ✅ A single, readable **agent loop** with append-only history — [agent/loop.py:231-475](../agent/loop.py#L231-L475) *(history append-only; run state not yet a pure reducer — `01`)*
- [x] ✅ A narrow **model-client interface**; provider/model via config; OpenAI-compatible — [llm/client.py:46-75](../llm/client.py#L46-L75)
- [ ] 🟡 **Capability detection + graceful degradation** for weak/local models — partial; thinking ignored, no prompted-tool fallback (`02`)
- [ ] 🟡 A small **tool registry** with **schema validation** and errors-as-feedback — errors-as-feedback ✅, schema validation 🔴 (`03`)
- [ ] 🔴 A **context manager** enforcing a token budget; compaction — absent (`04`)
- [ ] 🟡 **Guardrails**: iteration cap, **spend cap, timeout, no-progress** — iteration cap + timeouts ✅; spend cap 🔴; no-progress nudges-only (`06`)
- [ ] 🔴 **Least-privilege** tools + a policy seam for dangerous actions — absent (`06`)
- [ ] 🔴 **Tracing** (= serialized event log) + a **small eval suite** — event log produced but not persisted; no evals (`07`)

### Add only when justified (OPTIONAL)
- [ ] ⬜ Long-term memory / RAG — not present (correct to defer)
- [ ] ⬜ Subagents (read-only fan-out, as a tool) — not present (correct to defer)
- [x] ✅ MCP adapter (curated external tools) — present and well-built (`03`)
- [ ] ⬜ OTel export + trace UI; production sampling — not present
- [ ] ⬜ Sandboxing / approval UI — not present

### Avoid (until proven necessary)
- [x] ✅ Multi-agent by default — **avoided**
- [x] ✅ Heavyweight framework owning control flow/prompts — **avoided**
- [x] ✅ Mega-prompt / 30+ tool registry — **avoided**
- [x] ✅ Pre-loading large context / dumb RAG — **avoided**
- [x] ✅ Vendor/model hardcoding; secrets in context — **avoided**
- [x] ✅ Abstractions/DSLs with a single use site — **avoided**
- [ ] 🔴 Shipping without any evals — **currently violated** (`07`)

---

## Summary

The harness scores **clean on every "Avoid" item except one** (no evals) and on **all engineering anti-patterns except one** (no evals). The remaining red marks cluster predictably: the **context layer** (rot, no budget/compaction), **measurement** (no trace/evals), and **safety authorization** (no policy/approval). That is a healthy profile — the foundation is right, and the open items are additive. They are sequenced in `09-improvement-roadmap.md`.
