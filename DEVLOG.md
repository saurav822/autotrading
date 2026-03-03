# Development Log

Chronological notes on how SkopaqTrader was built. Reasoning, wrong turns, and what changed.

---

## Phase 0 — Starting point

Discovered TradingAgents (TauricResearch) while looking at LLM-based trading research. The framework had a solid multi-agent pipeline — market analyst, news analyst, researchers, risk manager, trader agent — but it was built for US equities (yfinance, Alpha Vantage) and had no concept of execution. It would give you a BUY/SELL/HOLD recommendation but couldn't actually place an order.

The gap was obvious: Indian equities (NSE/BSE), a real broker API, and a production-safe execution layer. That became the scope.

Decision: vendor the upstream as `tradingagents/` rather than forking or pip-installing. We'd need surgical modifications (per-role LLM injection, parallel analyst execution, INDstocks data vendor) that aren't cleanly achievable through configuration alone. Vendoring lets us track exactly what changed and why.

---

## Phase 1 — Config and project scaffold

First thing: get configuration right before anything else. Made the mistake in a previous project of hardcoding strings and env var names all over the place. This time: a single Pydantic Settings class with `env_prefix="SKOPAQ_"`, loaded from `.env`.

```python
class SkopaqConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SKOPAQ_",
        extra="ignore",
    )
    trading_mode: Literal["paper", "live"] = "paper"
    # ...
```

`extra="ignore"` is important — lets you add custom vars to `.env` without breaking validation.

The `env_bridge.py` module came from a specific pain point: upstream LangChain providers expect `GOOGLE_API_KEY`, not `SKOPAQ_GOOGLE_API_KEY`. Rather than modifying upstream to accept our key names, we just copy the values at startup:

```python
# SKOPAQ_GOOGLE_API_KEY → GOOGLE_API_KEY (if standard var not already set)
bridge_env_vars(config)
```

The "if not already set" clause means developers can override with standard env vars directly, which is useful in CI environments where you don't want the SKOPAQ_ prefix.

---

## Phase 2 — Broker integration (INDstocks)

INDstocks API was the main unknown. Spent significant time on the auth model: tokens (not keys), 24-hour TTLs, no OAuth refresh. The `TokenManager` stores the token encrypted at `~/.skopaq/token.json`. The encryption key is derived from the machine's hostname — not perfect, but sufficient to avoid plaintext tokens in home directories.

The API had several gotchas that took time to find:

**Symbol format:** `scrip-codes=NSE_2885` not `symbols=NSE:RELIANCE`. Every symbol has to be resolved to a numeric security ID first. Added `scrip_resolver.py` with a mapping for the NIFTY 50 + ability to look up arbitrary symbols.

**Timestamps:** Historical endpoint wants milliseconds, returns seconds. Off by 1000× is a silent bug that gives you wrong candles. Added explicit conversion comments everywhere this happens.

**Auth header:** `Authorization: TOKEN` not `Authorization: Bearer TOKEN`. The upstream HTTP clients all add "Bearer" by default — had to override.

The paper engine was built in parallel with the live client, sharing the same interface (`BaseClient`). This means the execution layer can't tell the difference between paper and live, which is exactly what we wanted.

---

## Phase 2.5 — Making the upstream actually work

Before adding anything, tried running the upstream `propagate()` on a NIFTY stock. Two immediate bugs:

1. **Technical indicator string splitting:** The upstream joins indicator names with spaces: `"macd rsi bbands"`. The INDstocks vendor and yfinance integration expected a comma-separated list. Fixed with `indicator_str.replace(",", " ").split()` → normalize on both ends.

2. **yfinance symbol suffixes:** The upstream's LLM agents generate symbols like `RELIANCE` (no suffix). yfinance needs `RELIANCE.NS` for NSE stocks. But the INDstocks vendor needs `RELIANCE` (no suffix). The router in `dataflows/interface.py` now adds/strips `.NS`/`.BO` based on which vendor is active.

These felt like minor bugs but caused silent failures (yfinance returning no data, or wrong data). The test suite for this is `tests/unit/dataflows/test_indstocks_vendor.py`.

---

## Phase 3 — Execution pipeline

The execution pipeline (`executor.py`, `safety_checker.py`, `order_router.py`) was designed around a single constraint: safety rules must be impossible to circumvent.

Initial approach was safety rules as a config dict. Realized that's insufficient — a config dict can be mutated at runtime, passed as a parameter, or loaded from an untrusted source. Changed to frozen dataclasses:

```python
@dataclass(frozen=True)
class SafetyRules:
    max_position_pct: float = 0.15
    # ...
```

Trying to set `SAFETY_RULES.max_position_pct = 0.5` raises `FrozenInstanceError` immediately. The daemon uses a tighter `DAEMON_SAFETY_RULES` variant (3 positions instead of 5, ₹2L order cap instead of ₹5L) because unattended = less tolerance for edge cases.

The position sizer came from a specific frustration with fixed lot sizes: they create wildly different risk profiles across stocks at different prices and volatilities. ATR sizing made the math sensible: every trade risks approximately the same INR amount relative to recent price movement.

```
quantity = (risk_per_trade_pct × capital) / (atr_multiplier × ATR × price)
```

With `risk_per_trade_pct=0.01` and `atr_multiplier=2.0`, each trade risks roughly 1% of capital with a 2×ATR stop. Confidence scaling then reduces this proportionally for low-confidence signals.

---

## Phase 4 — Multi-model tiering

Running all agents on Claude Opus is expensive (~$0.40/analysis at claude-opus-4-6 rates). Most analyst roles don't need the strongest model — they're doing structured data extraction, not complex judgment.

The tiering insight: only two roles actually need powerful reasoning:
- **Research Manager** — arbitrates between bull and bear arguments
- **Risk Manager** — assigns the confidence score that determines position size

Everything else (data extraction, drafting arguments, final trade formatting) runs fine on Gemini Flash.

The Grok assignment for Social Analyst came from practical testing: Grok has different training data (more Twitter/X context) and consistently gives different (and often more useful) sentiment signals than Gemini on social-driven moves.

Perplexity Sonar is scanner-only. It doesn't support tool calling (`bind_tools()` returns 404 via OpenRouter), so it can't serve as a LangGraph agent. The news analyst needs tool calling to fetch actual news articles. Perplexity's value is in the scanner: it's essentially a search engine with LLM synthesis, good at "what happened with this stock in the last 48 hours?"

```python
_ROLE_PREFERENCES = {
    "research_manager": [("anthropic", "claude-opus-4-6"), ("google", "gemini-3-flash-preview")],
    "risk_manager":     [("anthropic", "claude-opus-4-6"), ("google", "gemini-3-flash-preview")],
    "social_analyst":   [("openrouter", "x-ai/grok-3-mini"), ("google", "gemini-3-flash-preview")],
    # everything else: Gemini Flash
}
```

Graceful fallback: if Anthropic key isn't set, research manager falls back to Gemini. The system always runs — it just runs with less capable models.

---

## Phase 5 — Scanner

The full analysis pipeline is slow (60–120s per stock) and expensive. Running it on all 50 NIFTY stocks every morning would be both impractical and costly. Needed a fast pre-filter.

First attempt: simple price momentum screen. Discarded — too mechanical, missed news-driven moves.

Second attempt: single LLM with NIFTY 50 batch quotes. Better, but one model's blind spots become the system's blind spots.

Final: three LLMs in parallel, each with a different context:
- **Gemini Flash:** technicals + fundamentals (P/E relative to sector, price trend)
- **Grok:** social sentiment (unusual Twitter/X activity, retail interest)
- **Perplexity Sonar:** news search (recent announcements, catalyst detection)

Each screener outputs a score 0–100 and an urgency signal. The engine merges by weighted average and returns the top N candidates.

The 30-second cycle interval came from INDstocks rate limits + LLM latency. In practice, a scan cycle takes 15–25s, so 30s gives breathing room.

```python
# Parallel screening — all three fire at the same time
results = await asyncio.gather(
    gemini_screener.screen(quotes),
    grok_screener.screen(quotes),
    perplexity_screener.screen(quotes),
    return_exceptions=True,
)
```

`return_exceptions=True` means one screener failing doesn't kill the whole cycle. The surviving screeners contribute their scores.

---

## Phase 6 — Daemon FSM

Running the daemon as a single linear `async def` worked in testing but was fragile: any unhandled exception in one phase would exit the whole process, leaving open positions unmonitored.

Refactored into an FSM with explicit phase tracking:

```python
class DaemonPhase(str, Enum):
    IDLE = "idle"
    PRE_OPEN = "pre_open"
    SCANNING = "scanning"
    ANALYZING = "analyzing"
    TRADING = "trading"
    MONITORING = "monitoring"
    CLOSING = "closing"
    REPORTING = "reporting"
    SHUTDOWN = "shutdown"
```

The `_timed_phase()` wrapper records wall-clock time for each phase (useful for diagnosing slow LLM calls), and each phase method can fail independently without aborting subsequent phases.

Error recovery paths:
- Scanner failure → 0 candidates → skip to REPORTING (no positions to worry about)
- Individual analysis failure → log + skip candidate, continue to next
- All trades rejected → 0 positions → MONITORING exits immediately
- SIGTERM → stop event set → SCANNING returns empty → jump to CLOSING

The CLOSING phase is a safety net: it always runs, even after a crash in MONITORING. It checks for remaining open positions and force-sells them. In testing, this saved us from overnight exposure twice.

Local imports inside `_phase_pre_open()` were a consequence of a circular import problem: daemon imports CLI helpers (`_build_upstream_config`, `_run_scan`), CLI imports from everywhere else. Moving the imports inside the method body breaks the cycle at module load time while still working at call time. Not ideal — documented as a known technical debt in ARCHITECTURE.md.

---

## Phase 7 — Position monitor

Three-tier design evolved from thinking about failure modes:

**What if the hard stop doesn't trigger?** — Add AI analyst review every N cycles.
**What if the AI analyst makes a bad call?** — Keep the hard stop tier above it (always runs first).
**What if both fail near EOD?** — Force close at 15:20 IST regardless.

The AI tier being infrequent (every 6 polls ≈ 60s) is intentional. LLM calls take 3–10s. If we called the AI on every 10s poll, we'd have overlapping calls, timing issues, and the hard stop check would be delayed by LLM latency. Separating them by frequency means the hard stop always runs in <1s, and the AI gets enough time between calls to not overlap.

Min profit gate came from a real observation during paper testing: the AI sell analyst would sometimes recommend exiting a small gain, but the gain was smaller than the estimated ₹120 brokerage round-trip cost. Net result: a profitable trade turned into a small loss. Added a hard override in the monitor that blocks the sell if `net_pnl < min_profit_threshold_inr`.

---

## Phase 8 — Agent memory + reflection

The upstream already had a reflection mechanism — after a trade closes, you call `reflect_and_remember(returns)` and the agents update their in-memory BM25 stores with lessons learned. The problem: these stores lived only in memory. Process restart → everything lost.

Added `MemoryStore` backed by Supabase to persist the BM25 indexes:
- `load()`: called at session start, restores memories into the upstream graph's agent objects
- `save()`: called after each reflection, writes updated memories to Supabase
- TTL-based eviction: old memories automatically expire so the store doesn't grow indefinitely

The BM25 index is stored as JSON in Supabase. Not the most efficient format, but it's simple and the memory entries are small enough that deserialization isn't a bottleneck.

Reflection is triggered after each successful BUY trade (via `_run_lifecycle()` in the CLI). The agent receives the current session's realized P&L context and updates its memories for future sessions.

---

## Phase 9 — Crypto support

Added when a few users asked about using the platform for crypto trading (BTC, ETH). The architecture generalized well — the main additions were:

1. Three new analyst agents: OnChain (Blockchair API for transaction flow), DeFi (DeFiLlama/CoinGecko for protocol TVL and tokenomics), Funding (Binance Futures funding rates as a sentiment proxy)

2. Binance broker integration: REST client for spot orders, WebSocket for real-time feeds, authenticated trading client with API key/secret

3. Gas oracle and whale alert monitoring in `skopaq/blockchain/`

The crypto analysts activate when `asset_class=crypto` in config. The upstream graph's `setup.py` was modified to conditionally include them in the analyst fan-out.

One gotcha: the crypto analysts' output format is different from equity analysts. The debate consumers in `risk_mgmt/` needed updates to handle the new report fields. This was the most tedious part of the crypto integration — updating 9 files, each with similar but slightly different patterns.

---

## Phase 10 — Semantic LLM cache

Added Redis LangCache late in development, after noticing that re-analyzing the same stock within a session (e.g., scanning → full analysis → monitor asking "should I sell?") was hitting the LLM APIs multiple times for nearly identical prompts.

LangChain's `BaseCache` interface made this clean to implement. The key insight was cache key construction:

```python
# Key includes: model name + content hash
# This prevents:
# - Cross-model pollution (Claude and Gemini giving different answers to same prompt)
# - Stale hits after memory updates (new lessons → different content hash)
cache_key = f"{model_name}:{content_hash(prompt + memory_context)}"
```

Observed ~45× speedup on the second analysis of the same stock within a session. The cache is most valuable when running the daemon with scanner → full analysis, because the scanner's summary of each stock often appears verbatim in the analyst prompts.

---

## What I'd do differently

**Start with a simpler broker abstraction.** The `BaseClient` interface with both INDstocks and Binance implementations added complexity early. Should have built for INDstocks only first, then generalized.

**Fewer phases in the daemon.** PRE_OPEN and REPORTING could be merged with their neighbors. The 8-phase FSM feels over-engineered for what is essentially 4 meaningful operations (scan, analyze+trade, monitor, close).

**Better test coverage for the scanner.** The scanner unit tests mock the LLM responses, which means they test orchestration logic but not prompt quality. Should have kept a small set of live integration tests that run weekly with real LLM calls.

**The CLI helper module coupling.** `daemon.py` importing from `cli/main.py` for `_build_upstream_config` and `_run_scan` is a design smell. Those functions should live in a shared `skopaq/utils/` module that neither CLI nor daemon imports from the other.
