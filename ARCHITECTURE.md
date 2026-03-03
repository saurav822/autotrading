# Architecture

SkopaqTrader is two codebases glued together: a vendored multi-agent LLM framework (`tradingagents/`) and a custom execution and infrastructure layer (`skopaq/`). The split is intentional. The upstream handles the intelligence; the skopaq layer handles everything that has to work reliably in production.

---

## The core insight

The upstream TradingAgents framework is already good at generating reasoned BUY/SELL/HOLD decisions. What it doesn't have is:
- Indian market broker integration
- Safety enforcement that can't be bypassed
- A way to actually run unattended every day
- Cost optimization (running 10 Claude Opus calls per analysis adds up fast)

SkopaqTrader wraps the upstream as a black box and adds all of that around it.

---

## Module map

```
skopaq/
├── config.py           # Single source of truth for all configuration
├── constants.py        # Immutable safety rules — no runtime overrides
│
├── graph/              # The seam between upstream and skopaq
│   └── skopaq_graph.py # Calls propagate(), parses signal, routes to executor
│
├── execution/          # The production-grade order pipeline
│   ├── executor.py          # Orchestrates: sizer → safety → router
│   ├── safety_checker.py    # Enforces SafetyRules on every signal
│   ├── order_router.py      # Routes to paper engine or live broker
│   ├── position_monitor.py  # AI + hard stop + EOD exit loop
│   └── daemon.py            # FSM that runs the full daily session
│
├── llm/                # Multi-model intelligence layer
│   ├── model_tier.py        # Per-role LLM assignment (Gemini, Claude, Grok)
│   ├── env_bridge.py        # SKOPAQ_* → standard LLM env vars
│   └── cache.py             # Redis LangCache integration
│
├── broker/             # Market connectivity
│   ├── client.py            # INDstocks REST API
│   ├── paper_engine.py      # In-memory paper trading simulator
│   ├── models.py            # Typed order/position/quote models
│   ├── token_manager.py     # Encrypted token storage + health check
│   ├── scrip_resolver.py    # Symbol → NSE security_id mapping
│   ├── binance_client.py    # Binance REST
│   ├── binance_ws.py        # Binance WebSocket feeds
│   └── binance_auth.py      # Authenticated Binance trading
│
├── scanner/            # Pre-analysis opportunity finder
│   ├── engine.py            # Parallel 30s scan cycles on NIFTY 50
│   ├── screen.py            # 3-LLM screening logic
│   └── watchlist.py         # NIFTY 50 symbol list + INDstocks scrip codes
│
├── risk/               # Position sizing and market regime
│   ├── atr.py               # ATR calculation from OHLCV candles
│   ├── position_sizer.py    # Converts confidence + ATR → lot size
│   ├── regime.py            # VIX/NIFTY SMA market regime detector
│   ├── calendar.py          # NSE event calendar (RBI, F&O expiry, elections)
│   ├── concentration.py     # Sector concentration cap
│   └── drawdown.py          # Daily/weekly/monthly loss accumulator
│
├── memory/             # Persistent agent learning
│   ├── store.py             # Supabase-backed BM25 memory store
│   └── lifecycle.py         # Load at session start, save after reflection
│
├── agents/             # Skopaq-specific agents
│   └── sell_analyst.py      # LLM-powered exit decision agent
│
├── api/                # REST backend
│   └── server.py            # FastAPI app (health, analyze, trade endpoints)
│
├── db/                 # Database layer
│   ├── client.py            # Async Supabase client
│   ├── repositories.py      # CRUD for trades, drawdown, profiles
│   └── models.py            # Table schemas
│
├── blockchain/         # Crypto infrastructure
│   ├── gas.py               # Gas price oracle (ETH, Polygon, Arbitrum, Optimism)
│   └── whales.py            # Large transaction monitoring
│
└── cli/                # User interface
    ├── main.py              # Typer commands + shared helper functions
    ├── display.py           # Rich tables, panels, progress
    └── theme.py             # Color scheme
```

---

## Key abstractions

### SkopaqTradingGraph

The main integration point. Lives in `skopaq/graph/skopaq_graph.py`.

```python
graph = SkopaqTradingGraph(upstream_config, executor, selected_analysts=[...])
result = await graph.analyze_and_execute("RELIANCE", "2026-03-01")
```

Internally it:
1. Lazy-initializes `TradingAgentsGraph` (upstream) on first call
2. Calls `upstream.propagate(symbol, date)` — this runs all the agents
3. Parses the returned `(state, decision)` into a typed `TradingSignal`
4. Extracts confidence score from the risk debate state (regex → explicit keys → heuristic → fallback 50)
5. Routes the signal through `executor.execute_signal()`

The lazy init is important — upstream graph construction is slow (initializes all LLM clients and tools). We don't pay that cost until the first trade.

### SafetyChecker

Enforces `SafetyRules` before every order. Checks:
- Position size vs max_position_pct × capital
- Order value vs max_order_value_inr
- Number of open positions vs max_open_positions
- Market hours (skipped in paper mode via `PAPER_SAFETY_RULES`)
- Daily/weekly/monthly drawdown limits
- Sector concentration (via `ConcentrationChecker`)

`SafetyRules` is a frozen dataclass. Calling `SAFETY_RULES.max_position_pct = 0.5` raises `FrozenInstanceError`. This is deliberate — the rules exist precisely so they can't be changed by a buggy learning loop or misconfigured environment.

### TradingDaemon

The daemon (`skopaq/execution/daemon.py`) is a finite state machine with 8 phases:

```
IDLE → PRE_OPEN → SCANNING → ANALYZING → TRADING → MONITORING → CLOSING → REPORTING → SHUTDOWN
```

Each phase is a separate async method, wrapped in `_timed_phase()` which records wall-clock duration. The FSM doesn't need a state transition table — it's sequential Python with `asyncio.Event` for graceful shutdown.

Stop event propagation: the daemon checks `self._stop.is_set()` before every major operation, not just in sleep waits. This means Railway's SIGTERM → stop event → clean shutdown within one cycle.

The daemon doesn't import most of its dependencies at module level. Imports happen inside `_phase_pre_open()` to avoid circular import chains (the execution stack imports from CLI helpers, which import from config, etc.).

### PositionMonitor

Three-tier sell logic runs in a polling loop every N seconds:

```
Tier 1 (every poll):     Hard stop — if unrealized loss > hard_stop_pct, sell immediately
Tier 2 (every N polls):  AI analyst — LLM looks at current P&L + market data, decides to hold or exit
Tier 3 (at 15:20 IST):  EOD safety net — force close everything
```

The AI tier is intentionally infrequent (every 6 polls = ~60s by default). LLM calls are slow and we don't want latency to delay the hard stop check.

Min profit gate: the monitor won't submit a sell if estimated net P&L (after ₹120 brokerage) is negative or below threshold. This prevents the AI from generating churn that enriches the broker.

### Multi-model tiering

`skopaq/llm/model_tier.py` maintains a preference list per role:

```python
"research_manager": [
    ("anthropic", "claude-opus-4-6"),
    ("google", "gemini-3-flash-preview"),   # fallback
]
```

`build_llm_map()` iterates roles, checks if the preferred provider's API key is in the environment, and assigns the first available model. LLM instances are cached by `(provider, model)` key to avoid duplicate client construction.

The result is a `dict[str, BaseChatModel]` that gets passed into the upstream `TradingAgentsGraph` constructor via `config["llm_map"]`. The upstream graph uses it to route each agent to its assigned model.

### Semantic LLM cache

`skopaq/llm/cache.py` implements LangChain's `BaseCache` interface backed by Redis LangCache (Upstash). Cache keys include the model name and a content hash, so:
- Cross-model cache pollution is impossible
- Memory updates (new lessons → different content hash) automatically invalidate stale responses

Hit rates of 40–45× on re-analysis of the same stock within a session are realistic, since news and fundamentals don't change minute-to-minute.

---

## Data flow: a full trade

```
1. skopaq daemon --once --paper

2. PRE_OPEN phase
   TokenManager.get_health()           → validate INDstocks token
   INDstocksClient.__aenter__()        → open HTTP session
   bridge_env_vars(config)             → SKOPAQ_GOOGLE_API_KEY → GOOGLE_API_KEY
   build_llm_map()                     → {role: BaseChatModel}
   SkopaqTradingGraph(config, executor)→ lazy-init upstream on first call

3. SCANNING phase
   ScannerEngine.run_cycle()
     INDstocksClient.get_quotes(NIFTY50)           → batch live prices
     [Gemini, Grok, Perplexity].screen(quotes)     → parallel LLM scoring
     merge + rank by urgency                       → [ScannerCandidate]

4. ANALYZING phase (per candidate)
   SkopaqTradingGraph.analyze_and_execute(symbol, date)
     TradingAgentsGraph.propagate(symbol, date)
       [market, news, social, fundamentals].analyze()  → parallel, uses llm_map
       bull_researcher.research() + bear_researcher.research()
       research_manager.synthesize()
       aggressive/neutral/conservative.debate()
       risk_manager.evaluate()              → CONFIDENCE: 72
       trader.decide()                      → "BUY RELIANCE..."
     _parse_signal(state, decision)         → TradingSignal(action=BUY, confidence=72)
     executor.execute_signal(signal)
       position_sizer.size(signal, capital) → quantity = 14 shares
       safety_checker.check(order)          → pass
       order_router.route(order)            → PaperEngine.place_order()
                                            → fill at live quote price

5. MONITORING phase
   PositionMonitor.run()
     every 10s: check hard stop (4% loss → sell)
     every 60s: sell_analyst.should_exit(position, market_data) → HOLD/SELL
     at 15:20:  force close all

6. CLOSING phase
   OrderRouter.get_positions() → [Position]
   For each remaining: TradingSignal(SELL, quantity=all) → executor
```

---

## Design decisions and tradeoffs

### Why vendor upstream instead of pip-installing it?

We needed surgical modifications to the upstream graph: adding `llm_map` injection, parallel analyst execution (state reducers), the INDstocks data vendor, and crypto-specific agents. These aren't cleanly extensible via config. Vendoring lets us make precise changes while preserving the upstream structure. All 34 changes are documented in `UPSTREAM_CHANGES.md` with original → modified diffs.

Downside: upstream updates require manual rebasing. We track this with `UPSTREAM_CHANGES.md` and a `git diff upstream-v0.2.0..HEAD -- tradingagents/` command.

### Why a separate scanner phase instead of running analysis on all stocks?

Full multi-agent analysis costs ~$0.40–$0.80 per symbol in LLM API costs (without caching). Running it on all 50 NIFTY stocks daily would be expensive and slow. The scanner uses cheap, fast models (Gemini Flash + Grok + Perplexity) to narrow 50 stocks to 3–5 high-urgency candidates, then runs the expensive analysis only on those.

### Why ATR-based sizing instead of fixed lot sizes?

Fixed lot sizes create inconsistent risk profiles across different price ranges and volatilities. A 100-share order for Reliance (₹2800) is very different from the same order for HDFC Bank (₹1700). ATR sizing normalizes by volatility: `quantity = (risk_per_trade × capital) / (atr_multiplier × atr_value × price)`. Every trade risks roughly the same INR amount relative to recent volatility.

### Why frozen dataclasses for SafetyRules?

The learning loop (reflection → memory → strategy evolution) needs guardrails it literally cannot circumvent. A frozen dataclass raises `FrozenInstanceError` at runtime if anything tries to mutate it. Config files, environment variables, and API payloads all pass through Python — none of them can override a frozen dataclass field. This is a deliberate architectural constraint.

### Why local imports inside daemon phase methods?

The execution stack has a dependency cycle risk: `daemon.py` needs to import from `cli/main.py` for shared helpers (`_build_upstream_config`, `_run_scan`, etc.), and `cli/main.py` imports from everything else. Moving the imports inside the method bodies breaks the cycle at module load time while still getting the actual symbols at call time. It's a compromise — the alternative is refactoring the shared helpers into a separate module.

### Why paper mode as the default?

`config.trading_mode` defaults to `"paper"`. The CLI's `trade` and `daemon` commands require an explicit `--live` flag AND a confirmation prompt. This means a misconfigured environment or a new developer accidentally running the daemon doesn't place real orders. Paper mode uses the same code path (PaperEngine implements the same interface as the live broker client), so the fallback costs nothing in terms of correctness testing.

### Why async throughout?

The scanner runs 3 LLMs concurrently. The position monitor runs a polling loop while the broker client streams quotes. The daemon phases are async so the stop event can interrupt any sleep wait instantly. The upstream graph calls LangChain tools, which are async-compatible. Making the entire stack async avoids the thread pool overhead and lets the event loop handle all concurrency.

---

## Upstream modifications summary

See `UPSTREAM_CHANGES.md` for the full list. Key categories:

**Phase 2 (core integration):**
- `llm_map` parameter in `TradingAgentsGraph.__init__()` and `setup.py`
- INDstocks data vendor + registration in `dataflows/interface.py`
- Parallel analyst fan-out via LangGraph state reducers

**Phase 2.5 (robustness):**
- Comma-separated indicator string splitting (upstream had a bug)
- yfinance symbol suffix stripping (`.NS` / `.BO` confusion)

**Phase 3 (crypto):**
- 7 new analyst agents (on-chain, DeFi, funding)
- 9 modified debate consumers to handle new analyst reports
- `propagation.py` initial state extended for crypto report fields

**Confidence scoring:**
- Risk manager prompt addition: outputs `CONFIDENCE: <N>` in judge decision
- `_extract_confidence()` parser in `skopaq_graph.py`

---

## Known limitations

- **Single-threaded analysis:** Candidates are analyzed sequentially (capital depletes with each BUY, so ordering matters). Parallel analysis would require capital pre-allocation logic.
- **No intraday data for analysis:** The upstream agents use daily OHLCV from yfinance/INDstocks. Intraday patterns (opening range, VWAP) aren't in the agent context.
- **Regime detection is disabled by default:** `SKOPAQ_REGIME_DETECTION_ENABLED=false`. The VIX/NIFTY SMA regime detector needs calibration with live data before it's safe to rely on.
- **Memory isn't global:** Agent memories are stored per-session and loaded at daemon start. If two daemon instances run (shouldn't happen, but), they'd overwrite each other's memories.
- **LangCache miss on first run:** The semantic cache only helps on repeated analysis of the same stock. Cold starts always hit the LLM APIs.
