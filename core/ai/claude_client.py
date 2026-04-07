"""Claude API client for trading analysis."""

import json
import asyncio
from datetime import datetime, timezone

import anthropic
from loguru import logger

from core.ai.models import TradeAnalysis, SignalEvaluation, PerformanceReview
from core.ai.prompts import (
    POST_TRADE_ANALYSIS_PROMPT,
    SIGNAL_EVALUATION_PROMPT,
    DAILY_REVIEW_PROMPT,
    WEEKLY_REVIEW_PROMPT,
)


HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-20250514"


class ClaudeAnalyzer:
    """AI-powered trading analyzer using Claude API."""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self._daily_token_usage = 0
        self._daily_cost_usd = 0.0
        self._reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _track_usage(self, model: str, input_tokens: int, output_tokens: int):
        """Track token usage and estimated cost."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._reset_date:
            logger.info(
                f"[AI] Yesterday's usage: {self._daily_token_usage} tokens, "
                f"${self._daily_cost_usd:.4f}"
            )
            self._daily_token_usage = 0
            self._daily_cost_usd = 0.0
            self._reset_date = today

        self._daily_token_usage += input_tokens + output_tokens

        # Cost estimation (per million tokens)
        if "haiku" in model:
            cost = (input_tokens * 1.0 + output_tokens * 5.0) / 1_000_000
        else:  # sonnet
            cost = (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000
        self._daily_cost_usd += cost

    def _call_claude(self, prompt: str, model: str = HAIKU_MODEL,
                     max_tokens: int = 1024, timeout: float = 15.0) -> tuple[str, int]:
        """Make a synchronous Claude API call. Returns (response_text, total_tokens)."""
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout,
        )
        text = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        self._track_usage(model, input_tokens, output_tokens)
        return text, input_tokens + output_tokens

    async def analyze_trade(
        self,
        position_id: str,
        strategy_name: str,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        entry_time: str,
        exit_time: str,
        pnl: float,
        pnl_pct: float,
        duration_seconds: int,
        exit_reason: str,
        entry_indicators: dict,
        strategy_stats: dict,
        active_positions: list[dict] = None,
    ) -> TradeAnalysis | None:
        """Analyze a closed trade. Runs in background thread to avoid blocking."""
        try:
            # Format duration
            mins = duration_seconds // 60
            hours = mins // 60
            duration_str = f"{hours}h {mins % 60}m" if hours > 0 else f"{mins}m"

            # Format indicators
            indicator_lines = []
            for k, v in (entry_indicators or {}).items():
                if isinstance(v, float):
                    indicator_lines.append(f"- {k}: {v:.4f}")
                elif isinstance(v, dict):
                    indicator_lines.append(f"- {k}: {v}")
                else:
                    indicator_lines.append(f"- {k}: {v}")
            indicators_str = "\n".join(indicator_lines) if indicator_lines else "No indicators recorded"

            # Format strategy stats
            trades = strategy_stats.get("recent_results", [])
            wins = sum(1 for t in trades if t > 0)
            losses = sum(1 for t in trades if t <= 0)
            win_rate = (wins / len(trades) * 100) if trades else 0
            avg_win = sum(t for t in trades if t > 0) / max(wins, 1)
            avg_loss = sum(t for t in trades if t <= 0) / max(losses, 1)
            consec_wins = strategy_stats.get("consecutive_wins", 0)
            consec_losses = strategy_stats.get("consecutive_losses", 0)
            streak_type = "wins" if consec_wins > consec_losses else "losses"
            streak_count = max(consec_wins, consec_losses)

            # Format active positions
            pos_lines = []
            for p in (active_positions or []):
                pos_lines.append(
                    f"- {p.get('strategy', '?')}:{p.get('symbol', '?')} "
                    f"{p.get('side', '?')} @ ${p.get('entry_price', 0):,.2f}"
                )
            positions_str = "\n".join(pos_lines) if pos_lines else "None"

            outcome = "win" if pnl > 0 else "lose"

            prompt = POST_TRADE_ANALYSIS_PROMPT.format(
                strategy_name=strategy_name,
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                entry_time=entry_time,
                exit_time=exit_time,
                pnl=pnl,
                pnl_pct=pnl_pct,
                duration=duration_str,
                exit_reason=exit_reason,
                entry_indicators=indicators_str,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                streak_type=streak_type,
                streak_count=streak_count,
                active_positions=positions_str,
                outcome=outcome,
            )

            text, tokens = await asyncio.to_thread(
                self._call_claude, prompt, HAIKU_MODEL, 512
            )

            # Try to extract a suggestion from the last paragraph
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
            suggestion = paragraphs[-1] if paragraphs else ""

            analysis = TradeAnalysis(
                position_id=position_id,
                analysis_text=text,
                patterns_identified=[],
                suggestion=suggestion,
                model_used=HAIKU_MODEL,
                tokens_used=tokens,
            )

            logger.info(
                f"[AI] Trade analysis complete for {strategy_name}:{symbol} "
                f"({tokens} tokens, ${self._daily_cost_usd:.4f} today)"
            )
            return analysis

        except anthropic.APITimeoutError:
            logger.warning(f"[AI] Trade analysis timed out for {strategy_name}:{symbol}")
            return None
        except Exception as e:
            logger.error(f"[AI] Trade analysis failed: {e}")
            return None

    async def evaluate_signal(
        self,
        strategy_name: str,
        symbol: str,
        direction: str,
        confidence: float,
        signal_reasoning: str,
        indicators: dict,
        recent_trades: list[dict],
        open_positions: list[dict],
        similar_losses: list[dict] = None,
    ) -> SignalEvaluation | None:
        """Evaluate a signal before execution. Returns adjusted confidence."""
        try:
            # Calculate strategy stats from recent trades
            strategy_trades = [t for t in recent_trades if t.get("strategy_name") == strategy_name]
            trade_count = len(strategy_trades)
            winners = sum(1 for t in strategy_trades if (t.get("net_pnl") or 0) > 0)
            win_rate = (winners / trade_count * 100) if trade_count else 50
            net_pnl = sum(t.get("net_pnl", 0) for t in strategy_trades)

            last_5 = strategy_trades[:5]
            last_5_str = ", ".join(
                f"{'W' if (t.get('net_pnl') or 0) > 0 else 'L'} ${t.get('net_pnl', 0):,.2f}"
                for t in last_5
            ) or "No recent trades"

            # Portfolio state
            same_dir = sum(1 for p in open_positions if p.get("side") == direction)
            total_exposure = sum(
                p.get("quantity", 0) * p.get("entry_price", 0) for p in open_positions
            )
            unrealized = sum(p.get("unrealized_pnl", 0) for p in open_positions)

            # Format indicators
            ind_lines = [f"- {k}: {v}" for k, v in (indicators or {}).items()]
            indicators_str = "\n".join(ind_lines) if ind_lines else "None"

            # Format similar losses
            loss_lines = []
            for loss in (similar_losses or []):
                loss_lines.append(
                    f"- {loss.get('symbol')} {loss.get('exit_reason', '?')}: "
                    f"${loss.get('net_pnl', 0):,.2f} ({loss.get('duration_seconds', 0) // 60}min)"
                )
            losses_str = "\n".join(loss_lines) if loss_lines else "None in last 7 days"

            prompt = SIGNAL_EVALUATION_PROMPT.format(
                strategy_name=strategy_name,
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                signal_reasoning=signal_reasoning,
                indicators=indicators_str,
                recent_trade_count=trade_count,
                win_rate=win_rate,
                net_pnl=net_pnl,
                last_5_results=last_5_str,
                open_positions_count=len(open_positions),
                same_direction_count=same_dir,
                total_exposure=total_exposure,
                unrealized_pnl=unrealized,
                similar_losses=losses_str,
            )

            text, tokens = await asyncio.to_thread(
                self._call_claude, prompt, HAIKU_MODEL, 256, timeout=5.0
            )

            # Parse JSON response
            try:
                # Handle potential markdown wrapping
                clean = text.strip()
                if clean.startswith("```"):
                    clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                result = json.loads(clean)
            except json.JSONDecodeError:
                logger.warning(f"[AI] Could not parse signal evaluation response: {text[:100]}")
                return None

            # Enforce: never inflate confidence
            adj = min(result.get("adjusted_confidence", confidence), confidence)

            evaluation = SignalEvaluation(
                adjusted_confidence=adj,
                reasoning=result.get("reasoning", ""),
                risk_flags=result.get("risk_flags", []),
                model_used=HAIKU_MODEL,
                tokens_used=tokens,
            )

            action = "PASS" if adj >= 0.65 else "BLOCK"
            logger.info(
                f"[AI] Signal gate [{action}] {strategy_name}:{symbol} {direction} "
                f"{confidence:.2f} -> {adj:.2f} ({tokens} tok)"
            )
            return evaluation

        except anthropic.APITimeoutError:
            logger.warning(f"[AI] Signal evaluation timed out for {strategy_name}:{symbol}")
            return None
        except Exception as e:
            logger.error(f"[AI] Signal evaluation failed: {e}")
            return None

    async def generate_daily_review(
        self,
        review_date: str,
        trades: list[dict],
        trade_analyses: list[dict],
        week_trades: list[dict],
        prior_week_trades: list[dict],
        market_regime: str = "Unknown",
    ) -> PerformanceReview | None:
        """Generate a daily performance review."""
        try:
            # Calculate stats
            total = len(trades)
            winners = [t for t in trades if (t.get("net_pnl") or 0) > 0]
            losers = [t for t in trades if (t.get("net_pnl") or 0) <= 0]
            net_pnl = sum(t.get("net_pnl", 0) for t in trades)

            best = max(trades, key=lambda t: t.get("net_pnl", 0)) if trades else {}
            worst = min(trades, key=lambda t: t.get("net_pnl", 0)) if trades else {}

            # Per-strategy breakdown
            strategies = {}
            for t in trades:
                s = t.get("strategy_name", "unknown")
                if s not in strategies:
                    strategies[s] = {"trades": 0, "wins": 0, "pnl": 0.0}
                strategies[s]["trades"] += 1
                strategies[s]["pnl"] += t.get("net_pnl", 0)
                if (t.get("net_pnl") or 0) > 0:
                    strategies[s]["wins"] += 1

            strat_lines = []
            for name, data in sorted(strategies.items()):
                wr = (data["wins"] / data["trades"] * 100) if data["trades"] else 0
                strat_lines.append(
                    f"- {name}: {data['trades']} trades, {wr:.0f}% WR, ${data['pnl']:,.2f}"
                )
            strat_str = "\n".join(strat_lines) if strat_lines else "No trades today"

            # Trade analyses summary
            analysis_lines = []
            for a in (trade_analyses or [])[:10]:
                analysis_lines.append(f"- {a.get('suggestion', a.get('analysis_text', '')[:100])}")
            analyses_str = "\n".join(analysis_lines) if analysis_lines else "No AI analyses available"

            # Week stats
            week_pnl = sum(t.get("net_pnl", 0) for t in week_trades)
            week_wins = sum(1 for t in week_trades if (t.get("net_pnl") or 0) > 0)
            week_wr = (week_wins / len(week_trades) * 100) if week_trades else 0
            prior_pnl = sum(t.get("net_pnl", 0) for t in prior_week_trades)

            prompt = DAILY_REVIEW_PROMPT.format(
                review_date=review_date,
                total_trades=total,
                winners=len(winners),
                losers=len(losers),
                net_pnl=net_pnl,
                best_trade=best.get("net_pnl", 0),
                best_strategy=best.get("strategy_name", "N/A"),
                worst_trade=worst.get("net_pnl", 0),
                worst_strategy=worst.get("strategy_name", "N/A"),
                strategy_breakdown=strat_str,
                trade_analyses_summary=analyses_str,
                week_pnl=week_pnl,
                week_win_rate=week_wr,
                prior_week_pnl=prior_pnl,
                market_regime=market_regime,
            )

            text, tokens = await asyncio.to_thread(
                self._call_claude, prompt, SONNET_MODEL, 1024, timeout=30.0
            )

            # Parse suggestions from response
            suggestions = []
            in_suggestions = False
            for line in text.split("\n"):
                line = line.strip()
                if "suggestion" in line.lower() or "top 3" in line.lower():
                    in_suggestions = True
                    continue
                if in_suggestions and line.startswith(("1.", "2.", "3.", "-")):
                    suggestions.append({
                        "priority": len(suggestions) + 1,
                        "suggestion": line.lstrip("0123456789.-) "),
                    })

            review = PerformanceReview(
                review_date=review_date,
                period="daily",
                summary=text,
                strategy_insights=strategies,
                suggestions=suggestions[:3],
                model_used=SONNET_MODEL,
                tokens_used=tokens,
            )

            logger.info(f"[AI] Daily review complete ({tokens} tokens)")
            return review

        except Exception as e:
            logger.error(f"[AI] Daily review failed: {e}")
            return None

    async def recommend_capital_allocation(
        self,
        strategy_stats: dict,
        total_capital: float,
        recent_trades: list[dict],
    ) -> dict[str, float] | None:
        """AI-powered capital rebalancing. Returns {strategy_name: new_capital} or None.

        Uses Claude Haiku for fast, cheap analysis of strategy performance data.
        Returns specific dollar allocations that sum to total_capital.
        """
        # Build performance summary
        from collections import defaultdict
        perf = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0, "time_mins": 0, "avg_pnl": 0})

        for t in recent_trades:
            strat = t.get("strategy_name", "unknown")
            pnl = t.get("net_pnl") or 0
            dur = (t.get("duration_seconds") or 0) / 60
            perf[strat]["pnl"] += pnl
            perf[strat]["trades"] += 1
            perf[strat]["time_mins"] += dur
            if pnl > 0:
                perf[strat]["wins"] += 1

        perf_lines = []
        for strat, stats in sorted(strategy_stats.items()):
            p = perf.get(strat, {"pnl": 0, "trades": 0, "wins": 0, "time_mins": 0})
            current_cap = stats.get("capital", 0)
            wr = p["wins"] / p["trades"] * 100 if p["trades"] > 0 else 0
            pnl_hr = p["pnl"] / (p["time_mins"] / 60) if p["time_mins"] > 60 else 0
            perf_lines.append(
                f"- {strat}: ${current_cap:.0f} allocated | {p['trades']} trades | "
                f"{wr:.0f}% WR | ${p['pnl']:+.2f} PnL | ${pnl_hr:+.2f}/hr in market"
            )

        prompt = f"""You are a quantitative portfolio manager for a crypto futures trading bot.

TASK: Recommend capital allocation across strategies. Return ONLY a JSON object mapping strategy names to dollar amounts. The amounts MUST sum to exactly ${total_capital:.0f}.

CURRENT PERFORMANCE (last 7 days):
{chr(10).join(perf_lines)}

Total capital: ${total_capital:.0f}

RULES:
1. Allocate MORE capital to strategies with high PnL/hr and positive PnL
2. Allocate LESS (but minimum $200) to strategies that are losing money or have 0 trades
3. No single strategy gets more than 50% of total capital
4. Every strategy gets at least $200 (floor) to stay alive for regime changes
5. Strategies with 0 trades in 7 days should get minimum allocation ($200)
6. Weight PnL/hr MORE than raw PnL (efficiency matters more than volume)
7. Consider win rate — below 35% suggests no real edge

Respond with ONLY valid JSON, no explanation. Example format:
{{"trend_breakout": 2500, "order_flow": 1000, "smart_money": 800, "crash_momentum": 300, "funding_reversion": 200, "regime_short": 200, "refined_liq_cascade": 200, "failed_breakout_short": 300}}"""

        try:
            response_text, tokens = await asyncio.to_thread(
                self._call_claude, prompt, HAIKU_MODEL, 256, 10.0
            )

            # Parse JSON from response
            import json
            # Strip markdown code blocks if present
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            allocations = json.loads(text)

            # Validate: all values positive, sum reasonable
            if not isinstance(allocations, dict):
                logger.warning(f"[AI Rebalance] Invalid response type: {type(allocations)}")
                return None

            # Normalize to exact total
            raw_sum = sum(allocations.values())
            if raw_sum <= 0:
                return None

            scale = total_capital / raw_sum
            result = {}
            for name, amount in allocations.items():
                if name in strategy_stats:
                    result[name] = max(200.0, amount * scale)

            # Final normalize after floor
            final_sum = sum(result.values())
            if final_sum > 0:
                final_scale = total_capital / final_sum
                result = {k: v * final_scale for k, v in result.items()}

            logger.info(f"[AI Rebalance] Recommendation received ({tokens} tokens)")
            return result

        except Exception as e:
            logger.warning(f"[AI Rebalance] Failed: {e}")
            return None

    # ==========================================
    # v7.0.2: Full AI Trading Intelligence Suite
    # ==========================================

    async def learn_trade_patterns(self, trades: list[dict], strategy_names: list[str]) -> dict | None:
        """#1: Analyze ALL historical trades to discover winning vs losing patterns.

        Returns learned rules per strategy, e.g.:
        {"trend_breakout": {"avoid_symbols": ["AVAXUSDT"], "min_vol_ratio": 2.0, "best_hours_utc": [14,15,16], "notes": "..."}}
        Runs daily. Uses Sonnet for deeper analysis.
        """
        if not trades:
            return None

        # Build per-strategy summaries
        from collections import defaultdict
        strat_data = defaultdict(lambda: {"wins": [], "losses": [], "by_symbol": defaultdict(lambda: {"w": 0, "l": 0, "pnl": 0})})

        for t in trades:
            strat = t.get("strategy_name", "unknown")
            pnl = t.get("net_pnl") or 0
            symbol = t.get("symbol", "?")
            ind = t.get("indicators_at_entry") or {}
            entry_hour = ""
            if t.get("entry_time"):
                try:
                    entry_hour = t["entry_time"][11:13]
                except Exception:
                    pass

            record = {
                "symbol": symbol, "pnl": round(pnl, 2),
                "htf_strength": ind.get("htf_strength", "?"),
                "vol_ratio": ind.get("vol_ratio", "?"),
                "rsi": ind.get("rsi", "?"),
                "confidence": t.get("signal_confidence", "?"),
                "hour_utc": entry_hour,
                "duration_min": int((t.get("duration_seconds") or 0) / 60),
            }

            if pnl > 0:
                strat_data[strat]["wins"].append(record)
            else:
                strat_data[strat]["losses"].append(record)

            strat_data[strat]["by_symbol"][symbol]["pnl"] += pnl
            if pnl > 0:
                strat_data[strat]["by_symbol"][symbol]["w"] += 1
            else:
                strat_data[strat]["by_symbol"][symbol]["l"] += 1

        # Build compact summary for prompt
        summary_lines = []
        for strat in strategy_names:
            d = strat_data.get(strat)
            if not d:
                summary_lines.append(f"\n### {strat}: 0 trades")
                continue
            total = len(d["wins"]) + len(d["losses"])
            wr = len(d["wins"]) / total * 100 if total else 0
            summary_lines.append(f"\n### {strat}: {total} trades, {wr:.0f}% WR")
            # Symbol breakdown
            for sym, sd in d["by_symbol"].items():
                sym_total = sd["w"] + sd["l"]
                sym_wr = sd["w"] / sym_total * 100 if sym_total else 0
                summary_lines.append(f"  {sym}: {sym_total} trades, {sym_wr:.0f}% WR, ${sd['pnl']:+.2f}")
            # Sample wins/losses (last 5 each)
            if d["wins"]:
                summary_lines.append(f"  Recent wins: {d['wins'][-5:]}")
            if d["losses"]:
                summary_lines.append(f"  Recent losses: {d['losses'][-5:]}")

        prompt = f"""You are a quantitative analyst. Analyze these crypto futures trading results and discover PATTERNS that separate winners from losers.

TRADE DATA (last 30 days):
{"".join(summary_lines)}

For EACH strategy, return a JSON object with learned rules. Focus on:
1. **avoid_symbols**: Symbols with <25% WR and negative PnL — list them
2. **min_vol_ratio**: If wins cluster above a volume threshold, specify it (null if no pattern)
3. **min_htf_strength**: If wins need stronger HTF trend, specify minimum (null if no pattern)
4. **best_hours_utc**: Hours (0-23) when win rate is notably higher (empty list if no pattern)
5. **max_rsi_entry**: If wins cluster below an RSI level, specify ceiling (null if no pattern)
6. **min_confidence**: Minimum confidence that produces positive expectancy (null if no pattern)
7. **notes**: One sentence explaining the key insight for this strategy

Respond with ONLY valid JSON. Example:
{{"trend_breakout": {{"avoid_symbols": ["AVAXUSDT"], "min_vol_ratio": 2.0, "min_htf_strength": null, "best_hours_utc": [], "max_rsi_entry": null, "min_confidence": 0.72, "notes": "Wins cluster on high-volume breakouts; AVAX has 15% WR"}},
"crash_momentum": {{"avoid_symbols": [], "min_vol_ratio": null, "min_htf_strength": null, "best_hours_utc": [8,9,10], "max_rsi_entry": 35, "min_confidence": null, "notes": "Best during Asian session with RSI <35"}}}}"""

        try:
            response_text, tokens = await asyncio.to_thread(
                self._call_claude, prompt, SONNET_MODEL, 1024, 30.0
            )
            import json
            text = response_text.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:])
            if text.endswith("```"):
                text = text[:-3]
            rules = json.loads(text.strip())
            logger.info(f"[AI Pattern] Learned rules for {len(rules)} strategies ({tokens} tokens)")
            return rules
        except Exception as e:
            logger.warning(f"[AI Pattern] Failed: {e}")
            return None

    async def detect_regime(self, market_data: dict) -> dict | None:
        """#2 + #5: AI regime detection + market context from multiple signals.

        Returns: {"regime": "trending_bull|trending_bear|ranging|high_vol_crisis",
                  "confidence": 0.8, "risk_level": "low|medium|high",
                  "reasoning": "...", "bias": "long|short|neutral"}
        Runs hourly. Uses Haiku for speed.
        """
        prompt = f"""You are a crypto market regime classifier. Analyze these signals and classify the current market regime.

MARKET DATA:
- BTC price: ${market_data.get('btc_price', '?')}
- BTC 1h SMA20 vs SMA50: {market_data.get('btc_sma_status', '?')}
- BTC 4h trend: {market_data.get('btc_4h_trend', '?')}
- Fear & Greed Index: {market_data.get('fear_greed', '?')}
- BTC funding rate: {market_data.get('btc_funding', '?')}
- Taker buy/sell ratio: {market_data.get('taker_ratio', '?')}
- Top trader long %: {market_data.get('top_long_pct', '?')}
- Recent volatility (ATR%): {market_data.get('atr_pct', '?')}

Respond with ONLY valid JSON:
{{"regime": "trending_bull|trending_bear|ranging|high_vol_crisis", "confidence": 0.0-1.0, "risk_level": "low|medium|high", "bias": "long|short|neutral", "reasoning": "one sentence"}}"""

        try:
            response_text, tokens = await asyncio.to_thread(
                self._call_claude, prompt, HAIKU_MODEL, 200, 8.0
            )
            import json
            text = response_text.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:])
            if text.endswith("```"):
                text = text[:-3]
            result = json.loads(text.strip())
            logger.info(f"[AI Regime] {result.get('regime', '?')} (conf={result.get('confidence', '?')}, bias={result.get('bias', '?')}) | {result.get('reasoning', '')[:80]}")
            return result
        except Exception as e:
            logger.warning(f"[AI Regime] Failed: {e}")
            return None

    async def optimize_exit(self, position: dict, current_price: float,
                            market_context: dict) -> dict | None:
        """#3: AI exit decision for open positions.

        Returns: {"action": "hold|tighten_stop|close_now", "reasoning": "..."}
        Called for positions open > 30min. Uses Haiku.
        """
        pnl_pct = position.get("pnl_pct", 0)
        mins_open = position.get("mins_open", 0)
        side = position.get("side", "?")
        sl_pct = position.get("sl_pct", 0.02)
        tp_pct = position.get("tp_pct", 0.04)

        prompt = f"""You are a trade exit optimizer. Should this crypto futures position be held, tightened, or closed?

POSITION:
- Side: {side} | PnL: {pnl_pct:+.2%} | Open: {mins_open:.0f}min
- SL distance: {sl_pct:.2%} | TP distance: {tp_pct:.2%}
- Strategy: {position.get('strategy', '?')} | Symbol: {position.get('symbol', '?')}

MARKET CONTEXT:
- Regime: {market_context.get('regime', '?')} | Bias: {market_context.get('bias', '?')}
- Risk level: {market_context.get('risk_level', '?')}

RULES:
- "hold": Position is progressing well or needs more time
- "tighten_stop": Move SL to breakeven or tighter (position is profitable but momentum fading)
- "close_now": Exit immediately (regime changed against position, or stale with no momentum)
- Do NOT close winners early — let them run to TP
- Close if: losing + regime flipped against position, or 60+ min with <0.1% move

Respond with ONLY valid JSON: {{"action": "hold|tighten_stop|close_now", "reasoning": "one sentence"}}"""

        try:
            response_text, tokens = await asyncio.to_thread(
                self._call_claude, prompt, HAIKU_MODEL, 100, 5.0
            )
            import json
            text = response_text.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:])
            if text.endswith("```"):
                text = text[:-3]
            return json.loads(text.strip())
        except Exception as e:
            logger.warning(f"[AI Exit] Failed: {e}")
            return None

    async def tune_parameters(self, strategy_stats: dict, recent_trades: list[dict],
                               current_params: dict) -> dict | None:
        """#4: AI strategy parameter tuning. Returns recommended parameter changes.

        Returns: {"trend_breakout": {"sl_atr_mult": 1.5, "tp_atr_mult": 3.5}, ...}
        Runs daily. Uses Sonnet.
        """
        from collections import defaultdict
        perf = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0, "sl_exits": 0, "tp_exits": 0, "stale_exits": 0, "avg_win": 0, "avg_loss": 0})

        for t in recent_trades:
            strat = t.get("strategy_name", "unknown")
            pnl = t.get("net_pnl") or 0
            perf[strat]["trades"] += 1
            perf[strat]["pnl"] += pnl
            if pnl > 0:
                perf[strat]["wins"] += 1
            exit_r = t.get("exit_reasoning") or ""
            if "Stop loss" in exit_r:
                perf[strat]["sl_exits"] += 1
            elif "Take profit" in exit_r:
                perf[strat]["tp_exits"] += 1
            elif "Stale" in exit_r or "momentum" in exit_r.lower():
                perf[strat]["stale_exits"] += 1

        lines = []
        for strat, p in perf.items():
            params = current_params.get(strat, {})
            wr = p["wins"] / p["trades"] * 100 if p["trades"] else 0
            lines.append(
                f"- {strat}: {p['trades']} trades, {wr:.0f}% WR, ${p['pnl']:+.2f} | "
                f"SL exits: {p['sl_exits']}, TP exits: {p['tp_exits']}, Stale: {p['stale_exits']} | "
                f"Current params: SL={params.get('sl_atr_mult', '?')}x ATR, TP={params.get('tp_atr_mult', '?')}x ATR"
            )

        prompt = f"""You are a quantitative strategist tuning crypto trading parameters.

STRATEGY PERFORMANCE (last 7 days):
{chr(10).join(lines)}

For each strategy, recommend parameter adjustments. Consider:
- If SL exits >> TP exits: SL may be too tight (increase sl_atr_mult) or TP too ambitious (decrease tp_atr_mult)
- If stale exits are high: max_position_hours may be too long, or TP too far
- If WR < 30%: Consider if the strategy has any edge at all
- Only suggest changes if there's clear evidence. Use null for "no change needed"

Respond with ONLY valid JSON. Example:
{{"trend_breakout": {{"sl_atr_mult": 1.8, "tp_atr_mult": 2.5, "notes": "Too many SL exits — widen SL slightly"}},
"crash_momentum": {{"sl_atr_mult": null, "tp_atr_mult": 2.5, "notes": "TP too ambitious, reduce to capture more wins"}}}}"""

        try:
            response_text, tokens = await asyncio.to_thread(
                self._call_claude, prompt, SONNET_MODEL, 512, 20.0
            )
            import json
            text = response_text.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:])
            if text.endswith("```"):
                text = text[:-3]
            result = json.loads(text.strip())
            logger.info(f"[AI Tuning] Parameter recommendations for {len(result)} strategies ({tokens} tokens)")
            return result
        except Exception as e:
            logger.warning(f"[AI Tuning] Failed: {e}")
            return None

    async def select_symbols(self, symbol_stats: dict, available_symbols: list[str]) -> dict | None:
        """#6: AI symbol selection. Recommends best symbols to trade.

        Returns: {"avoid_long": ["AVAXUSDT", ...], "avoid_short": [], "reasoning": "..."}
        Runs every 12h. Uses Haiku.
        """
        lines = []
        for sym, stats in symbol_stats.items():
            wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] else 0
            lines.append(f"- {sym}: {stats['trades']} trades, {wr:.0f}% WR, ${stats['pnl']:+.2f}, avg vol: {stats.get('avg_volume', '?')}")

        prompt = f"""You are a crypto symbol selector for an algo trading bot on Binance USDT-M Futures.

SYMBOL PERFORMANCE (last 7 days):
{chr(10).join(lines)}

Available symbols: {', '.join(available_symbols)}

IMPORTANT: A symbol that loses on LONGS may be GREAT for SHORTS (weak coins crash harder).
Do NOT avoid a symbol entirely — specify DIRECTION.

Recommend which symbols to avoid for longs and which to avoid for shorts:
- Avoid LONG on symbols with <25% WR on buy trades and negative PnL
- Avoid SHORT on symbols with <25% WR on sell trades and negative PnL
- A symbol can appear in avoid_long but NOT avoid_short (and vice versa)
- Keep at least 4 symbols available per direction

Respond with ONLY valid JSON:
{{"avoid_long": ["AVAXUSDT"], "avoid_short": [], "reasoning": "one sentence"}}"""

        try:
            response_text, tokens = await asyncio.to_thread(
                self._call_claude, prompt, HAIKU_MODEL, 200, 8.0
            )
            import json
            text = response_text.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:])
            if text.endswith("```"):
                text = text[:-3]
            result = json.loads(text.strip())
            logger.info(f"[AI Symbols] Trade: {result.get('trade', [])}, Avoid: {result.get('avoid', [])} ({tokens} tokens)")
            return result
        except Exception as e:
            logger.warning(f"[AI Symbols] Failed: {e}")
            return None

    @property
    def daily_cost(self) -> float:
        """Get today's estimated API cost in USD."""
        return self._daily_cost_usd

    @property
    def daily_tokens(self) -> int:
        """Get today's total token usage."""
        return self._daily_token_usage
