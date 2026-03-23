#!/usr/bin/env python
"""Live trading engine for Binance USDT-M Futures.

Same strategies as paper_trade_simple.py with real order execution.
Run with --dry-run (default) to verify signals without placing orders.

Safety features:
- Exchange-side stop-loss orders (protection when bot is down)
- Max daily loss kill switch (--max-daily-loss)
- File-based emergency kill switch (touch KILL_SWITCH in project root)
- Balance validation before trades
- Position sync with exchange on startup
- reduceOnly on all close orders (prevents accidental position opening)
"""

import asyncio
import os
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import uuid

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from loguru import logger

from data.collectors.market_data import MarketDataCollector
from core.engine.order_manager import OrderManager

# Reuse all strategy logic from paper trader
from scripts.paper_trade_simple import (
    SimplePaperTrader,
    StrategyConfig,
    FundingMeanReversionGenerator,
    TrendBreakoutGenerator,
    TrendPullbackGenerator,
    OrderFlowGenerator,
    calculate_position_size,
)


KILL_SWITCH_FILE = project_root / "KILL_SWITCH"


class LiveTrader(SimplePaperTrader):
    """Live trading engine — extends SimplePaperTrader with real execution.

    In dry-run mode (default), behaves exactly like paper trader.
    In live mode, places real orders on Binance Futures.

    Inherited from SimplePaperTrader (unchanged):
    - _fetch_market_data: market data collection
    - _process_strategy_symbol: signal generation per strategy
    - _check_entry: cooldowns, circuit breaker, pileup block, then calls _open_position
    - _check_exit: SL/TP/trailing/time/RSI/signal checks, then calls _close_position
    - _save_signal, _maybe_save_hold: signal logging
    - _save_performance_snapshots: performance tracking
    - All helper methods (RSI, ATR, HTF, position sizing)

    Overridden for live trading:
    - start: adds exchange connection, safety checks in loop
    - _open_position: real market orders + exchange SL
    - _close_position: real market close + cancel exchange orders
    - _load_existing_positions: filters by exchange='binance_live'
    """

    def __init__(
        self,
        *args,
        dry_run: bool = True,
        max_daily_loss_pct: float = 0.05,
        capital_scale: float = 0.5,
        close_on_exit: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dry_run = dry_run
        self.max_daily_loss_pct = max_daily_loss_pct
        self.capital_scale = capital_scale
        self.close_on_exit = close_on_exit

        # Scale capital for live trading
        if capital_scale != 1.0:
            for name, stats in self.strategy_stats.items():
                stats["capital"] *= capital_scale
                stats["initial_capital"] *= capital_scale
            self.initial_capital = sum(
                s["initial_capital"] for s in self.strategy_stats.values()
            )

        # Live-specific state
        self.order_mgr: OrderManager | None = None
        self.daily_pnl = 0.0
        self.daily_pnl_reset_date = ""
        self._kill_switch_triggered = False

    # =========================================================================
    # Safety Checks
    # =========================================================================

    def _check_kill_switch(self) -> bool:
        """Check if kill switch file exists."""
        if KILL_SWITCH_FILE.exists():
            logger.warning("🚨 KILL SWITCH activated — stopping all trading")
            self._kill_switch_triggered = True
            return True
        return False

    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been hit."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.daily_pnl_reset_date:
            self.daily_pnl = 0.0
            self.daily_pnl_reset_date = today

        if self.initial_capital > 0:
            daily_loss_pct = self.daily_pnl / self.initial_capital
            if daily_loss_pct < -self.max_daily_loss_pct:
                logger.warning(
                    f"🚨 Daily loss limit: {daily_loss_pct:.2%} "
                    f"(limit: -{self.max_daily_loss_pct:.0%})"
                )
                return True
        return False

    # =========================================================================
    # Exchange Connection
    # =========================================================================

    async def _connect_exchange(self):
        """Connect to Binance Futures."""
        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")
        testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

        if not api_key or not api_secret:
            if self.dry_run:
                logger.warning("No API keys configured — dry-run only (no exchange connection)")
                return
            else:
                raise ValueError(
                    "BINANCE_API_KEY and BINANCE_API_SECRET required for live trading. "
                    "Set them in .env or use --dry-run."
                )

        self.order_mgr = OrderManager(api_key, api_secret, testnet=testnet)
        await self.order_mgr.connect(leverage=self.leverage)

        if not self.dry_run:
            balance = await self.order_mgr.get_balance()
            logger.info(
                f"💰 Account balance: ${balance['total']:,.2f} "
                f"(free: ${balance['free']:,.2f})"
            )
            if balance["free"] < self.initial_capital * 0.5:
                logger.warning(
                    f"⚠️ Free balance ${balance['free']:,.2f} may be insufficient "
                    f"for allocated capital ${self.initial_capital:,.2f}"
                )

    # =========================================================================
    # Exchange Position Sync
    # =========================================================================

    async def _check_exchange_fills(self):
        """Check if any exchange SL orders have been filled."""
        if self.dry_run or not self.order_mgr:
            return

        for pos_key in list(self.positions.keys()):
            pos = self.positions[pos_key]
            sl_order_id = pos.get("sl_order_id")
            if not sl_order_id:
                continue

            _, symbol = self._parse_pos_key(pos_key)
            try:
                order = await self.order_mgr.fetch_order(symbol, sl_order_id)
                if order["status"] == "closed":
                    fill_price = order["average"] or order["price"] or pos["stop_loss_price"]
                    fill_fee = order.get("fee", 0)
                    strategy_name = pos.get("strategy_name")
                    strategy = self.strategy_map.get(strategy_name)
                    if strategy:
                        logger.warning(
                            f"🔴 Exchange SL filled: [{strategy_name}] {symbol} "
                            f"@ ${fill_price:,.2f}"
                        )
                        await self._handle_exchange_sl_fill(
                            pos_key, fill_price, fill_fee, strategy
                        )
            except Exception as e:
                logger.debug(f"Could not check SL order {sl_order_id}: {e}")

    async def _handle_exchange_sl_fill(
        self, pos_key: str, fill_price: float, fill_fee: float,
        strategy: StrategyConfig,
    ):
        """Handle a stop-loss that was filled on the exchange."""
        position = self.positions.pop(pos_key, None)
        if not position:
            return

        strategy_name, symbol = self._parse_pos_key(pos_key)
        stats = self.strategy_stats[strategy.name]

        # Calculate PnL using actual fill
        if position["side"] == "long":
            pnl = (fill_price - position["entry_price"]) * position["quantity"]
        else:
            pnl = (position["entry_price"] - fill_price) * position["quantity"]
        pnl -= fill_fee

        pnl_pct = self._calculate_pnl_pct(position, fill_price)
        margin = position.get("margin", position["quantity"] * position["entry_price"] / self.leverage)
        roe = (pnl / margin) * 100 if margin > 0 else 0

        stats["capital"] += pnl
        stats["total_pnl"] += pnl
        stats["trade_count"] += 1
        if pnl > 0:
            stats["winning_trades"] += 1
        self.daily_pnl += pnl

        # Apply SL cooldowns (same logic as paper trader's _check_exit SL handling)
        now = datetime.now(timezone.utc)
        self.stop_loss_cooldowns[pos_key] = now + timedelta(minutes=self.stop_loss_cooldown_minutes)
        self.symbol_sl_cooldowns[symbol] = now + timedelta(minutes=self.symbol_sl_cooldown_minutes)
        self.daily_stop_losses[pos_key] = self.daily_stop_losses.get(pos_key, 0) + 1

        # Circuit breaker
        self.global_sl_timestamps.append(now)
        cutoff = now - timedelta(hours=self.circuit_breaker_window_hours)
        self.global_sl_timestamps = [t for t in self.global_sl_timestamps if t > cutoff]
        if len(self.global_sl_timestamps) >= self.circuit_breaker_threshold:
            self.circuit_breaker_until = now + timedelta(minutes=self.circuit_breaker_pause_minutes)
            logger.warning(
                f"🔌 CIRCUIT BREAKER: {len(self.global_sl_timestamps)} SLs in "
                f"{self.circuit_breaker_window_hours}h — ALL trading paused "
                f"for {self.circuit_breaker_pause_minutes}min"
            )

        duration = (now - position["entry_time"]).total_seconds()
        logger.info(
            f"❌ [LIVE] EXCHANGE SL [{strategy_name}] {symbol} @ ${fill_price:,.2f} | "
            f"PnL: ${pnl:,.2f} (ROE: {roe:+.1f}%) | {duration/60:.1f}min"
        )

        # Update DB
        try:
            exit_data = {
                "exit_price": fill_price,
                "exit_time": now.isoformat(),
                "gross_pnl": pnl + fill_fee,
                "net_pnl": pnl,
                "return_pct": pnl_pct * 100,
                "duration_seconds": int(duration),
                "exit_reasoning": "[LIVE] Exchange SL triggered",
            }
            await asyncio.to_thread(
                lambda: self.trade_repo.table.update(exit_data)
                .eq("position_id", position["position_id"])
                .execute()
            )
        except Exception as e:
            logger.error(f"Failed to update trade in DB: {e}")

    async def _sync_with_exchange(self):
        """Verify tracked positions match exchange state on startup."""
        if self.dry_run or not self.order_mgr:
            return

        try:
            exchange_positions = await self.order_mgr.get_positions()
            exchange_syms = {p["symbol"] for p in exchange_positions}

            # Check tracked positions exist on exchange
            for pos_key in list(self.positions.keys()):
                _, symbol = self._parse_pos_key(pos_key)
                ccxt_sym = self.order_mgr.ccxt_symbol(symbol)

                if ccxt_sym not in exchange_syms:
                    pos = self.positions[pos_key]
                    strategy = self.strategy_map.get(pos.get("strategy_name"))
                    if strategy:
                        logger.warning(
                            f"⚠️ {pos_key} in DB but not on exchange — "
                            f"SL likely filled while offline"
                        )
                        await self._handle_exchange_sl_fill(
                            pos_key, pos["stop_loss_price"], 0, strategy
                        )

            # Warn about unknown exchange positions
            tracked_syms = set()
            for pk in self.positions:
                _, sym = self._parse_pos_key(pk)
                tracked_syms.add(self.order_mgr.ccxt_symbol(sym))

            for ep in exchange_positions:
                if ep["symbol"] not in tracked_syms:
                    logger.warning(
                        f"⚠️ Unknown exchange position: {ep['symbol']} {ep['side']} "
                        f"qty={ep['quantity']} — NOT managed by this bot"
                    )
        except Exception as e:
            logger.error(f"Failed to sync with exchange: {e}")

    # =========================================================================
    # Main Loop (override)
    # =========================================================================

    async def start(self):
        """Start live trading."""
        self.running = True

        # Connect to exchange
        await self._connect_exchange()

        # Load existing positions from DB
        await self._load_existing_positions()

        # Sync with exchange
        await self._sync_with_exchange()

        mode = "DRY RUN" if self.dry_run else "🔴 LIVE"
        logger.info("=" * 70)
        logger.info(f"🚀 Starting LIVE Trading Engine ({mode})")
        logger.info(f"   Symbols: {self.symbols}")
        logger.info(f"   Total Capital: ${self.initial_capital:,.2f} | Leverage: {self.leverage}x")
        logger.info(f"   Capital Scale: {self.capital_scale:.0%} of paper amounts")
        logger.info(f"   Max Daily Loss: {self.max_daily_loss_pct:.0%}")
        if self.dry_run:
            logger.info("   Mode: DRY RUN — no real orders will be placed")
        logger.info("-" * 70)
        for s in self.strategies:
            scaled = self.strategy_stats[s.name]["capital"]
            logger.info(
                f"   📋 {s.name}: ${scaled:,.2f} | "
                f"SL={s.sl_atr_mult}x ATR({s.atr_timeframe}) TP={s.tp_atr_mult}x ATR | "
                f"Risk={s.risk_per_trade_pct:.1%}/trade"
            )
        logger.info("-" * 70)
        logger.info(f"   Kill switch file: {KILL_SWITCH_FILE}")
        logger.info(f"   Circuit breaker: {self.circuit_breaker_threshold} SLs in "
                     f"{self.circuit_breaker_window_hours}h → pause {self.circuit_breaker_pause_minutes}min")
        logger.info("=" * 70)

        collector = MarketDataCollector()

        try:
            while self.running:
                # Safety checks
                if self._check_kill_switch():
                    break

                # Check if exchange SL orders have filled
                await self._check_exchange_fills()

                # Main loop (inherited logic)
                for symbol in self.symbols:
                    market_data = await self._fetch_market_data(symbol, collector)
                    if market_data is None:
                        continue

                    for strategy in self.strategies:
                        # Daily loss check: skip entries but still monitor exits
                        await self._process_strategy_symbol(strategy, symbol, market_data)

                await self._save_performance_snapshots()
                self._print_status()

                await asyncio.sleep(self.check_interval)

        except asyncio.CancelledError:
            logger.info("Live trading cancelled")
        finally:
            if self._kill_switch_triggered or self.close_on_exit:
                logger.info("Emergency close: closing all positions...")
                await self._emergency_close_all(collector)
            else:
                logger.info(
                    "Shutting down — positions remain open with exchange SL protection. "
                    "Use --close-on-exit to close positions on shutdown."
                )

            if self.order_mgr:
                await self.order_mgr.disconnect()
            await collector.close()
            self._print_summary()

    # =========================================================================
    # Entry Check (override to add daily loss guard)
    # =========================================================================

    async def _check_entry(self, pos_key: str, symbol: str, signal: dict, price: float,
                           strategy: StrategyConfig, atr_value: float):
        """Check if should enter — adds daily loss check, then delegates to parent."""
        if self._check_daily_loss_limit():
            return
        await super()._check_entry(pos_key, symbol, signal, price, strategy, atr_value)

    # =========================================================================
    # Position Opening (override — real execution)
    # =========================================================================

    async def _open_position(self, pos_key: str, symbol: str, side: str, price: float,
                              signal: dict, strategy: StrategyConfig, atr_value: float,
                              htf_trend: dict = None):
        """Open a position with real exchange execution (or dry-run simulation)."""

        if self.dry_run:
            # Dry run: use paper trader simulation
            await super()._open_position(pos_key, symbol, side, price, signal, strategy, atr_value, htf_trend)
            return

        stats = self.strategy_stats[strategy.name]

        # --- Position sizing (same as paper) ---
        if atr_value <= 0:
            atr_value = price * 0.015

        sl_distance = atr_value * strategy.sl_atr_mult
        tp_distance = atr_value * strategy.tp_atr_mult
        trailing_act_distance = atr_value * strategy.trailing_atr_mult
        trailing_dist_distance = atr_value * strategy.trailing_dist_atr_mult

        sl_pct = sl_distance / price
        tp_pct = tp_distance / price
        trailing_act_pct = trailing_act_distance / price
        trailing_dist_pct = trailing_dist_distance / price

        if strategy.min_sl_pct > 0 and sl_pct < strategy.min_sl_pct:
            scale = strategy.min_sl_pct / sl_pct
            sl_pct = strategy.min_sl_pct
            tp_pct *= scale
            trailing_act_pct *= scale
            trailing_dist_pct *= scale

        effective_risk = self._get_effective_risk_pct(strategy, signal, symbol, side, htf_trend)
        quantity, margin = calculate_position_size(
            capital=stats["capital"],
            risk_pct=effective_risk,
            sl_distance_pct=sl_pct,
            leverage=self.leverage,
            price=price,
        )

        if quantity <= 0:
            logger.warning(f"   [{strategy.name}] {symbol}: Position size too small")
            return

        # --- Balance check ---
        if self.order_mgr:
            try:
                balance = await self.order_mgr.get_balance()
                if balance["free"] < margin * 1.1:
                    logger.warning(
                        f"   [{strategy.name}] {symbol}: Insufficient margin "
                        f"(need ${margin:.2f}, have ${balance['free']:.2f})"
                    )
                    return
            except Exception as e:
                logger.error(f"Balance check failed: {e}")
                return

        # --- REAL EXECUTION ---
        try:
            # 1. Place market entry order
            order_side = "buy" if side == "long" else "sell"
            qty = self.order_mgr.format_quantity(symbol, quantity)

            if qty <= 0:
                logger.warning(f"   [{strategy.name}] {symbol}: Quantity rounds to 0")
                return

            entry_order = await self.order_mgr.place_market_order(
                symbol, order_side, qty, leverage=self.leverage
            )

            if entry_order["status"] != "closed" or entry_order["filled"] <= 0:
                logger.error(f"Entry order not filled: {entry_order}")
                return

            # Use actual fill data
            actual_price = entry_order["average"] or price
            actual_qty = entry_order["filled"]
            actual_fee = entry_order["fee"]
            actual_margin = actual_qty * actual_price / self.leverage

            # 2. Calculate SL/TP prices from actual fill
            if side == "long":
                stop_price = actual_price * (1 - sl_pct)
                take_profit_price = actual_price * (1 + tp_pct)
                liquidation_price = actual_price * (1 - self.liquidation_pct)
            else:
                stop_price = actual_price * (1 + sl_pct)
                take_profit_price = actual_price * (1 - tp_pct)
                liquidation_price = actual_price * (1 + self.liquidation_pct)

            # 3. Place SL on exchange (with retries)
            sl_side = "sell" if side == "long" else "buy"
            sl_price_formatted = self.order_mgr.format_price(symbol, stop_price)

            sl_order = None
            for attempt in range(3):
                try:
                    sl_order = await self.order_mgr.place_stop_loss(
                        symbol, sl_side, actual_qty, sl_price_formatted
                    )
                    break
                except Exception as e:
                    logger.error(f"SL placement attempt {attempt + 1}/3 failed: {e}")
                    if attempt == 2:
                        # CRITICAL: position is unprotected — close immediately
                        logger.critical(
                            f"🚨 Cannot place SL for {pos_key} — closing position immediately"
                        )
                        try:
                            await self.order_mgr.place_market_order(
                                symbol, sl_side, actual_qty,
                                leverage=self.leverage, reduce_only=True
                            )
                        except Exception as close_err:
                            logger.critical(f"🚨 FAILED to close unprotected position: {close_err}")
                        return

            sl_order_id = sl_order["id"] if sl_order else None

            # 4. Store position in memory
            position_id = str(uuid.uuid4())
            position_value = actual_qty * actual_price

            self.positions[pos_key] = {
                "position_id": position_id,
                "side": side,
                "entry_price": actual_price,
                "quantity": actual_qty,
                "margin": actual_margin,
                "leverage": self.leverage,
                "entry_time": datetime.now(timezone.utc),
                "signal": signal,
                "trailing_stop_active": False,
                "peak_pnl_pct": 0.0,
                "stop_loss_price": stop_price,
                "take_profit_price": take_profit_price,
                "liquidation_price": liquidation_price,
                "strategy_name": strategy.name,
                "sl_pct": sl_pct,
                "tp_pct": tp_pct,
                "trailing_act_pct": trailing_act_pct,
                "trailing_dist_pct": trailing_dist_pct,
                "trailing_enabled": strategy.trailing_enabled,
                # Live-specific
                "entry_order_id": entry_order["id"],
                "sl_order_id": sl_order_id,
                "entry_fee": actual_fee,
            }

            logger.info(
                f"📈 [LIVE] OPEN [{strategy.name}] {side.upper()} {symbol} "
                f"@ ${actual_price:,.2f} | "
                f"Margin: ${actual_margin:,.2f} | Size: ${position_value:,.2f} ({self.leverage}x) | "
                f"Conf: {signal['confidence']:.0%} | Fee: ${actual_fee:.4f}"
            )
            logger.info(
                f"   └─ SL: ${stop_price:,.2f} ({sl_pct:.2%}) [exchange order: {sl_order_id}] | "
                f"TP: ${take_profit_price:,.2f} ({tp_pct:.2%}) [bot-managed]"
            )

            # 5. Log to Supabase
            try:
                indicators = signal.get("indicators", {})
                indicators["sl_pct"] = sl_pct
                indicators["tp_pct"] = tp_pct
                indicators["trailing_act_pct"] = trailing_act_pct
                indicators["trailing_dist_pct"] = trailing_dist_pct
                indicators["atr_value"] = atr_value
                indicators["sl_order_id"] = sl_order_id
                indicators["entry_order_id"] = entry_order["id"]
                indicators["entry_fee"] = actual_fee

                await self.trade_repo.log_trade({
                    "position_id": position_id,
                    "symbol": symbol,
                    "exchange": "binance_live",
                    "side": "buy" if side == "long" else "sell",
                    "entry_price": actual_price,
                    "entry_time": datetime.now(timezone.utc).isoformat(),
                    "quantity": actual_qty,
                    "strategy_name": strategy.name,
                    "signal_source": strategy.source_label,
                    "signal_confidence": signal["confidence"],
                    "entry_reasoning": f"[LIVE] {signal['reasoning']}",
                    "indicators_at_entry": indicators,
                })
            except Exception as e:
                logger.error(f"Failed to log trade to DB: {e}")

        except Exception as e:
            logger.error(f"Failed to open position {pos_key}: {e}")

    # =========================================================================
    # Position Closing (override — real execution)
    # =========================================================================

    async def _close_position(self, pos_key: str, price: float, reason: str,
                               strategy: StrategyConfig):
        """Close a position with real exchange execution (or dry-run simulation)."""

        if self.dry_run:
            await super()._close_position(pos_key, price, reason, strategy)
            # Track daily PnL even in dry run
            position = self.positions.get(pos_key)  # Already popped by super
            return

        position = self.positions.get(pos_key)
        if not position:
            return

        _, symbol = self._parse_pos_key(pos_key)
        actual_price = price
        actual_fee = 0

        if self.order_mgr:
            # 1. Cancel exchange SL order
            sl_order_id = position.get("sl_order_id")
            if sl_order_id:
                await self.order_mgr.cancel_order(symbol, sl_order_id)

            # 2. Place market close order (reduceOnly for safety)
            close_side = "sell" if position["side"] == "long" else "buy"
            qty = self.order_mgr.format_quantity(symbol, position["quantity"])

            try:
                close_order = await self.order_mgr.place_market_order(
                    symbol, close_side, qty,
                    leverage=self.leverage, reduce_only=True
                )
                actual_price = close_order["average"] or price
                actual_fee = close_order["fee"]
            except Exception as e:
                # Position may already be closed by exchange SL
                logger.warning(
                    f"Close order failed for {pos_key} (SL may have fired): {e}"
                )
                actual_price = position.get("stop_loss_price", price)

        # Remove from memory
        position = self.positions.pop(pos_key, None)
        if not position:
            return

        strategy_name = position.get("strategy_name", "")
        stats = self.strategy_stats[strategy.name]

        # Calculate PnL from actual fill
        if position["side"] == "long":
            pnl = (actual_price - position["entry_price"]) * position["quantity"]
        else:
            pnl = (position["entry_price"] - actual_price) * position["quantity"]

        entry_fee = position.get("entry_fee", 0)
        pnl -= (actual_fee + entry_fee)  # Deduct both entry and exit fees

        pnl_pct = self._calculate_pnl_pct(position, actual_price)
        margin = position.get("margin", position["quantity"] * position["entry_price"] / self.leverage)
        roe = (pnl / margin) * 100 if margin > 0 else 0

        stats["capital"] += pnl
        stats["total_pnl"] += pnl
        stats["trade_count"] += 1
        if pnl > 0:
            stats["winning_trades"] += 1
        self.daily_pnl += pnl

        duration = (datetime.now(timezone.utc) - position["entry_time"]).total_seconds()
        emoji = "✅" if pnl > 0 else "❌"
        logger.info(
            f"{emoji} [LIVE] CLOSE [{strategy_name}] {symbol} @ ${actual_price:,.2f} | "
            f"PnL: ${pnl:,.2f} (ROE: {roe:+.1f}%) | Fees: ${entry_fee + actual_fee:.4f} | "
            f"Reason: {reason} | {duration/60:.1f}min"
        )

        # Update DB
        try:
            exit_data = {
                "exit_price": actual_price,
                "exit_time": datetime.now(timezone.utc).isoformat(),
                "gross_pnl": pnl + entry_fee + actual_fee,
                "net_pnl": pnl,
                "return_pct": pnl_pct * 100,
                "duration_seconds": int(duration),
                "exit_reasoning": f"[LIVE] {reason}",
            }
            await asyncio.to_thread(
                lambda: self.trade_repo.table.update(exit_data)
                .eq("position_id", position["position_id"])
                .execute()
            )
        except Exception as e:
            logger.error(f"Failed to update trade in DB: {e}")

    # =========================================================================
    # Emergency Close All
    # =========================================================================

    async def _emergency_close_all(self, collector: MarketDataCollector):
        """Close all positions and cancel all exchange orders."""
        logger.warning("🚨 Emergency close: closing all positions...")

        for pos_key in list(self.positions.keys()):
            try:
                strategy_name, symbol = self._parse_pos_key(pos_key)
                strategy = self.strategy_map.get(strategy_name)
                if not strategy:
                    continue

                ticker = await collector.get_binance_ticker(symbol)
                price = (
                    ticker.get("price", self.positions[pos_key]["entry_price"])
                    if ticker
                    else self.positions[pos_key]["entry_price"]
                )
                await self._close_position(pos_key, price, "Emergency shutdown", strategy)
            except Exception as e:
                logger.error(f"Error closing {pos_key}: {e}")

        # Cancel any remaining exchange orders
        if self.order_mgr:
            for symbol in self.symbols:
                try:
                    cancelled = await self.order_mgr.cancel_all_orders(symbol)
                    if cancelled > 0:
                        logger.info(f"Cancelled {cancelled} remaining orders for {symbol}")
                except Exception as e:
                    logger.error(f"Error cancelling orders for {symbol}: {e}")

    # =========================================================================
    # Load Positions (override to filter by exchange='binance_live')
    # =========================================================================

    async def _load_existing_positions(self):
        """Load existing open live positions from database."""
        try:
            result = await asyncio.to_thread(
                lambda: self.trade_repo.table.select("*")
                .is_("exit_time", "null")
                .eq("exchange", "binance_live")
                .execute()
            )

            if not result.data:
                return

            for trade in result.data:
                symbol = trade.get("symbol")
                if not symbol:
                    continue

                strategy_name = trade.get("strategy_name", "")
                strategy = self.strategy_map.get(strategy_name)
                if not strategy:
                    logger.debug(f"Skipping position for inactive strategy: {strategy_name}")
                    continue

                entry_price = float(trade.get("entry_price", 0))
                quantity = float(trade.get("quantity", 0))
                side = "long" if trade.get("side") == "buy" else "short"

                if entry_price <= 0 or quantity <= 0:
                    continue

                entry_time_str = trade.get("entry_time")
                if entry_time_str:
                    try:
                        entry_time = datetime.fromisoformat(
                            entry_time_str.replace("Z", "+00:00")
                        )
                    except Exception:
                        entry_time = datetime.now(timezone.utc)
                else:
                    entry_time = datetime.now(timezone.utc)

                indicators = trade.get("indicators_at_entry", {}) or {}
                sl_pct = indicators.get("sl_pct", 0.02)
                tp_pct = indicators.get("tp_pct", 0.04)
                trailing_act_pct = indicators.get("trailing_act_pct", 0.03)
                trailing_dist_pct = indicators.get("trailing_dist_pct", 0.015)

                if side == "long":
                    stop_price = entry_price * (1 - sl_pct)
                    take_profit_price = entry_price * (1 + tp_pct)
                else:
                    stop_price = entry_price * (1 + sl_pct)
                    take_profit_price = entry_price * (1 - tp_pct)

                pos_key = self._pos_key(strategy_name, symbol)
                self.positions[pos_key] = {
                    "position_id": trade.get("position_id", str(uuid.uuid4())),
                    "side": side,
                    "entry_price": entry_price,
                    "quantity": quantity,
                    "entry_time": entry_time,
                    "signal": {"confidence": trade.get("signal_confidence", 0.5)},
                    "trailing_stop_active": False,
                    "peak_pnl_pct": 0.0,
                    "stop_loss_price": stop_price,
                    "take_profit_price": take_profit_price,
                    "strategy_name": strategy_name,
                    "sl_pct": sl_pct,
                    "tp_pct": tp_pct,
                    "trailing_act_pct": trailing_act_pct,
                    "trailing_dist_pct": trailing_dist_pct,
                    "trailing_enabled": strategy.trailing_enabled,
                    "sl_order_id": indicators.get("sl_order_id"),
                    "entry_order_id": indicators.get("entry_order_id"),
                    "entry_fee": indicators.get("entry_fee", 0),
                }

                hours_open = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
                logger.info(
                    f"📂 Loaded: [{strategy_name}] {symbol} {side.upper()} "
                    f"@ ${entry_price:,.2f} | Qty: {quantity:.6f} | Open for {hours_open:.1f}h"
                )

            loaded = len(self.positions)
            if loaded:
                logger.info(f"✅ Loaded {loaded} existing live position(s) from database")

        except Exception as e:
            logger.warning(f"Could not load existing positions: {e}")

    # =========================================================================
    # Status (override to show mode)
    # =========================================================================

    def _print_status(self):
        """Print status with live/dry-run indicator."""
        mode = "DRY" if self.dry_run else "LIVE"
        total_cap = self.total_capital
        total_trades = self.total_trades
        total_winning = self.total_winning
        win_rate = total_winning / total_trades * 100 if total_trades > 0 else 0

        logger.info(
            f"💰 [{mode}] Total: ${total_cap:,.2f} | "
            f"Trades: {total_trades} | WR: {win_rate:.0f}% | "
            f"Open: {len(self.positions)} | "
            f"Daily PnL: ${self.daily_pnl:+,.2f}"
        )

        for s in self.strategies:
            stats = self.strategy_stats[s.name]
            s_trades = stats["trade_count"]
            s_wr = stats["winning_trades"] / s_trades * 100 if s_trades > 0 else 0
            s_pnl = stats["total_pnl"]
            pnl_emoji = "🟢" if s_pnl >= 0 else "🔴"
            logger.info(
                f"   {pnl_emoji} {s.name}: ${stats['capital']:,.2f} | "
                f"PnL: ${s_pnl:+,.2f} | {s_trades} trades | WR: {s_wr:.0f}%"
            )

        for pos_key, pos in self.positions.items():
            strategy_name, symbol = self._parse_pos_key(pos_key)
            sl_status = "🔒" if pos.get("sl_order_id") else "⚠️"
            trailing = "🔒" if pos.get("trailing_stop_active") else "⏳"
            logger.info(
                f"   {trailing} [{strategy_name}] {symbol}: {pos['side']} "
                f"@ ${pos['entry_price']:,.2f} | "
                f"SL: ${pos.get('stop_loss_price', 0):,.2f} {sl_status} | "
                f"TP: ${pos.get('take_profit_price', 0):,.2f}"
            )


# =============================================================================
# Main
# =============================================================================

async def main():
    """Main function for live trading."""
    import argparse

    parser = argparse.ArgumentParser(description="Live Trading Engine — Binance USDT-M Futures")
    parser.add_argument(
        "--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "XRPUSDT"],
        help="Symbols to trade"
    )
    parser.add_argument(
        "--capital", type=float, default=1000.0,
        help="Base capital unit (same as paper trader)"
    )
    parser.add_argument(
        "--leverage", type=int, default=10,
        help="Leverage multiplier (default: 10x)"
    )
    parser.add_argument(
        "--interval", type=int, default=60,
        help="Check interval in seconds"
    )
    parser.add_argument(
        "--strategies", nargs="+",
        default=["funding_reversion", "trend_breakout", "trend_pullback", "order_flow"],
        choices=["funding_reversion", "trend_breakout", "trend_pullback", "order_flow"],
        help="Strategies to run"
    )
    # Live-specific args
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Dry run mode — no real orders (default: True)"
    )
    parser.add_argument(
        "--live", action="store_true", default=False,
        help="Enable REAL trading (overrides --dry-run)"
    )
    parser.add_argument(
        "--capital-scale", type=float, default=0.5,
        help="Scale factor for capital vs paper amounts (default: 0.5 = 50%%)"
    )
    parser.add_argument(
        "--max-daily-loss", type=float, default=0.05,
        help="Max daily loss as fraction of capital (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--close-on-exit", action="store_true", default=False,
        help="Close all positions on shutdown (default: keep open with SL)"
    )

    args = parser.parse_args()

    # --live overrides --dry-run
    dry_run = not args.live

    if not dry_run:
        logger.warning("=" * 70)
        logger.warning("⚠️  REAL MONEY TRADING MODE ⚠️")
        logger.warning("    Orders will be placed on Binance Futures")
        logger.warning(f"    Capital scale: {args.capital_scale:.0%} of paper amounts")
        logger.warning(f"    Max daily loss: {args.max_daily_loss:.0%}")
        logger.warning("=" * 70)
        logger.warning("Starting in 10 seconds... Press Ctrl+C to abort.")
        await asyncio.sleep(10)

    # Setup logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
        level="INFO",
    )
    mode_str = "live" if not dry_run else "dry"
    logger.add(
        f"logs/{mode_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        rotation="10 MB",
    )

    # Build strategy configs (same as paper trader)
    base_capital = args.capital
    capital_allocation = {
        "funding_reversion": base_capital * 0.5,
        "trend_breakout": base_capital * 1.0,
        "trend_pullback": base_capital * 0.75,
        "order_flow": base_capital * 0.75,
    }

    strategy_configs = []
    for name in args.strategies:
        if name == "funding_reversion":
            strategy_configs.append(StrategyConfig(
                name="funding_reversion", strategy_type="funding",
                generator=FundingMeanReversionGenerator(),
                sl_atr_mult=2.0, tp_atr_mult=4.0,
                trailing_atr_mult=3.0, trailing_dist_atr_mult=1.5,
                max_position_hours=12.0, risk_per_trade_pct=0.02,
                capital=capital_allocation.get(name, base_capital),
                atr_timeframe="1h", trailing_enabled=False,
            ))
        elif name == "trend_breakout":
            strategy_configs.append(StrategyConfig(
                name="trend_breakout", strategy_type="breakout",
                generator=TrendBreakoutGenerator(),
                sl_atr_mult=1.5, tp_atr_mult=3.0,
                trailing_atr_mult=2.5, trailing_dist_atr_mult=1.0,
                max_position_hours=6.0, risk_per_trade_pct=0.02,
                capital=capital_allocation.get(name, base_capital),
                atr_timeframe="15m", trailing_enabled=False,
            ))
        elif name == "trend_pullback":
            strategy_configs.append(StrategyConfig(
                name="trend_pullback", strategy_type="pullback",
                generator=TrendPullbackGenerator(),
                sl_atr_mult=1.5, tp_atr_mult=3.0,
                trailing_atr_mult=2.5, trailing_dist_atr_mult=1.0,
                max_position_hours=8.0, risk_per_trade_pct=0.02,
                capital=capital_allocation.get(name, base_capital),
                atr_timeframe="15m", trailing_enabled=False,
            ))
        elif name == "order_flow":
            strategy_configs.append(StrategyConfig(
                name="order_flow", strategy_type="flow",
                generator=OrderFlowGenerator(),
                sl_atr_mult=1.5, tp_atr_mult=3.0,
                trailing_atr_mult=2.5, trailing_dist_atr_mult=1.0,
                max_position_hours=6.0, risk_per_trade_pct=0.02,
                capital=capital_allocation.get(name, base_capital),
                atr_timeframe="15m", trailing_enabled=False,
            ))

    # Create trader
    trader = LiveTrader(
        symbols=args.symbols,
        strategies=strategy_configs,
        initial_capital=args.capital,
        leverage=args.leverage,
        check_interval=args.interval,
        # Live-specific
        dry_run=dry_run,
        max_daily_loss_pct=args.max_daily_loss,
        capital_scale=args.capital_scale,
        close_on_exit=args.close_on_exit,
    )

    # Handle shutdown
    loop = asyncio.get_event_loop()

    def handle_shutdown():
        logger.info("Shutdown signal received...")
        loop.create_task(trader.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_shutdown)

    await trader.start()


if __name__ == "__main__":
    asyncio.run(main())
