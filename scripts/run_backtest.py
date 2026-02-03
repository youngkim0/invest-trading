"""Backtesting script for trading strategies."""

import argparse
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
from loguru import logger

from config import get_settings
from config.strategies import StrategyConfig
from core.strategies.hybrid_strategy import HybridStrategy, create_hybrid_strategy
from data.collectors.price_collector import PriceCollector
from data.features.technical import TechnicalIndicators
from journal.performance import PerformanceAnalyzer, PerformanceMetrics


class Backtester:
    """Simple backtesting engine."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.reset()

    def reset(self):
        """Reset backtester state."""
        self.capital = self.initial_capital
        self.position = 0.0
        self.position_side = None
        self.entry_price = 0.0
        self.trades = []
        self.equity_curve = []

    def run(
        self,
        strategy: HybridStrategy,
        data: pd.DataFrame,
        symbol: str,
    ) -> dict:
        """Run backtest on historical data."""
        self.reset()

        logger.info(f"Running backtest on {len(data)} candles...")

        # Need enough data for indicators
        min_bars = 60

        for i in range(min_bars, len(data)):
            # Get data up to current bar
            current_data = data.iloc[:i+1].copy()
            current_price = float(current_data["close"].iloc[-1])
            timestamp = current_data.index[-1] if hasattr(current_data.index[-1], 'isoformat') else datetime.now()

            # Calculate equity
            if self.position != 0:
                if self.position_side == "long":
                    unrealized_pnl = (current_price - self.entry_price) * self.position
                else:
                    unrealized_pnl = (self.entry_price - current_price) * self.position
                equity = self.capital + unrealized_pnl
            else:
                equity = self.capital

            self.equity_curve.append({
                "timestamp": timestamp,
                "equity": equity,
                "price": current_price,
            })

            # Generate signals
            signals = strategy.generate_signals(symbol, current_data)

            if not signals:
                continue

            signal = signals[0]

            # Execute trades
            if self.position == 0:
                # No position - check for entry
                if signal.signal_type.value in ["buy", "strong_buy"]:
                    self._enter_long(current_price, timestamp)
                elif signal.signal_type.value in ["sell", "strong_sell"]:
                    self._enter_short(current_price, timestamp)

            else:
                # Have position - check for exit
                if self.position_side == "long":
                    if signal.signal_type.value in ["sell", "strong_sell"]:
                        self._exit_position(current_price, timestamp)
                elif self.position_side == "short":
                    if signal.signal_type.value in ["buy", "strong_buy"]:
                        self._exit_position(current_price, timestamp)

        # Close any remaining position
        if self.position != 0:
            final_price = float(data["close"].iloc[-1])
            final_time = data.index[-1] if hasattr(data.index[-1], 'isoformat') else datetime.now()
            self._exit_position(final_price, final_time)

        return self._calculate_results()

    def _enter_long(self, price: float, timestamp):
        """Enter long position."""
        position_size = self.capital * 0.95 / price  # Use 95% of capital
        commission = position_size * price * self.commission_rate

        self.position = position_size
        self.position_side = "long"
        self.entry_price = price
        self.capital -= commission

        logger.debug(f"LONG entry @ {price:.2f}, size: {position_size:.4f}")

    def _enter_short(self, price: float, timestamp):
        """Enter short position."""
        position_size = self.capital * 0.95 / price
        commission = position_size * price * self.commission_rate

        self.position = position_size
        self.position_side = "short"
        self.entry_price = price
        self.capital -= commission

        logger.debug(f"SHORT entry @ {price:.2f}, size: {position_size:.4f}")

    def _exit_position(self, price: float, timestamp):
        """Exit current position."""
        if self.position_side == "long":
            pnl = (price - self.entry_price) * self.position
        else:
            pnl = (self.entry_price - price) * self.position

        commission = self.position * price * self.commission_rate
        net_pnl = pnl - commission

        self.capital += net_pnl + (self.entry_price * self.position)

        self.trades.append({
            "entry_price": self.entry_price,
            "exit_price": price,
            "side": self.position_side,
            "size": self.position,
            "pnl": net_pnl,
            "timestamp": timestamp,
        })

        logger.debug(f"EXIT @ {price:.2f}, PnL: {net_pnl:.2f}")

        self.position = 0
        self.position_side = None
        self.entry_price = 0

    def _calculate_results(self) -> dict:
        """Calculate backtest results."""
        if not self.equity_curve:
            return {}

        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index("timestamp", inplace=True)

        # Calculate returns
        equity_series = equity_df["equity"]
        returns = equity_series.pct_change().dropna()

        total_return = (equity_series.iloc[-1] / self.initial_capital) - 1

        # Calculate metrics
        if len(returns) > 0:
            volatility = returns.std() * (252 ** 0.5)  # Annualized
            sharpe = (returns.mean() * 252) / volatility if volatility > 0 else 0

            # Drawdown
            cummax = equity_series.expanding().max()
            drawdown = (equity_series - cummax) / cummax
            max_drawdown = abs(drawdown.min())
        else:
            volatility = 0
            sharpe = 0
            max_drawdown = 0

        # Trade statistics
        if self.trades:
            winning_trades = [t for t in self.trades if t["pnl"] > 0]
            losing_trades = [t for t in self.trades if t["pnl"] <= 0]

            win_rate = len(winning_trades) / len(self.trades)

            gross_profit = sum(t["pnl"] for t in winning_trades)
            gross_loss = abs(sum(t["pnl"] for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            avg_win = gross_profit / len(winning_trades) if winning_trades else 0
            avg_loss = gross_loss / len(losing_trades) if losing_trades else 0
        else:
            win_rate = 0
            profit_factor = 0
            avg_win = 0
            avg_loss = 0

        return {
            "total_return": total_return,
            "final_equity": equity_series.iloc[-1],
            "sharpe_ratio": sharpe,
            "volatility": volatility,
            "max_drawdown": max_drawdown,
            "total_trades": len(self.trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "trades": self.trades,
            "equity_curve": equity_df,
        }


async def main():
    """Main backtest function."""
    parser = argparse.ArgumentParser(description="Run strategy backtest")

    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT",
        help="Trading symbol",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Days of historical data",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Candle timeframe",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000.0,
        help="Initial capital",
    )
    parser.add_argument(
        "--rl-model",
        type=str,
        default=None,
        help="Path to RL model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backtest_results",
        help="Output directory",
    )

    args = parser.parse_args()

    # Setup logging
    logger.add(
        f"logs/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        rotation="100 MB",
    )

    logger.info("Starting backtest...")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Period: {args.days} days")

    # Collect data
    settings = get_settings()
    collector = PriceCollector(
        binance_config=settings.binance,
        alpaca_config=settings.alpaca,
    )

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=args.days)

    logger.info("Collecting historical data...")

    if "/" in args.symbol:
        data = await collector.collect_crypto_history(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date,
        )
    else:
        data = await collector.collect_stock_history(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date,
        )

    if data is None or len(data) == 0:
        logger.error("No data collected!")
        return

    logger.info(f"Collected {len(data)} candles")

    # Create strategy
    strategy_config = StrategyConfig()
    strategy = create_hybrid_strategy(
        config=strategy_config,
        rl_model_path=args.rl_model,
    )

    # Run backtest
    backtester = Backtester(
        initial_capital=args.initial_capital,
        commission_rate=0.001,
    )

    results = backtester.run(strategy, data, args.symbol)

    # Print results
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.days} days")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    print("-" * 50)
    print(f"Final Equity: ${results['final_equity']:,.2f}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print("-" * 50)
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Avg Win: ${results['avg_win']:,.2f}")
    print(f"Avg Loss: ${results['avg_loss']:,.2f}")
    print("=" * 50)

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save equity curve
    results["equity_curve"].to_csv(
        output_dir / f"equity_curve_{timestamp}.csv"
    )

    # Save trades
    trades_df = pd.DataFrame(results["trades"])
    trades_df.to_csv(output_dir / f"trades_{timestamp}.csv", index=False)

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
