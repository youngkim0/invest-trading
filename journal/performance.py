"""Performance analytics for the investment journal."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from data.storage.repository import (
    DatabaseManager,
    PerformanceRepository,
    TradeLogRepository,
)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""

    # Returns
    total_return: float
    annualized_return: float
    daily_returns: list[float]

    # Risk metrics
    volatility: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdown
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration_days: int

    # Win/Loss
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float

    # PnL
    gross_profit: float
    gross_loss: float
    net_profit: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

    # Ratios
    avg_win_loss_ratio: float
    expectancy: float

    # Period
    start_date: datetime | None = None
    end_date: datetime | None = None
    trading_days: int = 0


class PerformanceAnalyzer:
    """Analyzes trading performance."""

    def __init__(
        self,
        db_manager: DatabaseManager | None = None,
        risk_free_rate: float = 0.02,  # 2% annual
    ):
        """Initialize performance analyzer.

        Args:
            db_manager: Database manager
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        self.db_manager = db_manager or DatabaseManager()
        self.risk_free_rate = risk_free_rate

    async def calculate_metrics(
        self,
        strategy_name: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics.

        Args:
            strategy_name: Filter by strategy
            start_date: Start date
            end_date: End date

        Returns:
            Performance metrics
        """
        end_date = end_date or datetime.utcnow()
        start_date = start_date or (end_date - timedelta(days=365))

        # Get data from database
        async with self.db_manager.session() as session:
            trade_repo = TradeLogRepository(session)
            perf_repo = PerformanceRepository(session)

            trades = await trade_repo.get_trades(
                strategy_name=strategy_name,
                start_time=start_date,
                end_time=end_date,
                limit=10000,
            )

            snapshots = await perf_repo.get_snapshots(
                start_time=start_date,
                end_time=end_date,
                strategy_name=strategy_name,
            )

        # Convert to DataFrames for analysis
        trades_df = self._trades_to_dataframe(trades)
        equity_curve = self._snapshots_to_equity_curve(snapshots)

        # Calculate all metrics
        return self._compute_metrics(trades_df, equity_curve, start_date, end_date)

    def _trades_to_dataframe(self, trades: list) -> pd.DataFrame:
        """Convert trade logs to DataFrame."""
        if not trades:
            return pd.DataFrame()

        data = []
        for trade in trades:
            data.append({
                "entry_time": trade.entry_time,
                "exit_time": trade.exit_time,
                "symbol": trade.symbol,
                "side": trade.side.value if trade.side else None,
                "entry_price": float(trade.entry_price) if trade.entry_price else 0,
                "exit_price": float(trade.exit_price) if trade.exit_price else 0,
                "quantity": float(trade.quantity) if trade.quantity else 0,
                "gross_pnl": float(trade.gross_pnl) if trade.gross_pnl else 0,
                "net_pnl": float(trade.net_pnl) if trade.net_pnl else 0,
                "return_pct": float(trade.return_pct) if trade.return_pct else 0,
                "duration_seconds": trade.duration_seconds,
            })

        return pd.DataFrame(data)

    def _snapshots_to_equity_curve(self, snapshots: list) -> pd.Series:
        """Convert performance snapshots to equity curve."""
        if not snapshots:
            return pd.Series(dtype=float)

        data = {
            s.timestamp: float(s.total_equity)
            for s in snapshots
        }

        return pd.Series(data).sort_index()

    def _compute_metrics(
        self,
        trades_df: pd.DataFrame,
        equity_curve: pd.Series,
        start_date: datetime,
        end_date: datetime,
    ) -> PerformanceMetrics:
        """Compute all performance metrics."""
        # Trade-based metrics
        if len(trades_df) == 0:
            return self._empty_metrics()

        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df["net_pnl"] > 0])
        losing_trades = len(trades_df[trades_df["net_pnl"] < 0])

        gross_profit = float(trades_df[trades_df["net_pnl"] > 0]["net_pnl"].sum())
        gross_loss = abs(float(trades_df[trades_df["net_pnl"] < 0]["net_pnl"].sum()))
        net_profit = gross_profit - gross_loss

        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0
        avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

        largest_win = float(trades_df["net_pnl"].max()) if len(trades_df) > 0 else 0
        largest_loss = float(trades_df["net_pnl"].min()) if len(trades_df) > 0 else 0

        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Equity curve metrics
        if len(equity_curve) > 1:
            daily_returns = equity_curve.pct_change().dropna().tolist()

            total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

            # Annualized metrics
            trading_days = len(equity_curve)
            annualized_return = ((1 + total_return) ** (252 / trading_days)) - 1

            volatility = np.std(daily_returns) if daily_returns else 0
            annualized_volatility = volatility * np.sqrt(252)

            # Sharpe Ratio
            excess_return = annualized_return - self.risk_free_rate
            sharpe_ratio = excess_return / annualized_volatility if annualized_volatility > 0 else 0

            # Sortino Ratio
            negative_returns = [r for r in daily_returns if r < 0]
            downside_std = np.std(negative_returns) if negative_returns else 0.0001
            downside_std_annual = downside_std * np.sqrt(252)
            sortino_ratio = excess_return / downside_std_annual if downside_std_annual > 0 else 0

            # Drawdown analysis
            cumulative = (1 + pd.Series(daily_returns)).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max

            max_drawdown = abs(float(drawdowns.min())) if len(drawdowns) > 0 else 0
            avg_drawdown = abs(float(drawdowns[drawdowns < 0].mean())) if len(drawdowns[drawdowns < 0]) > 0 else 0

            # Max drawdown duration
            max_drawdown_duration = self._calculate_max_dd_duration(drawdowns)

            # Calmar Ratio
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

        else:
            daily_returns = []
            total_return = 0
            annualized_return = 0
            volatility = 0
            annualized_volatility = 0
            sharpe_ratio = 0
            sortino_ratio = 0
            max_drawdown = 0
            avg_drawdown = 0
            max_drawdown_duration = 0
            calmar_ratio = 0
            trading_days = 0

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            daily_returns=daily_returns,
            volatility=volatility,
            annualized_volatility=annualized_volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            max_drawdown_duration_days=max_drawdown_duration,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            net_profit=net_profit,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_win_loss_ratio=avg_win_loss_ratio,
            expectancy=expectancy,
            start_date=start_date,
            end_date=end_date,
            trading_days=trading_days,
        )

    def _calculate_max_dd_duration(self, drawdowns: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        in_drawdown = drawdowns < 0
        groups = (in_drawdown != in_drawdown.shift()).cumsum()
        dd_periods = in_drawdown.groupby(groups).sum()
        return int(dd_periods.max()) if len(dd_periods) > 0 else 0

    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics."""
        return PerformanceMetrics(
            total_return=0,
            annualized_return=0,
            daily_returns=[],
            volatility=0,
            annualized_volatility=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            max_drawdown=0,
            avg_drawdown=0,
            max_drawdown_duration_days=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            profit_factor=0,
            gross_profit=0,
            gross_loss=0,
            net_profit=0,
            avg_win=0,
            avg_loss=0,
            largest_win=0,
            largest_loss=0,
            avg_win_loss_ratio=0,
            expectancy=0,
        )

    async def generate_report(
        self,
        strategy_name: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """Generate a comprehensive performance report.

        Args:
            strategy_name: Filter by strategy
            start_date: Start date
            end_date: End date

        Returns:
            Performance report dictionary
        """
        metrics = await self.calculate_metrics(strategy_name, start_date, end_date)

        return {
            "summary": {
                "period": f"{metrics.start_date} to {metrics.end_date}",
                "trading_days": metrics.trading_days,
                "total_return": f"{metrics.total_return:.2%}",
                "annualized_return": f"{metrics.annualized_return:.2%}",
                "net_profit": f"${metrics.net_profit:,.2f}",
            },
            "risk_metrics": {
                "volatility": f"{metrics.annualized_volatility:.2%}",
                "sharpe_ratio": f"{metrics.sharpe_ratio:.2f}",
                "sortino_ratio": f"{metrics.sortino_ratio:.2f}",
                "calmar_ratio": f"{metrics.calmar_ratio:.2f}",
                "max_drawdown": f"{metrics.max_drawdown:.2%}",
                "avg_drawdown": f"{metrics.avg_drawdown:.2%}",
                "max_dd_duration_days": metrics.max_drawdown_duration_days,
            },
            "trade_metrics": {
                "total_trades": metrics.total_trades,
                "winning_trades": metrics.winning_trades,
                "losing_trades": metrics.losing_trades,
                "win_rate": f"{metrics.win_rate:.2%}",
                "profit_factor": f"{metrics.profit_factor:.2f}",
                "avg_win": f"${metrics.avg_win:,.2f}",
                "avg_loss": f"${metrics.avg_loss:,.2f}",
                "avg_win_loss_ratio": f"{metrics.avg_win_loss_ratio:.2f}",
                "expectancy": f"${metrics.expectancy:,.2f}",
                "largest_win": f"${metrics.largest_win:,.2f}",
                "largest_loss": f"${metrics.largest_loss:,.2f}",
            },
            "raw_metrics": {
                "total_return": metrics.total_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
            },
        }

    async def compare_strategies(
        self,
        strategy_names: list[str],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Compare performance across multiple strategies.

        Args:
            strategy_names: List of strategy names
            start_date: Start date
            end_date: End date

        Returns:
            Comparison dictionary
        """
        results = {}

        for name in strategy_names:
            report = await self.generate_report(name, start_date, end_date)
            results[name] = report["raw_metrics"]

        return results
