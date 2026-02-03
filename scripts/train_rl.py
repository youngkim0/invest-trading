"""RL model training script."""

import argparse
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

from ai.rl.environment import TradingEnv, TradingEnvConfig
from ai.rl.trainer import RLTrainer, TrainingConfig
from config import get_settings
from data.collectors.price_collector import PriceCollector
from data.storage.repository import DatabaseManager


async def collect_training_data(
    symbols: list[str],
    days: int = 365,
    timeframe: str = "1h",
) -> dict:
    """Collect historical data for training."""
    settings = get_settings()
    collector = PriceCollector(
        binance_config=settings.binance,
        alpaca_config=settings.alpaca,
    )

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    all_data = {}

    for symbol in symbols:
        logger.info(f"Collecting data for {symbol}...")

        # Determine if crypto or stock
        if "/" in symbol:
            # Crypto symbol (e.g., "BTC/USDT")
            data = await collector.collect_crypto_history(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            # Stock symbol
            data = await collector.collect_stock_history(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )

        if data is not None and len(data) > 0:
            all_data[symbol] = data
            logger.info(f"Collected {len(data)} candles for {symbol}")
        else:
            logger.warning(f"No data collected for {symbol}")

    return all_data


def create_environment(
    data: dict,
    config: TradingEnvConfig,
) -> TradingEnv:
    """Create trading environment with data."""
    # Use first symbol's data for now
    symbol = list(data.keys())[0]
    df = data[symbol]

    env = TradingEnv(config=config)
    env.set_data(df)

    return env


async def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RL trading agent")

    parser.add_argument(
        "--agent",
        type=str,
        choices=["ppo", "dqn"],
        default="ppo",
        help="Agent type to train",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=["BTC/USDT"],
        help="Symbols to train on",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Days of historical data",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Candle timeframe",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for models",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Use walk-forward optimization",
    )
    parser.add_argument(
        "--hyperparam-search",
        action="store_true",
        help="Perform hyperparameter search",
    )

    args = parser.parse_args()

    # Setup logging
    logger.add(
        f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        rotation="100 MB",
    )

    logger.info("Starting RL training...")
    logger.info(f"Agent: {args.agent}")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Timesteps: {args.timesteps}")

    # Collect data
    logger.info("Collecting training data...")
    data = await collect_training_data(
        symbols=args.symbols,
        days=args.days,
        timeframe=args.timeframe,
    )

    if not data:
        logger.error("No training data collected!")
        return

    # Create environment config
    env_config = TradingEnvConfig(
        initial_balance=100000.0,
        commission_rate=0.001,
        lookback_window=60,
        max_position_size=1.0,
        reward_function="sharpe",
        action_type="discrete",
        include_technical_indicators=True,
    )

    # Create training config
    training_config = TrainingConfig(
        total_timesteps=args.timesteps,
        eval_freq=10000,
        n_eval_episodes=10,
        save_freq=50000,
        log_interval=1000,
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize trainer
    trainer = RLTrainer(
        env_config=env_config,
        training_config=training_config,
        output_dir=str(output_dir),
    )

    # Get training data
    symbol = list(data.keys())[0]
    train_data = data[symbol]

    if args.walk_forward:
        # Walk-forward optimization
        logger.info("Running walk-forward optimization...")
        results = trainer.walk_forward_optimization(
            data=train_data,
            agent_type=args.agent,
            n_splits=5,
        )

        logger.info("Walk-forward results:")
        for i, r in enumerate(results):
            logger.info(f"  Fold {i+1}: Return={r['test_return']:.2%}")

    elif args.hyperparam_search:
        # Hyperparameter search
        logger.info("Running hyperparameter search...")

        if args.agent == "ppo":
            param_grid = {
                "learning_rate": [1e-4, 3e-4, 1e-3],
                "n_steps": [1024, 2048],
                "batch_size": [64, 128],
                "gamma": [0.99, 0.995],
            }
        else:
            param_grid = {
                "learning_rate": [1e-4, 5e-4, 1e-3],
                "buffer_size": [50000, 100000],
                "batch_size": [32, 64],
                "gamma": [0.99, 0.995],
            }

        best_params, results = trainer.hyperparameter_search(
            data=train_data,
            agent_type=args.agent,
            param_grid=param_grid,
            n_trials=10,
        )

        logger.info(f"Best parameters: {best_params}")

    else:
        # Standard training
        logger.info("Starting standard training...")
        agent = trainer.train(
            data=train_data,
            agent_type=args.agent,
        )

        # Save model
        model_path = output_dir / f"{args.agent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        agent.save(str(model_path))
        logger.info(f"Model saved to {model_path}")

    logger.info("Training complete!")


if __name__ == "__main__":
    asyncio.run(main())
