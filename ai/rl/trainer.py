"""Training pipeline for RL trading agents.

Provides a unified interface for training, evaluation, and model management.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from loguru import logger

from ai.rl.agents.dqn_agent import DQNTradingAgent, create_dqn_agent
from ai.rl.agents.ppo_agent import PPOTradingAgent, create_ppo_agent
from ai.rl.environment import TradingEnvConfig


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""

    # Algorithm selection
    algorithm: Literal["ppo", "dqn"] = "ppo"

    # Training parameters
    total_timesteps: int = 1_000_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 5

    # Data split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Saving
    save_freq: int = 50_000
    keep_n_checkpoints: int = 5

    # Walk-forward
    use_walk_forward: bool = False
    walk_forward_windows: int = 5
    walk_forward_train_size: float = 0.6
    walk_forward_test_size: float = 0.2

    # Paths
    output_dir: Path = field(default_factory=lambda: Path("models"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.log_dir = Path(self.log_dir)


class RLTrainer:
    """Unified trainer for RL trading agents."""

    def __init__(
        self,
        config: TrainingConfig | None = None,
        env_config: TradingEnvConfig | None = None,
        agent_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize trainer.

        Args:
            config: Training configuration
            env_config: Environment configuration
            agent_kwargs: Additional agent-specific parameters
        """
        self.config = config or TrainingConfig()
        self.env_config = env_config or TradingEnvConfig()
        self.agent_kwargs = agent_kwargs or {}

        self.agent: PPOTradingAgent | DQNTradingAgent | None = None
        self.training_history: list[dict[str, Any]] = []

    def _split_data(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test sets.

        Args:
            df: Full dataset

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        n = len(df)
        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        logger.info(
            f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

        return train_df, val_df, test_df

    def _create_agent(self) -> PPOTradingAgent | DQNTradingAgent:
        """Create agent based on configuration."""
        if self.config.algorithm == "ppo":
            return create_ppo_agent(env_config=self.env_config, **self.agent_kwargs)
        elif self.config.algorithm == "dqn":
            return create_dqn_agent(env_config=self.env_config, **self.agent_kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

    def train(
        self,
        df: pd.DataFrame,
        experiment_name: str | None = None,
    ) -> dict[str, Any]:
        """Train agent on data.

        Args:
            df: OHLCV DataFrame
            experiment_name: Name for this training run

        Returns:
            Training results
        """
        # Setup directories
        experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self.config.output_dir / experiment_name
        log_dir = self.config.log_dir / experiment_name

        save_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Split data
        train_df, val_df, test_df = self._split_data(df)

        # Create agent
        self.agent = self._create_agent()

        # Train
        logger.info(f"Starting training with {self.config.algorithm.upper()}")

        train_results = self.agent.train(
            train_df=train_df,
            eval_df=val_df,
            total_timesteps=self.config.total_timesteps,
            eval_freq=self.config.eval_freq,
            n_eval_episodes=self.config.n_eval_episodes,
            save_path=save_dir,
            log_dir=log_dir,
            verbose=1,
        )

        # Evaluate on test set
        logger.info("Evaluating on test set")
        test_results = self.agent.evaluate(
            df=test_df,
            n_episodes=10,
            deterministic=True,
        )

        # Compile results
        results = {
            "experiment_name": experiment_name,
            "algorithm": self.config.algorithm,
            "train_results": train_results,
            "test_results": test_results,
            "config": {
                "training": self.config.__dict__,
                "environment": self.env_config.__dict__,
            },
            "timestamp": datetime.now().isoformat(),
        }

        # Save results
        results_path = save_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {results_path}")

        self.training_history.append(results)

        return results

    def train_walk_forward(
        self,
        df: pd.DataFrame,
        experiment_name: str | None = None,
    ) -> dict[str, Any]:
        """Train using walk-forward optimization.

        Trains multiple models on expanding/rolling windows.

        Args:
            df: OHLCV DataFrame
            experiment_name: Name for this training run

        Returns:
            Walk-forward results
        """
        experiment_name = experiment_name or f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_dir = self.config.output_dir / experiment_name
        save_dir.mkdir(parents=True, exist_ok=True)

        n = len(df)
        window_size = n // self.config.walk_forward_windows
        train_size = int(window_size * self.config.walk_forward_train_size)
        test_size = int(window_size * self.config.walk_forward_test_size)

        window_results = []

        for i in range(self.config.walk_forward_windows):
            logger.info(f"Walk-forward window {i + 1}/{self.config.walk_forward_windows}")

            # Define window boundaries
            window_start = i * window_size
            train_end = window_start + train_size
            test_end = min(train_end + test_size, n)

            train_df = df.iloc[window_start:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()

            if len(train_df) < 100 or len(test_df) < 20:
                logger.warning(f"Skipping window {i + 1}: insufficient data")
                continue

            # Create new agent for this window
            agent = self._create_agent()

            # Reduced timesteps for walk-forward
            timesteps = self.config.total_timesteps // self.config.walk_forward_windows

            # Train
            window_save_dir = save_dir / f"window_{i + 1}"
            window_log_dir = self.config.log_dir / experiment_name / f"window_{i + 1}"

            train_results = agent.train(
                train_df=train_df,
                total_timesteps=timesteps,
                save_path=window_save_dir,
                log_dir=window_log_dir,
                verbose=0,
            )

            # Evaluate
            test_results = agent.evaluate(
                df=test_df,
                n_episodes=5,
                deterministic=True,
            )

            window_results.append({
                "window": i + 1,
                "train_size": len(train_df),
                "test_size": len(test_df),
                "train_results": train_results,
                "test_results": test_results,
            })

        # Aggregate results
        aggregate_metrics = self._aggregate_walk_forward_results(window_results)

        results = {
            "experiment_name": experiment_name,
            "algorithm": self.config.algorithm,
            "walk_forward_windows": self.config.walk_forward_windows,
            "window_results": window_results,
            "aggregate_metrics": aggregate_metrics,
            "timestamp": datetime.now().isoformat(),
        }

        # Save results
        results_path = save_dir / "wf_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Walk-forward results saved to {results_path}")

        return results

    def _aggregate_walk_forward_results(
        self,
        window_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Aggregate results across walk-forward windows."""
        if not window_results:
            return {}

        # Extract test metrics
        returns = [w["test_results"].get("mean_return", 0) for w in window_results]
        win_rates = [w["test_results"].get("mean_win_rate", 0) for w in window_results]
        drawdowns = [w["test_results"].get("max_drawdown", 0) for w in window_results]

        import numpy as np

        return {
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
            "mean_win_rate": np.mean(win_rates),
            "mean_max_drawdown": np.mean(drawdowns),
            "worst_drawdown": max(drawdowns),
            "profitable_windows": sum(1 for r in returns if r > 0),
            "total_windows": len(window_results),
        }

    def hyperparameter_search(
        self,
        df: pd.DataFrame,
        param_grid: dict[str, list[Any]],
        n_trials: int = 10,
    ) -> dict[str, Any]:
        """Simple random hyperparameter search.

        Args:
            df: OHLCV DataFrame
            param_grid: Dictionary of parameter names to lists of values
            n_trials: Number of random combinations to try

        Returns:
            Search results with best parameters
        """
        import random

        logger.info(f"Starting hyperparameter search with {n_trials} trials")

        train_df, val_df, _ = self._split_data(df)

        results = []

        for trial in range(n_trials):
            # Sample random parameters
            params = {
                key: random.choice(values)
                for key, values in param_grid.items()
            }

            logger.info(f"Trial {trial + 1}/{n_trials}: {params}")

            try:
                # Update agent kwargs
                trial_agent_kwargs = {**self.agent_kwargs, **params}

                # Create agent with sampled params
                if self.config.algorithm == "ppo":
                    agent = create_ppo_agent(
                        env_config=self.env_config,
                        **trial_agent_kwargs
                    )
                else:
                    agent = create_dqn_agent(
                        env_config=self.env_config,
                        **trial_agent_kwargs
                    )

                # Quick training
                agent.train(
                    train_df=train_df,
                    total_timesteps=self.config.total_timesteps // 10,
                    verbose=0,
                )

                # Evaluate
                eval_results = agent.evaluate(df=val_df, n_episodes=5)

                results.append({
                    "trial": trial + 1,
                    "params": params,
                    "mean_return": eval_results["mean_return"],
                    "mean_win_rate": eval_results["mean_win_rate"],
                    "max_drawdown": eval_results["max_drawdown"],
                })

            except Exception as e:
                logger.error(f"Trial {trial + 1} failed: {e}")
                continue

        # Find best
        if results:
            best = max(results, key=lambda x: x["mean_return"])
            logger.info(f"Best params: {best['params']} with return {best['mean_return']:.4f}")
        else:
            best = None

        return {
            "all_results": results,
            "best_result": best,
        }

    def load_model(
        self,
        path: str | Path,
        df: pd.DataFrame | None = None,
    ) -> None:
        """Load a trained model.

        Args:
            path: Path to saved model
            df: Optional DataFrame for environment
        """
        self.agent = self._create_agent()
        self.agent.load(path, df)
        logger.info(f"Model loaded from {path}")

    def predict(
        self,
        observation: Any,
        deterministic: bool = True,
    ) -> tuple[Any, dict[str, Any]]:
        """Get prediction from loaded model.

        Args:
            observation: Environment observation
            deterministic: Use deterministic policy

        Returns:
            Tuple of (action, info)
        """
        if not self.agent:
            raise RuntimeError("No agent loaded. Call train() or load_model() first.")

        return self.agent.predict(observation, deterministic)


def train_agent(
    df: pd.DataFrame,
    algorithm: str = "ppo",
    total_timesteps: int = 1_000_000,
    output_dir: str = "models",
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience function to train an agent.

    Args:
        df: OHLCV DataFrame
        algorithm: "ppo" or "dqn"
        total_timesteps: Total training steps
        output_dir: Output directory
        **kwargs: Additional configuration

    Returns:
        Training results
    """
    config = TrainingConfig(
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        output_dir=Path(output_dir),
        **kwargs
    )

    trainer = RLTrainer(config=config)
    return trainer.train(df)
