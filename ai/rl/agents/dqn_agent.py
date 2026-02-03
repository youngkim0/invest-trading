"""DQN (Deep Q-Network) agent for trading.

Uses Stable-Baselines3 for the core DQN implementation.
Best suited for discrete action spaces.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ai.rl.environment import TradingEnv, TradingEnvConfig


class DQNTradingCallback(BaseCallback):
    """Custom callback for DQN training metrics."""

    def __init__(
        self,
        log_freq: int = 1000,
        verbose: int = 1,
    ):
        """Initialize callback.

        Args:
            log_freq: Logging frequency in timesteps
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.q_values: list[float] = []

    def _on_step(self) -> bool:
        """Called at each step."""
        if self.n_calls % self.log_freq == 0:
            # Log exploration rate
            if hasattr(self.model, "exploration_rate"):
                if self.logger:
                    self.logger.record("dqn/exploration_rate", self.model.exploration_rate)

        return True


class DQNTradingAgent:
    """DQN agent for discrete action trading."""

    def __init__(
        self,
        env_config: TradingEnvConfig | None = None,
        learning_rate: float = 1e-4,
        buffer_size: int = 100_000,
        learning_starts: int = 10_000,
        batch_size: int = 64,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int = 4,
        gradient_steps: int = 1,
        target_update_interval: int = 1000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        policy_kwargs: dict[str, Any] | None = None,
        device: str = "auto",
        seed: int | None = None,
    ):
        """Initialize DQN trading agent.

        Args:
            env_config: Trading environment configuration
            learning_rate: Learning rate
            buffer_size: Replay buffer size
            learning_starts: Steps before training starts
            batch_size: Batch size for training
            tau: Soft update coefficient
            gamma: Discount factor
            train_freq: Training frequency
            gradient_steps: Gradient steps per update
            target_update_interval: Target network update interval
            exploration_fraction: Fraction of training for exploration decay
            exploration_initial_eps: Initial exploration rate
            exploration_final_eps: Final exploration rate
            policy_kwargs: Additional policy network kwargs
            device: Device to use
            seed: Random seed
        """
        self.env_config = env_config or TradingEnvConfig(action_type="discrete")
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.device = device
        self.seed = seed

        # Ensure discrete action space
        if self.env_config.action_type != "discrete":
            logger.warning("DQN requires discrete actions. Overriding action_type to 'discrete'")
            self.env_config.action_type = "discrete"

        # Default policy kwargs with dueling architecture
        self.policy_kwargs = policy_kwargs or {
            "net_arch": [256, 256],
        }

        self.model: DQN | None = None
        self.vec_env: VecNormalize | None = None

    def create_env(
        self,
        df: pd.DataFrame,
        normalize: bool = True,
    ) -> VecNormalize | DummyVecEnv:
        """Create environment for training/evaluation.

        Args:
            df: OHLCV DataFrame
            normalize: Whether to normalize observations

        Returns:
            Vectorized environment
        """
        def make_env():
            return TradingEnv(df=df, config=self.env_config)

        vec_env = DummyVecEnv([make_env])

        if normalize:
            vec_env = VecNormalize(
                vec_env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0,
                gamma=self.gamma,
            )

        return vec_env

    def train(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame | None = None,
        total_timesteps: int = 500_000,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        save_path: str | Path | None = None,
        log_dir: str | Path | None = None,
        verbose: int = 1,
    ) -> dict[str, Any]:
        """Train the DQN agent.

        Args:
            train_df: Training data DataFrame
            eval_df: Evaluation data DataFrame
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes
            save_path: Path to save best model
            log_dir: Tensorboard log directory
            verbose: Verbosity level

        Returns:
            Training statistics
        """
        logger.info(f"Starting DQN training for {total_timesteps} timesteps")

        # Create environment
        self.vec_env = self.create_env(train_df)

        # Create model
        self.model = DQN(
            policy="MlpPolicy",
            env=self.vec_env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            learning_starts=self.learning_starts,
            batch_size=self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            target_update_interval=self.target_update_interval,
            exploration_fraction=self.exploration_fraction,
            exploration_initial_eps=self.exploration_initial_eps,
            exploration_final_eps=self.exploration_final_eps,
            policy_kwargs=self.policy_kwargs,
            tensorboard_log=str(log_dir) if log_dir else None,
            device=self.device,
            seed=self.seed,
            verbose=verbose,
        )

        # Setup callbacks
        callbacks = [DQNTradingCallback(log_freq=1000, verbose=verbose)]

        # Evaluation callback
        if eval_df is not None and save_path:
            eval_env = self.create_env(eval_df)
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(save_path),
                log_path=str(log_dir) if log_dir else None,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
            )
            callbacks.append(eval_callback)

        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=verbose > 0,
        )

        logger.info("DQN training completed")

        # Save final model
        if save_path:
            self.save(Path(save_path) / "final_model")

        return self._get_training_stats()

    def _get_training_stats(self) -> dict[str, Any]:
        """Get training statistics."""
        if not self.model:
            return {}

        return {
            "total_timesteps": self.model.num_timesteps,
            "exploration_rate": self.model.exploration_rate,
        }

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Predict action for given observation.

        Args:
            observation: Environment observation
            deterministic: Use deterministic policy

        Returns:
            Tuple of (action, info)
        """
        if not self.model:
            raise RuntimeError("Model not trained. Call train() first.")

        # Normalize observation
        if self.vec_env and hasattr(self.vec_env, "normalize_obs"):
            observation = self.vec_env.normalize_obs(observation)

        action, _states = self.model.predict(observation, deterministic=deterministic)

        # Get Q-values for info
        q_values = None
        if hasattr(self.model, "q_net"):
            import torch

            with torch.no_grad():
                obs_tensor = torch.as_tensor(observation).float().unsqueeze(0).to(self.model.device)
                q_values = self.model.q_net(obs_tensor).cpu().numpy()[0]

        return action, {"q_values": q_values}

    def evaluate(
        self,
        df: pd.DataFrame,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> dict[str, Any]:
        """Evaluate agent on data.

        Args:
            df: Evaluation data DataFrame
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic policy

        Returns:
            Evaluation statistics
        """
        if not self.model:
            raise RuntimeError("Model not trained. Call train() first.")

        logger.info(f"Evaluating DQN agent for {n_episodes} episodes")

        env = TradingEnv(df=df, config=self.env_config)

        episode_rewards = []
        episode_returns = []
        episode_trades = []
        episode_win_rates = []
        episode_drawdowns = []
        action_distribution = {0: 0, 1: 0, 2: 0}  # Hold, Buy, Sell

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                # Normalize observation
                if self.vec_env and hasattr(self.vec_env, "normalize_obs"):
                    obs_normalized = self.vec_env.normalize_obs(obs.reshape(1, *obs.shape))
                else:
                    obs_normalized = obs.reshape(1, *obs.shape)

                action, _ = self.model.predict(obs_normalized, deterministic=deterministic)
                action_int = int(action[0])
                action_distribution[action_int] += 1

                obs, reward, terminated, truncated, info = env.step(action_int)
                episode_reward += float(reward)
                done = terminated or truncated

            # Collect episode stats
            episode_rewards.append(episode_reward)
            if env.portfolio:
                episode_returns.append(
                    (env.portfolio.total_value - self.env_config.initial_balance)
                    / self.env_config.initial_balance
                )
                episode_trades.append(env.portfolio.total_trades)
                episode_win_rates.append(
                    env.portfolio.winning_trades / max(1, env.portfolio.total_trades)
                )
                episode_drawdowns.append(env.portfolio.drawdown)

        total_actions = sum(action_distribution.values())

        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_return": np.mean(episode_returns),
            "std_return": np.std(episode_returns),
            "mean_trades": np.mean(episode_trades),
            "mean_win_rate": np.mean(episode_win_rates),
            "mean_drawdown": np.mean(episode_drawdowns),
            "max_drawdown": max(episode_drawdowns),
            "action_distribution": {
                "hold": action_distribution[0] / total_actions,
                "buy": action_distribution[1] / total_actions,
                "sell": action_distribution[2] / total_actions,
            },
        }

    def save(self, path: str | Path) -> None:
        """Save model and normalizer.

        Args:
            path: Save path (without extension)
        """
        if not self.model:
            raise RuntimeError("Model not trained. Call train() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(str(path))

        if self.vec_env and hasattr(self.vec_env, "save"):
            self.vec_env.save(str(path) + "_vecnorm.pkl")

        logger.info(f"DQN model saved to {path}")

    def load(self, path: str | Path, df: pd.DataFrame | None = None) -> None:
        """Load model and normalizer.

        Args:
            path: Model path (without extension)
            df: Optional DataFrame for environment creation
        """
        path = Path(path)

        # Load model
        if df is not None:
            self.vec_env = self.create_env(df)
            self.model = DQN.load(str(path), env=self.vec_env)
        else:
            self.model = DQN.load(str(path))

        # Load normalizer if exists
        vecnorm_path = str(path) + "_vecnorm.pkl"
        if Path(vecnorm_path).exists() and self.vec_env:
            self.vec_env = VecNormalize.load(vecnorm_path, self.vec_env)

        logger.info(f"DQN model loaded from {path}")


def create_dqn_agent(
    env_config: TradingEnvConfig | None = None,
    **kwargs: Any,
) -> DQNTradingAgent:
    """Factory function to create DQN agent.

    Args:
        env_config: Environment configuration
        **kwargs: Additional DQN parameters

    Returns:
        Configured DQN agent
    """
    return DQNTradingAgent(env_config=env_config, **kwargs)
