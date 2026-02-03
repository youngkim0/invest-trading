"""PPO (Proximal Policy Optimization) agent for trading.

Uses Stable-Baselines3 for the core PPO implementation.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from ai.rl.environment import TradingEnv, TradingEnvConfig


class TradingMetricsCallback(BaseCallback):
    """Custom callback for logging trading metrics during training."""

    def __init__(
        self,
        log_freq: int = 1000,
        verbose: int = 1,
    ):
        """Initialize callback.

        Args:
            log_freq: Frequency of logging (in timesteps)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.win_rates: list[float] = []
        self.sharpe_ratios: list[float] = []

    def _on_step(self) -> bool:
        """Called at each step."""
        # Log metrics periodically
        if self.n_calls % self.log_freq == 0:
            # Get info from environment
            if hasattr(self.training_env, "envs"):
                for env in self.training_env.envs:
                    if hasattr(env, "portfolio") and env.portfolio:
                        win_rate = env.portfolio.winning_trades / max(1, env.portfolio.total_trades)
                        self.win_rates.append(win_rate)

                        # Log to tensorboard if available
                        if self.logger:
                            self.logger.record("trading/win_rate", win_rate)
                            self.logger.record("trading/total_trades", env.portfolio.total_trades)
                            self.logger.record("trading/drawdown", env.portfolio.drawdown)

        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout."""
        pass


class PPOTradingAgent:
    """PPO agent specialized for trading environments."""

    def __init__(
        self,
        env_config: TradingEnvConfig | None = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        policy_kwargs: dict[str, Any] | None = None,
        device: str = "auto",
        seed: int | None = None,
    ):
        """Initialize PPO trading agent.

        Args:
            env_config: Trading environment configuration
            learning_rate: Learning rate
            n_steps: Number of steps per update
            batch_size: Minibatch size
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm
            policy_kwargs: Additional policy network kwargs
            device: Device to use (auto, cpu, cuda)
            seed: Random seed
        """
        self.env_config = env_config or TradingEnvConfig()
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.seed = seed

        # Default policy kwargs
        self.policy_kwargs = policy_kwargs or {
            "net_arch": dict(pi=[256, 256], vf=[256, 256]),
        }

        self.model: PPO | None = None
        self.vec_env: VecNormalize | None = None

    def create_env(
        self,
        df: pd.DataFrame,
        n_envs: int = 1,
        use_subprocess: bool = False,
    ) -> VecNormalize:
        """Create vectorized training environment.

        Args:
            df: OHLCV DataFrame
            n_envs: Number of parallel environments
            use_subprocess: Use subprocess for parallelization

        Returns:
            Vectorized and normalized environment
        """
        def make_env():
            return TradingEnv(df=df, config=self.env_config)

        if n_envs == 1:
            vec_env = DummyVecEnv([make_env])
        else:
            if use_subprocess:
                vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])
            else:
                vec_env = DummyVecEnv([make_env for _ in range(n_envs)])

        # Wrap with normalization
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
        total_timesteps: int = 1_000_000,
        n_envs: int = 4,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        save_path: str | Path | None = None,
        log_dir: str | Path | None = None,
        verbose: int = 1,
    ) -> dict[str, Any]:
        """Train the PPO agent.

        Args:
            train_df: Training data DataFrame
            eval_df: Evaluation data DataFrame
            total_timesteps: Total training timesteps
            n_envs: Number of parallel environments
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes
            save_path: Path to save best model
            log_dir: Tensorboard log directory
            verbose: Verbosity level

        Returns:
            Training statistics
        """
        logger.info(f"Starting PPO training for {total_timesteps} timesteps")

        # Create training environment
        self.vec_env = self.create_env(train_df, n_envs=n_envs)

        # Create model
        self.model = PPO(
            policy="MlpPolicy",
            env=self.vec_env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            policy_kwargs=self.policy_kwargs,
            tensorboard_log=str(log_dir) if log_dir else None,
            device=self.device,
            seed=self.seed,
            verbose=verbose,
        )

        # Setup callbacks
        callbacks = [TradingMetricsCallback(log_freq=1000, verbose=verbose)]

        # Evaluation callback
        if eval_df is not None and save_path:
            eval_env = self.create_env(eval_df, n_envs=1)
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

        logger.info("Training completed")

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
            "n_updates": self.model._n_updates,
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

        # Normalize observation if we have the normalizer
        if self.vec_env:
            observation = self.vec_env.normalize_obs(observation)

        action, _states = self.model.predict(observation, deterministic=deterministic)

        return action, {}

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

        logger.info(f"Evaluating agent for {n_episodes} episodes")

        env = TradingEnv(df=df, config=self.env_config)

        episode_rewards = []
        episode_returns = []
        episode_trades = []
        episode_win_rates = []
        episode_drawdowns = []

        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                # Normalize observation
                if self.vec_env:
                    obs_normalized = self.vec_env.normalize_obs(obs.reshape(1, *obs.shape))
                else:
                    obs_normalized = obs.reshape(1, *obs.shape)

                action, _ = self.model.predict(obs_normalized, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action[0])
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

        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_return": np.mean(episode_returns),
            "std_return": np.std(episode_returns),
            "mean_trades": np.mean(episode_trades),
            "mean_win_rate": np.mean(episode_win_rates),
            "mean_drawdown": np.mean(episode_drawdowns),
            "max_drawdown": max(episode_drawdowns),
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

        if self.vec_env:
            self.vec_env.save(str(path) + "_vecnorm.pkl")

        logger.info(f"Model saved to {path}")

    def load(self, path: str | Path, df: pd.DataFrame | None = None) -> None:
        """Load model and normalizer.

        Args:
            path: Model path (without extension)
            df: Optional DataFrame for environment creation
        """
        path = Path(path)

        # Load model
        if df is not None:
            self.vec_env = self.create_env(df, n_envs=1)
            self.model = PPO.load(str(path), env=self.vec_env)
        else:
            self.model = PPO.load(str(path))

        # Load normalizer if exists
        vecnorm_path = str(path) + "_vecnorm.pkl"
        if Path(vecnorm_path).exists() and self.vec_env:
            self.vec_env = VecNormalize.load(vecnorm_path, self.vec_env)

        logger.info(f"Model loaded from {path}")


def create_ppo_agent(
    env_config: TradingEnvConfig | None = None,
    **kwargs: Any,
) -> PPOTradingAgent:
    """Factory function to create PPO agent.

    Args:
        env_config: Environment configuration
        **kwargs: Additional PPO parameters

    Returns:
        Configured PPO agent
    """
    return PPOTradingAgent(env_config=env_config, **kwargs)
