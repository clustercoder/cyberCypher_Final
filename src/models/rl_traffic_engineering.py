"""Reinforcement Learning-based traffic engineering using PPO.

Trains a PPO agent (stable-baselines3) in a Gym/Gymnasium environment that
wraps synthetic network telemetry, then exposes a suggest_action() interface
for the DeciderAgent.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from src.utils.logger import logger

try:
    import gymnasium as _gym
    from gymnasium import spaces

    _GYM_AVAILABLE = True
    _GYM_BACKEND = "gymnasium"
except ImportError:
    try:
        import gym as _gym  # type: ignore[no-redef]
        from gym import spaces  # type: ignore[no-redef]

        _GYM_AVAILABLE = True
        _GYM_BACKEND = "gym"
    except ImportError:
        _gym = None  # type: ignore[assignment]
        spaces = None  # type: ignore[assignment]
        _GYM_AVAILABLE = False
        _GYM_BACKEND = "none"

try:
    from stable_baselines3 import PPO

    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False
    PPO = None  # type: ignore[assignment]

if _GYM_AVAILABLE:
    _EnvBase = _gym.Env  # type: ignore[union-attr]
else:
    class _EnvBase:  # pragma: no cover - fallback only
        pass

_LINK_METRICS = ("utilization_pct", "latency_ms", "packet_loss_pct", "throughput_gbps")
_RL_AVAILABLE = _GYM_AVAILABLE and _SB3_AVAILABLE
_DEFAULT_RL_MODEL_PATH = "models/rl_traffic_engineer"

# Reward weights: utilization (40%), latency (30%), packet_loss (20%), SLA (10%)
_REWARD_WEIGHTS = {
    "utilization_pct": 0.4,
    "latency_ms": 0.3,
    "packet_loss_pct": 0.2,
    "sla_violation": 0.1,
}
_UTIL_PENALTY_THRESHOLD = 80.0
_LATENCY_PENALTY_THRESHOLD = 30.0
_PKTLOSS_PENALTY_THRESHOLD = 0.5


class NetworkSimEnv(_EnvBase):
    """Gym-compatible environment wrapping a network telemetry snapshot.

    Observation:
        Flattened vector of per-link metric values (utilization, latency,
        packet_loss, throughput) for all links in ``link_ids`` order.

    Action space:
        Discrete: one action per link (reroute that link's traffic).
        Action 0 means \"no change\".
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, link_ids: list[str], num_steps: int = 100) -> None:
        if not _GYM_AVAILABLE:
            raise ImportError("gymnasium (or gym) is required for NetworkSimEnv.")
        super().__init__()

        self.link_ids = link_ids
        self.num_links = len(link_ids)
        self.num_steps = num_steps
        self._current_step = 0
        self._metrics: dict[str, dict[str, float]] = {}
        self._rng = np.random.default_rng()

        obs_dim = self.num_links * len(_LINK_METRICS)
        self.observation_space = spaces.Box(  # type: ignore[union-attr]
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        # +1 for "no action"
        self.action_space = spaces.Discrete(self.num_links + 1)  # type: ignore[union-attr]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to a randomised baseline network state."""
        del options
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if hasattr(super(), "reset"):
            try:
                super().reset(seed=seed)  # type: ignore[misc]
            except TypeError:
                # Older gym variants may not accept keyword args.
                super().reset()  # type: ignore[misc]

        self._current_step = 0
        self._metrics = {
            link_id: {
                "utilization_pct": float(self._rng.uniform(20, 70)),
                "latency_ms": float(self._rng.uniform(2, 20)),
                "packet_loss_pct": float(self._rng.uniform(0, 0.3)),
                "throughput_gbps": float(self._rng.uniform(50, 200)),
            }
            for link_id in self.link_ids
        }
        return self._observation(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply action and advance one time step."""
        self._apply_action(int(action))
        self._advance_metrics()
        reward = self._compute_reward()
        self._current_step += 1
        terminated = self._current_step >= self.num_steps
        truncated = False
        return self._observation(), reward, terminated, truncated, {}

    def update_metrics(self, metrics: dict[str, dict[str, float]]) -> None:
        """Inject telemetry snapshot for inference mode."""
        for link_id, link_metrics in metrics.items():
            if link_id in self._metrics:
                self._metrics[link_id].update({k: float(v) for k, v in link_metrics.items()})
            else:
                self._metrics[link_id] = {k: float(v) for k, v in link_metrics.items()}

    def render(self) -> None:  # pragma: no cover - debug helper only
        return None

    def close(self) -> None:  # pragma: no cover - nothing to release
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _observation(self) -> np.ndarray:
        obs: list[float] = []
        for link_id in self.link_ids:
            m = self._metrics.get(link_id, {})
            obs.append(float(m.get("utilization_pct", 0.0)) / 100.0)
            obs.append(float(m.get("latency_ms", 0.0)) / 100.0)
            obs.append(float(m.get("packet_loss_pct", 0.0)) / 5.0)
            obs.append(float(m.get("throughput_gbps", 0.0)) / 400.0)
        return np.array(obs, dtype=np.float32)

    def _apply_action(self, action: int) -> None:
        """Action >0 reduces utilization on the selected link (simulated reroute)."""
        if action == 0 or action > self.num_links:
            return
        link_id = self.link_ids[action - 1]
        m = self._metrics.get(link_id, {})
        current_util = float(m.get("utilization_pct", 50.0))
        m["utilization_pct"] = max(10.0, current_util * 0.85)
        self._metrics[link_id] = m

    def _advance_metrics(self) -> None:
        """Simulate small random fluctuations each step."""
        for link_id in self.link_ids:
            m = self._metrics[link_id]
            m["utilization_pct"] = float(
                np.clip(m["utilization_pct"] + self._rng.normal(0, 1.5), 0.0, 100.0)
            )
            m["latency_ms"] = float(np.clip(m["latency_ms"] + self._rng.normal(0, 0.5), 0.5, 200.0))
            m["packet_loss_pct"] = float(
                np.clip(m["packet_loss_pct"] + self._rng.normal(0, 0.02), 0.0, 5.0)
            )

    def _compute_reward(self) -> float:
        util_penalty = 0.0
        latency_penalty = 0.0
        pktloss_penalty = 0.0
        sla_violations = 0

        for m in self._metrics.values():
            util = float(m.get("utilization_pct", 0.0))
            latency = float(m.get("latency_ms", 0.0))
            pkt = float(m.get("packet_loss_pct", 0.0))

            if util > _UTIL_PENALTY_THRESHOLD:
                util_penalty += (util - _UTIL_PENALTY_THRESHOLD) / 20.0
            if latency > _LATENCY_PENALTY_THRESHOLD:
                latency_penalty += (latency - _LATENCY_PENALTY_THRESHOLD) / 20.0
            if pkt > _PKTLOSS_PENALTY_THRESHOLD:
                pktloss_penalty += pkt
                sla_violations += 1

        total_penalty = (
            _REWARD_WEIGHTS["utilization_pct"] * util_penalty
            + _REWARD_WEIGHTS["latency_ms"] * latency_penalty
            + _REWARD_WEIGHTS["packet_loss_pct"] * pktloss_penalty
            + _REWARD_WEIGHTS["sla_violation"] * sla_violations
        )
        return -total_penalty


class RLTrafficEngineer:
    """PPO-based traffic engineering agent."""

    def __init__(self, link_ids: list[str], total_timesteps: int = 50_000) -> None:
        self.link_ids = link_ids
        self.total_timesteps = total_timesteps
        self._model: Any = None
        self._env: NetworkSimEnv | None = None
        self._available = _RL_AVAILABLE

        if not self._available:
            logger.warning(
                "RL unavailable: stable-baselines3 and gym/gymnasium are required."
            )

    def train(self) -> None:
        """Train PPO on the synthetic network simulation environment."""
        if not self._available:
            logger.warning("Skipping RL training: dependencies unavailable.")
            return

        logger.info("RL backend detected: {}", _GYM_BACKEND)
        self._env = NetworkSimEnv(self.link_ids)
        self._model = PPO(
            "MlpPolicy",
            self._env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
        )
        self._model.learn(total_timesteps=self.total_timesteps)

    def suggest_action(self, link_metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
        """Suggest a traffic engineering action for current link metrics."""
        if not self._available or self._model is None:
            return self._fallback_suggest(link_metrics)

        env = NetworkSimEnv(self.link_ids)
        obs, _ = env.reset()
        del obs
        env.update_metrics(link_metrics)
        current_obs = env._observation()

        action, _ = self._model.predict(current_obs, deterministic=True)
        action = int(action)

        if action == 0 or action > len(self.link_ids):
            return {
                "action_type": "no_action",
                "target_link": None,
                "confidence": 0.5,
                "rationale": "RL policy suggests no intervention.",
                "source": "rl_policy",
            }

        target_link = self.link_ids[action - 1]
        metrics = link_metrics.get(target_link, {})
        util = float(metrics.get("utilization_pct", 0.0))
        if util > _UTIL_PENALTY_THRESHOLD:
            confidence = min(0.95, 0.6 + (util - _UTIL_PENALTY_THRESHOLD) / 100.0)
        else:
            confidence = 0.6

        return {
            "action_type": "reroute",
            "target_link": target_link,
            "confidence": round(confidence, 4),
            "rationale": (
                f"RL PPO policy selected reroute on {target_link} "
                f"(utilization={util:.1f}%)."
            ),
            "source": "rl_policy",
        }

    def is_available(self) -> bool:
        """Return True if SB3 + gym backend are available."""
        return self._available

    def save(self, path: str = _DEFAULT_RL_MODEL_PATH) -> None:
        """Save trained PPO model to disk (SB3 appends .zip automatically)."""
        if not self._available or self._model is None:
            logger.warning("Cannot save RL model: model not trained or RL unavailable.")
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._model.save(path)
        logger.info("RL model saved to {}", path)

    def load(self, path: str = _DEFAULT_RL_MODEL_PATH) -> bool:
        """Load a saved PPO model from disk. Returns True on success."""
        if not self._available:
            return False
        zip_path = path if path.endswith(".zip") else f"{path}.zip"
        if not (os.path.exists(path) or os.path.exists(zip_path)):
            return False
        try:
            self._env = NetworkSimEnv(self.link_ids)
            self._model = PPO.load(path, env=self._env)
            logger.info("RL model loaded from {}", path)
            return True
        except Exception as exc:
            logger.warning("Failed to load RL model from {}: {}", path, exc)
            return False

    def _fallback_suggest(self, link_metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
        """Heuristic fallback: reroute the most congested link."""
        worst_link: str | None = None
        worst_util = 0.0
        for link_id, metrics in link_metrics.items():
            util = float(metrics.get("utilization_pct", 0.0))
            if util > worst_util:
                worst_util = util
                worst_link = link_id

        if worst_link and worst_util >= _UTIL_PENALTY_THRESHOLD:
            return {
                "action_type": "reroute",
                "target_link": worst_link,
                "confidence": 0.6,
                "rationale": (
                    f"Heuristic fallback: reroute on {worst_link} "
                    f"(utilization={worst_util:.1f}%)."
                ),
                "source": "heuristic_fallback",
            }

        return {
            "action_type": "no_action",
            "target_link": None,
            "confidence": 0.5,
            "rationale": "Heuristic fallback: no congestion detected.",
            "source": "heuristic_fallback",
        }

