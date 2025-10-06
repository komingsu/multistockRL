"""Factory helpers for building agent specifications from config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Type

try:
    from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
except ImportError:  # pragma: no cover - optional dependency
    A2C = DDPG = PPO = SAC = TD3 = None  # type: ignore[assignment]

from src.utils.config import Config


AlgorithmClass = Type[Any] | None

ALGO_REGISTRY: Mapping[str, AlgorithmClass] = {
    "PPO": PPO,
    "SAC": SAC,
    "DDPG": DDPG,
    "A2C": A2C,
    "TD3": TD3,
}

@dataclass(slots=True)
class AgentSpec:
    """Serializable agent description aligned with stable-baselines3."""

    name: str
    algo_cls: AlgorithmClass
    policy: str
    kwargs: Dict[str, Any]

    def instantiate(self, env: Any) -> Any:
        if self.algo_cls is None:
            raise ImportError(
                f"Algorithm '{self.name}' requires stable-baselines3 but it is not installed."
            )
        return self.algo_cls(self.policy, env, **self.kwargs)


def resolve_agent_class(name: str) -> AlgorithmClass:
    try:
        return ALGO_REGISTRY[name.upper()]
    except KeyError as exc:
        raise KeyError(f"Unsupported agent '{name}'. Known: {', '.join(ALGO_REGISTRY.keys())}") from exc


def build_agent_spec(config: Config, name: str | None = None) -> AgentSpec:
    policy_name = name or config.agents.get("default")
    if not policy_name:
        raise KeyError("Agent configuration missing 'default' entry")
    agent_settings = config.agent_settings(policy_name)
    policy = agent_settings.pop("policy", "MlpPolicy")
    algo_cls = resolve_agent_class(policy_name)
    return AgentSpec(name=policy_name, algo_cls=algo_cls, policy=policy, kwargs=agent_settings)
