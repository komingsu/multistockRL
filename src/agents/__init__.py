"""Agent scaffolding for multi-algorithm experimentation."""

from .factory import AgentSpec, build_agent_spec, resolve_agent_class

__all__ = [
    "AgentSpec",
    "build_agent_spec",
    "resolve_agent_class",
]
