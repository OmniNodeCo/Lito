"""Lito — lightest thinking AI agent + custom micro-LLM (stdlib-only)."""

__version__ = "0.3.0"  # Lito-Nano custom micro-LLM
__all__ = ["__version__", "Agent"]


def __getattr__(name: str):
    if name == "Agent":
        from .agent import Agent

        return Agent
    raise AttributeError(name)
