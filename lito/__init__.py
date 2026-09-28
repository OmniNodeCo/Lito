"""Lito — lightest thinking AI agent (stdlib-only, tiny RAM)."""

__version__ = "0.2.0"
__all__ = ["__version__", "Agent"]


def __getattr__(name: str):
    if name == "Agent":
        from .agent import Agent

        return Agent
    raise AttributeError(name)
