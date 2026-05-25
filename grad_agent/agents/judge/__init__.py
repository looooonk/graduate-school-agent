"""Judge agent package."""

__all__ = ["run_judge"]


def __getattr__(name: str) -> object:
    if name == "run_judge":
        from grad_agent.agents.judge.service import run_judge

        return run_judge
    raise AttributeError(name)
