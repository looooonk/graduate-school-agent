"""Top-level pipeline orchestration."""

__all__ = ["run_all_schools", "run_school"]


def __getattr__(name: str) -> object:
    if name in {"run_all_schools", "run_school"}:
        from grad_agent.orchestration import runner

        return getattr(runner, name)
    raise AttributeError(name)
