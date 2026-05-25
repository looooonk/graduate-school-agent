"""Retrieval agent package."""

__all__ = ["gap_fill_prompt", "run_gap_fill", "run_retrieval"]


def __getattr__(name: str) -> object:
    if name in {"gap_fill_prompt", "run_gap_fill"}:
        from grad_agent.agents.retrieval import gap_fill

        return getattr(gap_fill, name)
    if name == "run_retrieval":
        from grad_agent.agents.retrieval.service import run_retrieval

        return run_retrieval
    raise AttributeError(name)
