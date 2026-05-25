"""Fit assessment package."""

__all__ = ["calibrate_fit_confidence", "run_fit_assessment"]


def __getattr__(name: str) -> object:
    if name == "calibrate_fit_confidence":
        from grad_agent.agents.fit.confidence import calibrate_fit_confidence

        return calibrate_fit_confidence
    if name == "run_fit_assessment":
        from grad_agent.agents.fit.service import run_fit_assessment

        return run_fit_assessment
    raise AttributeError(name)
