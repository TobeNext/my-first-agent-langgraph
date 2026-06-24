from __future__ import annotations

import importlib.util
import os
from typing import Any


def agent_evals_available() -> bool:
    return importlib.util.find_spec("agentevals") is not None


def agent_evals_judge_enabled() -> bool:
    return agent_evals_available() and any(
        os.environ.get(key)
        for key in [
            "OPENAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "ZHIPU_API_KEY",
            "MODEL_API_KEY",
        ]
    )


def evaluate_with_agent_evals_or_skip(
    trajectory: list[dict[str, Any]],
    *,
    rule_evaluation: dict[str, Any],
) -> dict[str, Any]:
    if not agent_evals_available():
        return {
            "status": "skipped",
            "skippedReason": "agentevals is not installed.",
            "rule_evaluation": rule_evaluation,
        }
    if not agent_evals_judge_enabled():
        return {
            "status": "skipped",
            "skippedReason": "No eval model API key is configured.",
            "rule_evaluation": rule_evaluation,
        }

    # Keep the first integration conservative: AgentEvals package APIs have changed across
    # releases, so this adapter records the deterministic trajectory rule result and only
    # flips to active once a pinned API integration is added.
    return {
        "status": "skipped",
        "skippedReason": (
            "AgentEvals package is present, but no pinned trajectory judge is configured."
        ),
        "rule_evaluation": rule_evaluation,
        "trajectory_event_count": len(trajectory),
    }
