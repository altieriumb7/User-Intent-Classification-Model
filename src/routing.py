"""Intent-to-team routing simulation."""

from __future__ import annotations

from collections import Counter

import pandas as pd


INTENT_TO_TEAM = {
    "account_access": "Identity & Access Support",
    "billing": "Billing Operations",
    "cancellation": "Retention & Account Changes",
    "complaint": "Customer Experience",
    "escalation": "Escalation Desk",
    "product_question": "Product Specialists",
    "refund": "Refunds & Credits",
    "technical_issue": "Technical Support",
}

DEFAULT_TEAM = "General Support"


def route_intent(intent: str) -> str:
    """Map an intent label to the suggested support team."""
    return INTENT_TO_TEAM.get(intent, DEFAULT_TEAM)


def simulate_routing(true_intents: list[str], predicted_intents: list[str]) -> dict[str, object]:
    """Compute routing outcomes and misrouting rate from ground-truth and predicted labels."""
    if len(true_intents) != len(predicted_intents):
        raise ValueError("true_intents and predicted_intents must have the same length.")

    rows = []
    misrouted = 0
    per_team_misroutes: Counter[str] = Counter()

    for true_intent, predicted_intent in zip(true_intents, predicted_intents):
        true_team = route_intent(true_intent)
        predicted_team = route_intent(predicted_intent)
        is_misrouted = true_team != predicted_team
        if is_misrouted:
            misrouted += 1
            per_team_misroutes[true_team] += 1
        rows.append(
            {
                "true_intent": true_intent,
                "predicted_intent": predicted_intent,
                "true_team": true_team,
                "predicted_team": predicted_team,
                "misrouted": is_misrouted,
            }
        )

    total = len(rows)
    return {
        "total_tickets": total,
        "misrouted_tickets": misrouted,
        "misrouting_rate": misrouted / total if total else 0.0,
        "per_team_misroutes": dict(per_team_misroutes),
        "routes": rows,
    }


def routing_frame(true_intents: list[str], predicted_intents: list[str]) -> pd.DataFrame:
    """Return the routing simulation as a DataFrame for reporting or display."""
    return pd.DataFrame(simulate_routing(true_intents, predicted_intents)["routes"])
