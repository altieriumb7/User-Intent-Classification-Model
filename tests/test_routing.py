from src.routing import route_intent, simulate_routing


def test_route_intent_known_and_unknown():
    assert route_intent("account_access") == "Identity & Access Support"
    assert route_intent("unknown") == "General Support"


def test_simulate_routing_computes_misrouting_rate():
    result = simulate_routing(
        ["billing", "technical_issue", "refund"],
        ["billing", "refund", "refund"],
    )
    assert result["total_tickets"] == 3
    assert result["misrouted_tickets"] == 1
    assert result["misrouting_rate"] == 1 / 3
