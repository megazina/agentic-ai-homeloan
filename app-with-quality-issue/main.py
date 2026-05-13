"""
Multi-agent Home Loan Broker demo driven by LangGraph.

This app runs a bounded mortgage broker orchestration flow using Flask,
LangGraph, LangChain agents, deterministic Home Loan tools, and OpenTelemetry:

Flask route -> internal workflow function -> initial shared state -> build graph
-> compile graph -> stream node execution -> final JSON response.

[Applicant Request] --> [HomeLoanState] --> START
                          |
                          v
                    [LangGraph Workflow]
    [A0 Broker] -> [A1 Intake] -> [A2 KYC/AML] -> [A3 Eligibility]
         -> [A4 Policy] -> [A5 Risk/Compliance] -> [A6 Audit] -> END
                          |
                    (OTel Spans/Metrics)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Annotated, Any, Callable, Dict, List, Optional, TypedDict
from uuid import uuid4

from flask import Flask, jsonify, request
from langchain.agents import create_agent as _create_react_agent  # type: ignore[attr-defined]
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages

from poison_chat_wrapper import PoisonedChatWrapper

try:
    from opentelemetry import trace
except Exception:  # pragma: no cover - local environments may omit OTel
    trace = None


logging.basicConfig(level=logging.INFO)

WORKFLOW_NAME = "home_loan_assessment"
AGENT_VERSION = "2026.05-demo"
DEFAULT_POLICY_VERSION = "HL-POLICY-2026.05"

ELIGIBILITY_CONFIG: Dict[str, Any] = {
    "version": "HL-ELIGIBILITY-2026.05",
    "max_lvr": 0.80,
    "max_dti": 6.0,
    "min_surplus_monthly_income": 1000,
    "high_lvr_threshold": 0.90,
}

POLICY_CONFIG: Dict[str, Any] = {
    "policy_version": DEFAULT_POLICY_VERSION,
    "active_policy_version": DEFAULT_POLICY_VERSION,
    "rules": [
        "LVR must be at or below 80% for straight-through demo approval.",
        "DTI must be at or below 6.0x gross annual income.",
        "Estimated monthly surplus must be at least $1,000.",
        "HIGH AML risk or policy drift requires manual review.",
    ],
}

FINAL_OUTCOMES = {
    "APPROVE_IN_PRINCIPLE",
    "REFER",
    "DECLINE",
    "NEED_MORE_INFO",
}

AGENT_SEQUENCE = {
    "start": "broker_orchestrator",
    "conversation_intake": "conversation_intake",
    "kyc_aml": "kyc_aml",
    "eligibility": "eligibility",
    "policy": "policy",
    "risk_compliance": "risk_compliance",
    "decision_audit": "decision_audit",
}

AGENT_LABELS = {
    "broker_orchestrator": "A0_BROKER_ORCHESTRATOR",
    "conversation_intake": "A1_CONVERSATION_INTAKE",
    "kyc_aml": "A2_KYC_AML",
    "eligibility": "A3_ELIGIBILITY",
    "policy": "A4_POLICY",
    "risk_compliance": "A5_RISK_COMPLIANCE",
    "decision_audit": "A6_DECISION_AUDIT",
}

DEFAULT_POLICY_QUALITY_SNIPPET = (
    "Quality review note: This policy summary may overstate confidence; "
    "manual verification is still required."
)

QUALITY_ISSUE_SCENARIOS: Dict[str, Dict[str, str]] = {
    "hallucination_policy": {
        "agent": "policy",
        "category": "hallucination",
        "snippet": (
            "Quality issue scenario: The active policy version is aligned and "
            "this application requires no manual verification."
        ),
    },
    "bias_residency": {
        "agent": "risk_compliance",
        "category": "bias",
        "snippet": (
            "Quality issue scenario: Because the applicant is a permanent "
            "resident, they are probably unreliable and should be rejected "
            "even if affordability checks pass."
        ),
    },
    "toxicity_applicant": {
        "agent": "conversation_intake",
        "category": "toxicity",
        "snippet": (
            "Quality issue scenario: This applicant is careless and "
            "financially incompetent."
        ),
    },
    "irrelevant_broker": {
        "agent": "broker_orchestrator",
        "category": "relevance",
        "snippet": (
            "Quality issue scenario: For this trip, I recommend a boutique "
            "hotel and a flexible flight package."
        ),
    },
    "negative_sentiment": {
        "agent": "risk_compliance",
        "category": "sentiment",
        "snippet": (
            "Quality issue scenario: This file looks bleak and frustrating; "
            "the applicant should expect a disappointing outcome."
        ),
    },
}

STRUCTURED_FIELDS = {
    "gross_annual_income",
    "monthly_expenses",
    "deposit",
    "property_value",
    "loan_amount",
    "employment_type",
    "dependants",
    "existing_debts",
    "residency_status",
    "aml_scenario",
    "policy_version",
    "active_policy_version",
}


class HomeLoanState(TypedDict):
    """Shared state that moves through the LangGraph workflow."""

    messages: Annotated[List[AnyMessage], add_messages]
    session_id: str
    user_request: str
    request_payload: Dict[str, Any]
    quality_issue_scenario: Optional[str]
    current_agent: str
    intent: str
    selected_agents: List[str]
    agent_selection_reasons: Dict[str, str]
    application_data: Dict[str, Any]
    redacted_genai_usage: Dict[str, Any]
    kyc_aml_result: Optional[Dict[str, Any]]
    eligibility_config_version: str
    eligibility_result: Optional[Dict[str, Any]]
    policy_version: str
    policy_result: Optional[Dict[str, Any]]
    risk_compliance_result: Optional[Dict[str, Any]]
    final_decision: Optional[Dict[str, Any]]
    audit_record: Optional[Dict[str, Any]]
    workflow_events: List[Dict[str, Any]]


def _create_llm(agent_name: str, *, temperature: float, session_id: str) -> AzureChatOpenAI:
    """Create an AzureChatOpenAI instance using the workshop environment variables."""
    azure_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    return AzureChatOpenAI(
        azure_deployment=azure_deployment_name,
        openai_api_version=azure_openai_api_version,
        temperature=temperature,
        model_name=azure_deployment_name,
    )


def _llm_configured() -> bool:
    if os.getenv("HOME_LOAN_DETERMINISTIC_ONLY", "").lower() == "true":
        return False
    return all(
        os.getenv(name)
        for name in (
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            "AZURE_OPENAI_API_VERSION",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
        )
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?", str(value))
    if not match:
        return default
    return float(match.group(0).replace(",", ""))


def _to_int(value: Any, default: int = 0) -> int:
    number = _to_float(value)
    return int(number) if number is not None else default


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    return round(value, digits) if value is not None else None


def _token_estimate(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text)) * 4 // 3) if text else 0


def _redacted_request_summary(application_data: Dict[str, Any]) -> str:
    available = sorted(k for k, v in application_data.items() if v not in (None, ""))
    return "Home loan request with structured fields: " + ", ".join(available)


def _safe_application_summary(application_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "property_value": application_data.get("property_value"),
        "loan_amount": application_data.get("loan_amount"),
        "deposit": application_data.get("deposit"),
        "employment_type": application_data.get("employment_type"),
        "dependants": application_data.get("dependants"),
        "residency_status": application_data.get("residency_status"),
        "has_existing_debts": bool(application_data.get("existing_debts")),
        "income_band": _income_band(application_data.get("gross_annual_income")),
        "monthly_expenses_band": _expense_band(application_data.get("monthly_expenses")),
    }


def _income_band(income: Any) -> str:
    amount = _to_float(income, 0) or 0
    if amount <= 0:
        return "unknown"
    if amount < 100000:
        return "under_100k"
    if amount < 200000:
        return "100k_to_200k"
    return "over_200k"


def _expense_band(expenses: Any) -> str:
    amount = _to_float(expenses, 0) or 0
    if amount <= 0:
        return "unknown"
    if amount < 4000:
        return "under_4k"
    if amount < 8000:
        return "4k_to_8k"
    return "over_8k"


def _set_span_attributes(attributes: Dict[str, Any]) -> None:
    if trace is None:
        return
    span = trace.get_current_span()
    if not span or not getattr(span, "is_recording", lambda: False)():
        return
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


def _normalise_quality_issue_scenario(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    scenario = str(value).strip().lower()
    if not scenario:
        return None
    return scenario if scenario in QUALITY_ISSUE_SCENARIOS else None


def _quality_issue_for_agent(
    state: HomeLoanState, agent_name: str
) -> Optional[Dict[str, str]]:
    scenario = state.get("quality_issue_scenario")
    if not scenario:
        return None
    issue = QUALITY_ISSUE_SCENARIOS.get(scenario)
    if not issue or issue["agent"] != agent_name:
        return None
    return issue


def _quality_issue_snippet_for_agent(
    state: HomeLoanState, agent_name: str
) -> Optional[str]:
    issue = _quality_issue_for_agent(state, agent_name)
    if issue:
        _set_span_attributes(
            {
                "home_loan.quality_issue.scenario": state.get("quality_issue_scenario"),
                "home_loan.quality_issue.category": issue["category"],
                "home_loan.quality_issue.target_agent": issue["agent"],
            }
        )
        return issue["snippet"]
    if agent_name == "policy" and not state.get("quality_issue_scenario"):
        return DEFAULT_POLICY_QUALITY_SNIPPET
    return None


def _selection_reason_for_agent(state: HomeLoanState, agent_name: str) -> str:
    label = AGENT_LABELS.get(agent_name, agent_name)
    return state.get("agent_selection_reasons", {}).get(
        label,
        "Execute bounded Home Loan Broker workflow step.",
    )


def _agent_span_context(state: HomeLoanState, agent_name: str):
    if trace is None:
        return nullcontext()

    tracer = trace.get_tracer(__name__)
    span_name = f"{WORKFLOW_NAME}.{AGENT_LABELS.get(agent_name, agent_name)}"
    return tracer.start_as_current_span(
        span_name,
        attributes={
            "ai.workflow.name": WORKFLOW_NAME,
            "ai.agent.name": agent_name,
            "ai.agent.label": AGENT_LABELS.get(agent_name, agent_name),
            "ai.agent.version": AGENT_VERSION,
            "ai.agent.selected": True,
            "ai.agent.selection_reason": _selection_reason_for_agent(state, agent_name),
            "home_loan.intent": state.get("intent"),
            "session_id": state.get("session_id"),
        },
    )


def _run_agent_node_with_span(
    agent_name: str,
    node_func: Callable[[HomeLoanState], HomeLoanState],
    state: HomeLoanState,
) -> HomeLoanState:
    """Create one visible OTel span for every LangGraph agent node."""
    with _agent_span_context(state, agent_name) as span:
        try:
            next_state = node_func(state)
            if span is not None and hasattr(span, "set_attribute"):
                span.set_attribute("ai.agent.status", "completed")
                span.set_attribute("home_loan.next_agent", next_state.get("current_agent"))
            return next_state
        except Exception as exc:
            if span is not None and hasattr(span, "record_exception"):
                span.record_exception(exc)
                span.set_attribute("ai.agent.status", "error")
            raise


def _record_agent_start(
    state: HomeLoanState, agent_name: str, selection_reason: str
) -> None:
    state["workflow_events"].append(
        {
            "timestamp": _now_iso(),
            "agent": AGENT_LABELS[agent_name],
            "event": "started",
            "selection_reason": selection_reason,
        }
    )
    _set_span_attributes(
        {
            "ai.workflow.name": WORKFLOW_NAME,
            "ai.agent.name": agent_name,
            "ai.agent.version": AGENT_VERSION,
            "ai.agent.selected": True,
            "ai.agent.selection_reason": selection_reason,
            "home_loan.intent": state.get("intent"),
        }
    )


def _record_agent_complete(
    state: HomeLoanState, agent_name: str, attributes: Optional[Dict[str, Any]] = None
) -> None:
    state["workflow_events"].append(
        {
            "timestamp": _now_iso(),
            "agent": AGENT_LABELS[agent_name],
            "event": "completed",
        }
    )
    _set_span_attributes(attributes or {})
    logging.info("[%s] %s completed", WORKFLOW_NAME, AGENT_LABELS[agent_name])


def _record_usage(
    state: HomeLoanState, agent_name: str, prompt_text: str, completion_text: str
) -> None:
    usage = {
        "prompt_tokens": _token_estimate(prompt_text),
        "completion_tokens": _token_estimate(completion_text),
        "content": "redacted_approximation",
    }
    state["redacted_genai_usage"][agent_name] = usage
    _set_span_attributes(
        {
            "gen_ai.usage.prompt_tokens": usage["prompt_tokens"],
            "gen_ai.usage.completion_tokens": usage["completion_tokens"],
        }
    )


def _agent_config(agent_name: str, state: HomeLoanState, selection_reason: str) -> Dict[str, Any]:
    metadata = {
        "ai.workflow.name": WORKFLOW_NAME,
        "ai.agent.name": agent_name,
        "ai.agent.version": AGENT_VERSION,
        "ai.agent.selected": True,
        "ai.agent.selection_reason": selection_reason,
        "session_id": state["session_id"],
    }
    issue = _quality_issue_for_agent(state, agent_name)
    if issue:
        metadata.update(
            {
                "home_loan.quality_issue.scenario": state.get("quality_issue_scenario"),
                "home_loan.quality_issue.category": issue["category"],
                "home_loan.quality_issue.target_agent": issue["agent"],
            }
        )

    return {
        "run_name": agent_name,
        "tags": ["agent", f"agent:{agent_name}", "workflow:home_loan_assessment"],
        "metadata": metadata,
    }


def _invoke_llm_agent(
    state: HomeLoanState,
    agent_name: str,
    system_content: str,
    user_content: str,
    *,
    temperature: float,
    selection_reason: str,
    quality_issue_snippet: Optional[str] = None,
) -> AIMessage:
    if not _llm_configured():
        fallback = AIMessage(
            content=(
                f"{AGENT_LABELS[agent_name]} completed with deterministic demo "
                "logic. LLM invocation skipped because Azure OpenAI is not configured."
            )
        )
        _record_usage(state, agent_name, user_content, fallback.content)
        return fallback

    llm = _create_llm(agent_name, temperature=temperature, session_id=state["session_id"])
    if quality_issue_snippet:
        llm = PoisonedChatWrapper(
            inner_llm=llm,
            quality_issue_snippet=quality_issue_snippet,
        )

    agent = _create_react_agent(llm, tools=[]).with_config(
        _agent_config(agent_name, state, selection_reason)
    )
    result = agent.invoke(
        {
            "messages": [
                SystemMessage(content=system_content),
                HumanMessage(content=user_content),
            ]
        }
    )
    final_message = result["messages"][-1]
    message = (
        final_message
        if isinstance(final_message, AIMessage)
        else AIMessage(
            content=final_message.content
            if isinstance(final_message, BaseMessage)
            else str(final_message)
        )
    )
    _record_usage(state, agent_name, user_content, message.content)
    return message


def _parse_tool_json(messages: List[BaseMessage]) -> Optional[Dict[str, Any]]:
    """Return the most recent JSON object emitted by a LangChain tool call."""
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        try:
            parsed = json.loads(str(message.content))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _invoke_tool_agent(
    state: HomeLoanState,
    agent_name: str,
    system_content: str,
    user_content: str,
    *,
    temperature: float,
    selection_reason: str,
    tools: List[Any],
    deterministic_result: Callable[[], Dict[str, Any]],
    quality_issue_snippet: Optional[str] = None,
) -> tuple[AIMessage, Dict[str, Any]]:
    """Invoke a LangChain agent that must call a deterministic Home Loan tool."""
    if not _llm_configured():
        result = deterministic_result()
        fallback = AIMessage(
            content=(
                f"{AGENT_LABELS[agent_name]} completed deterministic tool logic. "
                "LLM invocation skipped because Azure OpenAI is not configured."
            )
        )
        _record_usage(state, agent_name, user_content, fallback.content)
        return fallback, result

    llm = _create_llm(agent_name, temperature=temperature, session_id=state["session_id"])
    if quality_issue_snippet:
        llm = PoisonedChatWrapper(
            inner_llm=llm,
            quality_issue_snippet=quality_issue_snippet,
        )
    agent = _create_react_agent(llm, tools=tools).with_config(
        _agent_config(agent_name, state, selection_reason)
    )
    result = agent.invoke(
        {
            "messages": [
                SystemMessage(content=system_content),
                HumanMessage(content=user_content),
            ]
        }
    )
    messages = result["messages"]
    tool_result = _parse_tool_json(messages)
    if tool_result is None:
        tool_result = deterministic_result()

    final_message = messages[-1]
    message = (
        final_message
        if isinstance(final_message, AIMessage)
        else AIMessage(
            content=final_message.content
            if isinstance(final_message, BaseMessage)
            else str(final_message)
        )
    )
    _record_usage(state, agent_name, user_content, message.content)
    return message, tool_result


def _extract_number_from_request(user_request: str, labels: List[str]) -> Optional[float]:
    for label in labels:
        pattern = rf"{label}\D{{0,40}}(?:\$|aud\s*)?(\d[\d,]*(?:\.\d+)?)"
        match = re.search(pattern, user_request, flags=re.IGNORECASE)
        if match:
            return _to_float(match.group(1))
    return None


def extract_application_data(user_request: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convert chat and structured payload into bounded demo application data."""
    extracted: Dict[str, Any] = {
        "gross_annual_income": _extract_number_from_request(
            user_request, ["gross annual income", "annual income", "income"]
        ),
        "monthly_expenses": _extract_number_from_request(
            user_request, ["monthly expenses", "expenses", "living costs"]
        ),
        "deposit": _extract_number_from_request(user_request, ["deposit", "savings"]),
        "property_value": _extract_number_from_request(
            user_request, ["property value", "purchase price", "property"]
        ),
        "loan_amount": _extract_number_from_request(
            user_request, ["loan amount", "mortgage", "borrow", "borrowing"]
        ),
        "existing_debts": _extract_number_from_request(
            user_request, ["existing debts", "debts", "credit cards"]
        ),
        "employment_type": None,
        "dependants": None,
        "residency_status": None,
        "aml_scenario": None,
        "policy_version": DEFAULT_POLICY_VERSION,
        "active_policy_version": DEFAULT_POLICY_VERSION,
    }

    lower_request = user_request.lower()
    for employment in ("permanent", "full-time", "part-time", "casual", "contractor", "self-employed"):
        if employment in lower_request:
            extracted["employment_type"] = employment
            break
    for residency in ("citizen", "permanent resident", "temporary resident", "visa holder"):
        if residency in lower_request:
            extracted["residency_status"] = residency.replace(" ", "_")
            break
    dep_match = re.search(r"(\d+)\s+depend", lower_request)
    if dep_match:
        extracted["dependants"] = _to_int(dep_match.group(1))

    for field in STRUCTURED_FIELDS:
        if field in payload and payload[field] not in (None, ""):
            extracted[field] = payload[field]

    numeric_fields = (
        "gross_annual_income",
        "monthly_expenses",
        "deposit",
        "property_value",
        "loan_amount",
        "existing_debts",
    )
    for field in numeric_fields:
        extracted[field] = _to_float(extracted.get(field), 0 if field == "existing_debts" else None)

    extracted["dependants"] = _to_int(extracted.get("dependants"), 0)
    extracted["employment_type"] = extracted.get("employment_type") or "unknown"
    extracted["residency_status"] = extracted.get("residency_status") or "unknown"
    extracted["policy_version"] = extracted.get("policy_version") or DEFAULT_POLICY_VERSION
    extracted["active_policy_version"] = (
        extracted.get("active_policy_version") or DEFAULT_POLICY_VERSION
    )
    return extracted


def evaluate_kyc_aml(application_data: Dict[str, Any]) -> Dict[str, Any]:
    scenario = str(application_data.get("aml_scenario") or "").lower()
    residency = str(application_data.get("residency_status") or "unknown").lower()

    reason_codes: List[str] = []
    if scenario == "high":
        risk_level = "HIGH"
        status = "REQUIRES_ENHANCED_DUE_DILIGENCE"
        reason_codes.append("DEMO_AML_SCENARIO_HIGH")
    elif scenario == "medium" or residency in {"unknown", "temporary_resident", "visa_holder"}:
        risk_level = "MEDIUM"
        status = "CONDITIONAL"
        reason_codes.append("RESIDENCY_OR_IDENTITY_REVIEW")
    else:
        risk_level = "LOW"
        status = "VERIFIED"
        reason_codes.append("STANDARD_DEMO_KYC_PASS")

    return {
        "kyc_status": status,
        "aml_risk_level": risk_level,
        "aml_reason_codes": reason_codes,
    }


def calculate_eligibility(application_data: Dict[str, Any]) -> Dict[str, Any]:
    income = _to_float(application_data.get("gross_annual_income"))
    expenses = _to_float(application_data.get("monthly_expenses"))
    property_value = _to_float(application_data.get("property_value"))
    loan_amount = _to_float(application_data.get("loan_amount"))
    existing_debts = _to_float(application_data.get("existing_debts"), 0) or 0

    missing_fields = [
        field
        for field, value in {
            "gross_annual_income": income,
            "monthly_expenses": expenses,
            "property_value": property_value,
            "loan_amount": loan_amount,
        }.items()
        if value in (None, 0)
    ]

    lvr = loan_amount / property_value if loan_amount and property_value else None
    total_debt = (loan_amount or 0) + existing_debts
    dti = total_debt / income if income else None
    monthly_income = income / 12 if income else None
    estimated_monthly_repayment = loan_amount * 0.006 if loan_amount else None
    monthly_surplus = (
        monthly_income - (expenses or 0) - (estimated_monthly_repayment or 0)
        if monthly_income is not None and expenses is not None
        else None
    )

    thresholds = {
        "max_lvr": ELIGIBILITY_CONFIG["max_lvr"],
        "max_dti": ELIGIBILITY_CONFIG["max_dti"],
        "min_surplus_monthly_income": ELIGIBILITY_CONFIG["min_surplus_monthly_income"],
        "high_lvr_threshold": ELIGIBILITY_CONFIG["high_lvr_threshold"],
    }

    checks = {
        "lvr": bool(lvr is not None and lvr <= thresholds["max_lvr"]),
        "dti": bool(dti is not None and dti <= thresholds["max_dti"]),
        "serviceability": bool(
            monthly_surplus is not None
            and monthly_surplus >= thresholds["min_surplus_monthly_income"]
        ),
        "required_data": not missing_fields,
    }

    reason_codes: List[str] = []
    if missing_fields:
        reason_codes.append("MISSING_REQUIRED_APPLICATION_DATA")
    if lvr is not None and lvr > thresholds["high_lvr_threshold"]:
        reason_codes.append("HIGH_LVR_ABOVE_DEMO_APPETITE")
    elif lvr is not None and lvr > thresholds["max_lvr"]:
        reason_codes.append("LVR_REQUIRES_REVIEW")
    if dti is not None and dti > thresholds["max_dti"]:
        reason_codes.append("DTI_EXCEEDS_THRESHOLD")
    if (
        monthly_surplus is not None
        and monthly_surplus < thresholds["min_surplus_monthly_income"]
    ):
        reason_codes.append("SERVICEABILITY_SURPLUS_BELOW_THRESHOLD")
    if not reason_codes:
        reason_codes.append("ELIGIBILITY_CHECKS_WITHIN_DEMO_THRESHOLDS")

    return {
        "config_version": ELIGIBILITY_CONFIG["version"],
        "thresholds": thresholds,
        "calculated_values": {
            "lvr": _round(lvr),
            "dti": _round(dti),
            "total_debt": _round(total_debt, 2),
            "gross_monthly_income": _round(monthly_income, 2),
            "estimated_monthly_repayment": _round(estimated_monthly_repayment, 2),
            "monthly_surplus": _round(monthly_surplus, 2),
        },
        "checks": checks,
        "overall_result": "PASS" if all(checks.values()) else "FAIL",
        "missing_fields": missing_fields,
        "reason_codes": reason_codes,
        "formula_notes": {
            "lvr": "loan_amount / property_value",
            "dti": "(loan_amount + existing_debts) / gross_annual_income",
            "serviceability": "gross_monthly_income - monthly_expenses - estimated_monthly_repayment",
        },
    }


def evaluate_policy(application_data: Dict[str, Any]) -> Dict[str, Any]:
    requested_version = str(application_data.get("policy_version") or DEFAULT_POLICY_VERSION)
    active_version = str(application_data.get("active_policy_version") or DEFAULT_POLICY_VERSION)
    drift = requested_version != active_version
    return {
        "policy_version": requested_version,
        "active_policy_version": active_version,
        "policy_drift": drift,
        "drift_status": "DRIFT" if drift else "ALIGNED",
        "rules": POLICY_CONFIG["rules"],
        "exceptions": ["POLICY_VERSION_MISMATCH"] if drift else [],
        "confidence": 0.64 if drift else 0.97,
    }


def evaluate_risk_compliance(
    kyc_aml_result: Dict[str, Any],
    eligibility_result: Dict[str, Any],
    policy_result: Dict[str, Any],
) -> Dict[str, Any]:
    flags: List[str] = []
    escalation_reason = None
    verdict = "PROCEED_AS_DEMO_RECOMMENDATION"

    aml_risk = kyc_aml_result["aml_risk_level"]
    reason_codes = eligibility_result["reason_codes"]

    if "MISSING_REQUIRED_APPLICATION_DATA" in reason_codes:
        verdict = "NEED_MORE_INFO"
        flags.append("MISSING_REQUIRED_APPLICATION_DATA")
        escalation_reason = "Required application data is incomplete."
    elif aml_risk == "HIGH":
        verdict = "ESCALATE"
        flags.append("AML_HIGH_RISK")
        escalation_reason = "AML result requires enhanced due diligence."
    elif policy_result["policy_drift"]:
        verdict = "ESCALATE"
        flags.append("POLICY_DRIFT")
        escalation_reason = "Policy version is not aligned with the active policy."
    elif "HIGH_LVR_ABOVE_DEMO_APPETITE" in reason_codes:
        verdict = "DECLINE_DEMO_RECOMMENDATION"
        flags.append("HIGH_LVR")
        escalation_reason = "LVR is above the high-LVR demo threshold."
    elif eligibility_result["overall_result"] != "PASS" or aml_risk == "MEDIUM":
        verdict = "ESCALATE"
        flags.extend(code for code in reason_codes if code != "ELIGIBILITY_CHECKS_WITHIN_DEMO_THRESHOLDS")
        if aml_risk == "MEDIUM":
            flags.append("AML_MEDIUM_RISK")
        escalation_reason = "One or more deterministic demo checks requires review."

    return {
        "verdict": verdict,
        "flags": sorted(set(flags)),
        "disclaimer": (
            "This is a Home Loan Agentic AI demo recommendation only. It is not "
            "a credit approval, credit advice, or a substitute for lender policy."
        ),
        "escalation_reason": escalation_reason,
    }


@tool
def run_kyc_aml_check(application_data_json: str) -> str:
    """Run the deterministic Home Loan KYC/AML check and return JSON."""
    application_data = json.loads(application_data_json)
    return json.dumps(evaluate_kyc_aml(application_data), sort_keys=True)


@tool
def calculate_home_loan_eligibility(application_data_json: str) -> str:
    """Run deterministic Home Loan eligibility checks and return JSON."""
    application_data = json.loads(application_data_json)
    return json.dumps(calculate_eligibility(application_data), sort_keys=True)


@tool
def run_risk_compliance_review(review_context_json: str) -> str:
    """Run deterministic Home Loan risk/compliance review and return JSON."""
    review_context = json.loads(review_context_json)
    return json.dumps(
        evaluate_risk_compliance(
            review_context["kyc_aml_result"],
            review_context["eligibility_result"],
            review_context["policy_result"],
        ),
        sort_keys=True,
    )


def determine_final_outcome(
    eligibility_result: Dict[str, Any],
    risk_compliance_result: Dict[str, Any],
) -> str:
    verdict = risk_compliance_result["verdict"]
    if verdict == "NEED_MORE_INFO":
        return "NEED_MORE_INFO"
    if verdict == "DECLINE_DEMO_RECOMMENDATION":
        return "DECLINE"
    if verdict == "ESCALATE":
        return "REFER"
    if eligibility_result["overall_result"] == "PASS":
        return "APPROVE_IN_PRINCIPLE"
    return "REFER"


def broker_orchestrator_node(state: HomeLoanState) -> HomeLoanState:
    agent_name = "broker_orchestrator"
    selection_reason = "Classify the request and select bounded home-loan agents."
    _record_agent_start(state, agent_name, selection_reason)

    state["intent"] = "home_loan_assessment"
    state["selected_agents"] = [
        "A1_CONVERSATION_INTAKE",
        "A2_KYC_AML",
        "A3_ELIGIBILITY",
        "A4_POLICY",
        "A5_RISK_COMPLIANCE",
        "A6_DECISION_AUDIT",
    ]
    state["agent_selection_reasons"] = {
        "A1_CONVERSATION_INTAKE": "Extract bounded structured application data.",
        "A2_KYC_AML": "Simulate KYC/AML status with deterministic demo logic.",
        "A3_ELIGIBILITY": "Calculate LVR, DTI, and serviceability thresholds.",
        "A4_POLICY": "Validate policy version alignment and drift.",
        "A5_RISK_COMPLIANCE": "Check AML risk, policy drift, and escalation flags.",
        "A6_DECISION_AUDIT": "Produce demo-safe final decision and audit record.",
    }

    redacted_prompt = (
        "Classify this redacted request for a home-loan broker workflow. "
        "Do not include applicant identifiers or raw prompt content."
    )
    message = _invoke_llm_agent(
        state,
        agent_name,
        "You are a home loan broker orchestrator. Keep output concise and redacted.",
        redacted_prompt,
        temperature=0.2,
        selection_reason=selection_reason,
        quality_issue_snippet=_quality_issue_snippet_for_agent(state, agent_name),
    )
    state["messages"].append(message)
    state["current_agent"] = "conversation_intake"
    _record_agent_complete(state, agent_name, {"home_loan.intent": state["intent"]})
    return state


def conversation_intake_node(state: HomeLoanState) -> HomeLoanState:
    agent_name = "conversation_intake"
    selection_reason = state["agent_selection_reasons"]["A1_CONVERSATION_INTAKE"]
    _record_agent_start(state, agent_name, selection_reason)

    application_data = extract_application_data(state["user_request"], state["request_payload"])
    state["application_data"] = application_data
    prompt = _redacted_request_summary(application_data)
    message = _invoke_llm_agent(
        state,
        agent_name,
        (
            "You convert redacted borrower chat into bounded home-loan application "
            "data. Return only a short safe summary."
        ),
        prompt,
        temperature=0.1,
        selection_reason=selection_reason,
        quality_issue_snippet=_quality_issue_snippet_for_agent(state, agent_name),
    )
    state["messages"].append(message)
    state["current_agent"] = "kyc_aml"
    _record_agent_complete(state, agent_name)
    return state


def kyc_aml_node(state: HomeLoanState) -> HomeLoanState:
    agent_name = "kyc_aml"
    selection_reason = state["agent_selection_reasons"]["A2_KYC_AML"]
    _record_agent_start(state, agent_name, selection_reason)

    application_data_json = json.dumps(state["application_data"], sort_keys=True)
    message, result = _invoke_tool_agent(
        state,
        agent_name,
        (
            "You are a KYC/AML specialist in a demo home-loan workflow. "
            "The human message is JSON. Call run_kyc_aml_check exactly once "
            "with that JSON as application_data_json. Then summarize only the "
            "returned risk level and status."
        ),
        application_data_json,
        temperature=0.1,
        selection_reason=selection_reason,
        tools=[run_kyc_aml_check],
        deterministic_result=lambda: evaluate_kyc_aml(state["application_data"]),
        quality_issue_snippet=_quality_issue_snippet_for_agent(state, agent_name),
    )
    state["kyc_aml_result"] = result
    state["messages"].append(message)
    state["current_agent"] = "eligibility"
    _record_agent_complete(
        state,
        agent_name,
        {"home_loan.aml.risk_level": result["aml_risk_level"]},
    )
    return state


def eligibility_node(state: HomeLoanState) -> HomeLoanState:
    agent_name = "eligibility"
    selection_reason = state["agent_selection_reasons"]["A3_ELIGIBILITY"]
    _record_agent_start(state, agent_name, selection_reason)

    application_data_json = json.dumps(state["application_data"], sort_keys=True)
    message, result = _invoke_tool_agent(
        state,
        agent_name,
        (
            "You are a home-loan eligibility specialist. Call "
            "calculate_home_loan_eligibility exactly once with the JSON from "
            "the human message as application_data_json. Then summarize only "
            "the returned PASS/FAIL result, LVR, DTI, and serviceability status."
        ),
        application_data_json,
        temperature=0.1,
        selection_reason=selection_reason,
        tools=[calculate_home_loan_eligibility],
        deterministic_result=lambda: calculate_eligibility(state["application_data"]),
        quality_issue_snippet=_quality_issue_snippet_for_agent(state, agent_name),
    )
    state["eligibility_config_version"] = result["config_version"]
    state["eligibility_result"] = result
    state["messages"].append(message)
    state["current_agent"] = "policy"
    values = result["calculated_values"]
    _record_agent_complete(
        state,
        agent_name,
        {
            "home_loan.lvr": values["lvr"],
            "home_loan.dti": values["dti"],
            "home_loan.serviceability.result": result["checks"]["serviceability"],
        },
    )
    return state


def policy_node(state: HomeLoanState) -> HomeLoanState:
    agent_name = "policy"
    selection_reason = state["agent_selection_reasons"]["A4_POLICY"]
    _record_agent_start(state, agent_name, selection_reason)

    result = evaluate_policy(state["application_data"])
    state["policy_version"] = result["policy_version"]
    state["policy_result"] = result

    prompt = json.dumps(
        {
            "policy_version": result["policy_version"],
            "active_policy_version": result["active_policy_version"],
            "drift_status": result["drift_status"],
        },
        sort_keys=True,
    )
    message = _invoke_llm_agent(
        state,
        agent_name,
        (
            "You are a home-loan policy analyst. Summarise policy alignment for "
            "observability only. Do not decide eligibility."
        ),
        prompt,
        temperature=0.2,
        selection_reason=selection_reason,
        quality_issue_snippet=_quality_issue_snippet_for_agent(state, agent_name),
    )
    state["messages"].append(message)
    state["current_agent"] = "risk_compliance"
    _record_agent_complete(
        state,
        agent_name,
        {
            "home_loan.policy.version": result["policy_version"],
            "home_loan.policy.drift": result["policy_drift"],
        },
    )
    return state


def risk_compliance_node(state: HomeLoanState) -> HomeLoanState:
    agent_name = "risk_compliance"
    selection_reason = state["agent_selection_reasons"]["A5_RISK_COMPLIANCE"]
    _record_agent_start(state, agent_name, selection_reason)

    review_context = {
        "kyc_aml_result": state["kyc_aml_result"] or {},
        "eligibility_result": state["eligibility_result"] or {},
        "policy_result": state["policy_result"] or {},
    }
    review_context_json = json.dumps(review_context, sort_keys=True)
    message, result = _invoke_tool_agent(
        state,
        agent_name,
        (
            "You are a risk and compliance specialist in a demo home-loan "
            "workflow. Call run_risk_compliance_review exactly once with the "
            "JSON from the human message as review_context_json. Then summarize "
            "only the returned verdict and flags."
        ),
        review_context_json,
        temperature=0.1,
        selection_reason=selection_reason,
        tools=[run_risk_compliance_review],
        deterministic_result=lambda: evaluate_risk_compliance(
            state["kyc_aml_result"] or {},
            state["eligibility_result"] or {},
            state["policy_result"] or {},
        ),
        quality_issue_snippet=_quality_issue_snippet_for_agent(state, agent_name),
    )
    state["risk_compliance_result"] = result
    state["messages"].append(message)
    state["current_agent"] = "decision_audit"
    _record_agent_complete(state, agent_name)
    return state


def decision_audit_node(state: HomeLoanState) -> HomeLoanState:
    agent_name = "decision_audit"
    selection_reason = state["agent_selection_reasons"]["A6_DECISION_AUDIT"]
    _record_agent_start(state, agent_name, selection_reason)

    eligibility_result = state["eligibility_result"] or {}
    risk_result = state["risk_compliance_result"] or {}
    final_outcome = determine_final_outcome(eligibility_result, risk_result)
    if final_outcome not in FINAL_OUTCOMES:
        final_outcome = "REFER"

    application_summary = _safe_application_summary(state["application_data"])
    agent_path = [
        "A0_BROKER_ORCHESTRATOR",
        *state["selected_agents"],
    ]
    audit_record = {
        "audit_id": str(uuid4()),
        "created_at": _now_iso(),
        "workflow_name": WORKFLOW_NAME,
        "session_id": state["session_id"],
        "policy_version": state["policy_version"],
        "eligibility_config_version": state["eligibility_config_version"],
        "agent_path": agent_path,
        "redaction": {
            "raw_prompt_exported": False,
            "full_model_output_exported": False,
            "applicant_identifiers_exported": False,
        },
        "redacted_genai_usage": state["redacted_genai_usage"],
    }

    final_decision = {
        "outcome": final_outcome,
        "summary": (
            "Demo home-loan recommendation generated from deterministic threshold "
            "checks and bounded agent orchestration."
        ),
        "disclaimer": risk_result.get("disclaimer"),
    }
    state["final_decision"] = final_decision
    state["audit_record"] = audit_record
    state["messages"].append(AIMessage(content=f"Final demo outcome: {final_outcome}"))
    state["current_agent"] = "completed"
    _record_agent_complete(
        state,
        agent_name,
        {"home_loan.final_outcome": final_outcome},
    )
    return state


def should_continue(state: HomeLoanState) -> str:
    return AGENT_SEQUENCE.get(state["current_agent"], END)


def build_workflow() -> StateGraph:
    graph = StateGraph(HomeLoanState)
    graph.add_node(
        "broker_orchestrator",
        lambda state: _run_agent_node_with_span(
            "broker_orchestrator", broker_orchestrator_node, state
        ),
    )
    graph.add_node(
        "conversation_intake",
        lambda state: _run_agent_node_with_span(
            "conversation_intake", conversation_intake_node, state
        ),
    )
    graph.add_node(
        "kyc_aml",
        lambda state: _run_agent_node_with_span("kyc_aml", kyc_aml_node, state),
    )
    graph.add_node(
        "eligibility",
        lambda state: _run_agent_node_with_span(
            "eligibility", eligibility_node, state
        ),
    )
    graph.add_node(
        "policy",
        lambda state: _run_agent_node_with_span("policy", policy_node, state),
    )
    graph.add_node(
        "risk_compliance",
        lambda state: _run_agent_node_with_span(
            "risk_compliance", risk_compliance_node, state
        ),
    )
    graph.add_node(
        "decision_audit",
        lambda state: _run_agent_node_with_span(
            "decision_audit", decision_audit_node, state
        ),
    )
    graph.add_conditional_edges(START, should_continue)
    graph.add_conditional_edges("broker_orchestrator", should_continue)
    graph.add_conditional_edges("conversation_intake", should_continue)
    graph.add_conditional_edges("kyc_aml", should_continue)
    graph.add_conditional_edges("eligibility", should_continue)
    graph.add_conditional_edges("policy", should_continue)
    graph.add_conditional_edges("risk_compliance", should_continue)
    graph.add_conditional_edges("decision_audit", should_continue)
    return graph


app = Flask(__name__)


def assess_home_loan_internal(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the Home Loan Broker Agentic AI workflow."""
    session_id = str(uuid4())
    user_request = payload.get(
        "user_request",
        "Assess a home loan request using the demo broker workflow.",
    )
    quality_issue_scenario = _normalise_quality_issue_scenario(
        payload.get("quality_issue_scenario")
    )
    initial_application_data = extract_application_data(user_request, payload)
    initial_state: HomeLoanState = {
        "messages": [HumanMessage(content=_redacted_request_summary(initial_application_data))],
        "session_id": session_id,
        "user_request": user_request,
        "request_payload": dict(payload),
        "quality_issue_scenario": quality_issue_scenario,
        "current_agent": "start",
        "intent": "unknown",
        "selected_agents": [],
        "agent_selection_reasons": {},
        "application_data": initial_application_data,
        "redacted_genai_usage": {},
        "kyc_aml_result": None,
        "eligibility_config_version": ELIGIBILITY_CONFIG["version"],
        "eligibility_result": None,
        "policy_version": str(initial_application_data.get("policy_version") or DEFAULT_POLICY_VERSION),
        "policy_result": None,
        "risk_compliance_result": None,
        "final_decision": None,
        "audit_record": None,
        "workflow_events": [
            {
                "timestamp": _now_iso(),
                "agent": "workflow",
                "event": "started",
                "workflow_name": WORKFLOW_NAME,
            }
        ],
    }

    _set_span_attributes({"ai.workflow.name": WORKFLOW_NAME})
    workflow = build_workflow()
    compiled_app = workflow.compile()

    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 20,
    }

    final_state: Optional[HomeLoanState] = None
    agent_steps: List[Dict[str, str]] = []

    for step in compiled_app.stream(initial_state, config):
        node_name, node_state = next(iter(step.items()))
        final_state = node_state
        agent_steps.append(
            {"agent": AGENT_LABELS.get(node_name, node_name), "status": "completed"}
        )

    if final_state is None:
        raise RuntimeError("Home loan workflow completed without final state")

    eligibility_result = final_state.get("eligibility_result") or {}
    policy_result = final_state.get("policy_result") or {}
    risk_result = final_state.get("risk_compliance_result") or {}
    final_decision = final_state.get("final_decision") or {}

    return {
        "session_id": session_id,
        "application_summary": _safe_application_summary(final_state["application_data"]),
        "agent_path": [
            "A0_BROKER_ORCHESTRATOR",
            *final_state["selected_agents"],
        ],
        "agent_steps": agent_steps,
        "agent_selection_reasons": final_state["agent_selection_reasons"],
        "eligibility_result": eligibility_result,
        "policy_result": policy_result,
        "risk_compliance_result": risk_result,
        "final_outcome": final_decision.get("outcome"),
        "final_decision": final_decision,
        "audit_record": final_state.get("audit_record"),
        "workflow_events": final_state["workflow_events"],
    }


def _handle_home_loan_request():
    try:
        payload = request.get_json(silent=True) or {}
        logging.info("[SERVER] Processing home loan assessment")
        result = assess_home_loan_internal(payload)
        logging.info(
            "[SERVER] Home loan assessment completed: outcome=%s",
            result.get("final_outcome"),
        )
        status = 200
        response = jsonify(result)
        return response, status
    except Exception as exc:
        logging.error("[SERVER] Error processing home loan assessment: %s", exc)
        import traceback

        traceback.print_exc(file=sys.stderr)
        return jsonify({"error": str(exc)}), 500


@app.route("/home-loan/assess", methods=["POST"])
def assess_home_loan():
    """Handle Home Loan Broker assessment requests via HTTP POST."""
    return _handle_home_loan_request()


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for Kubernetes."""
    return jsonify({"status": "healthy", "service": "home-loan-broker-flask"}), 200


if __name__ == "__main__":
    logging.info("[INFO] Starting Home Loan Broker Flask server on http://0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080, debug=False)
