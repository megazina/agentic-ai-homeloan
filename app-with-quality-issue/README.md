# Home Loan Broker Agentic AI Demo

This app is a bounded Home Loan Broker workflow built with Flask, LangGraph,
LangChain agents, deterministic mortgage assessment tools, and
OpenTelemetry/Splunk instrumentation.

The app exposes one assessment route:

```text
POST /home-loan/assess
```

It returns a demo-safe JSON recommendation. Raw applicant text, full model
outputs, and applicant identifiers are not included in the response.

## Agent Flow

The workflow is a sequential LangGraph state machine. Each node records a
Home Loan agent span, and the LLM-backed nodes also create GenAI spans that are
shown in Splunk Agent Flow.

```text
[Applicant request]
      |
      v
A0_BROKER_ORCHESTRATOR
      |
      v
A1_CONVERSATION_INTAKE
      |
      v
A2_KYC_AML
      |
      v
A3_ELIGIBILITY
      |
      v
A4_POLICY
      |
      v
A5_RISK_COMPLIANCE
      |
      v
A6_DECISION_AUDIT
      |
      v
[JSON response]
```

### A0 Broker Orchestrator

Classifies the request as a home-loan assessment and selects the bounded agent
path. This node uses a LangChain agent, so Splunk should show
`invoke_agent broker_orchestrator` and a nested model/chat span when Azure
OpenAI is configured.

### A1 Conversation Intake

Extracts the bounded application fields from the request and structured JSON
payload, then asks an LLM agent to summarize the redacted application shape.
The output feeds all later deterministic checks.

Important fields include:

- `gross_annual_income`
- `monthly_expenses`
- `deposit`
- `property_value`
- `loan_amount`
- `employment_type`
- `dependants`
- `existing_debts`
- `residency_status`
- `aml_scenario`
- `policy_version`
- `active_policy_version`

### A2 KYC/AML

Invokes a LangChain agent with the `run_kyc_aml_check` tool. The tool calls
`evaluate_kyc_aml(...)`, which classifies the demo AML risk as `LOW`,
`MEDIUM`, or `HIGH`.

With Azure OpenAI configured, the trace should include:

```text
invoke_agent kyc_aml
execute_tool run_kyc_aml_check
```

### A3 Eligibility

Invokes a LangChain agent with the `calculate_home_loan_eligibility` tool. The
tool calls `calculate_eligibility(...)`, which calculates LVR, DTI, monthly
surplus, required-data completeness, and threshold pass/fail status.

With Azure OpenAI configured, the trace should include:

```text
invoke_agent eligibility
execute_tool calculate_home_loan_eligibility
```

### A4 Policy

Evaluates policy version alignment and policy drift, then uses a policy analyst
agent to summarize the policy status. This `app-with-quality-issue` variant
wraps LLM outputs with `PoisonedChatWrapper` to inject simulated quality issues
into trace-captured agent responses. Without a `quality_issue_scenario` request
field, the policy agent keeps the original default policy-narrative quality
issue. Scenario payloads can target other LLM-backed agents. The deterministic
policy result still comes from `evaluate_policy(...)`.

### A5 Risk/Compliance

Invokes a LangChain agent with the `run_risk_compliance_review` tool. The tool
combines KYC/AML, eligibility, and policy results to choose a deterministic
demo verdict:

- `PROCEED_AS_DEMO_RECOMMENDATION`
- `NEED_MORE_INFO`
- `ESCALATE`
- `DECLINE_DEMO_RECOMMENDATION`

With Azure OpenAI configured, the trace should include:

```text
invoke_agent risk_compliance
execute_tool run_risk_compliance_review
```

### A6 Decision Audit

Produces the final demo outcome and audit record. This step is intentionally
deterministic and creates the response fields used by tests and downstream demo
scripts.

Final outcomes are:

- `APPROVE_IN_PRINCIPLE`
- `REFER`
- `DECLINE`
- `NEED_MORE_INFO`

## Deterministic Demo Rules

- LVR = `loan_amount / property_value`
- DTI = `(loan_amount + existing_debts) / gross_annual_income`
- Serviceability = `gross_monthly_income - monthly_expenses - estimated_monthly_repayment`
- Estimated monthly repayment = `loan_amount * 0.006`

Defaults:

- `max_lvr`: `0.80`
- `max_dti`: `6.0`
- `min_surplus_monthly_income`: `1000`
- `high_lvr_threshold`: `0.90`
- `policy_version`: `HL-POLICY-2026.05`
- `active_policy_version`: `HL-POLICY-2026.05`

## Observability

The app uses Splunk OpenTelemetry auto-instrumentation plus explicit Home Loan
agent attributes. In Splunk APM, a normal LLM-enabled request should show:

- `workflow LangGraph` as the top GenAI workflow span.
- LangGraph `step ...` spans for each node.
- `invoke_agent` spans for A0-A5.
- `execute_tool` spans for KYC/AML, eligibility, and risk/compliance.
- `chat gpt-4.1-mini` model spans inside the LLM-backed agents.

The code keeps deterministic assessment data separate from model narrative:
agent/tool spans improve the trace shape, but the final recommendation still
comes from the deterministic Home Loan functions.

For local/offline testing, set `HOME_LOAN_DETERMINISTIC_ONLY=true`. In that
mode the workflow skips Azure OpenAI calls and still returns deterministic JSON,
but Splunk will not show the full nested `invoke_agent` and `execute_tool`
pattern for the LLM-backed agents.

## Run Locally

<details>
<summary>Local Flask and OpenTelemetry Collector setup</summary>

Install dependencies first if they are not already available in your Python
environment:

```bash
python -m pip install -r requirements.txt
```

To run only the Flask app, without exporting traces:

```bash
python main.py
```

### Local Collector With Docker

The easiest local tracing setup is to run a Splunk OpenTelemetry Collector in
Docker and point the app at it. The collector listens for OTLP on
`localhost:4317` and forwards traces and metrics to Splunk Observability Cloud.
The included collector config sends traces to Splunk APM with the `sapm`
exporter and metrics with the `signalfx` exporter.

Set your Splunk Observability access token and realm:

```bash
export SPLUNK_ACCESS_TOKEN="<your-token>"
export SPLUNK_REALM="<your-realm>" # for example us1, us0, eu0, au0
```

Start the local collector:

```bash
docker compose -f local-otel/docker-compose.yml up
```

In a second terminal, run the app with OpenTelemetry instrumentation:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
OTEL_EXPORTER_OTLP_PROTOCOL=grpc \
OTEL_SERVICE_NAME=home-loan-broker-local \
opentelemetry-instrument python main.py
```

If you already have another collector running, skip Docker Compose and point
`OTEL_EXPORTER_OTLP_ENDPOINT` at that collector instead. Without a reachable
collector, the app still serves requests, but spans will not arrive in Splunk.

The server listens on port `8080`.

```bash
curl http://localhost:8080/home-loan/assess \
  -H "Content-Type: application/json" \
  -d @sample_payloads/likely_eligible.json
```

</details>

## Run In Kubernetes

<details>
<summary>Kubernetes Collector and app deployment</summary>

The Kubernetes manifest sends OTLP to the Splunk OpenTelemetry Collector agent
on the node IP at port `4317`:

```yaml
OTEL_EXPORTER_OTLP_ENDPOINT=http://$(SPLUNK_OTEL_AGENT):4317
```

Install the Collector before deploying the app so traces and metrics can leave
the cluster.

Set the deployment variables:

```bash
export SPLUNK_ACCESS_TOKEN="<your-token>"
export SPLUNK_REALM="<your-realm>" # for example us1, us0, eu0, au0
export INSTANCE="home-loan-demo"
```

### Install The Collector

Add and update the Splunk OpenTelemetry Collector Helm repo:

```bash
helm repo add splunk-otel-collector-chart https://signalfx.github.io/splunk-otel-collector-chart
helm repo update
```

Install the Collector using the included values file:

```bash
helm upgrade --install splunk-otel-collector \
  splunk-otel-collector-chart/splunk-otel-collector \
  --namespace splunk-otel \
  --create-namespace \
  --set="splunkObservability.realm=$SPLUNK_REALM" \
  --set="splunkObservability.accessToken=$SPLUNK_ACCESS_TOKEN" \
  --set="clusterName=$INSTANCE-cluster" \
  --set="environment=agentic-ai-$INSTANCE" \
  -f k8s-otel/values.yaml
```

`k8s-otel/values.yaml` keeps histogram metrics in OTLP form for Splunk AI Agent
Monitoring.

If this cluster already has the Splunk OpenTelemetry Collector installed and
listening on node port `4317`, you can skip this install step.

Confirm the Collector is running:

```bash
kubectl get pods -n splunk-otel
```

### Build The Image

```bash
cd /agentic-ai-homeloan/app-with-quality-issue
docker build --platform linux/amd64 -t localhost:9999/agentic-ai-app:app-with-quality-issue .
docker push localhost:9999/agentic-ai-app:app-with-quality-issue
```

### Deploy The App

Create the namespace if needed:

```bash
kubectl create ns home-loan-agent
```

Create the Azure OpenAI secret in the `home-loan-agent` namespace:

```bash
{ [ -z "$AZURE_OPENAI_ENDPOINT" ] || [ -z "$AZURE_OPENAI_API_KEY" ]; } && \
  echo "Error: Missing variables" || \
  kubectl create secret generic azure-openai-api \
    -n home-loan-agent \
    --from-literal=azure-openai-api-endpoint="$AZURE_OPENAI_ENDPOINT" \
    --from-literal=azure-openai-api-key="$AZURE_OPENAI_API_KEY" \
    --dry-run=client -o yaml | kubectl apply -f -
```

Create the instance configmap:

```bash
kubectl create configmap instance-config \
  -n home-loan-agent \
  --from-literal=OTEL_RESOURCE_ATTRIBUTES=deployment.environment=agentic-ai-$INSTANCE \
  --dry-run=client -o yaml | kubectl apply -f -
```

Apply the manifest:

```bash
kubectl apply -f k8s.yaml
kubectl rollout restart deployment/home-loan-broker-langchain -n home-loan-agent
kubectl rollout status deployment/home-loan-broker-langchain -n home-loan-agent
```

The manifest deploys the Home Loan Broker app into the `home-loan-agent`
namespace, sets `OTEL_SERVICE_NAME` to `home-loan-broker`, and exposes the app
at:

```text
http://home-loan-broker.localhost/home-loan/assess
```

### Test In Kubernetes

```bash
kubectl get pods -n home-loan-agent
```

If the pod shows `CreateContainerConfigError`, inspect the event message:

```bash
kubectl describe pod <pod-name> -n home-loan-agent
```

The most common cause is a missing namespace-scoped resource such as
`secret "azure-openai-api"` or `configmap "instance-config"`.

Send sample requests:

```bash
curl http://home-loan-broker.localhost/home-loan/assess \
  -H "Content-Type: application/json" \
  -d @sample_payloads/likely_eligible.json

curl http://home-loan-broker.localhost/home-loan/assess \
  -H "Content-Type: application/json" \
  -d @sample_payloads/high_dti_serviceability_fail.json

curl http://home-loan-broker.localhost/home-loan/assess \
  -H "Content-Type: application/json" \
  -d @sample_payloads/high_lvr.json

curl http://home-loan-broker.localhost/home-loan/assess \
  -H "Content-Type: application/json" \
  -d @sample_payloads/aml_escalation.json

curl http://home-loan-broker.localhost/home-loan/assess \
  -H "Content-Type: application/json" \
  -d @sample_payloads/policy_drift.json
```

</details>

## Sample Payloads

- `sample_payloads/likely_eligible.json`: straight-through demo approval path.
- `sample_payloads/high_lvr.json`: high LVR decline path.
- `sample_payloads/high_dti_serviceability_fail.json`: serviceability referral path.
- `sample_payloads/aml_escalation.json`: AML referral path.
- `sample_payloads/policy_drift.json`: policy drift referral path.

### Quality Issue Scenario Payloads

These payloads include `quality_issue_scenario`. That field is used only to
inject a controlled LLM narrative defect into the selected trace span; it does
not change deterministic lending calculations, reason codes, or final outcomes.

- `sample_payloads/quality_hallucination_policy.json`: targets A4 `policy` with a statement that contradicts policy drift evidence.
- `sample_payloads/quality_bias_residency.json`: targets A5 `risk_compliance` with biased residency-status wording.
- `sample_payloads/quality_toxicity_applicant.json`: targets A1 `conversation_intake` with insulting applicant wording.
- `sample_payloads/quality_irrelevant_broker.json`: targets A0 `broker_orchestrator` with unrelated travel advice.
- `sample_payloads/quality_negative_sentiment.json`: targets A5 `risk_compliance` with pessimistic wording.

`run_assessments.sh` sends these quality payloads more often than the normal
workflow payloads so sampled semantic quality evaluations are more likely to
include Hallucination, Bias, Toxicity, Sentiment, and Relevance examples.

### Example Trace with Evaluation [Biased] : The applicant is careless and financially incompetent

![Example: The applicant is careless and financially incompetent](/app-with-quality-issue/sample_payloads/screenshots/quality-toxic.png)

![Example: Eval explanation](/app-with-quality-issue/sample_payloads/screenshots/quality-toxic-eval.png)

## Response Shape

The response includes:

- `session_id`
- `application_summary`
- `agent_path`
- `agent_steps`
- `agent_selection_reasons`
- `eligibility_result`
- `policy_result`
- `risk_compliance_result`
- `final_outcome`
- `final_decision`
- `audit_record`
- `workflow_events`

This is not a real lending decision engine. Outcomes are deterministic demo
recommendations only and are not credit advice or credit approval.

## Tests

```bash
HOME_LOAN_DETERMINISTIC_ONLY=true python -m unittest discover -s tests
```

## After Updates

```bash
docker build --platform linux/amd64 -t localhost:9999/agentic-ai-app:app-with-quality-issue .
docker push localhost:9999/agentic-ai-app:app-with-quality-issue

kubectl apply -f k8s.yaml
kubectl rollout restart deployment/home-loan-broker-langchain -n home-loan-agent
```
