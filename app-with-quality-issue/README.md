# Home Loan Broker Agentic AI Demo

This app is the Home Loan Broker version of the Splunk Agentic AI workshop Travel Agent demo. It keeps the same Flask, LangGraph, LangChain, and OpenTelemetry execution pattern, but re-themes the workflow from travel planning to a bounded home-loan assessment.

## Travel To Home Loan Mapping

- Travel coordinator -> `A0_BROKER_ORCHESTRATOR`
- Flight, hotel, and activity specialists -> intake, KYC/AML, eligibility, policy, and risk/compliance specialists
- Travel plan synthesizer -> `A6_DECISION_AUDIT`
- `plan_travel_internal(...)` -> `assess_home_loan_internal(...)`
- `/travel/plan` -> `/home-loan/assess`
- `final_itinerary` -> `final_decision`

`/travel/plan` is still available as a deprecated alias for older workshop scripts. It runs the Home Loan Broker workflow and returns a deprecation header.

## Run Locally

```bash
opentelemetry-instrument python main.py
```

The server listens on port `8080`.

```bash
curl http://localhost:8080/home-loan/assess \
  -H "Content-Type: application/json" \
  -d @sample_payloads/likely_eligible.json
```

## Run In Kubernetes

### Build An Updated Docker Image

Build an updated Docker image with the quality-issue tag:

```bash
cd /agentic-ai-homeloan/app-with-quality-issue
docker build --platform linux/amd64 -t localhost:9999/agentic-ai-app:app-with-quality-issue .
docker push localhost:9999/agentic-ai-app:app-with-quality-issue
```

### Update The Kubernetes Manifest

Open `k8s.yaml` and confirm the container image points at the updated image:

```yaml
          image: localhost:9999/agentic-ai-app:app-with-quality-issue
```

The manifest deploys the Home Loan Broker app into the `home-loan-agent` namespace and sets `OTEL_SERVICE_NAME` to `home-loan-broker`.

### Deploy The Updated Application

Apply the manifest:

```bash
kubectl apply -f /agentic-ai-homeloan/app-with-quality-issue/k8s.yaml
```

### Test The Application In Kubernetes

Ensure the new application pod has started successfully:

```bash
kubectl get pods -n home-loan-agent
```

Send a test assessment request through the primary Home Loan Broker ingress host:

```bash
curl http://home-loan-broker.localhost/home-loan/assess \
  -H "Content-Type: application/json" \
  -d @sample_payloads/likely_eligible.json
```

The old workshop host and route are still available as a deprecated compatibility alias:

```bash
curl http://travel-planner.localhost/travel/plan \
  -H "Content-Type: application/json" \
  -d @sample_payloads/likely_eligible.json
```

## Sample Payloads

- `sample_payloads/likely_eligible.json`
- `sample_payloads/high_lvr.json`
- `sample_payloads/high_dti_serviceability_fail.json`
- `sample_payloads/aml_escalation.json`
- `sample_payloads/policy_drift.json`

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

## Observability And Safety

The app preserves the workshop OpenTelemetry/Splunk instrumentation settings, including GenAI content capture in `k8s.yaml`. LLM calls receive redacted/summarized inputs where practical, and the final JSON excludes raw prompts, full model outputs, and applicant identifiers.

This is not a real lending decision engine. Outcomes are deterministic demo recommendations only: `APPROVE_IN_PRINCIPLE`, `REFER`, `DECLINE`, or `NEED_MORE_INFO`.

This `app-with-quality-issue` variant keeps a simulated policy-narrative quality issue through `PoisonedChatWrapper` so it remains visible in traces. The injected text is not used for deterministic eligibility, policy, risk, or final decision data.

## Tests

```bash
HOME_LOAN_DETERMINISTIC_ONLY=true python -m unittest discover -s tests
```
