#!/bin/bash

PAYLOADS=(
  "quality_hallucination_policy"
  "likely_eligible"
  "quality_bias_residency"
  "high_dti_serviceability_fail"
  "quality_toxicity_applicant"
  "quality_irrelevant_broker"
  "high_lvr"
  "quality_negative_sentiment"
  "aml_escalation"
  "policy_drift"
  "quality_hallucination_policy"
  "quality_bias_residency"
  "quality_toxicity_applicant"
  "quality_irrelevant_broker"
  "quality_negative_sentiment"
)

END_TIME=$(($(date +%s) + 9 * 3600))

while [ $(date +%s) -lt $END_TIME ]; do
  for payload in "${PAYLOADS[@]}"; do
    # Check time before each request
    if [ $(date +%s) -ge $END_TIME ]; then
      echo "[$(date -u)] 9 hours reached, stopping."
      exit 0
    fi

    echo "[$(date -u)] Running: $payload"
    curl -s http://home-loan-broker.localhost/home-loan/assess \
      -H "Content-Type: application/json" \
      -d @sample_payloads/${payload}.json
    echo ""
    sleep $((RANDOM % 61 + 120))
  done
done

echo "[$(date -u)] 9 hours reached, stopping."
