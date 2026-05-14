#!/bin/bash
set -u

BASE_URL="${BASE_URL:-http://home-loan-broker.localhost/home-loan/assess}"
DURATION_HOURS="${DURATION_HOURS:-9}"
SLEEP_MIN_SECONDS="${SLEEP_MIN_SECONDS:-120}"
SLEEP_JITTER_SECONDS="${SLEEP_JITTER_SECONDS:-60}"

PAYLOADS=(
  "quality_hallucination_policy"
  "likely_eligible"
  "security_pii_identity_leak"
  "quality_bias_residency"
  "high_dti_serviceability_fail"
  "quality_toxicity_applicant"
  "quality_irrelevant_broker"
  "security_pci_fee_card_leak"
  "high_lvr"
  "quality_negative_sentiment"
  "aml_escalation"
  "policy_drift"
  "quality_hallucination_policy"
  "security_pii_identity_leak"
  "quality_bias_residency"
  "quality_toxicity_applicant"
  "security_pci_fee_card_leak"
  "quality_irrelevant_broker"
  "quality_negative_sentiment"
)

END_TIME=$(($(date +%s) + DURATION_HOURS * 3600))

echo "Sending Home Loan assessments to: ${BASE_URL}"
echo "Duration: ${DURATION_HOURS} hour(s)"

while [ $(date +%s) -lt $END_TIME ]; do
  for payload in "${PAYLOADS[@]}"; do
    # Check time before each request
    if [ $(date +%s) -ge $END_TIME ]; then
      echo "[$(date -u)] ${DURATION_HOURS} hour(s) reached, stopping."
      exit 0
    fi

    echo "[$(date -u)] Running: $payload"
    curl -s "$BASE_URL" \
      -H "Content-Type: application/json" \
      -d @sample_payloads/${payload}.json
    echo ""
    sleep $((RANDOM % (SLEEP_JITTER_SECONDS + 1) + SLEEP_MIN_SECONDS))
  done
done

echo "[$(date -u)] ${DURATION_HOURS} hour(s) reached, stopping."
