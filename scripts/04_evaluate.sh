#!/usr/bin/env bash
# Stage 4 — Statistical tests and evaluation reports
#
# Runs Holm-Bonferroni corrected paired t-tests on fold metrics,
# generates LaTeX tables, and produces summary CSVs.

set -euo pipefail

echo "=== CultureVLM: Evaluation & Statistical Tests ==="

METRICS="f1_macro f1_weighted mae qwk"

for METRIC in $METRICS; do
  echo "  Running Holm-Bonferroni for metric: $METRIC"
  uv run python src/evaluation/statistical_tests.py \
    --fold-metrics outputs/training/fold_metrics.csv \
    --metric "$METRIC" \
    --output "outputs/training/stats/holm_ttests_${METRIC}"
done

echo ""
echo "=== Evaluation complete ==="
echo "Stats in: outputs/training/stats/"
