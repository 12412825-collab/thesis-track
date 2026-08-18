# EXP-001 reproducibility check

Checked on 2026-08-18 using the frozen `config.yaml` and the repository-local virtual environment.

- Complete five-seed run: succeeded.
- Accuracy: exactly equal for all 350 archived versus reproduced rows.
- Balanced accuracy: exactly equal for all 350 rows.
- Log loss: numerically equal after CSV parsing; four Random Forest stable-only rows differed only in the final printed decimal place (at most approximately `6e-17`).
- Aggregate research summary and reported model rankings: unchanged.
- Sanity-check warnings: none.

The baseline is reproducible at scientific numerical precision. The raw CSV is not promised to be byte-identical because parallel floating-point probability calculations can vary at the final serialization digit. Frozen EXP-001 outputs have not been edited.

