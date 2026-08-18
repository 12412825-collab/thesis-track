# Phase 1 Smoke Test Report

Date: 2026-08-18  
Status: **PASS**  
Run ID: `PHASE1-FAMILY-CAPACITY-smoke-fef9a5cb6b20`

## Scope

The smoke run exercised all 15 frozen configurations, seed 0, `rho_train={0,0.9}`, reduced train/validation/test samples of 700/350/500, and the frozen IID/OOD fitting boundary. The wrong-line smoke scanned all three stable-boundary geometries, neutral and strong shortcut regimes, and every capacity tier using a noise-free probe.

## Result contract

- Expected/actual fits: 30 / 30
- Expected/actual master rows: 75 / 75
- Unique fit and result keys: yes
- Maximum OOD-gap reconstruction error: `0.000e+00`
- Fit failure fraction: `0.0000`
- Non-finite or out-of-bound primary metrics: none detected
- Capacity tier ordering by family: `{'gradient_boosting': True, 'logistic_regression': True, 'mlp': True, 'random_forest': True, 'rbf_svm': True}`

## Generator gates

- Maximum absolute requested-versus-realized shortcut correlation error: `0.0289`
- Positive-rate range: `0.4820` to `0.5371`
- Maximum absolute noise-target correlation: `0.0592`

## Capacity proxy smoke summary

```text
             family capacity_tier       mean        min        max
logistic_regression           low     3.3325     3.0035     3.6616
logistic_regression        medium     4.5446     4.3046     4.7847
logistic_regression          high     4.9117     4.8351     4.9884
            rbf_svm           low     2.3270     2.3164     2.3377
            rbf_svm        medium    10.6146    10.0994    11.1299
            rbf_svm          high   152.6230   131.7412   173.5048
      random_forest           low  2554.5000  2196.0000  2913.0000
      random_forest        medium  4501.0000  2625.0000  6377.0000
      random_forest          high 11347.0000  4750.0000 17944.0000
  gradient_boosting           low   645.0000   590.0000   700.0000
  gradient_boosting        medium  3208.5000  1585.0000  4832.0000
  gradient_boosting          high  4504.0000  2171.0000  6837.0000
                mlp           low   113.0000   113.0000   113.0000
                mlp        medium  4609.0000  4609.0000  4609.0000
                mlp          high 25601.0000 25601.0000 25601.0000
```

## Noise-free wrong-line gate

High-capacity mean boundary fidelity at neutral rho (gate: at least 0.95):

```text
             family  boundary_fidelity
  gradient_boosting             0.9658
logistic_regression             0.9961
                mlp             0.9697
      random_forest             0.9619
            rbf_svm             0.9658
```

Low and medium capacity values remain in `results/phase1/smoke/summaries/wrong_line_guard.csv` as diagnostics and are not stop gates.

## Runtime and warnings

- Grid runtime: `18.73` seconds
- Mean/max fit time: `0.412` / `2.326` seconds
- Captured fit warnings: `1`
- Master results SHA-256: `674783d428965f1a3c97a8592aa49f48fbfa4d9be01a4acddf4d0a09ba5314b8`

## Stop-gate failures

None.

## Authorization

All preregistered smoke gates passed. The complete Phase 1 grid is authorized.
