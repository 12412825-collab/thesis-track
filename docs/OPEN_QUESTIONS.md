# Open Questions

## Central unresolved question

If model-family identity explains little robustness variation after controlling task structure, what learned mechanism **M** actually explains why models trained on the same shifted data generalize differently?

The current repository does not answer this. The next study must identify an intervention and observable mediator that distinguish competing explanations rather than add another family comparison.

## Scientific uncertainties

- Which learned quantity changes when shortcut strength changes: representation, feature reliance, decision boundary geometry, optimization path, or another mechanism?
- Which measurements can distinguish learned mechanism M from capacity, regularization, optimization, and search-budget explanations?
- Which interventions can be applied without allowing OOD outcomes to influence fitting, tuning, or selection?
- How much of the observed behavior survives changes to the synthetic generator, and what is the narrowest justified scope of any claim?
- Can a preregistered behavior metric predict OOD failure more directly than the stopped Fidelity proxy?

## Restart rule

When restarting, **do not first add dataset, model family, or shift type**. Use the following order:

```text
Task Structure
→ Mechanism M
→ Model Behavior
→ Robustness / Selection
```

The first new experiment should be a small, controlled mechanism intervention with explicit competing explanations, fixed search budget, train-only mechanism measurement, independent evaluation, and a preregistered stop rule.

