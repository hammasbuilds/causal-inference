# causal-inference

What it costs to get the causal structure wrong, measured on data where the true effect
is planted and therefore known exactly.

| project | what is measured | result |
|---|---|---|
| [01_adjustment_choice](projects/01_adjustment_choice/) | bias from adjusting for a confounder, collider, mediator or instrument | the same adjustment removes a 25% bias in one structure and creates a 75% bias in another; R² prefers adjusting in all four and is right in one |

## Method

Data comes from linear structural equation models with chosen coefficients, so "bias" is
the estimate minus a number that was set, not a comparison against another estimator.
Every claim in a project README is checked against that project's `results.json` before
it ships.

No neural networks, no GPU. NumPy and SciPy.
