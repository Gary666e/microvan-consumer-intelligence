# Microvan Consumer Intelligence & Decision Optimization

**Business-facing Data Science Portfolio | Kangjie Lin**

A decision-oriented analysis of consumer preferences for a premium compact family vehicle. The project connects statistical learning and customer segmentation with targeting, experimental design, causal methods, constrained resource allocation and operational monitoring.

## Business question

Should an automotive company advance the microvan concept to controlled product validation, which consumers should it prioritize, and how can it allocate limited research and outreach resources?

**Recommendation:** prioritize luxury-oriented consumers with practical sizing needs, validate product positioning through controlled experiments, and gate further investment on independently measured commercial outcomes.

## Key findings

The following results use the original 400-respondent case survey and internal out-of-fold predictions.

| Finding | Result | Decision implication |
| --- | ---: | --- |
| Predictive RMSE | 2.236 vs. 2.701 mean baseline | 17.2% lower error with a five-factor regression |
| Model dimension | 30 attributes → 5 factors | Favor interpretable, parsimonious models |
| Targeting precision at 25% capacity | 62% vs. 30% prevalence | 2.07× offline targeting lift |
| Missed high-interest respondents | 58; 45 outside the core segment | Diagnose coverage gaps rather than optimize precision alone |
| Coverage policy | 10 minimum selections per segment; 62 hits retained | Explore broader segment coverage on independent data |
| Short survey | 10 vs. 30 items; RMSE 2.272 vs. 2.264 tuned Ridge | Validate a shorter fixed questionnaire before rollout |

## Analytical approach

1. Data audit, exploratory analysis and factor extraction.
2. Regression and classification with nested cross-validation; model comparison, calibration and uncertainty.
3. Hierarchical clustering, K-means, mixture models and stability testing.
4. Product metrics, targeting lift, segment-level error diagnosis and policy evaluation.
5. Experimental design and **simulation-based** A/B analysis, CUPED, propensity weighting, doubly robust estimation and uplift modeling.
6. Mixed-integer optimization under scenario costs and budget constraints.
7. Execution roadmap, offline monitoring, corrective actions and evaluation gates.

## Selected results

- [Predictive model benchmark](results/model_benchmark.csv)
- [Segment-level diagnosis](results/root_cause.csv)
- [Coverage-policy evaluation](results/coverage_policy.csv)
- [Causal estimators (simulation)](results/simulated_causal_estimators.csv)
- [Monitoring stress test](results/monitoring_stress.csv)
- [Offline monitoring example](src/monitor.py)

## Evidence and limitations

The empirical dataset is a 400-respondent sample supplied with the academic case *Grosse Pointe Associates and the “Microvan”* (Modern Marketing Research: Concepts, Methods, and Cases). The original academic analysis was a team case; subsequent Python modeling, experimental simulations and reporting are AI-assisted extensions. Concept liking is not purchase behavior. Generated experimental records and hypothetical outreach costs are separate from the original survey; they demonstrate analytical methods rather than realized sales, causal marketing effects or actual ROI. The decision system is a local prototype, not a production deployment. External validation and live experimentation remain future work.

## Repository status

This initial portfolio snapshot includes the project overview, selected analysis outputs and a monitoring example. The full analysis package, figures, application and formatted reports are maintained separately pending upload. No license is granted to third-party case materials.
