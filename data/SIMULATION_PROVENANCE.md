# Observed versus simulated inputs

`data/microvan.csv` is the original survey. `microvan_v4/synthetic_cost_scenario.csv` contains scenario costs, out-of-fold probabilities, descriptive segments and observed survey labels; it does not record actual outreach. `microvan_v5/synthetic_randomized_trial.csv`, `synthetic_observational_study.csv`, and `synthetic_experiment_events.csv` contain generated records and outcomes used for methods validation. `microvan_v5/uplift_holdout_predictions.csv` derives from the simulated randomized trial. None of the generated records augment the 400 survey observations.

The original case prompt and unredacted student team submission are not distributed; the case is cited in the reports and the numerical R outputs are included in the technical appendix.
