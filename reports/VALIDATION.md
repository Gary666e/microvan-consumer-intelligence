# Release validation

- Original survey file: Git blob SHA `edd8bf9721cdfd88b4e72e1940223239a639266b`, identical to the online repository at the time of packaging.
- Core statistical, advanced modeling, and causal scripts checked against original source using Git blob hashes after path/NumPy compatibility adjustments.
- Local archived model smoke test: all-five input liking 5.1723228089; MILP scenario budget $1,500, capacity 100, floor 10 returned 100 contacts, $1,500 cost, expected 52.7820 high-interest, 57 observed survey high-interest labels, gap 0.
- Input validation: one valid row and one 30-cell invalid row correctly detected.
- Offline monitoring: baseline PASS; mix-shift stress REVIEW, with score drift, recall degradation, and capacity alerts.
- All packaged Python scripts parsed successfully. Complete training was performed in earlier research runs, not repeated for every script during this packaging step. Streamlit UI was not run in the current environment.
- Cover plus writing sample: 9 PDF pages.
