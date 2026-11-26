"""Quick-start analysis on original case survey; historical detailed outputs are in results/.
This is a compact portfolio implementation, not the archived full research scripts.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score, silhouette_score
from scipy.optimize import milp, Bounds, LinearConstraint

ROOT = Path(__file__).resolve().parents[1]

def load_survey():
    d = pd.read_csv(ROOT / "data/microvan.csv")
    x = d.iloc[:, 2:32]
    assert len(d) == 400 and x.shape == (400, 30)
    assert d.subjnumb.is_unique and not d.isna().any().any()
    assert d.iloc[:, 1:32].apply(lambda s: s.between(1, 9).all()).all()
    return d, x, d.mvliking.to_numpy()

def factor_transform():
    return make_pipeline(StandardScaler(), FactorAnalysis(n_components=5, rotation="varimax", svd_method="lapack", random_state=42))

def benchmark(x, y):
    """Outer held-out predictions; tuning and factor extraction are fitted on training folds."""
    specs = {
        "Mean": None,
        "OLS": (LinearRegression(), {}),
        "Factor OLS": (make_pipeline(StandardScaler(), FactorAnalysis(n_components=5, rotation="varimax", svd_method="lapack", random_state=42), LinearRegression()), {}),
        "Ridge": (make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": [1,10,30,100]})
    }
    preds = {name: np.empty(len(y)) for name in specs}
    for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=42).split(x), 1):
        for name, spec in specs.items():
            if spec is None:
                preds[name][te] = y[tr].mean()
            else:
                m = GridSearchCV(clone(spec[0]), spec[1], cv=KFold(4, shuffle=True, random_state=100+fold), scoring="neg_root_mean_squared_error", n_jobs=1)
                m.fit(x.iloc[tr], y[tr])
                preds[name][te] = m.predict(x.iloc[te])
    rows = []
    positive = y >= 7
    for name, p in preds.items():
        top = np.argsort(-p, kind="stable")[:100]
        hits = int(positive[top].sum())
        rows.append({"model": name, "RMSE": mean_squared_error(y, p)**.5,
                     "AUC": roc_auc_score(positive, p), "AP": average_precision_score(positive,p),
                     "precision_at_100": hits/100, "recall_at_100": hits/positive.sum(),
                     "lift_at_100": (hits/100)/positive.mean()})
    return pd.DataFrame(rows).sort_values("RMSE"), pd.DataFrame(preds)

def segment(x):
    """Descriptive, full-sample segmentation; do not treat as prospective segment validation."""
    f = factor_transform().fit_transform(x)
    k = KMeans(n_clusters=3, n_init=30, random_state=42)
    labels = k.fit_predict(f)
    return labels, silhouette_score(f, labels)

def allocate(probabilities, costs, budget, capacity, groups=None, floor=0):
    """Scenario optimization only; no observed outreach costs or causal responses."""
    p = np.asarray(probabilities, float)
    c = np.asarray(costs, float)
    if p.shape != c.shape or p.ndim != 1 or not np.isfinite(p).all() or not np.isfinite(c).all() or (c<0).any():
        raise ValueError("Invalid probability or cost array")
    if ((p<0)|(p>1)).any() or budget<0 or capacity<0 or floor<0:
        raise ValueError("Invalid bounds")
    a = [c, np.ones(len(p))]
    lo, hi = [-np.inf, -np.inf], [budget, capacity]
    if groups is not None:
        g = np.asarray(groups)
        if len(g) != len(p):
            raise ValueError("Group count mismatch")
        for label in np.unique(g):
            a.append((g==label).astype(float))
            lo.append(floor)
            hi.append(np.inf)
    result = milp(-p, integrality=np.ones(len(p)), bounds=Bounds(0,1),
                  constraints=LinearConstraint(np.vstack(a), lo, hi),
                  options={"mip_rel_gap": .0001, "time_limit": 15})
    if result.x is None:
        return {"status": "infeasible or no incumbent", "message":result.message}
    selected = result.x > .5
    assert c[selected].sum() <= budget+1e-6 and selected.sum() <= capacity
    if groups is not None:
        assert all((selected & (g==label)).sum()>=floor for label in np.unique(g))
    return {"status": "optimal" if result.status==0 else "incumbent", "contacts":int(selected.sum()),
            "cost":float(c[selected].sum()), "expected_interest":float(p[selected].sum()),
            "gap":float(result.mip_gap)}

if __name__ == "__main__":
    d,x,y = load_survey()
    scores, preds = benchmark(x,y)
    labels, sil = segment(x)
    print(scores.to_string(index=False))
    print("Descriptive silhouette:", sil)
    print(d.assign(segment=labels).groupby("segment").mvliking.agg(["count","mean"]))
