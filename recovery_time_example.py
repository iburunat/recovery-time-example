"""
Worked example: estimating post-exercise recovery time from workout and person features.

ILLUSTRATIVE ONLY. The data are simulated (no real participants, no Garmin data or methods).
The point is the workflow a data scientist would follow: prepare data, build an interpretable
baseline, validate it on unseen participants, report accuracy with uncertainty, check subgroups,
and consider the cost of running it on a small device.
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

rng = np.random.default_rng(42)

# ------------------------------------------------------------------ 1. simulate data
def simulate(n_participants=150):
    rows = []
    for pid in range(n_participants):
        age = rng.uniform(20, 65)
        vo2 = np.clip(rng.normal(48 - 0.25 * (age - 30), 7), 25, 70)
        u = rng.normal(0, 1.5)                       # person-specific offset (hours)
        for _ in range(rng.integers(6, 11)):         # 6-10 sessions per person
            dur = rng.uniform(20, 120)               # minutes
            inten = rng.uniform(0.55, 0.90)          # mean fraction of max heart rate
            prior7 = rng.lognormal(mean=3.0, sigma=0.4)  # training load, previous 7 days
            sleep = np.clip(rng.normal(7.0, 1.0), 4, 10)
            acute = dur / 60 * (inten / 0.7) ** 3    # saturating dose of the session
            true = (4 + 9 * (1 - np.exp(-acute / 2.0))
                    + 0.08 * (age - 35) - 0.12 * (vo2 - 45)
                    + 0.6 * (np.log(prior7) - 3.0) - 0.8 * (sleep - 7.0) + u)
            obs = max(2.0, true + rng.normal(0, 1.2 * (1 + 0.3 * acute / 3)))  # noisy measurement
            rows.append(dict(pid=pid, age=age, vo2max=vo2, duration_min=dur, intensity=inten,
                             prior7_load=prior7, sleep_h=sleep, recovery_h=obs))
    return pd.DataFrame(rows)

df = simulate()
# ------------------------------------------------------------------ 2. features
df["acute_load"] = df.duration_min / 60 * (df.intensity / 0.7) ** 3
df["log_acute"] = np.log1p(df.acute_load)
df["log_prior7"] = np.log(df.prior7_load)
RIDGE_FEATS = ["log_acute", "age", "vo2max", "log_prior7", "sleep_h"]
RAW_FEATS = ["duration_min", "intensity", "age", "vo2max", "prior7_load", "sleep_h"]
y = df.recovery_h.values
groups = df.pid.values

models = {
    "Baseline (duration x intensity)": (lambda: LinearRegression(), ["duration_min"], True),
    "Ridge (5 interpretable features)": (lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)), RIDGE_FEATS, False),
    "Gradient boosting (6 raw features)": (lambda: HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, random_state=0), RAW_FEATS, False),
}

# ------------------------------------------------------------------ 3. validation on unseen people
gkf = GroupKFold(n_splits=5)
preds = {k: np.zeros(len(df)) for k in models}
intervals = {"lo": np.zeros(len(df)), "hi": np.zeros(len(df))}
for tr, te in gkf.split(df, y, groups):
    for name, (make, feats, baseline) in models.items():
        Xtr = df.iloc[tr][feats].copy(); Xte = df.iloc[te][feats].copy()
        if baseline:
            Xtr["x"] = df.iloc[tr].duration_min * df.iloc[tr].intensity
            Xte["x"] = df.iloc[te].duration_min * df.iloc[te].intensity
            Xtr, Xte = Xtr[["x"]], Xte[["x"]]
        m = make().fit(Xtr, y[tr])
        preds[name][te] = m.predict(Xte)
    # split-conformal 90% interval for the Ridge model, calibrated on held-out *participants* of the training set
    tr_pids = np.unique(groups[tr]); rng2 = np.random.default_rng(1); rng2.shuffle(tr_pids)
    cal_p = set(tr_pids[: int(0.2 * len(tr_pids))])
    cal = np.array([g in cal_p for g in groups[tr]])
    fit_idx, cal_idx = tr[~cal], tr[cal]
    mk = models["Ridge (5 interpretable features)"][0]()
    mk.fit(df.iloc[fit_idx][RIDGE_FEATS], y[fit_idx])
    q = np.quantile(np.abs(y[cal_idx] - mk.predict(df.iloc[cal_idx][RIDGE_FEATS])), 0.90)
    p = mk.predict(df.iloc[te][RIDGE_FEATS])
    intervals["lo"][te], intervals["hi"][te] = p - q, p + q

def boot_ci(err, groups, n=1000):
    """Bootstrap over participants, so the interval reflects between-person variation."""
    pids = np.unique(groups); out = []
    for _ in range(n):
        s = rng.choice(pids, len(pids), replace=True)
        out.append(np.mean(np.concatenate([err[groups == p] for p in s])))
    return np.percentile(out, [2.5, 97.5])

res = []
for name in models:
    e = np.abs(y - preds[name])
    lo, hi = boot_ci(e, groups)
    res.append(dict(model=name, MAE_h=e.mean(), MAE_lo=lo, MAE_hi=hi,
                    RMSE_h=np.sqrt(mean_squared_error(y, preds[name])), R2=r2_score(y, preds[name])))
res = pd.DataFrame(res)
print(res.round(2).to_string(index=False))

cover = np.mean((y >= intervals["lo"]) & (y <= intervals["hi"]))
width = np.mean(intervals["hi"] - intervals["lo"])
print(f"\nRidge 90% interval: coverage on unseen participants = {cover:.2f}, mean width = {width:.1f} h")

# ------------------------------------------------------------------ 4. subgroup check (Ridge)
df["pred"] = preds["Ridge (5 interpretable features)"]
df["err"] = df.pred - df.recovery_h
df["age_band"] = pd.cut(df.age, [0, 35, 50, 100], labels=["<35", "35-50", ">50"])
sub = df.groupby("age_band", observed=True).agg(n=("err", "size"), bias_h=("err", "mean"),
                                                  MAE_h=("err", lambda s: s.abs().mean()))
print("\nRidge error by age band (bias = mean predicted minus observed):\n", sub.round(2).to_string())

# ------------------------------------------------------------------ 5. cost on a small device
full = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(df[RIDGE_FEATS], y)
gbm = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, random_state=0).fit(df[RAW_FEATS], y)
t0 = time.perf_counter(); [full.predict(df[RIDGE_FEATS].iloc[[i]]) for i in range(300)]; t_r = (time.perf_counter() - t0) / 300
t0 = time.perf_counter(); [gbm.predict(df[RAW_FEATS].iloc[[i]]) for i in range(300)]; t_g = (time.perf_counter() - t0) / 300
print(f"\nParameters: Ridge = {len(RIDGE_FEATS) + 1}; boosting = about {gbm.n_iter_ * 31} tree leaves")
print(f"Prediction time per call (Python, desktop; relative only): Ridge {t_r*1e3:.2f} ms, boosting {t_g*1e3:.2f} ms")
print("Ridge coefficients (standardised, hours per 1 SD):",
      dict(zip(RIDGE_FEATS, full[-1].coef_.round(2))))

# ------------------------------------------------------------------ 6. figure
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ax[0].scatter(y, df.pred, s=6, alpha=.35, color="#1F3A5F")
ax[0].plot([2, 22], [2, 22], "k--", lw=1); ax[0].set(xlabel="Observed recovery (h)", ylabel="Predicted (h), unseen participants", title="Ridge model: predicted vs observed")
x = np.arange(len(res)); lab = ["Baseline", "Ridge", "Boosting"]
ax[1].bar(x, res.MAE_h, yerr=[res.MAE_h - res.MAE_lo, res.MAE_hi - res.MAE_h], color=["#9AA7B8", "#1F3A5F", "#5B7DA8"], capsize=4)
ax[1].set_xticks(x, lab); ax[1].set(ylabel="Mean absolute error (h)", title="Accuracy with 95% bootstrap CI")
ax[2].bar(sub.index.astype(str), sub.MAE_h, color="#1F3A5F"); ax[2].set(xlabel="Age band", ylabel="MAE (h)", title="Ridge error by age band")
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("recovery_time_example.png", dpi=150)
