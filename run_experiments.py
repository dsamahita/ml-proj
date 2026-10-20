# run_experiments.py
# CSV location: ./data/data.csv
# Auto-repairs the corrupted (single-quoted) header in this dataset.

import os
import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (recall_score, confusion_matrix,
                             ConfusionMatrixDisplay, make_scorer)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42
DATA_PATH    = os.path.join("data", "data.csv")
TARGET       = "Target"
PROTECTED    = ["Scholarship holder", "Gender"]


# ---------------------------------------------------------------
# Header repair: this file has line 0 = '"col1;col2;...;col37"'
# ---------------------------------------------------------------
def parse_raw(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()

    header = lines[0]
    body   = lines[1:]

    if header.startswith('"') and header.endswith('"'):
        header = header[1:-1]
    header = header.replace('""', '"').replace("\t", "")

    if ";" in header:
        cols = [c.strip().strip('"').strip("'").strip()
                for c in header.split(";")]
        sep = ";"
    else:
        cols = [c.strip().strip('"').strip("'").strip()
                for c in header.split(",")]
        sep = ","

    df = pd.read_csv(io.StringIO("\n".join(body)), sep=sep,
                     header=None, skipinitialspace=True)
    df.columns = cols
    df.columns = [
        str(c).replace("\ufeff", "").replace("\t", "")
              .strip().strip('"').strip("'").strip()
        for c in df.columns
    ]

    # locate target if renamed
    if TARGET not in df.columns:
        cands = [c for c in df.columns if c.strip().lower() == "target"]
        if not cands:
            raise KeyError(f"No target column. Got:\n{list(df.columns)}")
        df = df.rename(columns={cands[0]: TARGET})
        print(f"[info] Renamed '{cands[0]}' -> '{TARGET}'")

    return df, sep, len(cols)


def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. cwd={os.getcwd()}"
        )
    df, sep, n_cols = parse_raw(path)
    drop_cols = [c for c in df.columns if "2nd sem" in c.lower()]
    df = df.drop(columns=drop_cols)
    print(f"[info] Header parsed with {n_cols} columns, sep='{sep}'")
    print(f"[info] Shape: {df.shape[0]} rows x {df.shape[1]} cols "
          f"(after dropping {len(drop_cols)} leakage columns)")
    return df


def split_xy(df):
    return df.drop(columns=[TARGET]), df[TARGET]


# ---------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------
def build_preprocessor(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


# ---------------------------------------------------------------
# Models
# ---------------------------------------------------------------
def get_models():
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                               solver="lbfgs"),
            {"clf__C": [0.1, 1, 10],
             "clf__class_weight": [None, "balanced"]}
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"clf__max_depth": [4, 8, 12],
             "clf__min_samples_leaf": [5, 20, 50],
             "clf__class_weight": [None, "balanced"]}
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {"clf__n_estimators": [100],
             "clf__max_depth": [None, 8],
             "clf__class_weight": [None, "balanced"]}
        ),
    }


# ---------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------
def tpr_for_class(y_true, y_pred, cls):
    tp = np.sum((y_true == cls) & (y_pred == cls))
    fn = np.sum((y_true == cls) & (y_pred != cls))
    return tp / (tp + fn) if (tp + fn) > 0 else float("nan")


def positive_rate(y_pred, cls):
    return float(np.mean(y_pred == cls))


def fairness_audit(y_true, y_pred, protected, cls="Dropout"):
    y_true    = np.asarray(y_true)
    y_pred    = np.asarray(y_pred)
    protected = np.asarray(protected)
    groups = sorted(pd.unique(protected))
    if len(groups) != 2:
        return None
    g0, g1 = groups[0], groups[1]
    m0, m1 = protected == g0, protected == g1
    tpr0 = tpr_for_class(y_true[m0], y_pred[m0], cls)
    tpr1 = tpr_for_class(y_true[m1], y_pred[m1], cls)
    pp0  = positive_rate(y_pred[m0], cls)
    pp1  = positive_rate(y_pred[m1], cls)
    return {"group0": g0, "group1": g1,
            "TPR_g0": tpr0, "TPR_g1": tpr1,
            "EOD": tpr1 - tpr0, "DPD": pp1 - pp0}


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    df = load_data()
    X, y = split_xy(df)

    # ---------- TABLE 2 ----------
    print("\n===== TABLE 2: CLASS AND GROUP DISTRIBUTION =====")
    print("\nTarget counts:")
    print(df[TARGET].value_counts())
    print("\nTarget share (%):")
    print((df[TARGET].value_counts(normalize=True) * 100).round(1))
    for p in PROTECTED:
        if p in df.columns:
            print(f"\n{p} counts:")
            print(df[p].value_counts())
            print(f"{p} share (%):")
            print((df[p].value_counts(normalize=True) * 100).round(1))

    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pre = build_preprocessor(X)

    scoring = {
        "accuracy":        "accuracy",
        "macro_f1":        "f1_macro",
        "recall_dropout":  make_scorer(recall_score, labels=["Dropout"],
                                       average="macro", zero_division=0),
        "recall_enrolled": make_scorer(recall_score, labels=["Enrolled"],
                                       average="macro", zero_division=0),
        "recall_graduate": make_scorer(recall_score, labels=["Graduate"],
                                       average="macro", zero_division=0),
    }

    best_estimators, cv_macro, perf_rows = {}, {}, []

    # ---------- TRAIN + TUNE ----------
    for name, (clf, grid) in get_models().items():
        print(f"\n===== TRAINING: {name} =====")
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro",
                          n_jobs=-1, verbose=0)
        gs.fit(X, y)
        best_estimators[name] = gs.best_estimator_
        cv_macro[name]        = gs.best_score_
        print(f"Best params: {gs.best_params_}")
        print(f"Best CV Macro F1: {gs.best_score_:.4f}")

        r = cross_validate(gs.best_estimator_, X, y, cv=cv,
                           scoring=scoring, n_jobs=-1)
        perf_rows.append({
            "Model":           name,
            "Accuracy":        f"{r['test_accuracy'].mean():.3f} "
                               f"± {r['test_accuracy'].std():.3f}",
            "Macro_F1":        f"{r['test_macro_f1'].mean():.3f}",
            "Recall_Dropout":  f"{r['test_recall_dropout'].mean():.3f}",
            "Recall_Enrolled": f"{r['test_recall_enrolled'].mean():.3f}",
            "Recall_Graduate": f"{r['test_recall_graduate'].mean():.3f}",
        })

    # ---------- TABLE 4 ----------
    print("\n\n===== TABLE 4: THREE-CLASS PERFORMANCE (5-fold CV) =====")
    perf_df = pd.DataFrame(perf_rows)
    print(perf_df.to_string(index=False))

    best_name = max(cv_macro, key=cv_macro.get)
    print(f"\n*** BEST MODEL: {best_name} "
          f"(CV Macro F1 = {cv_macro[best_name]:.4f}) ***")

    # ---------- TABLE 5: FAIRNESS ----------
    print("\n\n===== TABLE 5: FAIRNESS AUDIT (Dropout class) =====")
    fair_rows = []
    for p in PROTECTED:
        if p not in df.columns:
            print(f"[skip] {p} missing")
            continue
        yt, yp_, pr = [], [], []
        for tr, te in cv.split(X, y):
            m = best_estimators[best_name].fit(X.iloc[tr], y.iloc[tr])
            pred = m.predict(X.iloc[te])
            yt.extend(y.iloc[te].values)
            yp_.extend(pred)
            pr.extend(df[p].iloc[te].values)
        res = fairness_audit(yt, yp_, pr, "Dropout")
        if res:
            fair_rows.append({
                "Model":     best_name,
                "Protected": p,
                f"TPR (g={res['group1']})": f"{res['TPR_g1']:.3f}",
                f"TPR (g={res['group0']})": f"{res['TPR_g0']:.3f}",
                "EOD":       f"{res['EOD']:+.3f}",
                "DPD":       f"{res['DPD']:+.3f}",
            })
    fair_df = pd.DataFrame(fair_rows)
    print(fair_df.to_string(index=False) if not fair_df.empty else "(no rows)")

    # ---------- TABLE 6: ABLATION ----------
    print("\n\n===== TABLE 6: ABLATION =====")
    abl_rows = []

    def run_lr(X_, y_, cw, label):
        pipe = Pipeline([
            ("pre", build_preprocessor(X_)),
            ("clf", LogisticRegression(max_iter=2000,
                                       random_state=RANDOM_STATE,
                                       class_weight=cw, solver="lbfgs"))
        ])
        r = cross_validate(
            pipe, X_, y_, cv=cv,
            scoring={
                "macro_f1": "f1_macro",
                "recall_dropout": make_scorer(recall_score,
                                              labels=["Dropout"],
                                              average="macro",
                                              zero_division=0),
            }, n_jobs=-1)
        abl_rows.append({
            "Setting":        label,
            "Macro_F1":       f"{r['test_macro_f1'].mean():.3f}",
            "Dropout_recall": f"{r['test_recall_dropout'].mean():.3f}",
        })

    run_lr(X, y, None,       "LR, no balancing")
    run_lr(X, y, "balanced", "LR, balanced")

    # leakage comparison: raw read, keep 2nd-sem columns
    df_leak, _, _ = parse_raw(DATA_PATH)
    X_l, y_l = df_leak.drop(columns=[TARGET]), df_leak[TARGET]
    run_lr(X_l, y_l, "balanced",
           "LR, WITH 2nd-sem (leakage, comparison only)")

    abl_df = pd.DataFrame(abl_rows)
    print(abl_df.to_string(index=False))

    # ---------- SAVE ----------
    os.makedirs("results/tables",  exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    perf_df.to_csv("results/tables/table4_performance.csv", index=False)
    if not fair_df.empty:
        fair_df.to_csv("results/tables/table5_fairness.csv", index=False)
    abl_df.to_csv("results/tables/table6_ablation.csv", index=False)

    # ---------- FIG 2 ----------
    classes = sorted(y.unique())
    tr, te = list(cv.split(X, y))[-1]
    m = best_estimators[best_name].fit(X.iloc[tr], y.iloc[tr])
    pred = m.predict(X.iloc[te])
    cm = confusion_matrix(y.iloc[te], pred, labels=classes)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {best_name} (last fold)")
    plt.tight_layout()
    plt.savefig("results/figures/fig2_confusion_matrix.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n[done] Tables -> results/tables/, "
          "figure -> results/figures/")


if __name__ == "__main__":
    main()