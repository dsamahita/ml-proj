
import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET = "Target"
PROTECTED = ["Scholarship holder", "Gender"]

# --------------------------------------------------
# 1. FIND DATASET
# --------------------------------------------------

if os.path.exists(os.path.join("data", "data.csv")):
    DATA_PATH = os.path.join("data", "data.csv")
elif os.path.exists("data.csv"):
    DATA_PATH = "data.csv"
else:
    raise FileNotFoundError(
        "data.csv not found. Put it inside the data folder "
        "or the main project folder."
    )

# --------------------------------------------------
# 2. LOAD AND REPAIR DATASET
# --------------------------------------------------

def load_dataset(path):
    try:
        df = pd.read_csv(path)
        if TARGET in df.columns:
            return df
    except Exception:
        pass

    df = pd.read_csv(path, header=None)

    header = df.iloc[0].astype(str).tolist()
    df = df.iloc[1:].copy()
    df.columns = [x.strip().strip("'").strip('"') for x in header]

    return df.reset_index(drop=True)


df = load_dataset(DATA_PATH)

df.columns = [
    str(c).strip().strip("'").strip('"')
    for c in df.columns
]

if TARGET not in df.columns:
    raise ValueError(
        f"Target column not found. Available columns: {df.columns.tolist()}"
    )

# Remove second-semester features to prevent information leakage.
second_sem_columns = [
    c for c in df.columns
    if "2nd sem" in c.lower()
]

df = df.drop(columns=second_sem_columns, errors="ignore")

# Convert target to clean strings.
df[TARGET] = df[TARGET].astype(str).str.strip()

X = df.drop(columns=[TARGET])
y = df[TARGET]

class_names = ["Dropout", "Enrolled", "Graduate"]

if set(y.unique()) != set(class_names):
    raise ValueError(
        f"Unexpected target classes: {sorted(y.unique())}"
    )

# --------------------------------------------------
# 3. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\nTRAINING RECORDS:", len(X_train))
print("TESTING RECORDS:", len(X_test))

# --------------------------------------------------
# 4. PREPROCESSING
# --------------------------------------------------

numeric_features = X_train.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = [
    c for c in X_train.columns
    if c not in numeric_features
]

try:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
except TypeError:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", encoder)
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# --------------------------------------------------
# 5. MODELS AND HYPERPARAMETER GRIDS
# --------------------------------------------------

models = {
    "Logistic Regression": (
        LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="lbfgs"
        ),
        {
            "clf__C": [0.1, 1, 10],
            "clf__class_weight": [None, "balanced"]
        }
    ),

    "Decision Tree": (
        DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        {
            "clf__max_depth": [4, 8, 12],
            "clf__min_samples_leaf": [5, 20, 50],
            "clf__class_weight": [None, "balanced"]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        {
            "clf__n_estimators": [100],
            "clf__max_depth": [None, 8],
            "clf__class_weight": [None, "balanced"]
        }
    )
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

best_models = {}
cv_scores = {}
best_parameters = {}

# --------------------------------------------------
# 6. HYPERPARAMETER TUNING
#    TRAINING DATA ONLY
# --------------------------------------------------

print("\n========== MODEL TRAINING ==========")

for name, (classifier, param_grid) in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", classifier)
    ])

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True
    )

    search.fit(X_train, y_train)

    best_models[name] = search.best_estimator_
    cv_scores[name] = search.best_score_
    best_parameters[name] = search.best_params_

    print("Best CV Macro F1:", round(search.best_score_, 4))
    print("Best Parameters:", search.best_params_)

# --------------------------------------------------
# 7. SELECT MODEL USING TRAINING CV ONLY
# --------------------------------------------------

best_name = max(cv_scores, key=cv_scores.get)
best_model = best_models[best_name]

print("\n===== BEST MODEL SELECTED BY TRAINING CV =====")
print(best_name)
print("Training CV Macro F1:", round(cv_scores[best_name], 4))

# --------------------------------------------------
# 8. BASELINE MODELS
# --------------------------------------------------

majority_model = DummyClassifier(strategy="most_frequent")
majority_model.fit(X_train, y_train)

stratified_model = DummyClassifier(
    strategy="stratified",
    random_state=RANDOM_STATE
)
stratified_model.fit(X_train, y_train)

# --------------------------------------------------
# 9. TEST-SET EVALUATION
# --------------------------------------------------

def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "Macro Precision": precision_score(
            y_test, predictions,
            average="macro",
            zero_division=0
        ),
        "Macro Recall": recall_score(
            y_test, predictions,
            average="macro",
            zero_division=0
        ),
        "Macro F1": f1_score(
            y_test, predictions,
            average="macro",
            zero_division=0
        ),
        "Dropout Recall": recall_score(
            y_test,
            predictions,
            labels=["Dropout"],
            average=None,
            zero_division=0
        )[0]
    }


results = []

results.append(
    evaluate_model(
        "Majority Dummy",
        majority_model,
        X_test,
        y_test
    )
)

results.append(
    evaluate_model(
        "Stratified Dummy",
        stratified_model,
        X_test,
        y_test
    )
)

for name, model in best_models.items():
    results.append(
        evaluate_model(name, model, X_test, y_test)
    )

performance_df = pd.DataFrame(results)

print("\n===== TABLE 7: TEST PERFORMANCE =====")
print(performance_df.to_string(index=False))

# --------------------------------------------------
# 10. BEST MODEL TEST PREDICTIONS
# --------------------------------------------------

y_pred = best_model.predict(X_test)

print("\n===== BEST MODEL =====")
print(best_name)

print("\n===== CONFUSION MATRIX =====")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=class_names
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {c}" for c in class_names],
    columns=[f"Predicted {c}" for c in class_names]
)

print(cm_df)

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred,
        labels=class_names,
        zero_division=0
    )
)

# --------------------------------------------------
# 11. FAIRNESS AUDIT
# --------------------------------------------------

def calculate_fairness(y_true, y_pred, group_values):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group_values = np.asarray(group_values)

    dropout_true = y_true == "Dropout"
    dropout_pred = y_pred == "Dropout"

    groups = sorted(pd.Series(group_values).dropna().unique())

    if len(groups) != 2:
        return None

    g0, g1 = groups[0], groups[1]

    mask0 = group_values == g0
    mask1 = group_values == g1

    actual0 = dropout_true & mask0
    actual1 = dropout_true & mask1

    tpr0 = (
        np.sum(dropout_pred & actual0) / np.sum(actual0)
        if np.sum(actual0) else np.nan
    )

    tpr1 = (
        np.sum(dropout_pred & actual1) / np.sum(actual1)
        if np.sum(actual1) else np.nan
    )

    rate0 = (
        np.mean(dropout_pred[mask0])
        if np.sum(mask0) else np.nan
    )

    rate1 = (
        np.mean(dropout_pred[mask1])
        if np.sum(mask1) else np.nan
    )

    return {
        "Group 1": g1,
        "Group 0": g0,
        "TPR Group 1": tpr1,
        "TPR Group 0": tpr0,
        "EOD": tpr1 - tpr0,
        "DPD": rate1 - rate0,
        "Actual Dropouts Group 1": np.sum(actual1),
        "Actual Dropouts Group 0": np.sum(actual0)
    }


fairness_results = []

for attribute in PROTECTED:

    if attribute not in X_test.columns:
        print(f"\nWARNING: {attribute} not found. Skipping.")
        continue

    fairness = calculate_fairness(
        y_test,
        y_pred,
        X_test[attribute].values
    )

    if fairness is not None:
        fairness["Protected Attribute"] = attribute
        fairness_results.append(fairness)

fairness_df = pd.DataFrame(fairness_results)

print("\n===== TABLE 10: TEST-SET FAIRNESS =====")
print(fairness_df.to_string(index=False))

# --------------------------------------------------
# 12. SAVE ALL TABLES
# --------------------------------------------------

os.makedirs("results/tables", exist_ok=True)

performance_df.to_csv(
    "results/tables/table7_test_performance.csv",
    index=False
)

hyperparameter_df = pd.DataFrame([
    {
        "Model": name,
        "Training CV Macro F1": cv_scores[name],
        "Best Hyperparameters": str(best_parameters[name]),
        "Selected Model": name == best_name
    }
    for name in best_models
])

hyperparameter_df.to_csv(
    "results/tables/table8_test_hyperparameters.csv",
    index=False
)

cm_df.to_csv(
    "results/tables/table9_confusion_matrix.csv"
)

fairness_df.to_csv(
    "results/tables/table10_test_fairness.csv",
    index=False
)

print("\n====================================")
print("TESTING COMPLETED SUCCESSFULLY")
print("All result tables saved in results/tables/")
print("Best model selected using training CV only.")
print("====================================")
