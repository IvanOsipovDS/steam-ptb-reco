import pandas as pd
from pathlib import Path
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from joblib import dump
import numpy as np

from ..utils.io import ROOT, ensure_dir, write_json

PROC = ROOT / "data" / "processed"
ART = ROOT / "models" / "artifacts"
REG = ROOT / "models" / "registry.json"


def load_data():
    """
    Load processed dataset built at step 4.
    Expect columns at least:
      appid, name, owners_mid, target_tier, pos_ratio, release_year, developer, publisher, desc
    """
    path = PROC / "dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    df = pd.read_csv(path)

    # Basic filtering: make sure target/owners exist
    df = df.dropna(subset=["owners_mid", "target_tier"])
    # Features and target
    X = df[["developer", "publisher", "pos_ratio", "release_year", "desc"]].copy()
    y = df["target_tier"].astype(str).copy()

    # Print basic diagnostics
    print(f"[train] Loaded dataset: {path}, shape={df.shape}")
    print(f"[train] Class distribution:\n{y.value_counts(normalize=True).round(3)}")

    return X, y, df


def build_preprocessor():
    """
    Preprocessor combines:
      - OneHotEncoder for categorical features
      - SimpleImputer(median) for numerical
      - TfidfVectorizer for text
    """
    cat_cols = ["developer", "publisher"]
    num_cols = ["pos_ratio", "release_year"]
    text_col = "desc"

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", max_categories=50), cat_cols),
            ("num", Pipeline(steps=[("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("txt", TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), text_col),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


def train_and_eval(model, X_train, X_val, y_train, y_val):
    """
    Fit preprocessor + model; return fitted pipeline and metrics.
    We compute multiclass ROC-AUC (OvR) and macro F1.
    """
    pre = build_preprocessor()
    pipe = Pipeline([("pre", pre), ("clf", model)])
    pipe.fit(X_train, y_train)

    # AUC (OvR) if proba available
    try:
        y_proba = pipe.predict_proba(X_val)
        auc = roc_auc_score(y_val, y_proba, multi_class="ovr")
    except Exception:
        auc = np.nan

    y_pred = pipe.predict(X_val)
    f1 = f1_score(y_val, y_pred, average="macro")

    return pipe, {"auc": float(auc) if auc == auc else np.nan, "f1": float(f1)}


def main():
    ensure_dir(ART)

    # Configure MLflow to use local folder 'mlruns'
    mlflow.set_tracking_uri("file:" + str(ROOT / "mlruns"))
    mlflow.set_experiment("steam_ptb")

    X, y, df = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    candidates = {
        "logreg": LogisticRegression(max_iter=300, n_jobs=1),
        "xgb": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
        ),
    }

    best_name, best_score, best_pipe = None, -1.0, None

    for name, model in candidates.items():
        with mlflow.start_run(run_name=name):
            pipe, metrics = train_and_eval(model, X_train, X_val, y_train, y_val)

            # Log params and metrics to MLflow
            mlflow.log_params({"model": name})
            mlflow.log_params(
                {"n_train": len(X_train), "n_val": len(X_val), "tfidf_max_features": 5000}
            )
            for k, v in metrics.items():
                if v == v:  # skip NaN
                    mlflow.log_metric(k, float(v))

            # Select best mainly by AUC (fallback to F1 if AUC NaN)
            score = metrics["auc"] if metrics["auc"] == metrics["auc"] else metrics["f1"]
            print(f"[train] {name}: AUC={metrics['auc']:.4f} | F1={metrics['f1']:.4f}")
            if score > best_score:
                best_name, best_score, best_pipe = name, score, pipe

    # Persist best model and update registry
    best_path = ART / f"best_{best_name}.joblib"
    dump(best_pipe, best_path)
    write_json(REG, {"best_model_path": str(best_path)})

    print(f"[train] Best model: {best_name} | score={best_score:.4f}")
    print(f"[train] Saved best model to: {best_path}")
    print(f"[train] Registry updated: {REG}")


if __name__ == "__main__":
    main()
