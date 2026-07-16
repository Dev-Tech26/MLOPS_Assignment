import os, sys, warnings, logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)

from pathlib import Path

base_dir = Path.cwd()
if "__file__" in globals():
    base_dir = Path(__file__).resolve().parent

src_dir = base_dir if base_dir.name == "src" else base_dir / "src"
src_dir = src_dir.resolve()
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

from preprocess import download_data, load_data, get_preprocessor, split_data, FEATURE_COLS

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from pathlib import Path

base_dir = Path.cwd()
if "__file__" in globals():
    base_dir = Path(__file__).resolve().parent

src_dir = base_dir if base_dir.name == "src" else base_dir / "src"
src_dir = src_dir.resolve()
project_root = src_dir.parent if src_dir.name == "src" else base_dir

# DATA_PATH = project_root / "data" / "heart.csv"
# MODEL_PATH = project_root / "models" / "best_model.joblib"
# REPORTS_DIR = project_root / "reports"
# EXPERIMENT_NAME = "heart-disease-classification"
# MLRUNS_DIR = project_root / "mlruns"
# TRACKING_URI = MLRUNS_DIR.resolve().as_uri()

# REPORTS_DIR.mkdir(exist_ok=True)
# MODEL_PATH.parent.mkdir(exist_ok=True)

# os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
# mlflow.set_tracking_uri(TRACKING_URI)
# mlflow.set_experiment(EXPERIMENT_NAME)
DATA_PATH = project_root / "data" / "heart.csv"
MODEL_PATH = project_root / "models" / "best_model.joblib"
REPORTS_DIR = project_root / "reports"
EXPERIMENT_NAME = "heart-disease-classification"

# Use SQLite DB for MLflow tracking
DB_URI = "sqlite:///mlflow.db"

REPORTS_DIR.mkdir(exist_ok=True)
MODEL_PATH.parent.mkdir(exist_ok=True)

mlflow.set_tracking_uri(DB_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def make_models():
    preprocessor = get_preprocessor()
    return {
        "Logistic Regression": Pipeline([
            ("pre", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("pre", preprocessor),
            ("clf", RandomForestClassifier(random_state=42))
        ]),
        "Gradient Boosting": Pipeline([
            ("pre", preprocessor),
            ("clf", GradientBoostingClassifier(random_state=42))
        ]),
        "SVM": Pipeline([
            ("pre", preprocessor),
            ("clf", SVC(probability=True, random_state=42))
        ]),
    }


def evaluate(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }, y_pred, y_prob


def plot_confusion_matrix(y_test, y_pred, name, path):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_roc(y_test, probs_dict, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_prob in probs_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves"); ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def main():
    # Load data
    if os.path.exists(DATA_PATH):
        df = load_data(DATA_PATH)
        logger.info(f"Loaded data from {DATA_PATH}: {df.shape}")
    else:
        logger.info("Downloading dataset from UCI...")
        df = download_data()
        df.to_csv(DATA_PATH, index=False)

    X_train, X_test, y_train, y_test = split_data(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = make_models()
    all_probs = {}
    best_auc = 0
    best_pipe = None
    best_name = ""

    # ── Train all base models ──
    for name, pipe in models.items():
        with mlflow.start_run(run_name=name):
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
            pipe.fit(X_train, y_train)
            metrics, y_pred, y_prob = evaluate(pipe, X_test, y_test)
            all_probs[name] = y_prob

            mlflow.log_params({"model": name, "cv_folds": 5})
            mlflow.log_metrics({**metrics,
                                "cv_roc_auc_mean": cv_scores.mean(),
                                "cv_roc_auc_std": cv_scores.std()})

            cm_path = f"{REPORTS_DIR}/cm_{name.replace(' ', '_')}.png"
            plot_confusion_matrix(y_test, y_pred, name, cm_path)
            mlflow.log_artifact(cm_path)
            mlflow.sklearn.log_model(
                pipe,
                name=name.replace(" ", "_"),
                signature=infer_signature(X_train, pipe.predict(X_train)),
                skops_trusted_types=["numpy.dtype"],
            )

            logger.info(f"{name}: AUC={metrics['roc_auc']:.3f} CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_pipe = pipe
                best_name = name

    # ── Hyperparameter tuning ──
    with mlflow.start_run(run_name="RandomForest_GridSearch"):
        params = {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 5, 10],
                  "clf__min_samples_split": [2, 5], "clf__min_samples_leaf": [1, 2]}
        rs = RandomizedSearchCV(models["Random Forest"], params, n_iter=12,
                                cv=cv, scoring="roc_auc", n_jobs=-1, random_state=42)
        rs.fit(X_train, y_train)
        metrics, y_pred, y_prob = evaluate(rs.best_estimator_, X_test, y_test)
        all_probs["RF Tuned"] = y_prob

        mlflow.log_params(rs.best_params_)
        mlflow.sklearn.log_model(
            rs.best_estimator_,
            name="RF_Tuned",
            signature=infer_signature(X_train, rs.best_estimator_.predict(X_train)),
            skops_trusted_types=["numpy.dtype"],
        )

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_pipe = rs.best_estimator_
            best_name = "RF Tuned"

        logger.info(f"RF Tuned: AUC={metrics['roc_auc']:.3f} Best Params={rs.best_params_}")

    # ── Save champion model ──
    joblib.dump(best_pipe, MODEL_PATH)
    logger.info(f"Champion: {best_name} (AUC={best_auc:.3f}) saved to {MODEL_PATH}")

    # ── ROC curve for all ──
    roc_path = f"{REPORTS_DIR}/roc_all_models.png"
    plot_roc(y_test, all_probs, roc_path)

    with mlflow.start_run(run_name="Champion_Summary"):
        mlflow.log_param("champion_model", best_name)
        mlflow.log_metric("champion_roc_auc", best_auc)
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(MODEL_PATH)


if __name__ == "__main__":
    main()