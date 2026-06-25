import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

EVAL_THRESHOLD = 0.70
LABELS = [0, 1, 2]


def build_model(params: dict):
    model_type = params.get("model_type", "random_forest")
    model_params = {k: v for k, v in params.items() if k != "model_type"}

    if model_type == "random_forest":
        return RandomForestClassifier(**model_params, random_state=42)
    if model_type == "gradient_boosting":
        allowed = {"n_estimators", "learning_rate", "max_depth", "min_samples_split"}
        model_params = {k: v for k, v in model_params.items() if k in allowed}
        return GradientBoostingClassifier(**model_params, random_state=42)
    if model_type == "logistic_regression":
        allowed = {"C", "solver", "class_weight", "multi_class"}
        model_params = {k: v for k, v in model_params.items() if k in allowed}
        return LogisticRegression(**model_params, random_state=42, max_iter=1000)

    raise ValueError(
        "model_type phai la mot trong: random_forest, gradient_boosting, logistic_regression"
    )


def label_distribution(y: pd.Series) -> dict[str, float]:
    ratios = y.value_counts(normalize=True).reindex(LABELS, fill_value=0.0)
    return {str(label): float(ratio) for label, ratio in ratios.items()}


def write_report(y_true: pd.Series, preds, acc: float, f1: float) -> None:
    os.makedirs("outputs", exist_ok=True)
    cm = confusion_matrix(y_true, preds, labels=LABELS)
    cls_report = classification_report(
        y_true,
        preds,
        labels=LABELS,
        target_names=["thap", "trung_binh", "cao"],
        zero_division=0,
    )

    with open("outputs/report.txt", "w", encoding="utf-8") as f:
        f.write("Bao cao hieu suat mo hinh\n")
        f.write("==========================\n\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 weighted: {f1:.4f}\n\n")
        f.write("Confusion matrix (labels: 0=thap, 1=trung_binh, 2=cao)\n")
        f.write(str(cm))
        f.write("\n\nPrecision/Recall/F1 theo lop\n")
        f.write(cls_report)


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho RandomForestClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    # TODO 1: Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # TODO 2: Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

    with mlflow.start_run():

        # TODO 3: Ghi nhan cac sieu tham so
        mlflow.log_params(params)

        # TODO 4: Khoi tao va huan luyen mo hinh
        model = build_model(params)
        model.fit(X_train, y_train)

        # TODO 5: Du doan tren tap danh gia va tinh chi so
        preds = model.predict(X_eval)
        acc = float(accuracy_score(y_eval, preds))
        f1 = float(f1_score(y_eval, preds, average="weighted"))

        # TODO 6: Ghi nhan chi so vao MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        # TODO 7: In ket qua ra man hinh
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # TODO 8: Luu metrics ra file outputs/metrics.json
        # File nay duoc doc boi GitHub Actions o Buoc 2
        distribution = label_distribution(y_train)
        for label, ratio in distribution.items():
            if ratio < 0.10:
                print(
                    f"WARNING: Lop {label} chiem {ratio:.2%} tap train (< 10%). "
                    "Co nguy co lech phan phoi du lieu."
                )

        write_report(y_eval, preds, acc, f1)

        with open("outputs/metrics.json", "w") as f:
            json.dump(
                {
                    "accuracy": acc,
                    "f1_score": f1,
                    "label_distribution": distribution,
                },
                f,
                indent=2,
            )

        mlflow.log_artifact("outputs/report.txt")

        # TODO 9: Luu mo hinh ra file models/model.pkl
        # File nay duoc upload len GCS o Buoc 2
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    # TODO 10: Tra ve acc
    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
