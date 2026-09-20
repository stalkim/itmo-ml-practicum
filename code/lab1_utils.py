"""Activity classification: explicit columns, participant split, held-out test."""
from pathlib import Path
import hashlib
import json
import time
import importlib.metadata
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

CLASSES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
           "SITTING", "STANDING", "LAYING"]
LABELS = {name: i for i, name in enumerate(CLASSES)}


def _check_frame(frame, group_column):
    if not {"Activity", group_column}.issubset(frame.columns):
        raise ValueError("Need Activity and participant column: " + group_column)
    if frame.columns.duplicated().any() or frame.empty:
        raise ValueError("Empty table or duplicate columns")
    if frame[group_column].isna().any():
        raise ValueError("Participant identifiers cannot be missing")
    mapped = frame["Activity"].map(LABELS)
    if mapped.isna().any():
        raise ValueError("Missing or unknown Activity; inspect the original labels")
    features = frame.drop(columns=["Activity", group_column])
    if not len(features.columns):
        raise ValueError("No feature columns")
    numeric = features.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("HAR features must be finite; inspect missing values first")
    return numeric, mapped.to_numpy(dtype=int)


def prepare_har(train, test, group_column="subject", seed=40):
    X, y = _check_frame(train, group_column)
    Xt, yt = _check_frame(test, group_column)
    if set(X.columns) != set(Xt.columns):
        raise ValueError("Train/test feature sets differ")
    Xt = Xt.loc[:, X.columns]
    groups = train[group_column].to_numpy()
    test_groups = test[group_column].to_numpy()
    if set(groups) & set(test_groups):
        raise ValueError("Participants overlap between course train and test")
    if len(set(groups)) < 5:
        raise ValueError("Need at least five training participants for this protocol")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, va = next(splitter.split(X, y, groups))
    for name, labels in [("training", y[tr]), ("validation", y[va]), ("test", yt)]:
        if set(labels) != set(range(6)):
            raise ValueError(name + " must contain all six classes; review the split")
    return dict(Xtr=X.iloc[tr], Xva=X.iloc[va], ytr=y[tr], yva=y[va],
                Xall=X, yall=y, Xtest=Xt, ytest=yt, seed=seed,
                train_rows=tr.tolist(), validation_rows=va.tolist(),
                training_subjects=sorted(set(groups[tr])),
                validation_subjects=sorted(set(groups[va])),
                test_subjects=sorted(set(test_groups)),
                features=X.columns.tolist(), group_column=group_column)


def load_har(folder="data", group_column="subject", seed=40):
    folder = Path(folder)
    files = [folder / "train.csv", folder / "test.csv"]
    data = prepare_har(*(pd.read_csv(p) for p in files),
                       group_column=group_column, seed=seed)
    data["input_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return data


def candidate_models(seed=40):
    models = {"D0": DummyClassifier(strategy="most_frequent")}
    for k in (3, 7, 15):
        models[f"K{k}"] = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    for C in (0.1, 1.0, 10.0):
        models[f"S{C:g}"] = make_pipeline(StandardScaler(),
            LinearSVC(C=C, dual="auto", max_iter=10000, random_state=seed))
    for depth in (8, None):
        models[f"R{depth}"] = RandomForestClassifier(
            n_estimators=200, max_depth=depth, random_state=seed, n_jobs=1)
    return models


def classification_metrics(actual, predicted):
    return dict(accuracy=float(accuracy_score(actual, predicted)),
                macro_f1=float(f1_score(actual, predicted, labels=range(6),
                                        average="macro", zero_division=0)))


def fit_validation(data, estimator):
    model = clone(estimator)
    start = time.perf_counter()
    model.fit(data["Xtr"], data["ytr"])
    elapsed = time.perf_counter() - start
    prediction = model.predict(data["Xva"])
    return dict(model=model, prediction=prediction, seconds_fit=elapsed,
                metrics=classification_metrics(data["yva"], prediction))


def final_test(data, selected_estimator):
    """Call only after the choice is fixed; do not select configurations here."""
    model = clone(selected_estimator).fit(data["Xall"], data["yall"])
    yhat = model.predict(data["Xtest"])
    return dict(model=model, prediction=yhat,
                metrics=classification_metrics(data["ytest"], yhat),
                report=classification_report(data["ytest"], yhat, labels=range(6),
                    target_names=CLASSES, zero_division=0, output_dict=True))


def save_har_result(data, results_table, selected_name, final, folder):
    import joblib
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    results_table.to_csv(folder / "validation_comparison.csv", index=False)
    pd.DataFrame(final["report"]).T.to_csv(folder / "test_report.csv")
    pd.DataFrame({"row_in_test": np.arange(len(data["ytest"])),
                  "actual": data["ytest"], "predicted": final["prediction"]}).to_csv(
        folder / "test_predictions.csv", index=False)
    manifest = {key: data[key] for key in ["seed", "train_rows", "validation_rows",
        "training_subjects", "validation_subjects", "test_subjects", "features", "group_column"]}
    manifest.update(selected=selected_name, labels=LABELS,
        input_sha256=data.get("input_sha256", {}), test_metrics=final["metrics"],
        model_parameters={k: repr(v) for k, v in final["model"].get_params().items()},
        versions={p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn")},
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    # NumPy integer group identifiers are converted to plain JSON scalars.
    def scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2,
        ensure_ascii=False, allow_nan=False, default=scalar), encoding="utf-8")
    joblib.dump(final["model"], folder / "model.joblib")
