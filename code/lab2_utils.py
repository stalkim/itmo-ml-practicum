"""Teaching helpers for lab 2. No course data or measured scores are bundled."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DROP_COLUMNS = ["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"]
SEED = 40


def check_tables(train, test):
    if not {"Id", "SalePrice"}.issubset(train.columns):
        raise ValueError("train.csv must contain Id and SalePrice")
    if "Id" not in test or "SalePrice" in test:
        raise ValueError("test.csv must contain Id and no SalePrice")
    if train.columns.duplicated().any() or test.columns.duplicated().any():
        raise ValueError("Column names must be unique")
    for table in (train, test):
        if table["Id"].isna().any() or not table["Id"].is_unique:
            raise ValueError("Id must be present and unique within each table")
    if set(train.columns) - {"SalePrice"} != set(test.columns):
        raise ValueError("Train and prediction feature columns differ")
    y = pd.to_numeric(train["SalePrice"], errors="raise").astype(float)
    if not np.isfinite(y.to_numpy()).all() or (y <= 0).any():
        raise ValueError("SalePrice must be finite and strictly positive")
    if len(y) < 10 or y.nunique() < 2:
        raise ValueError("Need at least 10 rows and a nonconstant target")
    return y


def prepare_frames(train, test, seed=SEED):
    """Split raw rows first; learn all imputation, scaling and encoding on Xtr."""
    y = check_tables(train, test)
    removed = [c for c in DROP_COLUMNS if c in train.columns]
    features = [c for c in train.columns if c not in ["Id", "SalePrice", *removed]]
    X = train.loc[:, features].copy()
    Xtest = test.loc[:, features].copy()
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    numeric = Xtr.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in features if c not in numeric]
    # This deterministic conversion learns no statistics or category vocabulary.
    Xtr, Xva = Xtr.copy(), Xva.copy()
    for frame in (Xtr, Xva, Xtest):
        for col in numeric:
            values = frame[col].to_numpy(dtype=float)
            if np.isinf(values).any():
                raise ValueError("Infinite numeric input: " + col)
        for col in categorical:
            values = frame[col].astype(object)
            frame[col] = values.where(pd.notna(values), np.nan)
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__",
                                  keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                  dtype=np.float32)),
    ])
    preprocess = ColumnTransformer([
        ("num", num_pipe, numeric), ("cat", cat_pipe, categorical)
    ], remainder="drop")
    Atr = preprocess.fit_transform(Xtr).astype(np.float32)
    Ava = preprocess.transform(Xva).astype(np.float32)
    Atest = preprocess.transform(Xtest).astype(np.float32)
    for array in (Atr, Ava, Atest):
        if array.ndim != 2 or array.shape[1] == 0 or not np.isfinite(array).all():
            raise ValueError("Preprocessing produced invalid features")
    if ytr.nunique() < 2:
        raise ValueError("Training target is constant after the split")
    target_scaler = StandardScaler()
    ztr = target_scaler.fit_transform(ytr.to_numpy().reshape(-1, 1)).astype(np.float32)
    zva = target_scaler.transform(yva.to_numpy().reshape(-1, 1)).astype(np.float32)
    return dict(
        Atr=Atr, Ava=Ava, Atest=Atest, ztr=ztr, zva=zva,
        ytr=ytr.to_numpy(), yva=yva.to_numpy(),
        train_ids=train.loc[Xtr.index, "Id"].to_numpy(),
        validation_ids=train.loc[Xva.index, "Id"].to_numpy(),
        prediction_ids=test["Id"].to_numpy(), preprocess=preprocess,
        target_scaler=target_scaler, removed_columns=removed,
        numeric_columns=numeric, categorical_columns=categorical,
        feature_names=preprocess.get_feature_names_out().tolist(),
    )


def load_course_data(data_dir="data", seed=SEED):
    folder = Path(data_dir)
    files = [folder / "train.csv", folder / "test.csv"]
    absent = [str(p) for p in files if not p.is_file()]
    if absent:
        raise FileNotFoundError("Add the original course files: " + ", ".join(absent))
    train, test = (pd.read_csv(p) for p in files)
    data = prepare_frames(train, test, seed)
    data["input_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }
    description = folder / "data_description.txt"
    if description.is_file():
        data["input_sha256"][description.name] = hashlib.sha256(description.read_bytes()).hexdigest()
    data["split_seed"] = seed
    return data


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if actual.shape != predicted.shape or not len(actual):
        raise ValueError("Actual and predicted values must have matching lengths")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite values")
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if np.var(actual) > 0 else None,
    }


def constant_baselines(data):
    return {
        "training_mean": regression_metrics(
            data["yva"], np.full_like(data["yva"], data["ytr"].mean())
        ),
        "training_median": regression_metrics(
            data["yva"], np.full_like(data["yva"], np.median(data["ytr"]))
        ),
    }


def run_experiment(data, hidden=(64,), epochs=100, batch_size=32,
                   optimizer="adam", learning_rate=1e-3, loss="mse",
                   seed=SEED, early_stop=False, verbose=0):
    """A reproducible baseline. Research choices and explanations are student work."""
    import tensorflow as tf
    from tensorflow import keras
    if not hidden or any(int(width) != width or width <= 0 for width in hidden):
        raise ValueError("Specify positive hidden-layer widths")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Epochs, batch size and learning rate must be positive")
    if loss not in {"mse", "mae"} or optimizer not in {"adam", "sgd"}:
        raise ValueError("Supported choices: mse/mae and adam/sgd")
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    layers = [keras.Input(shape=(data["Atr"].shape[1],))]
    layers.extend(keras.layers.Dense(int(width), activation="relu") for width in hidden)
    layers.append(keras.layers.Dense(1, activation="linear"))
    model = keras.Sequential(layers)
    opt_cls = keras.optimizers.Adam if optimizer == "adam" else keras.optimizers.SGD
    model.compile(optimizer=opt_cls(learning_rate=learning_rate), loss=loss,
                  metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    callbacks = []
    if early_stop:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_mae", mode="min", patience=15, restore_best_weights=True
        ))
    start = time.perf_counter()
    history = model.fit(
        data["Atr"], data["ztr"], validation_data=(data["Ava"], data["zva"]),
        epochs=epochs, batch_size=batch_size, shuffle=True,
        callbacks=callbacks, verbose=verbose,
    )
    elapsed = time.perf_counter() - start
    # Direct calls avoid an additional input pipeline for these small tables.
    val_scaled = model(data["Ava"], training=False).numpy().reshape(-1, 1)
    test_scaled = model(data["Atest"], training=False).numpy().reshape(-1, 1)
    val_predictions = data["target_scaler"].inverse_transform(val_scaled).reshape(-1)
    predictions = data["target_scaler"].inverse_transform(test_scaled).reshape(-1)
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite prediction: inspect optimization and inputs")
    curves = pd.DataFrame(history.history)
    curves.insert(0, "epoch", np.arange(1, len(curves) + 1))
    scale = float(data["target_scaler"].scale_[0])
    curves["mae_price"] = curves["mae"] * scale
    curves["val_mae_price"] = curves["val_mae"] * scale
    return dict(
        model=model, history=curves, predictions=predictions,
        validation_predictions=val_predictions,
        metrics=regression_metrics(data["yva"], val_predictions),
        config=dict(hidden=list(hidden), epochs_requested=epochs,
                    epochs_run=len(curves), batch_size=batch_size, optimizer=optimizer,
                    learning_rate=learning_rate, loss=loss, seed=seed,
                    early_stop=early_stop, n_parameters=model.count_params(),
                    seconds_fit=elapsed),
    )


def save_experiment(data, result, output_dir):
    """Do not silently overwrite earlier experimental evidence."""
    folder = Path(output_dir)
    if folder.exists():
        raise FileExistsError("Use a new directory for every experiment")
    predictions = np.asarray(result["predictions"]).reshape(-1)
    if len(predictions) != len(data["prediction_ids"]) or not np.isfinite(predictions).all():
        raise ValueError("Invalid prediction length or values")
    folder.mkdir(parents=True)
    pd.DataFrame({"Id": data["prediction_ids"], "SalePrice": predictions}).to_csv(
        folder / "predictions_raw.csv", index=False
    )
    pd.DataFrame({"Id": data["validation_ids"], "SalePrice": data["yva"],
                  "Prediction": result["validation_predictions"]}).to_csv(
        folder / "validation_predictions.csv", index=False
    )
    result["history"].to_csv(folder / "history.csv", index=False)
    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "tensorflow", "tensorflow-cpu", "keras"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest = dict(
        config=result["config"], validation_metrics=result["metrics"],
        baselines=constant_baselines(data), versions=versions,
        split_seed=data.get("split_seed", SEED),
        input_sha256=data.get("input_sha256", {}),
        train_ids=data["train_ids"].tolist(), validation_ids=data["validation_ids"].tolist(),
        removed_columns=data["removed_columns"], feature_names=data["feature_names"],
        target_mean=float(data["target_scaler"].mean_[0]),
        target_scale=float(data["target_scaler"].scale_[0]),
        negative_prediction_count=int((predictions < 0).sum()),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (folder / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    result["model"].save(folder / "model.keras")
    # The locally created preprocessing file is needed to repeat inference.
    import joblib
    joblib.dump({"preprocess": data["preprocess"], "target_scaler": data["target_scaler"]},
                folder / "preprocessing.joblib")


def plot_diagnostics(data, result):
    import matplotlib.pyplot as plt
    history = result["history"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    axes[0].plot(history["epoch"], history["mae_price"], label="train")
    axes[0].plot(history["epoch"], history["val_mae_price"], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="MAE (price units)")
    axes[0].legend()
    actual, predicted = data["yva"], result["validation_predictions"]
    low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    axes[1].scatter(actual, predicted, s=12, alpha=0.6)
    axes[1].plot([low, high], [low, high], color="black", linewidth=1)
    axes[1].set(xlabel="Actual price", ylabel="Predicted price")
    axes[2].scatter(predicted, actual - predicted, s=12, alpha=0.6)
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set(xlabel="Predicted price", ylabel="Residual: actual - predicted")
    fig.tight_layout()
    return fig
