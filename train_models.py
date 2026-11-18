"""Train and save the models used by the Streamlit app.

Run once, locally, with the Inside Airbnb Los Angeles listings file:

    python train_models.py --data data/listings.csv.gz

It repeats the cleaning steps from the notebook (Section 4), fits the
notebook's four nested OLS price models plus three tuned machine-learning
models, and writes everything the app needs to models/, artifacts/ and data/.
The app itself never trains anything; it only loads these files.
"""
import argparse
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

SEED = 42
ROOT = Path(__file__).parent
AMENITIES = ["wifi", "parking", "pool_spa", "air_conditioning", "kitchen", "washer_dryer"]
PROPERTY = ["bedrooms", "bathrooms_num", "accommodates"]
HOST = ["superhost", "availability_365"]
CATS = ["room_type", "neighbourhood_cleansed"]


# --------------------------------------------------------------------------
# Cleaning: same rules as notebook Section 4
# --------------------------------------------------------------------------
def parse_amenities(value):
    if pd.isna(value):
        return []
    try:
        result = json.loads(value)
        return [str(i).lower().strip() for i in result] if isinstance(result, list) else []
    except (ValueError, TypeError):
        return []


def has_amenity(items, group):
    for item in items:
        item = item.lower().strip()
        if re.search(r"\b(not available|not included)\b", item):
            continue
        if group == "wifi":
            if re.search(r"\b(?:wifi|wi-fi)\b", item):
                return 1
        elif group == "parking":
            if "parking" in item:
                return 1
        elif group == "pool_spa":
            if re.search(r"\bpool\s+(?:table|view|toys?|cover)\b", item):
                continue
            if re.search(r"^(?:(?:private|shared|outdoor|indoor|heated)\s+)*"
                         r"(?:pool|swimming pool|hot\s*tub|jacuzzi)\b", item) or item == "spa access":
                return 1
        elif group == "air_conditioning":
            if "air conditioning" in item:
                return 1
        elif group == "kitchen":
            if item == "kitchen" or item.startswith("kitchen "):
                return 1
        elif group == "washer_dryer":
            if re.search(r"^(?:(?:free|paid)\s+)?(?:washer|dryer|washing machine|drying machine)\b", item):
                return 1
    return 0


def load_and_clean(path):
    df = pd.read_csv(path, low_memory=False)
    df["price"] = pd.to_numeric(
        df["price"].astype("string").str.replace(r"[$,]", "", regex=True), errors="coerce"
    ).astype(float)

    county = {
        "n_listings": int(len(df)),
        "n_columns": int(df.shape[1]),
        "geo_counts": df["neighbourhood_group_cleansed"].value_counts().to_dict(),
        "room_counts": df["room_type"].value_counts().to_dict(),
        "price_missing_pct": float(df["price"].isna().mean() * 100),
        "rating_missing_pct": float(df["review_scores_rating"].isna().mean() * 100),
    }

    data = df.loc[df["neighbourhood_group_cleansed"] == "City of Los Angeles"].copy()
    data.loc[data["price"].notna() & (data["price"] <= 0), "price"] = np.nan
    data["log_price"] = np.log(data["price"])

    data["bathrooms_num"] = pd.to_numeric(
        data["bathrooms_text"].str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce"
    )
    half = data["bathrooms_text"].str.fullmatch(r"(?:private |shared )?half[- ]bath", case=False, na=False)
    data.loc[half, "bathrooms_num"] = 0.5
    for col in PROPERTY:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["superhost"] = data["host_is_superhost"].map({"t": 1, "f": 0})
    lists = data["amenities"].apply(parse_amenities)
    for g in AMENITIES:
        data[g] = lists.apply(lambda items: has_amenity(items, g))
    data["amenity_count"] = lists.apply(len)

    predictors = PROPERTY + ["room_type"] + HOST + ["neighbourhood_cleansed"] + AMENITIES
    price_data = data.dropna(subset=["log_price"] + predictors).copy()
    county["n_city"] = int(len(data))
    county["n_price_sample"] = int(len(price_data))
    county["n_ratings_sample"] = int(price_data["review_scores_rating"].notna().sum())
    return price_data, county


# --------------------------------------------------------------------------
# Model definitions
# --------------------------------------------------------------------------
SPECS = {
    "OLS 1: Property": PROPERTY + ["room_type"],
    "OLS 2: + Amenities": PROPERTY + ["room_type"] + AMENITIES,
    "OLS 3: + Host": PROPERTY + ["room_type"] + AMENITIES + HOST,
    "OLS 4: + Neighborhood": PROPERTY + ["room_type"] + AMENITIES + HOST + ["neighbourhood_cleansed"],
}
FULL = SPECS["OLS 4: + Neighborhood"]
FORMULAS = {
    "OLS 1: Property": "log_price ~ bedrooms + bathrooms_num + accommodates + C(room_type)",
}
FORMULAS["OLS 2: + Amenities"] = FORMULAS["OLS 1: Property"] + " + " + " + ".join(AMENITIES)
FORMULAS["OLS 3: + Host"] = FORMULAS["OLS 2: + Amenities"] + " + superhost + availability_365"
FORMULAS["OLS 4: + Neighborhood"] = FORMULAS["OLS 3: + Host"] + " + C(neighbourhood_cleansed)"


def onehot_pipe(features, estimator, scale=False):
    cats = [c for c in features if c in CATS]
    nums = [c for c in features if c not in CATS]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=None), cats),
        ("num", StandardScaler() if scale else "passthrough", nums),
    ])
    return Pipeline([("prep", pre), ("model", estimator)])


def hgb_pipe(features):
    cats = [c for c in features if c in CATS]
    nums = [c for c in features if c not in CATS]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1), cats),
        ("num", "passthrough", nums),
    ], verbose_feature_names_out=False)
    est = HistGradientBoostingRegressor(categorical_features=list(range(len(cats))), random_state=SEED)
    return Pipeline([("prep", pre), ("model", est)])


def evaluate(y_true_log, y_pred_log):
    p_true, p_pred = np.exp(y_true_log), np.exp(y_pred_log)
    return {
        "R2 (log price)": float(r2_score(y_true_log, y_pred_log)),
        "RMSE (log price)": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "MAE ($)": float(mean_absolute_error(p_true, p_pred)),
        "Median abs. error ($)": float(np.median(np.abs(p_true - p_pred))),
        "Median abs. % error": float(np.median(np.abs(p_pred - p_true) / p_true) * 100),
    }


def main(data_path, fast=False):
    t0 = time.time()
    price_data, county = load_and_clean(data_path)
    print(f"Pricing sample: {len(price_data):,} listings")

    # Notebook-style in-sample OLS fits (HC3), for fidelity with the report
    insample = {}
    for name, formula in FORMULAS.items():
        m = smf.ols(formula, data=price_data).fit(cov_type="HC3")
        insample[name] = {"R2": float(m.rsquared), "Adj R2": float(m.rsquared_adj), "n": int(m.nobs)}
        if name == "OLS 4: + Neighborhood":
            full_ols = m
    ci = full_ols.conf_int()
    coef_rows = []
    for v in AMENITIES + HOST + PROPERTY:
        coef_rows.append({"term": v, "coef": float(full_ols.params[v]), "lo": float(ci.loc[v, 0]),
                          "hi": float(ci.loc[v, 1]), "p": float(full_ols.pvalues[v])})

    rating_data = price_data.dropna(subset=["review_scores_rating"])
    rating_model = smf.ols(
        "review_scores_rating ~ log_price + bedrooms + bathrooms_num + accommodates + C(room_type)"
        " + superhost + " + " + ".join(AMENITIES) + " + C(neighbourhood_cleansed)",
        data=rating_data).fit(cov_type="HC3")
    rating_summary = {
        "r2": float(rating_model.rsquared),
        "n": int(rating_model.nobs),
        "log_price_coef": float(rating_model.params["log_price"]),
        "superhost_coef": float(rating_model.params["superhost"]),
        "spearman": float(rating_data[["log_price", "review_scores_rating"]].corr("spearman").iloc[0, 1]),
    }

    # Train/test split for an honest out-of-sample comparison
    X = price_data[FULL]
    y = price_data["log_price"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    cv = KFold(5, shuffle=True, random_state=SEED)

    results, best_params, preds = {}, {}, {"actual_log": y_te.to_numpy(),
                                             "room_type": X_te["room_type"].to_numpy()}
    models = {}

    for name, feats in SPECS.items():
        pipe = onehot_pipe(feats, LinearRegression()).fit(X_tr[feats], y_tr)
        models[name] = (pipe, feats)
        best_params[name] = {"(none)": "Ordinary least squares has no tuning parameters"}

    searches = {
        "Ridge": GridSearchCV(
            onehot_pipe(FULL, Ridge(), scale=True),
            {"model__alpha": [0.01, 0.1, 1, 3, 10, 30, 100, 300]},
            cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1),
        "Random Forest": RandomizedSearchCV(
            onehot_pipe(FULL, RandomForestRegressor(random_state=SEED, n_jobs=-1)),
            {"model__n_estimators": [150, 250],
             "model__max_depth": [12, 18, 24],
             "model__min_samples_leaf": [2, 4, 8],
             "model__max_features": [0.3, 0.5, "sqrt"]},
            n_iter=4 if fast else 12, cv=3 if fast else cv, scoring="neg_root_mean_squared_error",
            random_state=SEED, n_jobs=-1),
        "Gradient Boosting": RandomizedSearchCV(
            hgb_pipe(FULL),
            {"model__learning_rate": [0.03, 0.05, 0.1],
             "model__max_iter": [300, 600, 900],
             "model__max_leaf_nodes": [15, 31, 63],
             "model__min_samples_leaf": [10, 20, 40],
             "model__l2_regularization": [0.0, 0.1, 1.0]},
            n_iter=4 if fast else 20, cv=3 if fast else cv, scoring="neg_root_mean_squared_error",
            random_state=SEED, n_jobs=-1),
    }
    for name, search in searches.items():
        print(f"Tuning {name} ...", flush=True)
        search.fit(X_tr, y_tr)
        models[name] = (search.best_estimator_, FULL)
        best_params[name] = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        best_params[name]["CV RMSE (log)"] = round(float(-search.best_score_), 4)

    for name, (pipe, feats) in models.items():
        p = pipe.predict(X_te[feats])
        results[name] = evaluate(y_te, p)
        results[name]["Features"] = len(feats)
        preds[name] = p
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        joblib.dump({"pipeline": pipe, "features": feats, "name": name},
                    ROOT / "models" / f"{slug}.joblib", compress=3)
        print(f"  {name:24s} test R2 = {results[name]['R2 (log price)']:.4f}")

    # Defaults for the prediction tab: numeric/binary at their means, categoricals at their mode
    defaults = {c: float(price_data[c].mean()) for c in PROPERTY + HOST + AMENITIES}
    defaults["room_type"] = price_data["room_type"].mode()[0]
    defaults["neighbourhood_cleansed"] = price_data["neighbourhood_cleansed"].mode()[0]

    pd.DataFrame(preds).to_csv(ROOT / "artifacts" / "test_predictions.csv", index=False)
    keep = ["price", "log_price", "room_type", "neighbourhood_cleansed", "review_scores_rating",
            "amenity_count"] + PROPERTY + HOST + AMENITIES
    price_data[keep].to_csv(ROOT / "data" / "price_data.csv.gz", index=False, compression="gzip")

    meta = {
        "county": county,
        "test_metrics": results,
        "best_params": best_params,
        "insample_ols": insample,
        "full_ols_coefficients": coef_rows,
        "rating_model": rating_summary,
        "defaults": defaults,
        "room_types": sorted(price_data["room_type"].unique().tolist()),
        "neighbourhoods": sorted(price_data["neighbourhood_cleansed"].unique().tolist()),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "model_files": {n: re.sub(r"[^a-z0-9]+", "_", n.lower()).strip("_") + ".joblib" for n in models},
    }
    with open(ROOT / "artifacts" / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/listings.csv.gz")
    ap.add_argument("--fast", action="store_true", help="smaller search, for quick checks")
    a = ap.parse_args()
    main(a.data, a.fast)
