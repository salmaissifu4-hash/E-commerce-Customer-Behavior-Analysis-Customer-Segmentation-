from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "age",
    "annual_income_k",
    "spending_score",
    "app_sessions_per_week",
    "online_orders_per_month",
    "store_visits_per_month",
    "promo_response_rate",
    "loyalty_months",
    "family_size",
    "seasonal_spending_index",
    "returns_rate",
    "order_value_avg",
]

CATEGORICAL_FEATURES = ["gender", "preferred_channel", "membership_tier", "city_tier"]


@dataclass(frozen=True)
class SegmentArtifacts:
    pipeline: Pipeline
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]


def generate_customer_dataset(n_customers: int = 360, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    customer_id = np.arange(1001, 1001 + n_customers)
    age = np.clip(rng.normal(36, 12, n_customers).round(), 18, 70).astype(int)
    annual_income_k = np.clip(rng.normal(72, 24, n_customers), 22, 160).round(1)
    spending_score = np.clip(
        52 + (annual_income_k - annual_income_k.mean()) * 0.18 - (age - 36) * 0.25 + rng.normal(0, 14, n_customers),
        1,
        100,
    ).round(0).astype(int)

    gender = rng.choice(["Female", "Male", "Non-binary"], size=n_customers, p=[0.47, 0.48, 0.05])
    city_tier = rng.choice(["Tier 1", "Tier 2", "Tier 3"], size=n_customers, p=[0.32, 0.43, 0.25])

    preferred_channel = np.where(
        spending_score >= 70,
        rng.choice(["App", "Website"], size=n_customers, p=[0.68, 0.32]),
        rng.choice(["Store", "Website", "App"], size=n_customers, p=[0.48, 0.28, 0.24]),
    )

    app_sessions_per_week = np.clip(
        rng.normal(6, 2.5, n_customers)
        + np.where(preferred_channel == "App", 4, 0)
        + np.where(preferred_channel == "Website", 1.5, 0),
        0.5,
        18,
    ).round(1)

    online_orders_per_month = np.clip(
        rng.normal(3, 1.6, n_customers)
        + np.where(preferred_channel == "App", 3, 0)
        + np.where(spending_score > 65, 1.5, 0),
        0,
        20,
    ).round(1)

    store_visits_per_month = np.clip(
        rng.normal(5, 2, n_customers)
        + np.where(preferred_channel == "Store", 4, 0)
        - np.where(online_orders_per_month > 6, 1.2, 0),
        0,
        18,
    ).round(1)

    loyalty_months = np.clip(rng.gamma(2.5, 8, n_customers), 1, 72).round(0).astype(int)
    family_size = rng.choice([1, 2, 3, 4, 5, 6], size=n_customers, p=[0.16, 0.23, 0.24, 0.19, 0.12, 0.06])

    promo_response_rate = np.clip(
        0.18
        + 0.004 * (spending_score)
        + np.where(city_tier == "Tier 1", 0.05, 0)
        + rng.normal(0, 0.08, n_customers),
        0,
        1,
    ).round(3)

    returns_rate = np.clip(
        0.06 + np.where(family_size >= 4, 0.05, 0) + np.where(preferred_channel == "Store", 0.02, 0) + rng.normal(0, 0.03, n_customers),
        0,
        0.35,
    ).round(3)

    seasonal_spending_index = np.clip(
        0.75 + 0.0035 * spending_score + np.where(city_tier == "Tier 1", 0.12, 0) + rng.normal(0, 0.12, n_customers),
        0.4,
        1.8,
    ).round(3)

    order_value_avg = np.clip(
        annual_income_k * (0.05 + spending_score / 220) + rng.normal(0, 2.5, n_customers),
        3,
        35,
    ).round(2)

    membership_tier = pd.cut(
        loyalty_months,
        bins=[0, 12, 30, 60, 100],
        labels=["Bronze", "Silver", "Gold", "Platinum"],
        include_lowest=True,
    ).astype(str)

    monthly_spend_k = np.clip(
        10 + annual_income_k * 0.11 + spending_score * 0.19 + online_orders_per_month * 0.8 + app_sessions_per_week * 0.25,
        5,
        120,
    ).round(1)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "gender": gender,
            "age": age,
            "annual_income_k": annual_income_k,
            "spending_score": spending_score,
            "app_sessions_per_week": app_sessions_per_week,
            "online_orders_per_month": online_orders_per_month,
            "store_visits_per_month": store_visits_per_month,
            "promo_response_rate": promo_response_rate,
            "loyalty_months": loyalty_months,
            "family_size": family_size,
            "city_tier": city_tier,
            "preferred_channel": preferred_channel,
            "seasonal_spending_index": seasonal_spending_index,
            "returns_rate": returns_rate,
            "order_value_avg": order_value_avg,
            "membership_tier": membership_tier,
            "monthly_spend_k": monthly_spend_k,
        }
    )

    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


def save_customer_dataset(output_path: str | Path, n_customers: int = 360, random_state: int = 42) -> pd.DataFrame:
    df = generate_customer_dataset(n_customers=n_customers, random_state=random_state)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def build_preprocessor(
    numeric_features: Iterable[str] = NUMERIC_FEATURES,
    categorical_features: Iterable[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_features)),
            ("categorical", categorical_pipeline, list(categorical_features)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = (
        renamed.columns.str.strip()
        .str.lower()
        .str.replace("%", "pct", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return renamed


def validate_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    validated = df.copy()
    validated = validated.drop_duplicates(subset=["customer_id"]).reset_index(drop=True)
    validated = validated[validated["age"].between(18, 70)]
    validated = validated[validated["annual_income_k"].between(20, 180)]
    validated = validated[validated["spending_score"].between(1, 100)]
    validated = validated[validated["promo_response_rate"].between(0, 1)]
    validated = validated[validated["returns_rate"].between(0, 0.4)]
    return validated.reset_index(drop=True)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    return df[feature_cols].copy()


def attach_segment_names(df: pd.DataFrame, cluster_col: str = "kmeans_cluster") -> pd.DataFrame:
    profile_rules = {
        0: "Digital-first Professionals",
        1: "Premium Loyal Customers",
        2: "Discount-driven Families",
        3: "Value Seekers",
        4: "High-Spend Omnichannel Buyers",
    }
    named = df.copy()
    named["segment_name"] = named[cluster_col].map(profile_rules).fillna("Emerging Segment")
    return named


def predict_cluster_label(model_bundle: dict, new_customer: pd.DataFrame) -> int:
    preprocessor = model_bundle["preprocessor"]
    model = model_bundle["model"]
    features = new_customer[model_bundle["feature_columns"]]
    transformed = preprocessor.transform(features)
    return int(model.predict(transformed)[0])
