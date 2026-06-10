# preprocessing.py

from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from scipy.stats import skew

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)


def _looks_like_datetime(series: pd.Series) -> bool:
    non_null_values = series.dropna().astype(str)

    if non_null_values.empty:
        return False

    sample = non_null_values.head(25)
    date_pattern = (
        r"(?:\d{4}[-/]\d{1,2}[-/]\d{1,2})"
        r"|(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
        r"|(?:[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})"
    )

    return bool(
        sample
        .str.contains(
            date_pattern,
            regex=True,
            na=False
        )
        .mean() >= 0.80
    )


# ==========================================================
# DUPLICATE REMOVAL
# ==========================================================

def remove_duplicates(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict]:

    df = df.copy()

    before = len(df)

    df = df.drop_duplicates()

    metadata = {
        "duplicates_removed":
        before - len(df)
    }

    return df, metadata


# ==========================================================
# CONSTANT FEATURE REMOVAL
# ==========================================================

def remove_constant_features(
    df: pd.DataFrame,
    target_column: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:

    df = df.copy()

    constant_cols = [
        col
        for col in df.columns
        if col != target_column
        if df[col].nunique(dropna=False) <= 1
    ]

    df.drop(
        columns=constant_cols,
        inplace=True,
        errors="ignore"
    )

    metadata = {
        "constant_columns_removed":
        constant_cols
    }

    return df, metadata


# ==========================================================
# IDENTIFIER DETECTION
# ==========================================================

def detect_identifier_columns(
    df: pd.DataFrame,
    threshold: float = 0.95,
    target_column: Optional[str] = None
) -> List[str]:

    identifier_cols = []

    for col in df.columns:
        if col == target_column:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        unique_ratio = (
            df[col].nunique(dropna=False)
            / len(df)
        )

        if unique_ratio >= threshold:

            identifier_cols.append(col)

    return identifier_cols


# ==========================================================
# DATETIME FEATURE ENGINEERING
# ==========================================================

def engineer_datetime_features(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    datetime_columns: Optional[List[str]] = None,
    drop_original: bool = True
) -> Tuple[pd.DataFrame, Dict]:

    df = df.copy()

    created_features = []
    parsed_columns = []
    skipped_columns = []

    candidate_columns = (
        datetime_columns
        if datetime_columns is not None
        else list(df.columns)
    )

    for col in candidate_columns:
        if col == target_column:
            continue

        if col not in df.columns:
            skipped_columns.append(col)
            continue

        if pd.api.types.is_datetime64_any_dtype(
            df[col]
        ):

            dt = df[col]

        else:
            if not (
                pd.api.types.is_object_dtype(df[col])
                or pd.api.types.is_string_dtype(df[col])
            ):
                skipped_columns.append(col)
                continue

            if not _looks_like_datetime(df[col]):
                skipped_columns.append(col)
                continue

            try:
                dt = pd.to_datetime(
                    df[col],
                    errors="coerce"
                )
            except Exception:
                skipped_columns.append(col)
                continue

            parse_rate = float(dt.notna().mean())

            if parse_rate < 0.90:
                skipped_columns.append(col)
                continue

        df[f"{col}_year"] = dt.dt.year
        df[f"{col}_month"] = dt.dt.month
        df[f"{col}_day"] = dt.dt.day
        df[f"{col}_weekday"] = dt.dt.weekday
        df[f"{col}_quarter"] = dt.dt.quarter

        created_features.extend([
            f"{col}_year",
            f"{col}_month",
            f"{col}_day",
            f"{col}_weekday",
            f"{col}_quarter"
        ])

        parsed_columns.append(col)

        if drop_original:
            df.drop(
                columns=[col],
                inplace=True
            )

    metadata = {
        "datetime_features_created":
        created_features,
        "datetime_columns_parsed":
        parsed_columns,
        "datetime_columns_skipped":
        skipped_columns,
        "drop_original_datetime_columns":
        drop_original
    }

    return df, metadata


# ==========================================================
# MISSING VALUE HANDLING
# ==========================================================

def handle_missing_values(
    df: pd.DataFrame,
    target_column: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:

    df = df.copy()

    metadata = {
        "missing_before":
        int(df.isna().sum().sum()),
        "numerical_strategy": {},
        "categorical_strategy":
        "most_frequent",
        "imputers": {},
        "target_column_ignored":
        target_column
    }

    numerical_cols = df.select_dtypes(
        include="number"
    ).columns.drop(
        [target_column],
        errors="ignore"
    )

    categorical_cols = df.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.drop(
        [target_column],
        errors="ignore"
    )

    for col in numerical_cols:

        if df[col].isna().sum() == 0:
            continue

        try:

            col_skew = skew(
                df[col].dropna(),
                bias = False
            )

        except Exception:

            col_skew = 0

        strategy = (
            "median"
            if abs(col_skew) > 1
            else "mean"
        )

        imputer = SimpleImputer(
            strategy=strategy
        )

        df[[col]] = imputer.fit_transform(
            df[[col]]
        )

        metadata[
            "numerical_strategy"
        ][col] = strategy
        metadata["imputers"][col] = imputer

    for col in categorical_cols:

        imputer = SimpleImputer(
            strategy="most_frequent"
        )

        df[[col]] = (
            imputer.fit_transform(
                df[[col]]
            )
        )
        metadata["imputers"][col] = imputer

    metadata["missing_after"] = int(
        df.isna().sum().sum()
    )

    return df, metadata


def transform_missing_values(
    df: pd.DataFrame,
    imputers: Dict[str, Any],
    target_column: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()
    transformed_columns = []

    for col, imputer in imputers.items():
        if col == target_column or col not in df.columns:
            continue

        df[[col]] = imputer.transform(df[[col]])
        transformed_columns.append(col)

    metadata = {
        "missing_after":
        int(df.isna().sum().sum()),
        "columns_transformed":
        transformed_columns,
        "target_column_ignored":
        target_column
    }

    return df, metadata


# ==========================================================
# OUTLIER CAPPING
# ==========================================================

def cap_outliers_iqr(
    df: pd.DataFrame,
    multiplier: float = 1.5,
    target_column: Optional[str] = None,
    bounds: Optional[Dict[str, Dict[str, float]]] = None
) -> Tuple[pd.DataFrame, Dict]:

    df = df.copy()

    metadata = {}

    numerical_cols = df.select_dtypes(
        include="number"
    ).columns.drop(
        [target_column],
        errors="ignore"
    )

    for col in numerical_cols:

        if bounds and col in bounds:
            lower = bounds[col]["lower_bound"]
            upper = bounds[col]["upper_bound"]
        else:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr

        outlier_count = (
            (
                df[col] < lower
            ) |
            (
                df[col] > upper
            )
        ).sum()

        df[col] = df[col].clip(
            lower,
            upper
        )

        metadata[col] = {
            "outliers_capped":
            int(outlier_count),
            "lower_bound":
            float(lower),
            "upper_bound":
            float(upper),
            "fitted":
            not bool(bounds and col in bounds)
        }

    return df, metadata


# ==========================================================
# FREQUENCY ENCODING
# ==========================================================

def frequency_encode_column(
    train_series: pd.Series,
    test_series: pd.Series,
    freq_map: Optional[Dict[Any, float]] = None
):

    if freq_map is None:
        freq_map = (
            train_series
            .value_counts(normalize=True)
            .to_dict()
        )

    train_encoded = (
        train_series.map(freq_map)
    )

    test_encoded = (
        test_series.map(freq_map)
        .fillna(0)
    )

    return (
        train_encoded,
        test_encoded,
        freq_map
    )


# ==========================================================
# CATEGORICAL ENCODING
# ==========================================================

def encode_categorical_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cardinality_threshold: int = 15,
    encoders: Optional[Dict[str, Any]] = None,
    frequency_maps: Optional[Dict[str, Dict[Any, float]]] = None
):

    X_train = X_train.copy()
    X_test = X_test.copy()

    metadata = {}
    fitted_encoders = {}
    fitted_frequency_maps = {}
    encoders = encoders or {}
    frequency_maps = frequency_maps or {}

    categorical_cols = (
        X_train
        .select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        )
        .columns
    )

    for col in categorical_cols:

        unique_count = (
            X_train[col]
            .nunique()
        )

        # ----------------------------------
        # LOW CARDINALITY -> ONE HOT
        # ----------------------------------

        if unique_count <= cardinality_threshold:

            if col in encoders:
                encoder = encoders[col]
                train_encoded = (
                    encoder.transform(
                        X_train[[col]]
                    )
                )
            else:
                encoder = OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )

                train_encoded = encoder.fit_transform(
                    X_train[[col]]
                )
                fitted_encoders[col] = encoder

            train_encoded = np.asarray(train_encoded)
            

            test_encoded = (
                encoder.transform(
                    X_test[[col]]
                )
            )

            test_encoded = np.asarray(test_encoded)

            feature_names = (
                encoder
                .get_feature_names_out(
                    [col]
                ).tolist()
            )

            train_encoded = pd.DataFrame(
                train_encoded,
                columns=feature_names,
                index=X_train.index
            )

            test_encoded = pd.DataFrame(
                test_encoded,
                columns=feature_names,
                index=X_test.index
            )

            X_train.drop(
                columns=[col],
                inplace=True
            )

            X_test.drop(
                columns=[col],
                inplace=True
            )

            X_train = pd.concat(
                [
                    X_train,
                    train_encoded
                ],
                axis=1
            )

            X_test = pd.concat(
                [
                    X_test,
                    test_encoded
                ],
                axis=1
            )

            metadata[col] = {
                "encoding":
                "one_hot",
                "encoded_columns":
                feature_names
            }

        # ----------------------------------
        # HIGH CARDINALITY
        # ----------------------------------

        else:

            (
                train_encoded,
                test_encoded,
                freq_map
            ) = frequency_encode_column(
                X_train[col],
                X_test[col],
                frequency_maps.get(col)
            )

            X_train[col] = train_encoded
            X_test[col] = test_encoded

            metadata[col] = {
                "encoding":
                "frequency",
                "unknown_category_value":
                0
            }
            fitted_frequency_maps[col] = freq_map

    metadata["_artifacts"] = {
        "one_hot_encoders":
        {**encoders, **fitted_encoders},
        "frequency_maps":
        {**frequency_maps, **fitted_frequency_maps}
    }

    return (
        X_train,
        X_test,
        metadata
    )


# ==========================================================
# FEATURE SCALING
# ==========================================================

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_type: str = "robust",
    scaler: Optional[Any] = None,
    identifier_columns: Optional[List[str]] = None
):

    X_train = X_train.copy()
    X_test = X_test.copy()

    identifier_cols = identifier_columns or (
        detect_identifier_columns(
            X_train
        )
    )

    numerical_cols = [
        col
        for col in
        X_train.select_dtypes(
            include="number"
        ).columns
        if col not in identifier_cols
    ]

    if scaler is not None:
        X_train[numerical_cols] = (
            scaler.transform(
                X_train[numerical_cols]
            )
        )

        X_test[numerical_cols] = (
            scaler.transform(
                X_test[numerical_cols]
            )
        )

        metadata = {
            "scaler":
            scaler.__class__.__name__,
            "scaled_columns":
            numerical_cols,
            "ignored_identifier_columns":
            identifier_cols,
            "fitted":
            False
        }

        return (
            X_train,
            X_test,
            scaler,
            metadata
        )

    if scaler_type == "standard":

        scaler = StandardScaler()

    elif scaler_type == "minmax":

        scaler = MinMaxScaler()

    elif scaler_type == "robust":

        scaler = RobustScaler()

    else:

        raise ValueError(
            f"Unsupported scaler: "
            f"{scaler_type}"
        )

    X_train[numerical_cols] = (
        scaler.fit_transform(
            X_train[numerical_cols]
        )
    )

    X_test[numerical_cols] = (
        scaler.transform(
            X_test[numerical_cols]
        )
    )

    metadata = {
        "scaler":
        scaler_type,
        "scaled_columns":
        numerical_cols,
        "ignored_identifier_columns":
        identifier_cols,
        "fitted":
        True
    }

    return (
        X_train,
        X_test,
        scaler,
        metadata
    )
