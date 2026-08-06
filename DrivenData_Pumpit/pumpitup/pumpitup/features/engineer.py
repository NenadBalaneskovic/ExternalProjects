import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Full feature engineering pipeline for Pump-It-Up competition.
    Handles:
    - cleaning
    - derived features
    - encoding
    - scikit-learn compatibility
    """

    def __init__(self):
        # High-cardinality categorical features
        self.high_card_cols = [
            "funder", "installer", "subvillage", "ward", "lga"
        ]
        self.target_encoders = {
            col: TargetEncoder(cols=[col], smoothing=0.3)
            for col in self.high_card_cols
        }

        # Low-cardinality categorical features
        self.low_card_cols = [
            "basin", "region", "management", "management_group",
            "payment", "payment_type", "water_quality",
            "quality_group", "quantity", "quantity_group",
            "source", "source_type", "source_class",
            "waterpoint_type", "waterpoint_type_group",

            # Missing columns added
            "wpt_name",
            "recorded_by",
            "scheme_name",
            "scheme_management",   # ← NEW FIX
            "extraction_type",
            "extraction_type_group",
            "extraction_type_class"
        ]

        # Numerical columns
        self.num_cols = [
            "amount_tsh", "gps_height", "longitude", "latitude",
            "population", "construction_year"
        ]

    # ---------------------------------------------------------
    # Cleaning
    # ---------------------------------------------------------
    def clean(self, df):
        df = df.copy()

        # Replace zeros with NaN for known problematic fields
        zero_as_nan = [
            "gps_height", "longitude", "latitude",
            "population", "construction_year"
        ]
        for col in zero_as_nan:
            df[col] = df[col].replace(0, np.nan)

        # Fill missing categoricals
        for col in self.low_card_cols + self.high_card_cols:
            df[col] = df[col].fillna("unknown")

        # Fill missing numericals with median
        for col in self.num_cols:
            df[col] = df[col].fillna(df[col].median())

        return df

    # ---------------------------------------------------------
    # Derived features
    # ---------------------------------------------------------
    def add_derived_features(self, df):
        df = df.copy()
    
        # Extract year and month from date_recorded
        df["recorded_year"] = pd.to_datetime(df["date_recorded"]).dt.year
        df["recorded_month"] = pd.to_datetime(df["date_recorded"]).dt.month
    
        # Drop original datetime column (critical fix)
        df = df.drop(columns=["date_recorded"])
    
        # Pump age
        df["pump_age"] = df["recorded_year"] - df["construction_year"]
        df["pump_age"] = df["pump_age"].clip(lower=0, upper=100)
    
        # Binary indicators
        df["has_scheme"] = df["scheme_name"].notna().astype(int)
        df["has_permit"] = df["permit"].astype(int)
    
        return df


    # ---------------------------------------------------------
    # Encoding
    # ---------------------------------------------------------
    def fit(self, df, y=None):
        df = self.clean(df)
        df = self.add_derived_features(df)

        # Fit target encoders
        if y is not None:
            for col, enc in self.target_encoders.items():
                enc.fit(df[col], y)

        return self

    def transform(self, df):
        df = self.clean(df)
        df = self.add_derived_features(df)
    
        # Apply target encoders correctly
        for col, enc in self.target_encoders.items():
            encoded = enc.transform(df[[col]])      # MUST pass DataFrame
            df[col] = encoded[col].astype(float).values  # extract 1-D array
    
        # Ordinal encoding for low-cardinality categoricals
        for col in self.low_card_cols:
            df[col] = df[col].astype("category").cat.codes
    
        # Drop raw categorical columns CatBoost cannot handle
        drop_cols = ["permit", "scheme_name"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
        # Keep only numeric columns
        df = df.select_dtypes(include=[np.number])
    
        return df



