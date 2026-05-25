import numpy as np
import pandas as pd


POINT_IN_TIME_FEATURES = [
    "user_total_interactions",
    "user_total_sessions",
    "item_total_interactions",
    "item_unique_users",
    "item_avg_price",
    "hour_of_day",
    "day_of_week",
    "user_session_interaction_count",
    "item_session_popularity",
    "recalled_by_long_term",
    "recalled_by_session",
]


def add_point_in_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction-level features using only events that happened before each row.

    These columns are intended for model training. Snapshot feature tables are still
    produced separately for offline serving lookup.
    """
    required_cols = {"event_time", "user_id", "product_id", "custom_session_id"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns for point-in-time features: {sorted(missing_cols)}")

    result = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(result["event_time"]):
        result["event_time"] = pd.to_datetime(result["event_time"], utc=True)

    result["_original_order"] = np.arange(len(result))
    result = result.sort_values(
        by=["event_time", "user_id", "custom_session_id", "product_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    result["user_total_interactions"] = result.groupby("user_id").cumcount()
    result["item_total_interactions"] = result.groupby("product_id").cumcount()

    session_order = (
        result[["user_id", "custom_session_id", "event_time"]]
        .drop_duplicates(["user_id", "custom_session_id"])
        .sort_values(["user_id", "event_time", "custom_session_id"], kind="mergesort")
    )
    session_order["user_total_sessions"] = session_order.groupby("user_id").cumcount()
    result = result.merge(
        session_order[["user_id", "custom_session_id", "user_total_sessions"]],
        on=["user_id", "custom_session_id"],
        how="left",
    )

    first_user_item_event = ~result.duplicated(["product_id", "user_id"])
    result["item_unique_users"] = (
        first_user_item_event.groupby(result["product_id"]).cumsum()
        - first_user_item_event.astype(int)
    )

    if "price" in result.columns:
        price = pd.to_numeric(result["price"], errors="coerce")
        prior_price_sum = price.groupby(result["product_id"]).cumsum() - price.fillna(0)
        prior_price_count = price.notna().groupby(result["product_id"]).cumsum() - price.notna().astype(int)
        result["item_avg_price"] = (prior_price_sum / prior_price_count.replace(0, np.nan)).fillna(0)
    else:
        result["item_avg_price"] = 0.0

    result["hour_of_day"] = result["event_time"].dt.hour
    result["day_of_week"] = result["event_time"].dt.dayofweek

    # Calculate user_session_interaction_count (point-in-time)
    result["user_session_interaction_count"] = result.groupby(["user_id", "custom_session_id"]).cumcount()

    # Calculate item_session_popularity (point-in-time)
    first_item_session = ~result.duplicated(["product_id", "custom_session_id"])
    result["item_session_popularity"] = (
        first_item_session.groupby(result["product_id"]).cumsum()
        - first_item_session.astype(int)
    )

    # In training data (historical interactions), every row represents a session event,
    # so we assume it could be recalled by both channels.
    result["recalled_by_long_term"] = 1.0
    result["recalled_by_session"] = 1.0

    result[POINT_IN_TIME_FEATURES] = result[POINT_IN_TIME_FEATURES].fillna(0)
    return result.sort_values("_original_order").drop(columns=["_original_order"]).reset_index(drop=True)


def extract_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the latest user feature snapshot for serving.
    """
    user_features = df.groupby("user_id").size().reset_index(name="user_total_interactions")

    if "custom_session_id" in df.columns:
        user_sessions = (
            df.groupby("user_id")["custom_session_id"]
            .nunique()
            .reset_index(name="user_total_sessions")
        )
        user_features = user_features.merge(user_sessions, on="user_id", how="left")

    return user_features


def extract_item_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the latest item feature snapshot for serving.
    """
    item_features = df.groupby("product_id").size().reset_index(name="item_total_interactions")

    item_users = (
        df.groupby("product_id")["user_id"]
        .nunique()
        .reset_index(name="item_unique_users")
    )
    item_features = item_features.merge(item_users, on="product_id", how="left")

    if "price" in df.columns:
        item_price = (
            df.groupby("product_id")["price"]
            .mean()
            .reset_index(name="item_avg_price")
        )
        item_features = item_features.merge(item_price, on="product_id", how="left")

    # Extract static features category_code and brand from the latest event per product
    if "category_code" in df.columns or "brand" in df.columns:
        cols_to_keep = [col for col in ["product_id", "category_code", "brand"] if col in df.columns]
        static_info = df.sort_values("event_time").drop_duplicates("product_id", keep="last")[cols_to_keep]
        item_features = item_features.merge(static_info, on="product_id", how="left")

    # Extract unique sessions per item for item_session_popularity
    if "custom_session_id" in df.columns:
        item_sessions = (
            df.groupby("product_id")["custom_session_id"]
            .nunique()
            .reset_index(name="item_session_popularity")
        )
        item_features = item_features.merge(item_sessions, on="product_id", how="left")

    return item_features

