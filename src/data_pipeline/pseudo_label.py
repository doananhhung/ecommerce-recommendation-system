import pandas as pd

try:
    from src.data_pipeline.featurizer import POINT_IN_TIME_FEATURES
except ModuleNotFoundError:
    from featurizer import POINT_IN_TIME_FEATURES


EVENT_LABELS = {"view": 0, "cart": 1, "purchase": 1}
EVENT_SAMPLE_WEIGHTS = {"view": 0.1, "cart": 0.5, "purchase": 1.0}
EVENT_STRENGTH = {"view": 0, "cart": 1, "purchase": 2}


def apply_pseudo_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create one training row per session-user-product and keep the strongest signal.
    """
    required_cols = {"custom_session_id", "product_id", "user_id", "event_type", "event_time"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns for pseudo-labeling: {sorted(missing_cols)}")

    work_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(work_df["event_time"]):
        work_df["event_time"] = pd.to_datetime(work_df["event_time"], utc=True)

    work_df["label"] = work_df["event_type"].map(EVENT_LABELS).fillna(0).astype(int)
    work_df["sample_weight"] = work_df["event_type"].map(EVENT_SAMPLE_WEIGHTS).fillna(0.1).astype(float)
    work_df["_event_strength"] = work_df["event_type"].map(EVENT_STRENGTH).fillna(0).astype(int)

    group_cols = ["custom_session_id", "product_id", "user_id"]
    context_cols = [
        col
        for col in ["event_time", "event_type", "category_id", "category_code", "brand", "price"]
        if col in work_df.columns
    ]
    feature_cols = [col for col in POINT_IN_TIME_FEATURES if col in work_df.columns]

    best_rows = (
        work_df.sort_values(
            group_cols + ["_event_strength", "event_time"],
            kind="mergesort",
        )
        .groupby(group_cols, as_index=False)
        .tail(1)
    )

    output_cols = group_cols + context_cols + feature_cols + ["label", "sample_weight"]
    labeled_df = best_rows[output_cols].copy()
    return labeled_df.sort_values("event_time").reset_index(drop=True)
