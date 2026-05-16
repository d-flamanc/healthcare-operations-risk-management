
# =========================
# Block 00: Imports + Config
# =========================

import numpy as np
import pandas as pd
import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List

from IPython.display import display

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.utils.class_weight import compute_sample_weight

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -------------------------
# Decision + workload knobs
# -------------------------
HIGH_UTIL_THRESHOLD = 0.85
CALIBRATION_QUANTILE = 0.95
SAFETY_WEIGHT = 5
MAX_FN_ALLOWED = 1
AUTO_PRECISION_TARGET = 0.80
MAX_FLAG_RATE = 0.25

HORIZONS = [1, 2, 3, 4, 5, 6, 7]
REVIEW_GRID = np.round(np.arange(0.01, 0.31, 0.01), 2)
AUTO_GRID = np.round(np.arange(0.50, 0.99, 0.01), 2)

VAL_WINDOW_DAYS = 180
STEP_DAYS = 90
MIN_TRAIN_DAYS = 365
MAX_FOLDS = 12

COALITION_COUNT = 4
MIN_STAFFED_CAPACITY = 20

# -------------------------
# Operational state enums
# -------------------------
class OpsDecision(str, Enum):
    NO_ACTION = "NO_ACTION"
    REVIEW = "REVIEW"
    AUTO_ESCALATE = "AUTO_ESCALATE"
    CRISIS = "CRISIS"

@dataclass(frozen=True)
class PolicyLock:
    thr_review: float
    thr_auto: float
    auto_enabled: bool

@dataclass(frozen=True)
class CrisisFlags:
    crisis_declared: bool = False
    critical_resource_shortage: bool = False

# -------------------------
# Core policy logic (FIXED)
# -------------------------
def apply_two_tier_policy(p_gbm, p_svm, lock: PolicyLock):
    """
    AUTO: both models >= thr_auto (if enabled)
    REVIEW: either model >= thr_review and not AUTO
    """
    p_gbm = np.asarray(p_gbm)
    p_svm = np.asarray(p_svm)

    if lock.auto_enabled:
        auto = (p_gbm >= lock.thr_auto) & (p_svm >= lock.thr_auto)
    else:
        auto = np.zeros_like(p_gbm, dtype=bool)

    # FIX: OR logic was broken in your file; must be "|" not line-break
    review = ((p_gbm >= lock.thr_review) | (p_svm >= lock.thr_review)) & (~auto)

    decision = np.select(
        [auto, review],
        [OpsDecision.AUTO_ESCALATE.value, OpsDecision.REVIEW.value],
        default=OpsDecision.NO_ACTION.value
    )

    return decision, auto, review

def crisis_override(decision, flags: CrisisFlags):
    if flags.crisis_declared or flags.critical_resource_shortage:
        return np.array([OpsDecision.CRISIS.value] * len(decision), dtype=object)
    return decision

# -------------------------
# Utility: confusion metrics
# -------------------------
def confusion_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "recall": recall, "precision": precision}

# -------------------------
# Utility: probability extraction
# -------------------------
def proba_high(pipeline: Pipeline, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    proba = pipeline.predict_proba(X)
    classes = pipeline.named_steps["clf"].classes_
    pos_idx = list(classes).index(1)
    return proba[:, pos_idx], proba


# In[112]:


# ======================
# Block 01: Load raw data
# ======================

FILE_PATH = r"C:\Users\flemid9\OneDrive - Caterpillar\Desktop\SEIS-763 Project\healthcare_data.csv"  # CSV file path.
df_raw = pd.read_csv(FILE_PATH)  # Load CSV to raw dataframe.

display(df_raw.head())  # Preview raw data rows.
display(pd.DataFrame({"rows": [df_raw.shape[0]], "columns": [df_raw.shape[1]]}))  # Show shape summary.


# In[113]:


# ==============================
# Block 02: Clean and normalize
# ==============================

df_clean = df_raw.copy()

TEXT_COLS = [
    "Name", "Gender", "Medical Condition", "Doctor", "Hospital",
    "Insurance Provider", "Admission Type", "Medication", "Test Results",
]

for col in TEXT_COLS:
    df_clean[col] = df_clean[col].astype(str).str.strip().str.title()

DATE_COLS = ["Date of Admission", "Discharge Date"]
for col in DATE_COLS:
    df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")

df_clean["Room Number"] = pd.to_numeric(df_clean["Room Number"], errors="coerce").astype("Int64")

df_clean["hospital_name_raw"] = df_clean["Hospital"].astype(str)

LEGAL_SUFFIXES = {"LLC", "INC", "LTD", "PLC", "GROUP", "CO", "CORP", "CORPORATION", "COMPANY"}
TOKEN_REPLACEMENTS = {"&": "AND", "INC.": "INC", "LTD.": "LTD", "PLC.": "PLC", "CO.": "CO", "CORP.": "CORP"}
SUFFIX_TYPO_MAP = {"LNC": "LLC"}

def _normalize_tokens(name: str) -> List[str]:
    s = str(name).strip().upper()
    s = s.replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = [TOKEN_REPLACEMENTS.get(t, t) for t in s.split(" ") if t]
    return tokens

def canonicalize_hospital_name(name: str) -> Tuple[str, bool, str, str]:
    tokens = _normalize_tokens(name)

    if not tokens:
        return "", True, "", "empty_name"

    typo_found = False
    suspect_reason = ""
    corrected: List[str] = []

    for t in tokens:
        if t in SUFFIX_TYPO_MAP:
            corrected.append(SUFFIX_TYPO_MAP[t])
            typo_found = True
            suspect_reason = f"typo_suffix:{t}->{SUFFIX_TYPO_MAP[t]}"
        else:
            corrected.append(t)

    tokens = corrected

    # Normalize AND SONS phrase ordering
    if ("AND" in tokens) and ("SONS" in tokens):
        root = [t for t in tokens if t not in {"AND", "SONS"}]
        if len(root) >= 1:
            tokens = root + ["AND", "SONS"]

    # Rotate legal suffix if it appears as first token
    if len(tokens) > 1 and tokens[0] in LEGAL_SUFFIXES and tokens[-1] not in LEGAL_SUFFIXES:
        tokens = tokens[1:] + [tokens[0]]

    entity_suffix = tokens[-1] if tokens[-1] in LEGAL_SUFFIXES else ""
    canonical = " ".join(tokens)
    suffix_suspect = typo_found

    return canonical, suffix_suspect, entity_suffix, suspect_reason

tmp = df_clean["hospital_name_raw"].apply(canonicalize_hospital_name)
df_clean["hospital_name_canonical"] = tmp.apply(lambda x: x[0])
df_clean["hospital_suffix_suspect"] = tmp.apply(lambda x: x[1])
df_clean["hospital_entity_suffix"] = tmp.apply(lambda x: x[2])
df_clean["hospital_suspect_reason"] = tmp.apply(lambda x: x[3])

dq = pd.DataFrame({
    "column": TEXT_COLS + DATE_COLS + ["Room Number"],
    "missing_count": [int(df_clean[c].isna().sum()) for c in (TEXT_COLS + DATE_COLS + ["Room Number"])],
    "missing_rate": [float(df_clean[c].isna().mean()) for c in (TEXT_COLS + DATE_COLS + ["Room Number"])],
})
display(dq)

canon_preview = (
    df_clean.groupby("hospital_name_canonical")["hospital_name_raw"]
    .agg(rows="size", n_raw_variants="nunique", raw_variants=lambda x: sorted(set(x))[:5])
    .reset_index()
    .sort_values("rows", ascending=False)
)
display(canon_preview.head(25))

suspects = df_clean[df_clean["hospital_suffix_suspect"]].copy()
display(suspects[["hospital_name_raw", "hospital_name_canonical", "hospital_suspect_reason"]].head(25))


# In[114]:


# ==========================================
# Block 03: Create operational coalition IDs
# ==========================================

df_model = df_clean.copy()  # Create modeling dataframe copy.
df_model["hospital_id"] = df_model["hospital_name_canonical"]  # Define hospital identifier.

coalition_labels = [f"COALITION_{i:02d}" for i in range(1, COALITION_COUNT + 1)]  # Coalition names list.
hospital_counts = df_model["hospital_id"].value_counts(dropna=False)  # Count rows per hospital.

coalition_totals = {c: 0 for c in coalition_labels}  # Initialize coalition totals.
hospital_to_coalition = {}  # Initialize hospital mapping dictionary.

for hosp, cnt in hospital_counts.items():  # Iterate hospitals by size.
    target = min(coalition_totals, key=coalition_totals.get)  # Find least-loaded coalition.
    hospital_to_coalition[hosp] = target  # Assign hospital to coalition.
    coalition_totals[target] += int(cnt)  # Update coalition row totals.

df_model["coalition_id"] = df_model["hospital_id"].map(hospital_to_coalition)  # Map hospital to coalition.
df_model["entity_id"] = df_model["coalition_id"]  # Define modeling entity identifier.
df_model["entity_level"] = "OPERATIONAL_COALITION"  # Label entity granularity.
df_model["entity_assignment_source"] = "hospital_id_balanced_greedy"  # Document assignment method.

display(pd.DataFrame({"coalition_id": list(coalition_totals.keys()), "assigned_rows": list(coalition_totals.values())}))  # Show balance.
display(df_model["coalition_id"].value_counts())  # Show coalition row counts.
display(pd.DataFrame(list(hospital_to_coalition.items()), columns=["hospital_id", "coalition_id"]).head(20))  # Preview mapping.


# In[115]:


# ==========================================
# Block 04: Build stays and event stream only
# ==========================================

# 1) Build stay-level table for occupancy reconstruction
df_occ = df_model[["entity_id", "hospital_id", "Room Number", "Date of Admission", "Discharge Date"]].copy()

# 2) Drop rows missing required fields
df_occ = df_occ.dropna(subset=["entity_id", "hospital_id", "Room Number", "Date of Admission", "Discharge Date"]).copy()

# 3) Enforce correct dtypes
df_occ["Room Number"] = df_occ["Room Number"].astype("Int64")
df_occ["Date of Admission"] = pd.to_datetime(df_occ["Date of Admission"], errors="coerce")
df_occ["Discharge Date"] = pd.to_datetime(df_occ["Discharge Date"], errors="coerce")

# 4) Drop invalid date parses
df_occ = df_occ.dropna(subset=["Date of Admission", "Discharge Date"]).copy()

# 5) Drop invalid intervals (discharge before admit)
df_occ = df_occ[df_occ["Discharge Date"] >= df_occ["Date of Admission"]].copy()

# 6) Create collision-proof room ID (critical after aggregation)
df_occ["room_uid"] = df_occ["hospital_id"].astype(str) + "::" + df_occ["Room Number"].astype(str)

# 7) Create admit events (+1 on admit date)
df_admit = df_occ.rename(columns={"Date of Admission": "date"})[["entity_id", "room_uid", "date"]].copy()
df_admit["delta"] = 1

# 8) Create discharge events (-1 on discharge date)
df_dis = df_occ.rename(columns={"Discharge Date": "date"})[["entity_id", "room_uid", "date"]].copy()
df_dis["delta"] = -1

# 9) Combine events and collapse duplicates to net delta
df_events = pd.concat([df_admit, df_dis], ignore_index=True)
df_events = (
    df_events.groupby(["entity_id", "room_uid", "date"])["delta"]
    .sum()
    .reset_index()
)

# 10) Sanity checks
display(df_events.head())

display(pd.DataFrame({
    "metric": [
        "df_occ_rows",
        "df_events_rows",
        "unique_entities",
        "unique_hospitals",
        "unique_room_uids",
        "date_min",
        "date_max"
    ],
    "value": [
        int(len(df_occ)),
        int(len(df_events)),
        int(df_occ["entity_id"].nunique()),
        int(df_occ["hospital_id"].nunique()),
        int(df_occ["room_uid"].nunique()),
        str(df_occ["Date of Admission"].min().date()),
        str(df_occ["Discharge Date"].max().date()),
    ]
}))


# In[116]:


# ==========================================
# Block 05: Build coalition-day census and flow
# ==========================================

# 1) Prepare stay intervals for occupied-day expansion
df_stays = df_occ[["entity_id", "room_uid", "Date of Admission", "Discharge Date"]].copy()
df_stays = df_stays.rename(columns={"Date of Admission": "admit_date", "Discharge Date": "discharge_date"})

# 2) Ensure datetime types (safe conversion)
df_stays["admit_date"] = pd.to_datetime(df_stays["admit_date"], errors="coerce")
df_stays["discharge_date"] = pd.to_datetime(df_stays["discharge_date"], errors="coerce")
df_stays = df_stays.dropna(subset=["admit_date", "discharge_date"]).copy()

# 3) Occupied days convention: admit_date through (discharge_date - 1 day)
df_stays["occupied_dates"] = df_stays.apply(
    lambda r: pd.date_range(r["admit_date"], r["discharge_date"] - pd.Timedelta(days=1), freq="D"),
    axis=1
)

# 4) Expand to one row per occupied room-day
df_occ_days = df_stays[["entity_id", "room_uid", "occupied_dates"]].explode("occupied_dates")
df_occ_days = df_occ_days.rename(columns={"occupied_dates": "date"}).dropna(subset=["date"]).copy()

# 5) Daily census: count unique occupied rooms per coalition-day
df_census = (
    df_occ_days.groupby(["entity_id", "date"])["room_uid"]
    .nunique()
    .reset_index(name="rooms_occupied")
)

# 6) Daily flow: admissions and discharges per coalition-day (from events)
df_flow = df_events[["entity_id", "date", "delta"]].copy()
df_flow["admissions_today"] = (df_flow["delta"] == 1).astype(int)
df_flow["discharges_today"] = (df_flow["delta"] == -1).astype(int)

df_flow_day = (
    df_flow.groupby(["entity_id", "date"])[["admissions_today", "discharges_today"]]
    .sum()
    .reset_index()
)
df_flow_day["net_flow"] = df_flow_day["admissions_today"] - df_flow_day["discharges_today"]

# 7) Merge census and flow (fill missing flow with 0)
df_entity_day = df_census.merge(df_flow_day, on=["entity_id", "date"], how="left")
df_entity_day[["admissions_today", "discharges_today", "net_flow"]] = (
    df_entity_day[["admissions_today", "discharges_today", "net_flow"]].fillna(0)
)

# 8) Calendar features (seasonality)
df_entity_day["dow"] = df_entity_day["date"].dt.dayofweek
df_entity_day["is_weekend"] = (df_entity_day["dow"] >= 5).astype(int)
df_entity_day["month"] = df_entity_day["date"].dt.month
df_entity_day["week_of_year"] = df_entity_day["date"].dt.isocalendar().week.astype(int)

df_entity_day["dow_sin"] = np.sin(2 * np.pi * df_entity_day["dow"] / 7)
df_entity_day["dow_cos"] = np.cos(2 * np.pi * df_entity_day["dow"] / 7)

# 9) Sort for downstream lag/rolling features
df_entity_day = df_entity_day.sort_values(["entity_id", "date"]).reset_index(drop=True)

# 10) Sanity checks
display(df_entity_day.head())

display(pd.DataFrame({
    "metric": ["entity_day_rows", "unique_entities", "date_min", "date_max",
               "avg_rooms_occupied", "max_rooms_occupied"],
    "value": [
        int(len(df_entity_day)),
        int(df_entity_day["entity_id"].nunique()),
        str(df_entity_day["date"].min().date()),
        str(df_entity_day["date"].max().date()),
        float(df_entity_day["rooms_occupied"].mean()),
        int(df_entity_day["rooms_occupied"].max())
    ]
}))


# In[117]:


# ==========================================
# Block 06: Calibrate staffed capacity and features
# ==========================================

# 1) Estimate staffed capacity per coalition using busy-day quantile
busy = df_entity_day.groupby("entity_id")["rooms_occupied"].quantile(CALIBRATION_QUANTILE)

# 2) Convert busy-day occupancy to staffed capacity at 85% target
staffed_capacity = np.ceil(busy / HIGH_UTIL_THRESHOLD).astype(int)

# 3) Enforce minimum capacity floor (should not bind here)
staffed_capacity = staffed_capacity.clip(lower=MIN_STAFFED_CAPACITY)

# 4) Create feature dataframe
df_feat = df_entity_day.copy()

# 5) Map staffed capacity into daily rows
df_feat["staffed_capacity"] = df_feat["entity_id"].map(staffed_capacity)

# 6) Compute staffed utilization rate
df_feat["utilization_rate"] = df_feat["rooms_occupied"] / df_feat["staffed_capacity"]

# 7) Create same-day high flag at 85% threshold
df_feat["is_high_today"] = (df_feat["utilization_rate"] >= HIGH_UTIL_THRESHOLD).astype(int)

# 8) Create gap-to-threshold feature (positive means below 85%)
df_feat["gap_to_85"] = HIGH_UTIL_THRESHOLD - df_feat["utilization_rate"]

# 9) Group for lag/rolling features
g = df_feat.groupby("entity_id")

# 10) Utilization lags (past-only)
for k in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]:
    df_feat[f"util_lag{k}"] = g["utilization_rate"].shift(k)

# 11) Net-flow lag (past-only)
df_feat["netflow_lag1"] = g["net_flow"].shift(1)

# 12) Shifted histories for rolling windows (prevents leakage)
util_hist = g["utilization_rate"].shift(1)
flow_hist = g["net_flow"].shift(1)

# 13) Rolling utilization means (past-only)
df_feat["util_roll7"] = util_hist.transform(lambda x: x.rolling(7, min_periods=3).mean())
df_feat["util_roll14"] = util_hist.transform(lambda x: x.rolling(14, min_periods=5).mean())
df_feat["util_roll28"] = util_hist.transform(lambda x: x.rolling(28, min_periods=10).mean())
df_feat["util_roll56"] = util_hist.transform(lambda x: x.rolling(56, min_periods=20).mean())

# 14) Rolling utilization volatility (past-only)
df_feat["util_vol7"] = util_hist.transform(lambda x: x.rolling(7, min_periods=3).std())
df_feat["util_vol14"] = util_hist.transform(lambda x: x.rolling(14, min_periods=5).std())

# 15) Rolling net-flow means (past-only)
df_feat["netflow_roll7"] = flow_hist.transform(lambda x: x.rolling(7, min_periods=3).mean())
df_feat["netflow_roll14"] = flow_hist.transform(lambda x: x.rolling(14, min_periods=5).mean())
df_feat["netflow_roll28"] = flow_hist.transform(lambda x: x.rolling(28, min_periods=10).mean())

# 16) Prior high-day count (past-only)
high_hist = g["is_high_today"].shift(1)
df_feat["high_count_14d"] = high_hist.transform(lambda x: x.rolling(14, min_periods=5).sum())

# 17) Fill early NaNs from lags/rolls
df_feat = df_feat.fillna(0)

# 18) Define modeling feature list (single source of truth)
FEATURE_COLS = [
    # utilization lags
    "util_lag1","util_lag2","util_lag3","util_lag4","util_lag5","util_lag6","util_lag7",
    "util_lag14","util_lag21","util_lag28",

    # utilization rolling/volatility
    "util_roll7","util_roll14","util_roll28","util_roll56","util_vol7","util_vol14",

    # flow features
    "admissions_today","discharges_today","net_flow","netflow_lag1","netflow_roll7","netflow_roll14","netflow_roll28",

    # calendar features (from Block 05)
    "month","week_of_year","is_weekend","dow_sin","dow_cos",

    # threshold aligned
    "gap_to_85","high_count_14d",
]

# 19) Diagnostics
display(pd.DataFrame({
    "metric": ["rows","unique_entities","util_mean","util_max","high_rate_today","high_count_today"],
    "value": [
        int(len(df_feat)),
        int(df_feat["entity_id"].nunique()),
        float(df_feat["utilization_rate"].mean()),
        float(df_feat["utilization_rate"].max()),
        float(df_feat["is_high_today"].mean()),
        int(df_feat["is_high_today"].sum()),
    ]
}))

# 20) Show calibrated capacity values per coalition
display(staffed_capacity.rename("staffed_capacity").to_frame())

# 21) Preview output
display(df_feat.head())


# In[118]:


# ==========================================
# Block 07: Create future high-occupancy targets
# ==========================================

# 1) Ensure proper time ordering
df_feat = df_feat.sort_values(["entity_id", "date"]).reset_index(drop=True)

# 2) Group by coalition for horizon shifting
g = df_feat.groupby("entity_id")

# 3) Create future labels (shift(-h) = h days ahead)
for h in HORIZONS:
    df_feat[f"is_high_t{h}"] = g["is_high_today"].shift(-h)

# 4) Horizon prevalence summary (sanity check)
prev_rows = []
for h in HORIZONS:
    col = f"is_high_t{h}"
    tmp = df_feat.dropna(subset=[col])
    prev_rows.append({
        "horizon_days_ahead": h,
        "rows_available": int(len(tmp)),
        "high_rate": float(tmp[col].mean()),
        "high_count": int(tmp[col].sum())
    })

prev_df = pd.DataFrame(prev_rows)
display(prev_df)

# 5) Quick preview of labels
display(
    df_feat[["entity_id", "date", "is_high_today", "is_high_t1", "is_high_t3", "is_high_t7"]]
    .head(12)
)


# In[119]:


# ==========================================
# Block 08: Time-safe split + preprocessing
# ==========================================

# 1) Select forecasting horizon for the modeling run
HORIZON = 1
TARGET_COL = f"is_high_t{HORIZON}"

# 2) Drop rows without a target (end-of-series NaNs)
df_modeling = df_feat.dropna(subset=[TARGET_COL]).copy()

# 3) Ensure date is datetime and sort for time-safe splitting
df_modeling["date"] = pd.to_datetime(df_modeling["date"], errors="coerce")
df_modeling = df_modeling.sort_values(["entity_id", "date"]).reset_index(drop=True)

# 4) Time-safe split per entity (coalition): train earliest, test latest
def time_split_3way(df_in, group_col="entity_id", time_col="date", test_frac=0.20, val_frac=0.10):
    df_in = df_in.sort_values([group_col, time_col]).copy()
    train_idx, val_idx, test_idx = [], [], []

    for _, grp in df_in.groupby(group_col):
        n = len(grp)
        cut_test = int(np.floor((1 - test_frac) * n))
        cut_val = int(np.floor((1 - test_frac - val_frac) * n))

        train_idx.extend(grp.index[:cut_val])
        val_idx.extend(grp.index[cut_val:cut_test])
        test_idx.extend(grp.index[cut_test:])

    return df_in.loc[train_idx], df_in.loc[val_idx], df_in.loc[test_idx]

train_df, val_df, test_df = time_split_3way(df_modeling)

# 5) Build feature matrices and labels
X_train = train_df[FEATURE_COLS].copy()
y_train = train_df[TARGET_COL].astype(int).copy()

X_val = val_df[FEATURE_COLS].copy()
y_val = val_df[TARGET_COL].astype(int).copy()

X_test = test_df[FEATURE_COLS].copy()
y_test = test_df[TARGET_COL].astype(int).copy()

# 6) Split prevalence sanity check
split_summary = pd.DataFrame({
    "set": ["train", "val", "test"],
    "rows": [len(y_train), len(y_val), len(y_test)],
    "high_rate": [float(y_train.mean()), float(y_val.mean()), float(y_test.mean())],
    "high_count": [int(y_train.sum()), int(y_val.sum()), int(y_test.sum())]
})
display(split_summary)

# 7) Preprocessing: numeric scaling only (all FEATURE_COLS are numeric)
preprocess = ColumnTransformer(
    transformers=[("num", StandardScaler(), FEATURE_COLS)],
    remainder="drop"
)

# 8) Quick check: no missing values in model matrices
display(pd.DataFrame({
    "dataset": ["X_train", "X_val", "X_test"],
    "missing_cells": [
        int(pd.DataFrame(X_train).isna().sum().sum()),
        int(pd.DataFrame(X_val).isna().sum().sum()),
        int(pd.DataFrame(X_test).isna().sum().sum()),
    ]
}))


# In[120]:


# ==========================================
# Block 09: Train GBM + SVM and compute probabilities
# ==========================================

# 1) Compute safety-weighted samples for GBM training
gbm_sw = compute_sample_weight(
    class_weight={0: 1, 1: SAFETY_WEIGHT},
    y=y_train
)

# 2) Build GBM pipeline (scale -> model)
gbm_model = Pipeline([
    ("prep", preprocess),
    ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE))
])

# 3) Fit GBM with sample weights (safety-first)
gbm_model.fit(X_train, y_train, clf__sample_weight=gbm_sw)

# 4) Build SVM pipeline (scale -> model)
svm_model = Pipeline([
    ("prep", preprocess),
    ("clf", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        class_weight={0: 1, 1: SAFETY_WEIGHT},
        random_state=RANDOM_STATE
    ))
])

# 5) Fit SVM (class_weight handles imbalance emphasis)
svm_model.fit(X_train, y_train)

# 6) Probability extraction helper (P(y=1))
def proba_high(pipeline, X):
    proba = pipeline.predict_proba(X)
    classes = pipeline.named_steps["clf"].classes_
    pos_idx = list(classes).index(1)
    return proba[:, pos_idx], proba

# 7) Compute probabilities on validation and test
p_gbm_val, proba_gbm_val = proba_high(gbm_model, X_val)
p_svm_val, proba_svm_val = proba_high(svm_model, X_val)

p_gbm_test, proba_gbm_test = proba_high(gbm_model, X_test)
p_svm_test, proba_svm_test = proba_high(svm_model, X_test)

# 8) Baseline evaluation at threshold 0.50 (for reference only)
def eval_at_threshold(model_name, y_true, p_high, proba_2col, thr=0.50):
    y_pred = (p_high >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    pred_rate = float(y_pred.mean())
    actual_rate = float(np.mean(y_true))
    ll = float(log_loss(y_true, proba_2col, labels=[0, 1]))

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "model": model_name,
        "threshold": thr,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "recall_high": float(recall),
        "precision_high": float(precision),
        "f1_high": float(f1),
        "predicted_high_rate": float(pred_rate),
        "actual_high_rate": float(actual_rate),
        "log_loss": ll
    }

baseline_rows = [
    eval_at_threshold("GBM_val", y_val.values, p_gbm_val, proba_gbm_val, thr=0.50),
    eval_at_threshold("SVM_val", y_val.values, p_svm_val, proba_svm_val, thr=0.50),
    eval_at_threshold("GBM_test", y_test.values, p_gbm_test, proba_gbm_test, thr=0.50),
    eval_at_threshold("SVM_test", y_test.values, p_svm_test, proba_svm_test, thr=0.50),
]

baseline_df = pd.DataFrame(baseline_rows)
display(baseline_df)


# In[121]:


# ==========================================
# Block 10: Tune thresholds under heavy REVIEW constraint
# ==========================================

# -----------------------------
# 1) Tune THR_REVIEW on validation with workload cap
#    Priority: minimize FN, then FP, then flag_rate, then higher threshold
# -----------------------------
review_rows = []

for thr_review in REVIEW_GRID:
    flagged_val = ((p_gbm_val >= thr_review) | (p_svm_val >= thr_review)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val.values, flagged_val, labels=[0, 1]).ravel()

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    review_rows.append({
        "thr_review": float(thr_review),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "recall_high": float(recall),
        "precision_high": float(precision),
        "flag_rate": float(flagged_val.mean()),
        "flag_count": int(flagged_val.sum()),
    })

df_review = pd.DataFrame(review_rows)

# Apply heavy workload cap
df_review_cap = df_review[df_review["flag_rate"] <= MAX_FLAG_RATE].copy()

cap_summary = pd.DataFrame({
    "MAX_FLAG_RATE": [MAX_FLAG_RATE],
    "options_under_cap": [int(len(df_review_cap))],
    "min_flag_rate_under_cap": [float(df_review_cap["flag_rate"].min()) if len(df_review_cap) else None],
    "min_FN_under_cap": [int(df_review_cap["FN"].min()) if len(df_review_cap) else None],
})
display(cap_summary)

if len(df_review_cap) == 0:
    # No feasible threshold under cap → pick minimum flag_rate overall
    best_review = df_review.sort_values(
        ["flag_rate", "FN", "FP", "thr_review"],
        ascending=[True, True, True, False]
    ).head(1)
    review_feasible = False
else:
    best_review = df_review_cap.sort_values(
        ["FN", "FP", "flag_rate", "thr_review"],
        ascending=[True, True, True, False]
    ).head(1)
    review_feasible = True

display(best_review)

THR_REVIEW = float(best_review["thr_review"].iloc[0])

# -----------------------------
# 2) Tune THR_AUTO on validation (AUTO requires both models >= thr_auto)
#    Priority: meet AUTO_PRECISION_TARGET; else best available precision
# -----------------------------
auto_rows = []

for thr_auto in AUTO_GRID:
    if thr_auto < THR_REVIEW:
        continue

    auto_val = (p_gbm_val >= thr_auto) & (p_svm_val >= thr_auto)
    auto_flag = auto_val.astype(int)

    tn_a, fp_a, fn_a, tp_a = confusion_matrix(y_val.values, auto_flag, labels=[0, 1]).ravel()
    auto_precision = tp_a / (tp_a + fp_a) if (tp_a + fp_a) else np.nan
    auto_recall = tp_a / (tp_a + fn_a) if (tp_a + fn_a) else 0.0

    auto_rows.append({
        "thr_auto": float(thr_auto),
        "AUTO_TP": int(tp_a),
        "AUTO_FP": int(fp_a),
        "AUTO_count": int(auto_val.sum()),
        "AUTO_precision": float(auto_precision) if not np.isnan(auto_precision) else np.nan,
        "AUTO_recall": float(auto_recall),
    })

df_auto = pd.DataFrame(auto_rows)

df_auto_ok = df_auto[df_auto["AUTO_precision"].fillna(0) >= AUTO_PRECISION_TARGET].copy()

auto_summary = pd.DataFrame({
    "AUTO_PRECISION_TARGET": [AUTO_PRECISION_TARGET],
    "options_meeting_target": [int(len(df_auto_ok))],
    "max_AUTO_TP_meeting_target": [int(df_auto_ok["AUTO_TP"].max()) if len(df_auto_ok) else None],
})
display(auto_summary)

if len(df_auto_ok) == 0:
    best_auto = df_auto.sort_values(
        ["AUTO_precision", "AUTO_TP", "AUTO_count", "thr_auto"],
        ascending=[False, False, True, False]
    ).head(1)
    auto_met_target = False
else:
    best_auto = df_auto_ok.sort_values(
        ["AUTO_TP", "AUTO_count", "thr_auto"],
        ascending=[False, True, False]
    ).head(1)
    auto_met_target = True

display(best_auto)

THR_AUTO = float(best_auto["thr_auto"].iloc[0])

# -----------------------------
# 3) Apply 3-level policy on TEST and summarize
#    AUTO if both >= THR_AUTO
#    REVIEW if either >= THR_REVIEW and not AUTO
# -----------------------------
auto_test = (p_gbm_test >= THR_AUTO) & (p_svm_test >= THR_AUTO)
review_test = ((p_gbm_test >= THR_REVIEW) | (p_svm_test >= THR_REVIEW)) & (~auto_test)
flagged_test = (auto_test | review_test).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test.values, flagged_test, labels=[0, 1]).ravel()
recall = tp / (tp + fn) if (tp + fn) else 0.0
precision = tp / (tp + fp) if (tp + fp) else 0.0

# AUTO precision on test
tn_a, fp_a, fn_a, tp_a = confusion_matrix(y_test.values, auto_test.astype(int), labels=[0, 1]).ravel()
auto_precision_test = tp_a / (tp_a + fp_a) if (tp_a + fp_a) else np.nan

policy_summary = pd.DataFrame({
    "THR_REVIEW": [THR_REVIEW],
    "THR_AUTO": [THR_AUTO],
    "review_feasible_under_cap_val": [bool(review_feasible)],
    "auto_met_precision_target_val": [bool(auto_met_target)],
    "TP": [int(tp)], "FP": [int(fp)], "FN": [int(fn)], "TN": [int(tn)],
    "recall_high": [float(recall)],
    "precision_high": [float(precision)],
    "flag_rate_test": [float(flagged_test.mean())],
    "auto_count_test": [int(auto_test.sum())],
    "review_count_test": [int(review_test.sum())],
    "AUTO_precision_test": [float(auto_precision_test) if not np.isnan(auto_precision_test) else np.nan],
    "test_high_count": [int(y_test.sum())],
    "test_rows": [int(len(y_test))]
})
display(policy_summary)

decision_counts = pd.Series(
    np.select([auto_test, review_test], ["AUTO_ESCALATE", "REVIEW"], default="NO_ACTION")
).value_counts()
display(decision_counts)


# In[143]:


# ======================================================
# Block 12B: Apply locked THR_REVIEW per horizon on TEST
#   - Train GBM+SVM per horizon (time-safe split)
#   - Use THR_REVIEW_LOCK_median (or p75) from 12A
#   - Compute policy metrics + ROC/AUC per horizon
#   - Plot workload/safety/performance across horizons
# ======================================================

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------
# 1) Choose which lock to use
#    - median = typical operating point
#    - p75 = more conservative (lower workload, lower recall)
# -----------------------------
USE_LOCK = "median"   # options: "median" or "p75"
LOCK_COL = "THR_REVIEW_LOCK_median" if USE_LOCK == "median" else "THR_REVIEW_LOCK_p75"

# -----------------------------
# 2) Optional AUTO settings (simple, stable)
#    We keep AUTO fixed and guardrail it.
#    AUTO enabled only if enough val positives and precision meets target.
# -----------------------------
THR_AUTO_FIXED = 0.87
MIN_VAL_HIGHS_FOR_AUTO = 20
MIN_AUTO_COUNT_FOR_AUTO = 10

def auto_supported(y_val, p_gbm_val, p_svm_val, thr_auto, precision_target):
    auto_val = (p_gbm_val >= thr_auto) & (p_svm_val >= thr_auto)
    auto_count = int(auto_val.sum())
    val_high_count = int(np.sum(y_val))

    if val_high_count < MIN_VAL_HIGHS_FOR_AUTO:
        return False, auto_count, np.nan

    if auto_count < MIN_AUTO_COUNT_FOR_AUTO:
        return False, auto_count, np.nan

    tn, fp, fn, tp = confusion_matrix(y_val, auto_val.astype(int), labels=[0, 1]).ravel()
    auto_precision = tp / (tp + fp) if (tp + fp) else np.nan

    return (auto_precision >= precision_target), auto_count, auto_precision

# -----------------------------
# 3) Run across horizons
# -----------------------------
rows = []

for H in HORIZONS:
    target_col = f"is_high_t{H}"

    # Drop missing targets for this horizon
    df_modeling = df_feat.dropna(subset=[target_col]).copy()
    df_modeling = df_modeling.sort_values(["entity_id", "date"]).reset_index(drop=True)

    # Split (time-safe per coalition)
    train_df, val_df, test_df = time_split_3way(df_modeling, "entity_id", "date")

    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df[target_col].astype(int).values

    X_val = val_df[FEATURE_COLS].copy()
    y_val = val_df[target_col].astype(int).values

    X_test = test_df[FEATURE_COLS].copy()
    y_test_h = test_df[target_col].astype(int).values

    # Train GBM
    gbm_sw = compute_sample_weight(class_weight={0: 1, 1: SAFETY_WEIGHT}, y=y_train)
    gbm_model_h = Pipeline([
        ("prep", preprocess),
        ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE))
    ])
    gbm_model_h.fit(X_train, y_train, clf__sample_weight=gbm_sw)

    # Train SVM
    svm_model_h = Pipeline([
        ("prep", preprocess),
        ("clf", SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True,
            class_weight={0: 1, 1: SAFETY_WEIGHT},
            random_state=RANDOM_STATE
        ))
    ])
    svm_model_h.fit(X_train, y_train)

    # Probabilities
    p_gbm_val_h, _ = proba_high(gbm_model_h, X_val)
    p_svm_val_h, _ = proba_high(svm_model_h, X_val)

    p_gbm_test_h, _ = proba_high(gbm_model_h, X_test)
    p_svm_test_h, _ = proba_high(svm_model_h, X_test)

    # Locked THR_REVIEW for this horizon
    thr_review_lock = float(review_lock_by_horizon_df.loc[review_lock_by_horizon_df["horizon"] == H, LOCK_COL].iloc[0])

    # Auto guardrail (optional)
    auto_ok, auto_count_val, auto_precision_val = auto_supported(
        y_val, p_gbm_val_h, p_svm_val_h, THR_AUTO_FIXED, AUTO_PRECISION_TARGET
    )

    # Apply policy on test
    auto_test = (p_gbm_test_h >= THR_AUTO_FIXED) & (p_svm_test_h >= THR_AUTO_FIXED) if auto_ok else np.zeros_like(y_test_h, dtype=bool)
    review_test = ((p_gbm_test_h >= thr_review_lock) | (p_svm_test_h >= thr_review_lock)) & (~auto_test)
    flagged_test = (auto_test | review_test).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test_h, flagged_test, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    flag_rate = float(flagged_test.mean())

    # AUCs (continuous scores)
    score_review = np.maximum(p_gbm_test_h, p_svm_test_h)
    auc_gbm = roc_auc_score(y_test_h, p_gbm_test_h)
    auc_svm = roc_auc_score(y_test_h, p_svm_test_h)
    auc_review = roc_auc_score(y_test_h, score_review)

    # AUTO precision on test (if auto enabled)
    if auto_ok and int(auto_test.sum()) > 0:
        tn_a, fp_a, fn_a, tp_a = confusion_matrix(y_test_h, auto_test.astype(int), labels=[0, 1]).ravel()
        auto_precision_test = tp_a / (tp_a + fp_a) if (tp_a + fp_a) else np.nan
    else:
        auto_precision_test = np.nan

    rows.append({
        "horizon": H,
        "THR_REVIEW_LOCK": thr_review_lock,
        "AUTO_enabled": bool(auto_ok),
        "AUTO_count_val": int(auto_count_val),
        "AUTO_precision_val": float(auto_precision_val) if not np.isnan(auto_precision_val) else np.nan,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "recall_high": float(recall),
        "precision_high": float(precision),
        "flag_rate_test": float(flag_rate),
        "auto_count_test": int(auto_test.sum()),
        "review_count_test": int(review_test.sum()),
        "AUC_gbm": float(auc_gbm),
        "AUC_svm": float(auc_svm),
        "AUC_review": float(auc_review),
        "test_high_count": int(y_test_h.sum()),
        "test_rows": int(len(y_test_h)),
    })

results_locked_by_horizon_df = pd.DataFrame(rows)
display(results_locked_by_horizon_df)


# In[144]:


# -----------------------------
# 4) Plot: Flag rate vs cap (TEST) — Color Standard
#   Workload (flag rate) = Purple
# -----------------------------
import matplotlib.pyplot as plt

COLOR_WORKLOAD = "#9c27b0"  # purple
COLOR_TEXT = "black"

fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
ax.set_facecolor("white")

ax.plot(
    results_locked_by_horizon_df["horizon"],
    results_locked_by_horizon_df["flag_rate_test"],
    marker="o",
    linewidth=2,
    color=COLOR_WORKLOAD,
    label="Review workload (Flag rate, TEST)"
)

ax.axhline(
    MAX_FLAG_RATE,
    color=COLOR_WORKLOAD,
    linestyle="--",
    linewidth=1.5,
    alpha=0.9,
    label="Workload cap"
)

ax.set_xlabel("Horizon (days ahead)", color=COLOR_TEXT)
ax.set_ylabel("Flag rate (AUTO + REVIEW)", color=COLOR_TEXT)
ax.set_title(f"Workload on TEST using {USE_LOCK.upper()} REVIEW locks", color=COLOR_TEXT)

# Black ticks/spines
ax.tick_params(colors=COLOR_TEXT)
for spine in ax.spines.values():
    spine.set_color(COLOR_TEXT)

leg = ax.legend(frameon=True)
leg.get_frame().set_facecolor("white")
leg.get_frame().set_edgecolor("black")
for t in leg.get_texts():
    t.set_color("black")

ax.grid(True, axis="y", alpha=0.25, color="gray")

plt.tight_layout()
plt.show()


# In[145]:


# -----------------------------
# 5) Plot: Recall & Precision vs horizon (TEST) — Color Standard
#   Recall (Sensitivity) = Blue
#   Precision = Green (kept consistent with "good/positive" line)
# -----------------------------
import matplotlib.pyplot as plt

COLOR_RECALL = "#5b9bd5"  # blue
COLOR_PREC = "#2e7d32"    # green
COLOR_TEXT = "black"

fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
ax.set_facecolor("white")

ax.plot(
    results_locked_by_horizon_df["horizon"],
    results_locked_by_horizon_df["recall_high"],
    marker="o",
    linewidth=2,
    color=COLOR_RECALL,
    label="Sensitivity (Recall, HIGH)"
)

ax.plot(
    results_locked_by_horizon_df["horizon"],
    results_locked_by_horizon_df["precision_high"],
    marker="o",
    linewidth=2,
    color=COLOR_PREC,
    label="Precision (HIGH)"
)

ax.set_xlabel("Horizon (days ahead)", color=COLOR_TEXT)
ax.set_ylabel("Metric", color=COLOR_TEXT)
ax.set_title("Policy Safety vs Noise (TEST)", color=COLOR_TEXT)

ax.tick_params(colors=COLOR_TEXT)
for spine in ax.spines.values():
    spine.set_color(COLOR_TEXT)

leg = ax.legend(frameon=True)
leg.get_frame().set_facecolor("white")
leg.get_frame().set_edgecolor("black")
for t in leg.get_texts():
    t.set_color("black")

ax.grid(True, axis="y", alpha=0.25, color="gray")

plt.tight_layout()
plt.show()


# In[146]:


# -----------------------------
# 6) Plot: AUC vs horizon (TEST) — Color Standard
#   GBM AUC = Blue
#   SVM AUC = Green
#   Ensemble Review AUC = Black (system summary)
# -----------------------------
import matplotlib.pyplot as plt

COLOR_GBM = "#5b9bd5"      # blue
COLOR_SVM = "#2e7d32"      # green
COLOR_ENS = "black"        # neutral
COLOR_TEXT = "black"

fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
ax.set_facecolor("white")

ax.plot(
    results_locked_by_horizon_df["horizon"],
    results_locked_by_horizon_df["AUC_gbm"],
    marker="o",
    linewidth=2,
    color=COLOR_GBM,
    label="GBM AUC"
)

ax.plot(
    results_locked_by_horizon_df["horizon"],
    results_locked_by_horizon_df["AUC_svm"],
    marker="o",
    linewidth=2,
    color=COLOR_SVM,
    label="SVM AUC"
)

ax.plot(
    results_locked_by_horizon_df["horizon"],
    results_locked_by_horizon_df["AUC_review"],
    marker="o",
    linewidth=2.5,
    color=COLOR_ENS,
    label="Ensemble Review AUC"
)

ax.set_xlabel("Horizon (days ahead)", color=COLOR_TEXT)
ax.set_ylabel("AUC", color=COLOR_TEXT)
ax.set_title("Ranking Power vs Horizon (TEST)", color=COLOR_TEXT)

ax.tick_params(colors=COLOR_TEXT)
for spine in ax.spines.values():
    spine.set_color(COLOR_TEXT)

leg = ax.legend(frameon=True, loc="lower left")
leg.get_frame().set_facecolor("white")
leg.get_frame().set_edgecolor("black")
for t in leg.get_texts():
    t.set_color("black")

ax.grid(True, axis="y", alpha=0.25, color="gray")

plt.tight_layout()
plt.show()


# In[149]:


# ===== FIGURE 7A: Sensitivity (blue) vs Specificity (green) + Workload (purple) =====
# White background, black fonts, no MAX_FLAG_RATE cap line.

import matplotlib.pyplot as plt
import numpy as np

story = results_locked_by_horizon_df.copy()
story["specificity"] = story["TN"] / (story["TN"] + story["FP"])
x = story["horizon"].values

fig, ax = plt.subplots(figsize=(13, 5), facecolor="white")
ax.set_facecolor("white")

# Force black spines/ticks/fonts
for spine in ax.spines.values():
    spine.set_color("black")
ax.tick_params(axis="both", colors="black")
ax.xaxis.label.set_color("black")
ax.yaxis.label.set_color("black")
ax.title.set_color("black")

# --- Sensitivity (Recall) as BLUE bars ---
ax.bar(
    x,
    story["recall_high"],
    width=0.65,
    color="#5b9bd5",        # BLUE
    edgecolor="black",
    label="Sensitivity (Recall)"
)

# --- Specificity as GREEN line ---
ax.plot(
    x,
    story["specificity"],
    marker="o",
    color="#2e7d32",        # GREEN
    linewidth=2,
    label="Specificity"
)

# Secondary axis: Workload (Flag rate) as PURPLE line
ax2 = ax.twinx()
ax2.set_facecolor("white")

for spine in ax2.spines.values():
    spine.set_color("black")
ax2.tick_params(axis="both", colors="black")
ax2.yaxis.label.set_color("black")

ax2.plot(
    x,
    story["flag_rate_test"],
    marker="o",
    color="#9c27b0",        # PURPLE
    linewidth=2,
    label="Review workload (Flag rate)"
)

# Axis formatting
ax.set_ylim(0, 1.05)
ax2.set_ylim(0, max(0.30, story["flag_rate_test"].max() + 0.05))

ax.set_xlabel("Forecast horizon (days ahead)", color="black")
ax.set_ylabel("Sensitivity / Specificity", color="black")
ax2.set_ylabel("Review workload (Flag rate)", color="black")

ax.set_title("Sensitivity vs Specificity with Review Workload (Test)", color="black")

# Light grid (neutral)
ax.grid(True, axis="y", color="gray", alpha=0.25)

# Combine legends (black text, white box)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

legend = ax.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper right",
    frameon=True
)
legend.get_frame().set_facecolor("white")
legend.get_frame().set_edgecolor("black")
for t in legend.get_texts():
    t.set_color("black")

plt.tight_layout()

# Optional save (PowerPoint-ready)
plt.savefig("fig07A_sens_spec_workload_colorcoded.png", dpi=250, bbox_inches="tight")

plt.show()


# In[152]:


# ===== TWO-PANEL SUMMARY: High vs Not High (WHITE BACKGROUND, BLACK FONT) =====

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

story = results_locked_by_horizon_df.copy()
story["specificity"] = story["TN"] / (story["TN"] + story["FP"])
x = story["horizon"].values

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True, facecolor="white")

# Force white axes background
ax1.set_facecolor("white")
ax2.set_facecolor("white")

# -----------------------------
# Background regions
# -----------------------------
for ax in (ax1, ax2):
    ax.axvspan(0.5, 2.5, color="#d9f2d9", alpha=0.6)   # Actionable
    ax.axvspan(2.5, 5.5, color="#fff2cc", alpha=0.6)   # Planning
    ax.axvspan(5.5, 7.5, color="#f8d7da", alpha=0.4)   # Lower reliability
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)

    # ✅ Force black ticks & spines
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")

# -----------------------------
# Panel 1: Sensitivity (High)
# -----------------------------
bars1 = ax1.bar(
    x,
    story["recall_high"],
    width=0.65,
    color="grey",
    edgecolor="black"
)

ax1.set_ylabel("Sensitivity (Recall)", color="black")
ax1.set_title(
    "Sensitivity: Detecting HIGH (>85%) staffed capacity days",
    color="black"
)

# Annotate bars
for rect, val in zip(bars1, story["recall_high"]):
    ax1.text(
        rect.get_x() + rect.get_width() / 2,
        rect.get_height() + 0.02,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=12,
        color="black"
    )

# -----------------------------
# Panel 2: Specificity (Not High)
# -----------------------------
ax2.plot(
    x,
    story["specificity"],
    marker="o",
    linewidth=2,
    color="#2e7d32"
)

ax2.set_ylabel("Specificity", color="black")
ax2.set_title(
    "Specificity: Correctly identifying NOT HIGH days",
    color="black"
)

for xi, val in zip(x, story["specificity"]):
    ax2.text(
        xi,
        val + 0.02,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=12,
        color="black"
    )

# X-axis
ax2.set_xticks(x)
ax2.set_xticklabels([f"T+{h}" for h in x], color="black")
ax2.set_xlabel("Forecast horizon (days ahead)", color="black")

# -----------------------------
# Region legend (black text)
# -----------------------------
region_legend = [
    Patch(facecolor="#d9f2d9", label="Actionable (T+1–T+2)"),
    Patch(facecolor="#fff2cc", label="Planning (T+3–T+5)"),
    Patch(facecolor="#f8d7da", label="Lower reliability (T+6–T+7)"),
]

plt.tight_layout()

# Optional save (PowerPoint-safe)
plt.savefig("summary_high_vs_not_high_black_font.png", dpi=250, bbox_inches="tight")

plt.show()


# In[137]:


# ===== LEGEND A1: Region shading legend only =====

region_handles = [
    Patch(facecolor="#d9f2d9", edgecolor="none"),
    Patch(facecolor="#fff2cc", edgecolor="none"),
    Patch(facecolor="#f8d7da", edgecolor="none"),
]
region_labels = [
    "Actionable (T+1–T+2)",
    "Planning (T+3–T+5)",
    "Lower reliability (T+6–T+7)",
]

legend_only(region_handles, region_labels, title="Region Key", ncol=1)
save_fig("legendA1_regions.png")
plt.show()


# In[138]:


# ===== LEGEND A2: Metric legend only =====

metric_handles = [
    Patch(facecolor=COLOR_SENS, edgecolor="black"),
    Line2D([0], [0], color=COLOR_SPEC, marker="o", linewidth=2),
]
metric_labels = [
    "Sensitivity (Recall) — HIGH detected",
    "Specificity — NOT HIGH correctly identified",
]

legend_only(metric_handles, metric_labels, title="Metric Key", ncol=1)
save_fig("legendA2_metrics.png")
plt.show()


# In[108]:


# ===== LEGEND B1: Region shading legend only (reuse) =====

legend_only(region_handles, region_labels, title="Region Key", ncol=1)
save_fig("legendB1_regions.png")
plt.show()


# In[ ]:




