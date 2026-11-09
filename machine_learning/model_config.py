
import os

# ---- data grid ---------------------------------------------------------------------------------
# observations are averaged onto a regular grid of this size
BIN_MINUTES = 5
# a gap of up to 2 bins is bridged by carrying the last value forward
MAX_FFILL_BINS = 2
# minutes; a horizon is trained only if the data supports it
SUPPORTED_HORIZONS = (15, 30, 60)
# ---- congestion definition (documented, relative to each road's own reference speed) -----------
# ratio = speed / reference_speed.  reference_speed = segment.free_flow_speed_kmh if provided,
# otherwise the 85th percentile of the segment's TRAINING-period speeds (a common free-flow proxy).
# "heavy" is the brief's "high congestion"
CLASS_LABELS = ("low", "moderate", "heavy")
LOW_RATIO = 0.70                # ratio >= 0.70  -> low
HEAVY_RATIO = 0.40              # ratio <  0.40  -> heavy ; otherwise moderate
REFERENCE_PERCENTILE = 85
# ---- cleaning ----------------------------------------------------------------------------------
# above this is a sensor error on urban roads (flagged, then excluded)
MAX_PLAUSIBLE_SPEED_KMH = 140
MAX_PLAUSIBLE_COUNT = 5000      # vehicles per interval; above this is flagged
# ---- features ----------------------------------------------------------------------------------
# local hours, [start, end) - Nairobi peaks; verify with your data
RUSH_HOURS = ((7, 9), (16, 19))
# Africa/Nairobi; time features use local time
LOCAL_UTC_OFFSET_HOURS = 3
# previous speeds at t-1.. t-12 bins
LAG_BINS = (1, 2, 3, 6, 12)
# rolling mean/std windows in bins
ROLL_WINDOWS = (6, 12)
INCIDENT_RADIUS_KM = 1.5
INCIDENT_WINDOW_MIN = 60
# the incident feature is used only if enough incidents exist
MIN_INCIDENT_ROWS_FOR_FEATURE = 30
# ---- training / acceptance ---------------------------------------------------------------------
# remaining 15% is the untouched test period
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
MIN_SPAN_DAYS = 7               # need at least a week to see weekly patterns
MIN_TRAIN_ROWS = 2000
MIN_TEST_ROWS = 200
RANDOM_STATE = 42
# A model is deployed only if it beats persistence ("same speed as now") on the TEST period by at least
# this relative margin in MAE, and also beats the currently deployed model (if any).
MIN_MAE_IMPROVEMENT_VS_PERSISTENCE = 0.02
# ---- serving -----------------------------------------------------------------------------------
# newest measurement older than this -> prediction is 'stale', not served
MAX_OBSERVATION_AGE_MIN = 20
MODELS_DIR = os.environ.get("ML_MODELS_DIR", os.path.join(
    os.path.dirname(__file__), "saved_models"))
KEEP_VERSIONS = 3               # older model versions kept for rollback
