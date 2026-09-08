"""
Shared configuration: constants, analysis choices, and the reasoning behind them.

Every non-obvious choice in this file is a declared assumption. The memo's
"What I assumed" section is derived from here.
"""
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data"            # the seven raw CSVs handed to us (input)
DELIVERABLES = ROOT / "deliverables"   # memo and deck, with their sources
OUT = ROOT / "outputs"         # everything the pipeline generates
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- extract & censoring
# The brief states the extract was taken at 2026-06-30 23:59 IST. Everything
# after this instant is unobserved, so any rate whose denominator includes
# captains who have not had time to reach the outcome will be biased downward.
EXTRACT_TS = pd.Timestamp("2026-06-30 23:59:00")

# Longest signup -> decision latency actually observed in the data is 14.11 days
# (see outputs/quality_report.txt). We require 15 days of observation before we
# treat a signup's onboarding outcome as final. This yields 1 residual
# `in_progress` captain out of 22,561 (0.004%), which we keep in the denominator
# and count as not-approved (conservative).
MAX_DECISION_LATENCY_DAYS = 15
COHORT_CUTOFF = pd.Timestamp("2026-06-16 00:00:00")  # = EXTRACT - 15 days, rounded to midnight
COHORT_START = pd.Timestamp("2026-01-01 00:00:00")

# Months spanned by the mature cohort, used to convert totals into run-rates.
COHORT_MONTHS = (COHORT_CUTOFF - COHORT_START).days / 30.4375  # 5.45

# ---------------------------------------------------------------- document model
# Stated required sequence. PERMIT applies to Auto and Cab only, so ERickshaw
# captains require 5 documents, not 6. Getting this wrong makes every ERickshaw
# captain look like a funnel failure.
DOC_SEQUENCE = ["DL", "RC", "AADHAAR", "PERMIT", "FITNESS", "INSURANCE"]
PERMIT_EXEMPT_VEHICLES = {"ERickshaw"}

# Physical form of each document. This split is not in the data dictionary; it is
# our reading of why failure rates differ so sharply (see funnel.py). DL, Aadhaar
# and Permit are pocket-sized laminated cards; RC, Fitness and Insurance are
# full-page paper documents with dense small print.
A4_PAPER_DOCS = ["RC", "FITNESS", "INSURANCE"]
CARD_DOCS = ["DL", "AADHAAR", "PERMIT"]

# Failure reasons that are a photograph problem rather than a document problem.
# A captain with a valid document who is told "blurred" can be saved by better
# capture UX. A captain whose document is genuinely expired cannot.
IMAGE_QUALITY_REASONS = ["image_blurred", "ocr_low_confidence", "details_not_legible"]
SUBSTANTIVE_REASONS = ["document_expired", "name_mismatch",
                       "wrong_document_type", "duplicate_document"]

MAX_ATTEMPTS = 3

def required_docs(vehicle_type: str) -> list:
    """Documents this vehicle type must clear, in order."""
    return [d for d in DOC_SEQUENCE
            if not (d == "PERMIT" and vehicle_type in PERMIT_EXEMPT_VEHICLES)]

# ---------------------------------------------------------------- campaign
CAMPAIGN_UNDER_REVIEW = "CAMP_WA_002"

# ---------------------------------------------------------------- airport
# Night window derived from the data, not assumed: hours where terminal fill rate
# collapses below ~75% (see airport.py, hourly profile).
NIGHT_HOURS = [20, 21, 22, 23, 0, 1, 2, 3]

# Average city driving speed used to convert distance into occupied time.
# Sensitivity to this assumption is reported in airport.py.
CITY_SPEED_KMPH = 22.0

# A captain who drops a passenger and gets no follow-on fare is assumed to wait
# the 20-minute window in the data, then drive back the same distance.
RETURN_FARE_WINDOW_MIN = 20

RANDOM_SEED = 20260630
