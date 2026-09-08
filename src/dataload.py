"""
Loading and derived-feature construction.

Design decision that matters: we rebuild every funnel measure from `doc_events`
rather than trusting the summary columns in `approvals.csv`. `docs_cleared`
disagrees with the event log for 1.8% of captains and always in the same
direction (it overstates progress). See quality.py.
"""
import numpy as np
import pandas as pd

from . import config as CFG


# ---------------------------------------------------------------- raw loaders
def load_raw() -> dict:
    """Read the seven CSVs with correct dtypes and parsed timestamps."""
    captains = pd.read_csv(CFG.RAW / "captains.csv", parse_dates=["signup_ts"])
    doc_events = pd.read_csv(CFG.RAW / "doc_events.csv", parse_dates=["event_ts"])
    approvals = pd.read_csv(CFG.RAW / "approvals.csv", parse_dates=["decision_ts"])
    activation = pd.read_csv(CFG.RAW / "activation.csv", parse_dates=["first_order_ts"])
    nudges = pd.read_csv(CFG.RAW / "nudges.csv", parse_dates=["sent_ts"])
    airport_hourly = pd.read_csv(CFG.RAW / "airport_hourly.csv", parse_dates=["hour_ts"])
    airport_trips = pd.read_csv(CFG.RAW / "airport_trips.csv", parse_dates=["request_ts"])
    return dict(captains=captains, doc_events=doc_events, approvals=approvals,
                activation=activation, nudges=nudges,
                airport_hourly=airport_hourly, airport_trips=airport_trips)


# ---------------------------------------------------------------- doc-event features
def enrich_doc_events(doc_events: pd.DataFrame, captains: pd.DataFrame) -> pd.DataFrame:
    """Attach vehicle type, signup time, sequence position and relative timing."""
    d = doc_events.merge(
        captains[["captain_id", "vehicle_type", "signup_ts", "device_tier",
                  "acquisition_channel", "city"]],
        on="captain_id", how="left", validate="m:1")
    d["seq_position"] = d.doc_type.map({v: i for i, v in enumerate(CFG.DOC_SEQUENCE)})
    d["hours_since_signup"] = (d.event_ts - d.signup_ts).dt.total_seconds() / 3600
    d["doc_form"] = np.where(d.doc_type.isin(CFG.A4_PAPER_DOCS), "A4_paper", "card")
    d["is_image_quality_fail"] = (
        (d.event_type == "verification_fail")
        & d.failure_reason.isin(CFG.IMAGE_QUALITY_REASONS)
    )
    return d


def verification_pairs(doc_events: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (captain, doc, attempt): the upload and its verdict.
    Used to measure per-upload failure rates and verification turnaround.
    """
    up = (doc_events.loc[doc_events.event_type == "upload_success",
                         ["captain_id", "doc_type", "attempt_no", "event_ts"]]
          .rename(columns={"event_ts": "upload_ts"}))
    ver = (doc_events.loc[doc_events.event_type != "upload_success",
                          ["captain_id", "doc_type", "attempt_no", "event_ts",
                           "event_type", "failure_reason"]]
           .rename(columns={"event_ts": "verdict_ts"}))
    pairs = up.merge(ver, on=["captain_id", "doc_type", "attempt_no"], how="left")
    pairs["verification_lag_h"] = (
        (pairs.verdict_ts - pairs.upload_ts).dt.total_seconds() / 3600)
    pairs["failed"] = (pairs.event_type == "verification_fail").astype("Int64")
    pairs["image_quality_fail"] = (
        (pairs.event_type == "verification_fail")
        & pairs.failure_reason.isin(CFG.IMAGE_QUALITY_REASONS)).astype(int)
    pairs["awaiting_verdict"] = pairs.verdict_ts.isna()
    return pairs


# ---------------------------------------------------------------- captain-level spine
def build_captain_spine(raw: dict) -> pd.DataFrame:
    """
    One row per signup, with every onboarding measure rebuilt from the event log.

    Returns all 25,000 signups. Cohort filtering happens downstream so that the
    censoring decision stays visible rather than baked in.
    """
    captains, approvals = raw["captains"], raw["approvals"]
    doc = enrich_doc_events(raw["doc_events"], captains)

    s = captains.merge(approvals, on="captain_id", how="left", validate="1:1")
    s["n_required_docs"] = s.vehicle_type.map(
        lambda v: len(CFG.required_docs(v)))

    passes = doc[doc.event_type == "verification_pass"]
    uploads = doc[doc.event_type == "upload_success"]
    fails = doc[doc.event_type == "verification_fail"]

    s["n_docs_passed"] = s.captain_id.map(
        passes.groupby("captain_id").doc_type.nunique()).fillna(0).astype(int)
    s["n_docs_uploaded"] = s.captain_id.map(
        uploads.groupby("captain_id").doc_type.nunique()).fillna(0).astype(int)
    s["n_upload_events"] = s.captain_id.map(
        uploads.groupby("captain_id").size()).fillna(0).astype(int)
    s["n_failures"] = s.captain_id.map(
        fails.groupby("captain_id").size()).fillna(0).astype(int)
    s["n_image_quality_failures"] = s.captain_id.map(
        fails[fails.failure_reason.isin(CFG.IMAGE_QUALITY_REASONS)]
        .groupby("captain_id").size()).fillna(0).astype(int)

    # terminal event in the funnel, used to classify how a captain dropped out
    last = (doc.sort_values("event_ts").groupby("captain_id")
            .agg(last_event_type=("event_type", "last"),
                 last_event_doc=("doc_type", "last"),
                 last_event_ts=("event_ts", "last")))
    s = s.merge(last, on="captain_id", how="left")

    # time the captain effectively left the funnel: a decision if there was one,
    # otherwise their final observed action
    s["exit_ts"] = s.decision_ts.fillna(s.last_event_ts).fillna(s.signup_ts)
    s["exit_day"] = (s.exit_ts - s.signup_ts).dt.total_seconds() / 86400
    s["days_to_decision"] = (s.decision_ts - s.signup_ts).dt.total_seconds() / 86400
    s["observation_days"] = (CFG.EXTRACT_TS - s.signup_ts).dt.total_seconds() / 86400

    s["cleared_all_docs"] = s.n_docs_passed >= s.n_required_docs
    s["approved"] = (s.final_status == "approved").astype(int)
    s["is_mature"] = s.signup_ts < CFG.COHORT_CUTOFF

    # timing of the 2nd document clearance: this is the trigger point for
    # CAMP_WA_002 and the alignment point for its evaluation
    p = passes.sort_values(["captain_id", "event_ts"]).copy()
    p["pass_rank"] = p.groupby("captain_id").cumcount() + 1
    for k in (1, 2):
        s[f"t_doc{k}_pass"] = s.captain_id.map(
            p.loc[p.pass_rank == k].set_index("captain_id").event_ts)

    # first uncleared required document = the stage the captain is stuck on
    passed_sets = passes.groupby("captain_id").doc_type.agg(set)
    def stuck_on(row):
        got = passed_sets.get(row.captain_id, set())
        for d in CFG.required_docs(row.vehicle_type):
            if d not in got:
                return d
        return None
    s["stuck_on_doc"] = s.apply(stuck_on, axis=1)

    # activation
    act = raw["activation"]
    s = s.merge(act, on="captain_id", how="left", validate="1:1")
    s["activated"] = s.first_order_ts.notna().astype(int)
    s["days_to_first_order"] = (
        (s.first_order_ts - s.decision_ts).dt.total_seconds() / 86400)

    # nudge exposure
    nud = raw["nudges"]
    s["n_nudges"] = s.captain_id.map(nud.groupby("captain_id").size()).fillna(0).astype(int)
    for camp in sorted(nud.campaign_id.unique()):
        c = nud[nud.campaign_id == camp].groupby("captain_id").agg(
            sent_ts=("sent_ts", "min"), delivered=("delivered", "max"),
            clicked=("clicked", "max"))
        s[f"{camp}_sent_ts"] = s.captain_id.map(c.sent_ts)
        s[f"{camp}_delivered"] = s.captain_id.map(c.delivered)
        s[f"{camp}_clicked"] = s.captain_id.map(c.clicked)
        s[f"{camp}_received"] = s[f"{camp}_sent_ts"].notna().astype(int)

    return s


def mature(spine: pd.DataFrame) -> pd.DataFrame:
    """The analysis cohort: signups with a full 15 days of observation."""
    return spine[spine.is_mature].copy()
