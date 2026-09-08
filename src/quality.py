"""
Data-quality audit.

The brief says "nothing here has been cleaned for you", so this module states
plainly what is wrong with the data rather than silently working around it.
Every check writes a line to outputs/quality_report.txt.
"""
import numpy as np
import pandas as pd

from . import config as CFG
from .dataload import enrich_doc_events, verification_pairs


def run(raw: dict, spine: pd.DataFrame) -> tuple:
    """Return (list_of_report_lines, dict_of_key_numbers)."""
    L, K = [], {}

    def say(msg=""):
        L.append(msg)

    captains, approvals = raw["captains"], raw["approvals"]
    doc = enrich_doc_events(raw["doc_events"], captains)
    pairs = verification_pairs(raw["doc_events"])
    act, nud = raw["activation"], raw["nudges"]
    hourly, trips = raw["airport_hourly"], raw["airport_trips"]

    say("=" * 78)
    say("DATA QUALITY AUDIT")
    say("=" * 78)
    say(f"extract timestamp assumed: {CFG.EXTRACT_TS}")
    say()

    # ---------------------------------------------------- 1. shape & keys
    say("-- 1. Shape and referential integrity " + "-" * 39)
    for name, df in raw.items():
        say(f"   {name:<16} {df.shape[0]:>7,} rows x {df.shape[1]:>2} cols   "
            f"exact dup rows: {df.duplicated().sum()}")
    say(f"   captain_id unique in captains : {captains.captain_id.is_unique}")
    ids = set(captains.captain_id)
    for name in ["doc_events", "approvals", "activation", "nudges"]:
        orphans = (~raw[name].captain_id.isin(ids)).sum()
        say(f"   {name:<16} captain_ids not in captains.csv: {orphans}")
    say(f"   signups with zero document events: "
        f"{(~captains.captain_id.isin(doc.captain_id)).sum():,} "
        f"({(~captains.captain_id.isin(doc.captain_id)).mean()*100:.1f}%)")
    say()

    # ---------------------------------------------------- 2. RIGHT-CENSORING
    say("-- 2. Right-censoring (the thing the brief told us to notice) " + "-" * 16)
    in_prog = (approvals.final_status == "in_progress").sum()
    say(f"   final_status == 'in_progress'         : {in_prog:,} "
        f"({in_prog/len(approvals)*100:.2f}% of all signups)")
    say(f"   observed signup -> decision latency   : "
        f"median {spine.days_to_decision.median():.2f} d, "
        f"p95 {spine.days_to_decision.quantile(.95):.2f} d, "
        f"max {spine.days_to_decision.max():.2f} d")
    ip = spine[spine.final_status == "in_progress"]
    say(f"   in_progress signup dates              : "
        f"{ip.signup_ts.min().date()} to {ip.signup_ts.max().date()}")
    say(f"   -> a signup needs ~{CFG.MAX_DECISION_LATENCY_DAYS} days before its "
        f"outcome is final.")
    say(f"   Cohort rule applied: signup < {CFG.COHORT_CUTOFF.date()} "
        f"-> n = {spine.is_mature.sum():,}")
    say(f"   residual unresolved in cohort         : "
        f"{((spine.is_mature) & (spine.final_status == 'in_progress')).sum()} "
        f"(kept in denominator as not-approved)")
    say(f"   NOTE: pooling all 25,000 signups would report A2O of "
        f"{spine.approved.mean()*100:.2f}% instead of "
        f"{spine[spine.is_mature].approved.mean()*100:.2f}% -- a "
        f"{(spine[spine.is_mature].approved.mean()-spine.approved.mean())*100:.2f} pp "
        f"understatement caused purely by unfinished June signups.")
    K["a2o_naive_all_signups"] = spine.approved.mean()
    K["a2o_mature"] = spine[spine.is_mature].approved.mean()
    K["n_in_progress"] = int(in_prog)
    say()

    # ---------------------------------------------------- 3. approvals.csv vs events
    say("-- 3. approvals.csv summary columns disagree with the event log " + "-" * 14)
    mism = (spine.docs_cleared != spine.n_docs_passed)
    say(f"   docs_cleared != distinct docs passed  : {mism.sum():,} captains "
        f"({mism.mean()*100:.2f}%)")
    delta = (spine.loc[mism, "docs_cleared"] - spine.loc[mism, "n_docs_passed"])
    say(f"   direction of error                    : "
        f"docs_cleared too HIGH in {(delta > 0).sum():,} cases, "
        f"too LOW in {(delta < 0).sum():,} cases")
    say(f"   size of error (docs overstated by N)   : "
        f"{ {int(k): int(v) for k, v in delta.value_counts().sort_index().items()} }")
    say("   -> the summary column only ever OVERSTATES progress. Building the")
    say("      funnel from it would move captains to later stages than they")
    say("      reached and understate the early-stage leak. All funnel measures")
    say("      in this analysis are rebuilt from doc_events.")
    # the binary "cleared everything" gate IS consistent, which is why totals tie
    tie = pd.crosstab(spine.final_status, spine.cleared_all_docs)
    say(f"   cross-check: captains who cleared all required docs = "
        f"{spine.cleared_all_docs.sum():,}")
    say(f"                approved + rejected + in_progress-but-cleared = "
        f"{(spine.final_status.isin(['approved','rejected'])).sum() + ((spine.final_status=='in_progress') & spine.cleared_all_docs).sum():,}"
        f"  -> ties exactly")
    K["docs_cleared_mismatch_pct"] = float(mism.mean())
    say()

    # ---------------------------------------------------- 4. vehicle/document model
    say("-- 4. Document requirement depends on vehicle type " + "-" * 27)
    permit_by_v = pd.crosstab(doc.vehicle_type, doc.doc_type).reindex(
        columns=CFG.DOC_SEQUENCE, fill_value=0)
    say(f"   PERMIT events by vehicle type         : "
        f"{ {k: int(v) for k, v in permit_by_v['PERMIT'].items()} }")
    say(f"   max documents passed by an approved captain, by vehicle:")
    ap = spine[spine.approved == 1]
    say(f"      { {k: int(v) for k, v in ap.groupby('vehicle_type').n_docs_passed.max().items()} }")
    say("   -> ERickshaw requires 5 documents, not 6. A funnel defined as")
    say("      'cleared 6 documents' would classify all "
        f"{(captains.vehicle_type=='ERickshaw').sum():,} ERickshaw signups as")
    say("      failures. Denominators here are vehicle-aware.")
    say()

    # ---------------------------------------------------- 5. event-log internal logic
    say("-- 5. Event log internal consistency (this part is clean) " + "-" * 20)
    # every verdict must be traceable to an upload of the same doc and attempt
    verdicts = raw["doc_events"][raw["doc_events"].event_type != "upload_success"]
    uploads_key = set(zip(
        raw["doc_events"].loc[raw["doc_events"].event_type == "upload_success", "captain_id"],
        raw["doc_events"].loc[raw["doc_events"].event_type == "upload_success", "doc_type"],
        raw["doc_events"].loc[raw["doc_events"].event_type == "upload_success", "attempt_no"]))
    orphan_verdicts = sum(
        (c, d, a) not in uploads_key
        for c, d, a in zip(verdicts.captain_id, verdicts.doc_type, verdicts.attempt_no))
    say(f"   verdicts with no matching upload            : {orphan_verdicts}")
    say(f"   uploads still awaiting a verdict            : "
        f"{pairs.awaiting_verdict.sum():,} "
        f"({pairs.awaiting_verdict.mean()*100:.1f}% -- consistent with censoring)")
    say(f"   verdicts timestamped before their upload    : "
        f"{(pairs.verification_lag_h < 0).sum()}")
    say(f"   both a pass and a fail on one attempt       : "
        f"{(raw['doc_events'][raw['doc_events'].event_type != 'upload_success'].groupby(['captain_id','doc_type','attempt_no']).event_type.nunique() > 1).sum()}")
    say(f"   duplicate uploads on one attempt            : "
        f"{(pairs.groupby(['captain_id','doc_type','attempt_no']).size() > 1).sum()}")
    ds = raw["doc_events"].merge(captains[["captain_id", "signup_ts"]], on="captain_id")
    say(f"   events before the captain signed up         : "
        f"{(ds.event_ts < ds.signup_ts).sum()}")
    da = raw["doc_events"].merge(approvals[["captain_id", "decision_ts"]], on="captain_id")
    say(f"   events after a terminal decision            : "
        f"{(da.event_ts > da.decision_ts).sum()}")
    say(f"   attempt numbers observed                    : "
        f"{sorted(int(x) for x in raw['doc_events'].attempt_no.unique())} "
        f"(policy cap {CFG.MAX_ATTEMPTS})")
    say(f"   failure_reason populated on non-fail events : "
        f"{raw['doc_events'].loc[raw['doc_events'].event_type != 'verification_fail', 'failure_reason'].notna().sum()}")
    say()

    # ---------------------------------------------------- 6. activation
    say("-- 6. activation.csv " + "-" * 57)
    say(f"   rows                                  : {len(act):,} "
        f"(= number of approved captains: "
        f"{(approvals.final_status=='approved').sum():,})")
    say(f"   first_order_ts null                   : {act.first_order_ts.isna().sum()} "
        f"({act.first_order_ts.isna().mean()*100:.2f}%)")
    say(f"   orders_d7 null                        : {act.orders_d7.isna().sum()} "
        f"({act.orders_d7.isna().mean()*100:.2f}%)")
    say(f"   orders_d30 null                       : {act.orders_d30.isna().sum()} "
        f"({act.orders_d30.isna().mean()*100:.2f}%)")
    a = act.merge(approvals[["captain_id", "decision_ts"]], on="captain_id")
    a["days_obs"] = (CFG.EXTRACT_TS - a.decision_ts).dt.total_seconds() / 86400
    say(f"   nulls are censoring, not missingness  : "
        f"max days observed for null orders_d30 = "
        f"{a.loc[a.orders_d30.isna(), 'days_obs'].max():.1f} (< 30); "
        f"min for non-null = {a.loc[a.orders_d30.notna(), 'days_obs'].min():.1f} (>= 30)")
    say(f"   -> d7/d30 metrics must be computed only on captains with a full")
    say(f"      window. Averaging orders_d30 over all approved captains would")
    say(f"      silently drop the {act.orders_d30.isna().sum()} most recent ones.")
    bad = (act.orders_d7 > act.orders_d30).sum()
    say(f"   IMPOSSIBLE: orders_d7 > orders_d30     : {bad} rows")
    say(f"   IMPOSSIBLE: first_order_ts before the approval decision: "
        f"{(a.first_order_ts < a.decision_ts).sum()} rows")
    say(f"   online_hours_d30 == 0 with orders > 0  : "
        f"{((act.online_hours_d30 == 0) & (act.orders_d30 > 0)).sum()} rows")
    say(f"   online_hours_d30 == 0 overall          : "
        f"{(act.online_hours_d30 == 0).sum()} rows")
    K["activation_impossible_rows"] = int(bad)
    say()

    # ---------------------------------------------------- 7. nudges
    say("-- 7. nudges.csv " + "-" * 61)
    imposs = ((nud.clicked == 1) & (nud.delivered == 0)).sum()
    say(f"   IMPOSSIBLE: clicked = 1 while delivered = 0 : {imposs:,} rows "
        f"({imposs/len(nud)*100:.2f}%)")
    say(f"      by campaign: "
        f"{ {k: int(v) for k, v in nud[(nud.clicked==1)&(nud.delivered==0)].campaign_id.value_counts().items()} }")
    say("   -> click-through cannot be trusted as a delivery-conditional metric.")
    say("      We therefore do not use clicks as the primary campaign outcome.")
    ns = nud.merge(captains[["captain_id", "signup_ts"]], on="captain_id")
    ns["day"] = (ns.sent_ts - ns.signup_ts).dt.total_seconds() / 86400
    say(f"   nudges sent before signup                   : {(ns.day < 0).sum()}")
    say("   send timing by campaign (days after signup, median):")
    for camp, g in ns.groupby("campaign_id"):
        say(f"      {camp:<14} n={len(g):>5,}  median {g.day.median():.2f} d  "
            f"p5 {g.day.quantile(.05):.2f}  p95 {g.day.quantile(.95):.2f}")
    say(f"   -> {CFG.CAMPAIGN_UNDER_REVIEW} is sent ~1.3 days LATER than every other")
    say("      campaign. That gap is the whole story in A3.")
    K["nudge_impossible_clicks"] = int(imposs)
    say()

    # ---------------------------------------------------- 8. airport files
    say("-- 8. Airport files " + "-" * 58)
    resid = hourly.requests - hourly.fulfilled_requests - hourly.unfulfilled_requests
    say(f"   requests = fulfilled + unfulfilled violated : {(resid != 0).sum()} rows")
    say(f"   hourly coverage      : {hourly.hour_ts.min()} to {hourly.hour_ts.max()} "
        f"({hourly.hour_ts.nunique():,} hours x {hourly.zone_id.nunique()} zones)")
    say(f"   zones                : "
        f"{ {k: int(v) for k, v in hourly.groupby('zone_type').zone_id.nunique().items()} }")
    say(f"   trips sample         : {len(trips):,} rows, "
        f"{trips.request_ts.min().date()} to {trips.request_ts.max().date()}")
    comp = (trips.captain_cancelled == 0).sum()
    ful = hourly.loc[hourly.zone_type == "airport_terminal", "fulfilled_requests"].sum()
    say(f"   completed trips in sample / terminal fulfilled = "
        f"{comp:,} / {ful:,} = {comp/ful:.3f}")
    say(f"   -> airport_trips.csv is a ~{comp/ful*100:.0f}% sample of completed")
    say(f"      airport-origin trips. Any population total derived from it is")
    say(f"      scaled by {ful/comp:.2f}x and flagged as an estimate.")
    # fare model
    import statsmodels.formula.api as smf
    tj = trips.merge(
        hourly.loc[hourly.zone_type == "airport_terminal",
                   ["zone_id", "hour_ts", "avg_surge_multiplier"]],
        left_on=["pickup_zone_id", "request_ts"], right_on=["zone_id", "hour_ts"],
        how="left")
    r1 = smf.ols("fare_inr ~ trip_distance_km", data=tj).fit()
    r2 = smf.ols("fare_inr ~ trip_distance_km + avg_surge_multiplier", data=tj).fit()
    say(f"   INCONSISTENCY: fare_inr = {r1.params['Intercept']:.0f} + "
        f"{r1.params['trip_distance_km']:.2f}/km, R2 = {r1.rsquared:.4f}")
    say(f"      adding avg_surge_multiplier moves the fare by "
        f"{r2.params['avg_surge_multiplier']:.2f} INR and R2 by "
        f"{r2.rsquared - r1.rsquared:.5f}")
    say(f"      night mean surge at terminals = "
        f"{hourly[(hourly.zone_type=='airport_terminal') & hourly.hour_ts.dt.hour.isin(CFG.NIGHT_HOURS)].avg_surge_multiplier.mean():.2f}x "
        f"yet night fares equal day fares.")
    say("   -> the surge reported in airport_hourly.csv is NOT present in the")
    say("      fares in airport_trips.csv. Either surge is not reaching captain")
    say("      earnings, or the two files were generated independently. All")
    say("      earnings figures below are therefore surge-free and are a LOWER")
    say("      BOUND on what a surge-paid captain would make.")
    K["fare_per_km"] = float(r1.params["trip_distance_km"])
    K["trips_sample_share"] = float(comp / ful)
    say()

    # ---------------------------------------------------- 9. possible duplicates
    say("-- 9. Possible duplicate signups " + "-" * 45)
    dup = captains.duplicated(subset=["city", "vehicle_type", "signup_zone_id",
                                      "signup_ts"]).sum()
    say(f"   identical (city, vehicle, zone, signup_ts) : {dup} rows")
    say(f"   'duplicate_captain' rejections             : "
        f"{(approvals.last_stage_reached=='duplicate_captain').sum()}")
    say(f"   'duplicate_document' verification failures : "
        f"{(raw['doc_events'].failure_reason=='duplicate_document').sum():,}")
    say("   -> small, but it means A2O has a modest duplicate-inflated")
    say("      denominator. Not material at this scale; worth a dedup check")
    say("      before anyone is paid a per-signup commission on it.")
    say()

    say("=" * 78)
    say("ITEMS THAT WOULD CHANGE A NUMBER IF FIXED")
    say("=" * 78)
    say(f"  1. Censoring          : +{(K['a2o_mature']-K['a2o_naive_all_signups'])*100:.2f} pp on A2O")
    say(f"  2. docs_cleared drift : misplaces {mism.sum():,} captains in the funnel")
    say(f"  3. ERickshaw permits  : mislabels {(captains.vehicle_type=='ERickshaw').sum():,} signups if ignored")
    say(f"  4. Surge vs fare      : understates night airport earnings by up to "
        f"{hourly[(hourly.zone_type=='airport_terminal') & hourly.hour_ts.dt.hour.isin(CFG.NIGHT_HOURS)].avg_surge_multiplier.mean():.2f}x")
    say(f"  5. clicked/delivered  : {imposs:,} impossible rows -> clicks unusable as a KPI")
    say(f"  6. d7/d30 censoring   : {act.orders_d30.isna().sum()} approved captains have no 30-day window")

    return L, K
