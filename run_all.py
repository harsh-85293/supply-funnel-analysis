"""
End-to-end analysis for the Captain Acquisition take-home.

Runs from the seven raw CSVs and writes every number quoted in the memo and the
deck to outputs/. Deterministic: no sampling except the parametric bootstrap in
campaign.py, which is seeded.

    python run_all.py
"""
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config as CFG
from src import dataload, quality, funnel, campaign, airport, charts

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)


def _w(df, name):
    """Write a table to outputs/ and return it."""
    path = CFG.OUT / f"{name}.csv"
    df.to_csv(path, index=False)
    return df


def _section(fh, title):
    fh.write("\n\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78 + "\n")


def _table(fh, df, note=""):
    if note:
        fh.write(note.rstrip() + "\n")
    fh.write(df.to_string(index=False) + "\n")


def _kv(fh, d, skip=()):
    for k, v in d.items():
        if k in skip or isinstance(v, (pd.DataFrame, pd.Series)):
            continue
        fh.write(f"  {k:<52} {v}\n")


def main():
    print("Loading raw data ...")
    raw = dataload.load_raw()
    doc = dataload.enrich_doc_events(raw["doc_events"], raw["captains"])
    pairs = dataload.verification_pairs(raw["doc_events"])
    spine = dataload.build_captain_spine(raw)
    coh = dataload.mature(spine)
    print(f"  {len(spine):,} signups loaded; mature cohort = {len(coh):,}")

    # ------------------------------------------------------------ quality
    print("Running data-quality audit ...")
    qlines, qkeys = quality.run(raw, spine)
    (CFG.OUT / "quality_report.txt").write_text("\n".join(qlines), encoding="utf-8")
    _w(spine, "captain_spine")

    results = open(CFG.OUT / "RESULTS.txt", "w", encoding="utf-8")
    results.write("CAPTAIN ONBOARDING & AIRPORT SUPPLY - FULL RESULTS\n")
    results.write(f"generated from raw CSVs; extract assumed {CFG.EXTRACT_TS}\n")
    results.write(f"mature cohort: signups {CFG.COHORT_START.date()} to "
                  f"{(CFG.COHORT_CUTOFF - pd.Timedelta(days=1)).date()} "
                  f"(n={len(coh):,}, {CFG.COHORT_MONTHS:.2f} months)\n")

    # ------------------------------------------------------------ A1
    print("A1: building the funnel ...")
    _section(results, "A1  THE FUNNEL")
    hf = _w(funnel.headline_funnel(coh), "a1_headline_funnel")
    _table(results, hf, "\nHeadline funnel, mature cohort:\n")

    df_ = _w(funnel.document_funnel(coh, doc), "a1_document_funnel")
    _table(results, df_, "\nDocument-by-document funnel, conditioned on reaching the stage:\n")

    dt = _w(funnel.drop_taxonomy(coh, doc), "a1_drop_taxonomy")
    _table(results, dt, "\nHow the captains who never finished actually left:\n")

    ac = funnel.activation_check(coh)
    results.write("\nR2A / activation check:\n")
    _kv(results, ac)

    # ------------------------------------------------------------ A2
    print("A2: segmenting the leak ...")
    _section(results, "A2  WHERE THE FIXABLE LOSS IS")
    seg = _w(funnel.segment_a2o(coh), "a2_segment_a2o")
    _table(results, seg, "\nA2O by signup attribute:\n")

    fr = _w(funnel.failure_reason_table(doc, coh), "a2_failure_reasons")
    _table(results, fr, "\nVerification failures by document and reason:\n")

    dd = _w(funnel.device_document_interaction(pairs, coh), "a2_device_document")
    _table(results, dd, "\nFailure rate per upload, device tier x document "
                        "(the key cut):\n")

    rb = funnel.retry_behaviour(doc, coh)
    results.write("\nRetry behaviour after a rejection:\n")
    _table(results, _w(rb["by_attempt"], "a2_retry_by_attempt"))
    _table(results, _w(rb["by_reason"], "a2_retry_by_reason"),
           "\nRetry rate by rejection reason (attempts still available):\n")
    _table(results, _w(rb["by_doc"], "a2_retry_outcome_by_doc"),
           "\nDid retrying work?\n")
    results.write(f"\n  captains who never retried a first rejection: "
                  f"{rb['never_retried']:,} ({rb['never_retried_per_month']}/month)\n")

    sla = funnel.verification_sla_test(pairs, doc, coh)
    results.write("\nHypothesis tested and REJECTED - verification turnaround:\n")
    _table(results, _w(sla["by_bucket"], "a2_verification_sla"))
    _kv(results, sla, skip=("by_bucket",))

    cp = _w(funnel.channel_productivity(coh), "a2_channel_productivity")
    _table(results, cp, "\nChannels judged on orders, not just approvals:\n")

    sz = _w(funnel.size_opportunities(coh, doc, pairs), "a2_opportunity_sizing")
    _table(results, sz, "\nSizing by leak: extra approved captains per month if fixed:\n")

    szd = _w(funnel.size_opportunities_deduplicated(coh, doc),
             "a2_opportunity_sizing_dedup")
    _table(results, szd,
           "\nSame prize with NO double counting - every unapproved captain assigned\n"
           "to exactly one primary reason, valued at the onward approval rate of\n"
           "captains who cleared the document they are stuck on:\n")

    # ------------------------------------------------------------ A3
    print("A3: evaluating CAMP_WA_002 ...")
    _section(results, "A3  CAMP_WA_002")
    trig = campaign.recover_trigger(coh, doc)
    results.write("\nRecovered assignment rule:\n")
    _kv(results, trig)
    _table(results, _w(campaign.receipt_by_progress(coh), "a3_receipt_by_progress"),
           "\nP(received) by documents eventually cleared:\n")
    _table(results, _w(campaign.other_campaigns(coh, raw["nudges"]), "a3_all_campaigns"),
           "\nEvery campaign, for context:\n")

    ladder, est = campaign.estimate_ladder(coh)
    _w(ladder, "a3_estimate_ladder")
    _table(results, ladder, "\nFrom the claimed number to a defensible one:\n")
    results.write(f"\n  adjusted average marginal effect: {est['ame_pp']:+.2f} pp "
                  f"95% CI [{est['ci_pp'][0]:+.2f}, {est['ci_pp'][1]:+.2f}]\n")

    al = est["aligned_frame"]
    _table(results, _w(campaign.covariate_balance(al), "a3_covariate_balance"),
           "\nCovariate balance among eligible captains "
           "(receipt looks random on everything observable):\n")

    pl = _w(campaign.placebo_tests(al), "a3_placebo_tests")
    _table(results, pl, "\nPlacebo tests - would a real message effect survive them?\n")

    su = campaign.scale_up_headroom(coh, effect_pp_range=(0.0, est["ame_pp"]))
    results.write("\nThe 5x request:\n")
    _kv(results, su)

    hd = _w(campaign.holdout_design(coh), "a3_holdout_design")
    _table(results, hd, "\nWhat a real measurement would cost:\n")

    # ------------------------------------------------------------ Part B
    print("Part B: airport supply ...")
    _section(results, "B1  THE AIRPORT MISMATCH")
    h, t = airport._prep(raw["airport_hourly"], raw["airport_trips"])
    _table(results, _w(airport.zone_type_summary(h), "b1_zone_type_summary"),
           "\nTerminals against every other zone type:\n")
    _table(results, _w(airport.terminal_hourly_profile(h), "b1_hourly_profile"),
           "\nTerminal demand and supply by hour of day:\n")
    _table(results, _w(airport.night_day_split(h), "b1_night_day_split"),
           "\nNight versus day:\n")
    mv = _w(airport.marginal_value_of_a_captain(h), "b1_marginal_value")
    _table(results, mv, "\nWhat one extra online captain-hour is worth:\n")

    cb = airport.capacity_balance(h)
    results.write("\nCapacity balance:\n")
    _kv(results, cb, skip=("per_hour_table",))
    _table(results, _w(cb["per_hour_table"], "b1_night_capacity_gap"),
           "\nNight-hour capacity gap per terminal:\n")

    _section(results, "B2  WHAT HAPPENS AFTER AN AIRPORT TRIP")
    _table(results, _w(airport.post_trip_economics(t), "b2_post_trip_economics"),
           "\nPost-trip economics by destination type:\n")
    ca = airport.cancellation_analysis(t)
    results.write("\nCancellation:\n")
    _kv(results, ca, skip=("by_zone", "logit", "night_by_zone_type"))
    _table(results, _w(ca["by_zone"], "b2_cancel_by_zone"), "\nBy drop zone:\n")
    _table(results, _w(ca["logit"], "b2_cancel_logit"),
           "\nLogit on cancellation:\n")
    _table(results, _w(ca["night_by_zone_type"], "b2_cancel_night"),
           "\nNight versus day, by destination:\n")
    _table(results, _w(airport.is_deadhead_avoidable(h), "b2_dropzone_demand"),
           "\nIs there spare demand in the drop zones to match to?\n")

    _section(results, "B3  IS TARGETED ACQUISITION THE RIGHT INTERVENTION?")
    a2o = float(coh.approved.mean())
    si = _w(airport.size_interventions(h, t, cb, a2o), "b3_intervention_sizing")
    _table(results, si, "\nOptions, costed per incremental fulfilled trip:\n")
    af = airport.acquisition_feasibility(t, h, a2o, cb)
    results.write("\nThe case against the proposal as framed:\n")
    _kv(results, af)

    # ------------------------------------------------------------ headline numbers
    _section(results, "HEADLINE NUMBERS QUOTED IN THE MEMO AND DECK")
    head = {
        "mature cohort signups": f"{len(coh):,}",
        "signups per month": f"{len(coh)/CFG.COHORT_MONTHS:,.0f}",
        "A2O": f"{coh.approved.mean()*100:.2f}%",
        "approvals per month": f"{coh.approved.sum()/CFG.COHORT_MONTHS:,.0f}",
        "R2A": f"{coh.activated.mean()*100:.2f}%",
        "activation rate given approved": f"{ac['activation_rate_pct']:.2f}%",
        "A2O if censoring ignored": f"{qkeys['a2o_naive_all_signups']*100:.2f}%",
        "docs_cleared column error rate": f"{qkeys['docs_cleared_mismatch_pct']*100:.2f}%",
        "loss at or before RC": f"{df_.loc[df_.document.isin(['DL','RC']), 'total_lost'].sum():,} "
                                f"({df_.loc[df_.document.isin(['DL','RC']), 'pct_of_all_signup_loss'].sum():.0f}% of all loss)",
        "voluntary abandonment share of dropouts":
            f"{dt.loc[dt.how_they_left.str.startswith('Quit'), 'pct_of_dropouts'].sum():.0f}%",
        "total recoverable approvals per month (deduplicated)":
            f"{szd.iloc[-1].extra_approvals_pm_low:.0f}-{szd.iloc[-1].extra_approvals_pm_high:.0f}"
            f" on a base of {szd.iloc[-1].current_approvals_pm:.0f}"
            f" (+{szd.iloc[-1].pct_uplift_on_current_approvals:.0f}% at the top end)",
        "policy lock-out share of dropouts":
            f"{dt.loc[dt.how_they_left.str.startswith('Locked'), 'pct_of_dropouts'].sum():.1f}%",
        "low-tier image-fail rate, A4 paper docs":
            f"{dd.loc[dd.doc_type.isin(CFG.A4_PAPER_DOCS), 'image_quality_fail_low_pct'].mean():.1f}%",
        "high-tier image-fail rate, A4 paper docs":
            f"{dd.loc[dd.doc_type.isin(CFG.A4_PAPER_DOCS), 'image_quality_fail_high_pct'].mean():.1f}%",
        "same gap on card documents":
            f"{dd.loc[dd.doc_type.isin(CFG.CARD_DOCS), 'image_quality_fail_low_pct'].mean():.1f}% low vs "
            f"{dd.loc[dd.doc_type.isin(CFG.CARD_DOCS), 'image_quality_fail_high_pct'].mean():.1f}% high (no gap)",
        "CAMP_WA_002 claimed lift": f"{ladder.loc[0,'lift_pp']:+.1f} pp "
                                    f"({ladder.loc[0,'relative_lift_pct']:+.0f}% relative)",
        "CAMP_WA_002 defensible upper bound": f"{est['ame_pp']:+.2f} pp "
                                              f"CI [{est['ci_pp'][0]:+.2f}, {est['ci_pp'][1]:+.2f}]",
        "CAMP_WA_002 max headroom": su["max_headroom_inside_trigger"],
        "terminal fill rate": f"{airport.zone_type_summary(h).query('zone_type==\"airport_terminal\"').fill_rate_pct.iloc[0]:.1f}%",
        "other zones fill rate": "96-97%",
        "night share of terminal unmet demand": f"{af['night_share_of_terminal_unmet_pct']:.0f}%",
        "marginal trips per captain-hour, night": f"{mv.loc[mv.window.str.startswith('night'),'extra_trips_per_extra_online_captain'].iloc[0]:.2f}",
        "marginal trips per captain-hour, day": f"{mv.loc[mv.window.str.startswith('day'),'extra_trips_per_extra_online_captain'].iloc[0]:.2f}",
        "idle daytime captain-hours": f"{cb['idle_captain_hours_in_surplus_hours']:,}",
        "night deficit captain-hours": f"{cb['deficit_captain_hours_in_short_hours']:,}",
        "suburban cancel rate": f"{ca['by_zone'].query('drop_zone_type==\"suburban\"').cancel_rate_pct.mean():.1f}%",
        "city-core cancel rate": f"{ca['by_zone'].query('drop_zone_type==\"city_core\"').cancel_rate_pct.mean():.1f}%",
        "corr(cancel, return-fare) across drop zones": f"{ca['corr_cancel_vs_return_fare']:.2f}",
    }
    _kv(results, head)
    results.close()

    _w(pd.DataFrame([{"metric": k, "value": v} for k, v in head.items()]),
       "headline_numbers")

    # ------------------------------------------------------------ deck exhibits
    print("Building chart exhibits ...")
    made = charts.build_all(h, pairs, coh, ladder)
    for p in made:
        print(f"  {p.relative_to(CFG.ROOT)}")

    print(f"\nDone. Wrote {len(list(CFG.OUT.glob('*')))} files to {CFG.OUT}")
    print("  outputs/RESULTS.txt        - every table, in reading order")
    print("  outputs/quality_report.txt - the data-quality audit")
    print("  outputs/*.csv              - individual tables")


if __name__ == "__main__":
    main()
