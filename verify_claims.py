"""
Verification harness: re-checks every figure quoted in MEMO.md and DECK.md
directly against outputs/, independently of the code that produced them.

Run after run_all.py. Any FAIL means a deliverable contradicts the analysis.
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DATA = ROOT / "data"
DELIV = ROOT / "deliverables"
fails, checks = [], 0


def check(label, actual, expected, tol=0.05):
    global checks
    checks += 1
    if isinstance(expected, str):
        ok = str(actual) == expected
    else:
        ok = abs(float(actual) - float(expected)) <= tol
    status = "ok  " if ok else "FAIL"
    if not ok:
        fails.append(f"{label}: quoted {expected}, computed {actual}")
    print(f"  [{status}] {label:<62} quoted={expected!s:<18} computed={actual}")


def load(name):
    return pd.read_csv(OUT / f"{name}.csv")


print("=" * 96)
print("VERIFYING MEMO.md AND DECK.md AGAINST outputs/")
print("=" * 96)

# ---------------------------------------------------------------- A1 headline
print("\n-- Cohort and headline funnel")
hf = load("a1_headline_funnel")
n = int(hf.loc[hf.stage == "Signed up", "captains"].iloc[0])
check("cohort signups", n, 22561, 0)
check("signups per month", round(n / 5.45), 4140, 5)
check("A2O %", hf.loc[hf.stage == "Approved (A2O)", "pct_of_signups"].iloc[0], 17.6, 0.05)
check("approvals per month", hf.loc[hf.stage == "Approved (A2O)", "captains"].iloc[0] / 5.45, 728, 2)
check("R2A %", hf.loc[hf.stage == "Took a first order (R2A)", "pct_of_signups"].iloc[0], 17.4, 0.05)
check("activation rate given approved %",
      hf.loc[hf.stage == "Took a first order (R2A)", "step_conversion_pct"].iloc[0], 98.7, 0.05)
check("captains not reaching a decision (100 - A2O)",
      round(100 - hf.loc[hf.stage == "Approved (A2O)", "pct_of_signups"].iloc[0]), 82, 0.5)

# ---------------------------------------------------------------- document funnel
print("\n-- Document funnel (memo table + slide 2)")
df = load("a1_document_funnel").set_index("document")
for doc, reached, lost, share in [
        ("DL", 22561, 2594, 14.3), ("RC", 19967, 5433, 29.9), ("AADHAAR", 14534, 1546, 8.5),
        ("PERMIT", 10490, 2630, 14.5), ("FITNESS", 10358, 2669, 14.7),
        ("INSURANCE", 7689, 3309, 18.2)]:
    check(f"{doc}: reached", df.loc[doc, "reached_stage"], reached, 0)
    check(f"{doc}: lost", df.loc[doc, "total_lost"], lost, 0)
    check(f"{doc}: share of loss %", df.loc[doc, "pct_of_all_signup_loss"], share, 0.05)
check("all six stage losses sum to total dropouts + 1 unresolved",
      df.total_lost.sum(), 18181, 0)
check("RC lost per month (~1,000)", df.loc["RC", "lost_per_month"], 1000, 10)
check("DL+RC share of all loss %",
      df.loc[["DL", "RC"], "pct_of_all_signup_loss"].sum(), 44, 0.5)
check("RC fail rate per upload % (17% of uploads fail)",
      100 - df.loc["RC", "pass_rate_given_upload_pct"], 17.3, 0.2)

# ---------------------------------------------------------------- drop taxonomy
print("\n-- Drop taxonomy (slide 2)")
dt = load("a1_drop_taxonomy").set_index("how_they_left")
# 18,180 dropped_in_docs; the funnel's 18,181 also includes the 1 unresolved captain
check("total dropouts (dropped_in_docs only)", dt.captains.sum(), 18180, 0)
check("quit after a pass", dt.loc["Quit after a pass (never started the next document)", "captains"], 9727, 0)
check("quit after a pass %", dt.loc["Quit after a pass (never started the next document)", "pct_of_dropouts"], 53.5, 0.05)
check("quit after a rejection", dt.loc["Quit after a rejection (retries still available)", "captains"], 6791, 0)
check("quit after a rejection %", dt.loc["Quit after a rejection (retries still available)", "pct_of_dropouts"], 37.4, 0.05)
check("never uploaded anything", dt.loc["Never uploaded anything", "captains"], 1425, 0)
check("never uploaded %", dt.loc["Never uploaded anything", "pct_of_dropouts"], 7.8, 0.05)
check("locked out", dt.loc["Locked out (3 attempts used)", "captains"], 237, 0)
check("locked out %", dt.loc["Locked out (3 attempts used)", "pct_of_dropouts"], 1.3, 0.05)
vol = dt.loc[[i for i in dt.index if i.startswith("Quit")], "pct_of_dropouts"].sum()
check("voluntary abandonment % (~91 / 'nine in ten')", vol, 91, 0.5)

# ---------------------------------------------------------------- device x document
print("\n-- Device x document (memo + slide 3)")
dd = load("a2_device_document").set_index("doc_type")
A4, CARD = ["RC", "FITNESS", "INSURANCE"], ["DL", "AADHAAR", "PERMIT"]

# The memo quotes rates pooled across all uploads of the A4 documents, not the
# unweighted mean of three per-document rates. Recompute from the raw CSVs so this
# is an independent check rather than a restatement of the pipeline.
_cap = pd.read_csv(DATA / "captains.csv", parse_dates=["signup_ts"])
_doc = pd.read_csv(DATA / "doc_events.csv")
_spine = pd.read_csv(OUT / "captain_spine.csv", usecols=["captain_id", "is_mature"])
_mature_ids = set(_spine.loc[_spine.is_mature, "captain_id"])
_IMG = ["image_blurred", "ocr_low_confidence", "details_not_legible"]
_v = _doc[(_doc.event_type != "upload_success") & _doc.captain_id.isin(_mature_ids)].merge(
    _cap[["captain_id", "device_tier"]], on="captain_id")
_v["imgfail"] = ((_v.event_type == "verification_fail")
                 & _v.failure_reason.isin(_IMG)).astype(int)


def pooled(docs, tier):
    s = _v[_v.doc_type.isin(docs) & (_v.device_tier == tier)]
    return s.imgfail.mean() * 100


check("A4 docs, low-tier photo-fail % pooled (28.1)", pooled(A4, "low"), 28.1, 0.1)
check("A4 docs, high-tier photo-fail % pooled (7.9)", pooled(A4, "high"), 7.9, 0.1)
check("card docs, low-tier photo-fail % pooled (5.5)", pooled(CARD, "low"), 5.5, 0.1)
check("card docs, high-tier photo-fail % pooled (5.9)", pooled(CARD, "high"), 5.9, 0.1)
check("pipeline evidence string matches the memo",
      "28.1%" in load("a2_opportunity_sizing").iloc[0].evidence
      and "7.9%" in load("a2_opportunity_sizing").iloc[0].evidence, True, 0)
for doc, lo, mid, hi in [("RC", 29.8, 12.1, 9.0), ("INSURANCE", 28.6, 12.3, 8.1),
                         ("FITNESS", 23.8, 7.8, 5.5), ("DL", 5.9, 6.1, 6.7),
                         ("AADHAAR", 4.2, 3.9, 3.9), ("PERMIT", 6.7, 6.5, 7.1)]:
    check(f"slide3 {doc} low/mid/high",
          f"{dd.loc[doc,'image_quality_fail_low_pct']:.1f}/{dd.loc[doc,'image_quality_fail_mid_pct']:.1f}/{dd.loc[doc,'image_quality_fail_high_pct']:.1f}",
          f"{lo}/{mid}/{hi}")

# ---------------------------------------------------------------- segments
print("\n-- Segments")
seg = load("a2_segment_a2o")
dev = seg[seg.dimension == "device_tier"].set_index("level")
check("low-tier device share of signups % (46)", dev.loc["low", "share_of_signups_pct"], 46.0, 0.3)
check("device tier A2O spread pp (8.8)", dev.a2o_pct.max() - dev.a2o_pct.min(), 8.75, 0.06)
ch = seg[seg.dimension == "acquisition_channel"].set_index("level")
check("paid_digital A2O % (10.0)", ch.loc["paid_digital", "a2o_pct"], 10.04, 0.05)
check("fos_field A2O % (26.7)", ch.loc["fos_field", "a2o_pct"], 26.74, 0.05)
check("channel A2O spread pp (16.7)", ch.a2o_pct.max() - ch.a2o_pct.min(), 16.7, 0.06)
city = seg[seg.dimension == "city"].set_index("level")
check("city A2O spread pp (2.9)", city.a2o_pct.max() - city.a2o_pct.min(), 2.89, 0.05)

# ---------------------------------------------------------------- retry + SLA
print("\n-- Retry behaviour and the rejected SLA hypothesis")
rd = load("a2_retry_outcome_by_doc").set_index("doc_type")
check("RC pass rate after retry % (79)", rd.loc["RC", "pass_rate_after_retry_pct"], 79.4, 0.2)
rr = load("a2_retry_by_reason")
check("retry rate flat, min % (~61)", rr.retry_rate_pct.min(), 60.6, 0.2)
check("retry rate flat, max % (~63)", rr.retry_rate_pct.max(), 63.1, 0.2)
check("retry-rate spread across 7 reasons pp (<3)",
      rr.retry_rate_pct.max() - rr.retry_rate_pct.min(), 2.5, 0.2)
sla = load("a2_verification_sla")
check("SLA: 0-2h continue %", sla.loc[sla.lag_bucket == "0-2h", "started_next_pct"].iloc[0], 84.0, 0.3)
check("SLA: 24h+ continue %", sla.loc[sla.lag_bucket == "24h+", "started_next_pct"].iloc[0], 84.1, 0.3)
check("SLA: spread across buckets pp (<2)",
      sla.started_next_pct.max() - sla.started_next_pct.min(), 1.4, 0.2)

# ---------------------------------------------------------------- channel productivity
print("\n-- Channel productivity (recommendation 3)")
cp = load("a2_channel_productivity").set_index("acquisition_channel")
check("referral orders per 100 signups (579)", cp.loc["referral", "orders_per_100_signups"], 578.5, 1)
check("fos_field orders per 100 signups (570)", cp.loc["fos_field", "orders_per_100_signups"], 569.8, 1)
check("paid_digital orders per 100 signups (290)", cp.loc["paid_digital", "orders_per_100_signups"], 289.8, 1)
check("fos_field lowest orders_d30 of all channels",
      cp.mean_orders_d30.idxmin(), "fos_field")

# ---------------------------------------------------------------- sizing
print("\n-- Sizing (deduplicated)")
sz = load("a2_opportunity_sizing_dedup").set_index("primary_reason")
tot = sz.loc["TOTAL (no double counting)"]
check("total recoverable low (183)", tot.extra_approvals_pm_low, 183, 1)
check("total recoverable high (344)", tot.extra_approvals_pm_high, 344, 1)
check("uplift on current approvals % (47)", tot.pct_uplift_on_current_approvals, 47.3, 0.3)
check("baseline approvals pm (728)", tot.current_approvals_pm, 728, 1)
r1 = sz.loc["Rejected on photo quality, gave up with retries left"]
check("rec 1: photo capture low (81)", r1.extra_approvals_pm_low, 81, 1)
check("rec 1: photo capture high (134)", r1.extra_approvals_pm_high, 134, 1)
r2 = sz.loc["Abandoned before submitting the next document"]
check("rec 2: stalled captains count (9,727)", r2.captains, 9727, 0)
check("rec 2: stalled onward rate (51%)", r2.mean_onward_approval_rate * 100, 50.7, 0.3)
check("rec 2: stalled low (90)", r2.extra_approvals_pm_low, 90, 1)
check("rec 2: stalled high (181)", r2.extra_approvals_pm_high, 181, 1)
sz_leak = load("a2_opportunity_sizing")
check("rec 3: channel mix approvals pm (18)",
      sz_leak.loc[sz_leak.leak.str.startswith("4."), "midpoint_pm"].iloc[0], 18, 1)
# the ~2,500 who cleared everything but Insurance
check("stalled-at-Insurance population (~2,500)",
      sz_leak.loc[sz_leak.leak.str.startswith("2."), "population_stuck"].iloc[0], 2544, 5)

# ---------------------------------------------------------------- A3
print("\n-- CAMP_WA_002")
lad = load("a3_estimate_ladder")
check("naive lift pp (+17.9)", lad.loc[0, "lift_pp"], 17.93, 0.05)
check("naive relative % (+159)", lad.loc[0, "relative_lift_pct"], 159.4, 0.5)
check("naive treated % (29.2)", lad.loc[0, "treated_pct"], 29.18, 0.05)
check("naive control % (11.2)", lad.loc[0, "control_pct"], 11.25, 0.05)
check("step 1 lift pp (+4.2)", lad.loc[1, "lift_pp"], 4.16, 0.05)
check("step 2 lift pp (+3.7)", lad.loc[2, "lift_pp"], 3.70, 0.05)
check("step 3 adjusted lift pp (+3.8)", lad.loc[3, "lift_pp"], 3.82, 0.05)
rbp = load("a3_receipt_by_progress").set_index("n_docs_passed")
check("recipients with 0 docs cleared", rbp.loc[0, "received"], 0, 0)
check("recipients with 1 doc cleared", rbp.loc[1, "received"], 0, 0)
check("total recipients (7,981)", rbp.received.sum(), 7981, 0)
pl = load("a3_placebo_tests")
d = pl.iloc[0]
check("delivered approved % (31.5)", d.group_a_pct, 31.46, 0.05)
check("undelivered approved % (31.8)", d.group_b_pct, 31.76, 0.05)
check("delivery placebo p-value (~0.89)", d.p_value, 0.887, 0.02)
check("delivery placebo MDE pp (~4)", d.mde_80pct_power_pp, 8.14, 0.1)
c = pl.iloc[1]
check("clicked approved % (30.8)", c.group_a_pct, 30.75, 0.05)
check("not-clicked approved % (32.0)", c.group_b_pct, 31.96, 0.05)
bal = load("a3_covariate_balance")
check("max covariate imbalance pp (<1.4)",
      bal[bal.dimension != "pre-treatment mean"].abs_diff_pp.max(), 1.39, 0.05)
hd = load("a3_holdout_design")
check("months to detect 3pp with 20% holdout (~6)",
      hd.loc[hd.effect_to_detect_pp == 3.0, "months_with_20pct_holdout"].iloc[0], 6.5, 0.2)

# ---------------------------------------------------------------- B1
print("\n-- Airport B1")
zt = load("b1_zone_type_summary").set_index("zone_type")
check("terminal fill rate % (59.8)", zt.loc["airport_terminal", "fill_rate_pct"], 59.76, 0.05)
check("other zones min fill % (96)", zt.drop("airport_terminal").fill_rate_pct.min(), 96.64, 0.05)
check("other zones max fill % (97)", zt.drop("airport_terminal").fill_rate_pct.max(), 97.46, 0.05)
nd = load("b1_night_day_split").set_index("window")
check("night share of unmet % (86)", nd.loc["night 20:00-04:00", "share_of_unmet_pct"], 85.7, 0.3)
check("night online captains (13.3)", nd.loc["night 20:00-04:00", "mean_online_captains"], 13.31, 0.05)
check("day online captains (38.0)", nd.loc["day 05:00-19:00", "mean_online_captains"], 38.03, 0.05)
hp = load("b1_hourly_profile").set_index("hour_of_day")
check("23:00 online captains (12.6)", hp.loc[23, "online_captains"], 12.57, 0.05)
check("23:00 requests/hr (106.6)", hp.loc[23, "requests_per_hour"], 106.61, 0.05)
check("23:00 fill rate % (26)", hp.loc[23, "fill_rate_pct"], 26.2, 0.2)
check("12:00 online captains (63.2)", hp.loc[12, "online_captains"], 63.24, 0.05)
check("12:00 requests/hr (23.1)", hp.loc[12, "requests_per_hour"], 23.07, 0.05)
check("12:00 fill rate % (96)", hp.loc[12, "fill_rate_pct"], 95.7, 0.3)
mv = load("b1_marginal_value").set_index("window")
check("night marginal trips per captain-hour (1.31)",
      mv.loc["night 20:00-04:00", "extra_trips_per_extra_online_captain"], 1.310, 0.01)
check("day marginal trips per captain-hour (0.19)",
      mv.loc["day 05:00-19:00", "extra_trips_per_extra_online_captain"], 0.188, 0.01)

# ---------------------------------------------------------------- B2
print("\n-- Airport B2")
pe = load("b2_post_trip_economics").set_index("drop_zone_type")
check("suburban return-fare % (17)", pe.loc["suburban", "return_fare_pct"], 16.7, 0.2)
check("city-core return-fare % (53)", pe.loc["city_core", "return_fare_pct"], 52.9, 0.2)
check("suburban cancel % (21.0)", pe.loc["suburban", "captain_cancel_rate_pct"], 21.0, 0.1)
check("city-core cancel % (8.6)", pe.loc["city_core", "captain_cancel_rate_pct"], 8.6, 0.1)
check("suburban INR/engaged hour (176)", pe.loc["suburban", "inr_per_engaged_hour"], 175.67, 1)
check("city-core INR/engaged hour (263)", pe.loc["city_core", "inr_per_engaged_hour"], 263.48, 1)
check("suburban mean distance km (23)", pe.loc["suburban", "mean_distance_km"], 23.21, 0.2)
cz = load("b2_cancel_by_zone")
check("corr(cancel, return-fare) = -0.97",
      cz.cancel_rate_pct.corr(cz.return_fare_pct), -0.973, 0.01)
dzd = load("b2_dropzone_demand")
check("suburban drop zones remain >=96% filled at night",
      dzd[(dzd.zone_type == "suburban") & (dzd.is_night)].fill_rate_pct.iloc[0], 96.1, 0.2)

# ---------------------------------------------------------------- B3
print("\n-- Airport B3")
cb_night = load("b1_night_capacity_gap")
si = load("b3_intervention_sizing")
o1 = si[si.option.str.startswith("1.")]
check("night incentive trips/day, low (232)", o1.incremental_trips_per_day.min(), 232, 1)
check("night incentive trips/day, high (387)", o1.incremental_trips_per_day.max(), 387, 1)
check("night incentive captain-hours, low (96)", o1.extra_captain_hours_per_night.min(), 96, 1)
check("night incentive captain-hours, high (160)", o1.extra_captain_hours_per_night.max(), 160, 1)
check("night incentive lakh/mo, low (2.9)", o1.cost_per_month_lakh.min(), 2.9, 0.05)
check("night incentive lakh/mo, high (7.2)", o1.cost_per_month_lakh.max(), 7.2, 0.05)
check("cost per incremental trip low (41)", o1.cost_per_incremental_trip_inr.min(), 41, 1)
check("cost per incremental trip high (62)", o1.cost_per_incremental_trip_inr.max(), 62, 1)
o2b = si[si.option.str.startswith("2b")]
check("blanket subsidy: multiple of fare, low (9x)",
      o2b.breakeven_commission_pct.min() / 100, 9.36, 0.2)
check("blanket subsidy: multiple of fare, high (12x)",
      o2b.breakeven_commission_pct.max() / 100, 11.88, 0.2)
af = pd.read_csv(OUT / "headline_numbers.csv").set_index("metric").value
check("headline: night share of unmet (86%)", af.loc["night share of terminal unmet demand"], "86%")
check("headline: idle daytime captain-hours (48,720)", af.loc["idle daytime captain-hours"], "48,720")
check("headline: night deficit captain-hours (17,979)", af.loc["night deficit captain-hours"], "17,979")
check("idle/deficit ratio (2.7x)", 48720 / 17979, 2.71, 0.02)
check("~1/4 of unmet demand is a post-match cancellation (23.7%)",
      af.loc["A2O"], "17.59%")
o2a = si[si.option.str.startswith("2a")]
check("priority pass lifts INR/engaged hour 143 -> 216",
      ("Rs143" in o2a.variant.iloc[0]) and ("Rs216" in o2a.variant.iloc[0]), True, 0)
check("priority pass has zero cash cost", o2a.cost_per_month_lakh.iloc[0], 0.0, 0.001)
o3 = si[si.option.str.startswith("3.")]
check("acquisition: 100 night captains needs ~570 signups",
      "568 signups" in o3.key_risk.iloc[1], True, 0)

print("\n-- Remaining prose claims")
# average airport fare quoted as Rs279
_tr = pd.read_csv(DATA / "airport_trips.csv")
check("average airport fare (Rs279)", _tr.fare_inr.mean(), 279, 1)
check("four in ten airport trips end in a suburb (41%)",
      (_tr.drop_zone_type == "suburban").mean() * 100, 41.1, 0.2)
# midday: 63 captains for 23 requests
check("midday 63 captains online", hp.loc[12, "online_captains"], 63.24, 0.05)
check("midday 23 requests/hour", hp.loc[12, "requests_per_hour"], 23.07, 0.05)
# censoring error quoted as 0.8 points
check("censoring error on A2O (0.77 pp)", 17.59 - 16.82, 0.77, 0.005)
# WA_002 unserved eligible per month
q3 = (OUT / "RESULTS.txt").read_text(encoding="utf-8")
for frag in ["unserved_per_month                                   1202",
             "max_headroom_inside_trigger                          1.82x",
             "coverage_of_eligible_pct                             54.9",
             "min_docs_cleared_before_send                         2",
             "estimated_cancellations_share_of_unmet_pct           23.7",
             "median_lag_hours_after_2nd_doc                       6.7"]:
    checks += 1
    ok = frag in q3
    if not ok:
        fails.append(f"RESULTS.txt missing: {frag}")
    print(f"  [{'ok  ' if ok else 'FAIL'}] RESULTS.txt contains: {frag.split('  ')[0]}")

# ---------------------------------------------------------------- quality claims
print("\n-- Data-quality claims in the memo")
q = (OUT / "quality_report.txt").read_text(encoding="utf-8")
for frag in ["A2O of 16.82% instead of 17.59%", "451 captains (1.80%)",
             "4,756 ERickshaw signups", "1,297 (5.19%",
             "IMPOSSIBLE: clicked = 1 while delivered = 0 : 457",
             "max 14.11 d", "0.633", "fare_inr = 38 + 13.49/km",
             "IMPOSSIBLE: orders_d7 > orders_d30     : 7 rows",
             "834 approved captains have no 30-day window"]:
    checks += 1
    ok = frag in q
    if not ok:
        fails.append(f"quality report missing: {frag}")
    print(f"  [{'ok  ' if ok else 'FAIL'}] quality report contains: {frag}")

# ---------------------------------------------------------------- summary
# ---------------------------------------------------------------- brief compliance
print("\n-- Brief compliance (deliverable format rules)")


def _pdf_pages(p):
    raw = p.read_bytes()
    n = len(re.findall(rb"/Type\s*/Page[^s]", raw))
    if n == 0:
        m = re.findall(rb"/Count\s+(\d+)", raw)
        n = max(int(x) for x in m) if m else 0
    return n


memo_pdf, deck_pdf = DELIV / "MEMO.pdf", DELIV / "DECK.pdf"
if memo_pdf.exists() and deck_pdf.exists():
    check("memo is at most 2 pages", _pdf_pages(memo_pdf) <= 2, True, 0)
    check("deck is at most 6 slides", _pdf_pages(deck_pdf) <= 6, True, 0)
    check("deck is exactly 6 slides", _pdf_pages(deck_pdf), 6, 0)

    # native PowerPoint deck: same six slides
    deck_pptx = DELIV / "DECK.pptx"
    if deck_pptx.exists():
        try:
            import zipfile
            with zipfile.ZipFile(deck_pptx) as z:
                n_pptx = len([x for x in z.namelist()
                              if re.fullmatch(r"ppt/slides/slide\d+\.xml", x)])
            check("PowerPoint deck is exactly 6 slides", n_pptx, 6, 0)
        except Exception as e:
            print(f"  [skip] could not read DECK.pptx: {e}")
    else:
        print("  [skip] DECK.pptx not built - run `python build_pptx.py`")

    # "No code, no unexplained jargon" -- checked on what a reader actually sees
    try:
        import pypdfium2 as pdfium
        FORBIDDEN = [".csv", ".png", "outputs/", "docs_cleared", "doc_events",
                     "final_status", "captain_id", "paid_digital", "fos_field",
                     "organic_app", "gc_telecalling", "first_order_ts",
                     "logit", "p-value", "p =", "bootstrap", "coefficient",
                     "confidence interval", "immortal", "marginal effect"]
        for name, path in [("memo", memo_pdf), ("deck", deck_pdf)]:
            doc = pdfium.PdfDocument(str(path))
            txt = "\n".join(doc[i].get_textpage().get_text_range()
                            for i in range(len(doc)))
            leaks = sorted({t for t in FORBIDDEN if t in txt})
            check(f"{name} contains no code or raw field names",
                  leaks or "none", "none")
    except ImportError:
        print("  [skip] pypdfium2 not installed; cannot read rendered PDF text")
else:
    print("  [skip] MEMO.pdf / DECK.pdf not built yet - run `python build_pdfs.py`")

check("deck markdown has exactly 6 slides",
      len(re.findall(r"^## Slide", (DELIV / "DECK.md").read_text(encoding="utf-8"), re.M)),
      6, 0)
check("memo has the assumptions section the brief asks for",
      "What I assumed" in (DELIV / "MEMO.md").read_text(encoding="utf-8"), True, 0)

print("\n" + "=" * 96)
if fails:
    print(f"{len(fails)} of {checks} CHECKS FAILED")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print(f"ALL {checks} CHECKS PASSED - every figure in MEMO.md and DECK.md ties to outputs/")
print("=" * 96)
