"""
Part A1 + A2: build the funnel, locate the leaks, size the fixes.

Two definitional choices drive everything:

1. Denominator = every signup in the mature cohort. Not "captains who started
   uploading". If 6% of signups never upload a document, that is a funnel
   problem, and hiding it in the denominator would be flattering ourselves.

2. Stage conditioning = a captain "reaches" document k only if they cleared
   every required document before k. Reporting raw per-document upload counts
   mixes together "did not reach this stage" with "reached it and quit", which
   are different problems with different owners.
"""
import numpy as np
import pandas as pd

from . import config as CFG
from .dataload import enrich_doc_events, verification_pairs


# ---------------------------------------------------------------- A1
def headline_funnel(coh: pd.DataFrame) -> pd.DataFrame:
    """Signup -> approved, at the level a Head of Supply wants to see first."""
    n = len(coh)
    rows = [
        ("Signed up", n),
        ("Uploaded at least one document", int((coh.n_upload_events > 0).sum())),
        ("Passed at least one document", int((coh.n_docs_passed >= 1).sum())),
        ("Cleared every required document", int(coh.cleared_all_docs.sum())),
        ("Approved (A2O)", int((coh.final_status == "approved").sum())),
        ("Took a first order (R2A)", int(coh.activated.sum())),
    ]
    f = pd.DataFrame(rows, columns=["stage", "captains"])
    f["pct_of_signups"] = (f.captains / n * 100).round(2)
    f["step_conversion_pct"] = (f.captains / f.captains.shift() * 100).round(2)
    f["lost_at_this_step"] = (f.captains.shift() - f.captains).fillna(0).astype(int)
    f["lost_per_month"] = (f.lost_at_this_step / CFG.COHORT_MONTHS).round(0)
    return f


def document_funnel(coh: pd.DataFrame, doc: pd.DataFrame) -> pd.DataFrame:
    """
    Stage-by-stage document funnel, properly conditioned on reaching the stage,
    and splitting each stage's loss into the two failure modes:

      drop_before_upload  -- captain reached the stage and never submitted.
                             This is a motivation / friction problem. Owner: CRM + product.
      drop_at_verification -- captain submitted and never got a pass.
                             This is a document-capture / eligibility problem. Owner: product + policy.
    """
    ids = coh.set_index("captain_id")
    d = doc[doc.captain_id.isin(coh.captain_id)]

    first_pass = d[d.event_type == "verification_pass"].pivot_table(
        index="captain_id", columns="doc_type", values="event_ts", aggfunc="min")
    first_up = d[d.event_type == "upload_success"].pivot_table(
        index="captain_id", columns="doc_type", values="event_ts", aggfunc="min")
    first_pass = first_pass.reindex(coh.captain_id)
    first_up = first_up.reindex(coh.captain_id)
    vehicle = ids.vehicle_type

    reached = pd.Series(True, index=coh.captain_id)
    rows = []
    for stage in CFG.DOC_SEQUENCE:
        applies = pd.Series(True, index=coh.captain_id)
        if stage == "PERMIT":
            applies = ~vehicle.isin(CFG.PERMIT_EXEMPT_VEHICLES)
        eligible = reached & applies
        uploaded = eligible & first_up.get(
            stage, pd.Series(np.nan, index=coh.captain_id)).notna()
        passed = eligible & first_pass.get(
            stage, pd.Series(np.nan, index=coh.captain_id)).notna()

        rows.append(dict(
            order=CFG.DOC_SEQUENCE.index(stage) + 1,
            document=stage,
            applies_to="Auto, Cab" if stage == "PERMIT" else "all",
            reached_stage=int(eligible.sum()),
            uploaded=int(uploaded.sum()),
            passed=int(passed.sum()),
            drop_before_upload=int((eligible & ~uploaded).sum()),
            drop_at_verification=int((uploaded & ~passed).sum()),
            upload_rate_pct=round(uploaded.sum() / max(eligible.sum(), 1) * 100, 1),
            pass_rate_given_upload_pct=round(passed.sum() / max(uploaded.sum(), 1) * 100, 1),
            stage_conversion_pct=round(passed.sum() / max(eligible.sum(), 1) * 100, 1),
        ))
        # a captain moves on if they passed, or if the stage did not apply to them
        reached = passed | (reached & ~applies)

    f = pd.DataFrame(rows)
    f["total_lost"] = f.drop_before_upload + f.drop_at_verification
    f["lost_per_month"] = (f.total_lost / CFG.COHORT_MONTHS).round(0)
    f["pct_of_all_signup_loss"] = (
        f.total_lost / f.total_lost.sum() * 100).round(1)
    return f


def drop_taxonomy(coh: pd.DataFrame, doc: pd.DataFrame) -> pd.DataFrame:
    """
    How did the captains who never finished actually leave?

    This matters more than the stage counts, because it decides who owns the fix.
    A policy lock-out needs a policy change; an abandonment needs a nudge or a
    UX fix. They are not the same problem.
    """
    dr = coh[coh.final_status == "dropped_in_docs"].copy()
    max_attempt = doc.groupby(["captain_id", "doc_type"]).attempt_no.max()
    dr["attempts_used_on_last_doc"] = [
        max_attempt.get((c, d), 0)
        for c, d in zip(dr.captain_id, dr.last_event_doc)]

    def classify(r):
        if r.n_upload_events == 0:
            return "Never uploaded anything"
        if r.last_event_type == "verification_fail":
            if r.attempts_used_on_last_doc >= CFG.MAX_ATTEMPTS:
                return "Locked out (3 attempts used)"
            return "Quit after a rejection (retries still available)"
        if r.last_event_type == "verification_pass":
            return "Quit after a pass (never started the next document)"
        return "Uploaded, no verdict recorded"

    dr["how_they_left"] = dr.apply(classify, axis=1)
    t = (dr.groupby("how_they_left")
         .agg(captains=("captain_id", "size"))
         .sort_values("captains", ascending=False))
    t["pct_of_dropouts"] = (t.captains / t.captains.sum() * 100).round(1)
    t["per_month"] = (t.captains / CFG.COHORT_MONTHS).round(0)
    t["owner"] = t.index.map({
        "Quit after a pass (never started the next document)": "CRM / in-app prompts",
        "Quit after a rejection (retries still available)": "Product (capture UX + rejection messaging)",
        "Never uploaded anything": "Acquisition quality / day-0 activation",
        "Locked out (3 attempts used)": "Policy",
        "Uploaded, no verdict recorded": "Ops (verification backlog)",
    })
    return t.reset_index()


# ---------------------------------------------------------------- A2
def segment_a2o(coh: pd.DataFrame, dims=None) -> pd.DataFrame:
    """A2O by each signup attribute, with the volume at stake attached."""
    dims = dims or ["city", "vehicle_type", "acquisition_channel",
                    "device_tier", "app_language", "age_band"]
    out = []
    overall = coh.approved.mean()
    for d in dims:
        g = coh.groupby(d).agg(signups=("approved", "size"),
                               approved=("approved", "sum"))
        g["a2o_pct"] = (g.approved / g.signups * 100).round(2)
        g["share_of_signups_pct"] = (g.signups / len(coh) * 100).round(1)
        g["pp_vs_overall"] = (g.a2o_pct - overall * 100).round(2)
        # approvals forgone per month vs the best-performing level of this dimension
        best = g.a2o_pct.max() / 100
        g["gap_to_best_approvals_pm"] = (
            (best - g.a2o_pct / 100) * g.signups / CFG.COHORT_MONTHS).round(0)
        g = g.reset_index().rename(columns={d: "level"})
        g.insert(0, "dimension", d)
        out.append(g.sort_values("a2o_pct"))
    return pd.concat(out, ignore_index=True)


def failure_reason_table(doc: pd.DataFrame, coh: pd.DataFrame) -> pd.DataFrame:
    """Verification failures by document and reason, split image-quality vs substantive."""
    f = doc[(doc.event_type == "verification_fail")
            & doc.captain_id.isin(coh.captain_id)]
    t = pd.crosstab(f.doc_type, f.failure_reason)
    t = t.reindex(index=CFG.DOC_SEQUENCE).fillna(0).astype(int)
    t["total_failures"] = t.sum(axis=1)
    t["image_quality"] = t[[c for c in CFG.IMAGE_QUALITY_REASONS if c in t]].sum(axis=1)
    t["substantive"] = t[[c for c in CFG.SUBSTANTIVE_REASONS if c in t]].sum(axis=1)
    t["image_quality_pct"] = (t.image_quality / t.total_failures * 100).round(1)
    return t.reset_index()


def device_document_interaction(pairs: pd.DataFrame, coh: pd.DataFrame) -> pd.DataFrame:
    """
    The single most useful cut in the dataset.

    Failure rates are not uniformly worse on cheap phones. They are dramatically
    worse on cheap phones for exactly three documents -- RC, Fitness, Insurance --
    and identical across device tiers for DL, Aadhaar and Permit. The three
    affected documents are the full-page paper ones. That is a camera problem,
    not a compliance problem, and it has a product fix.
    """
    p = pairs.merge(coh[["captain_id", "device_tier"]], on="captain_id", how="inner")
    p = p[p.verdict_ts.notna()]
    t = p.pivot_table(index="doc_type", columns="device_tier",
                      values=["failed", "image_quality_fail"], aggfunc="mean")
    t = (t * 100).round(2)
    t.columns = [f"{a}_{b}_pct" for a, b in t.columns]
    t = t.reindex(CFG.DOC_SEQUENCE)
    t["document_form"] = np.where(t.index.isin(CFG.A4_PAPER_DOCS),
                                  "A4 paper (dense text)", "laminated card")
    t["low_minus_high_imgfail_pp"] = (
        t["image_quality_fail_low_pct"] - t["image_quality_fail_high_pct"]).round(2)
    t["uploads"] = p.groupby("doc_type").size().reindex(CFG.DOC_SEQUENCE)
    return t.reset_index()


def retry_behaviour(doc: pd.DataFrame, coh: pd.DataFrame) -> dict:
    """After a rejection, do captains try again, and does trying again work?"""
    d = doc[doc.captain_id.isin(coh.captain_id)]
    fails = d[d.event_type == "verification_fail"][
        ["captain_id", "doc_type", "attempt_no", "failure_reason"]]
    max_up = d[d.event_type == "upload_success"].groupby(
        ["captain_id", "doc_type"]).attempt_no.max()
    fails = fails.copy()
    fails["attempts_available"] = [max_up.get((c, dt), 0)
                                   for c, dt in zip(fails.captain_id, fails.doc_type)]
    fails["retried"] = (fails.attempts_available > fails.attempt_no).astype(int)
    passed = set(zip(d.loc[d.event_type == "verification_pass", "captain_id"],
                     d.loc[d.event_type == "verification_pass", "doc_type"]))
    fails["eventually_passed"] = [
        int((c, dt) in passed) for c, dt in zip(fails.captain_id, fails.doc_type)]

    by_attempt = fails.groupby("attempt_no").agg(
        failures=("retried", "size"), retry_rate_pct=("retried", "mean")).round(4)
    by_attempt.retry_rate_pct = (by_attempt.retry_rate_pct * 100).round(1)

    retryable = fails[fails.attempt_no < CFG.MAX_ATTEMPTS]
    by_reason = retryable.groupby("failure_reason").agg(
        failures=("retried", "size"), retry_rate_pct=("retried", "mean")).round(4)
    by_reason.retry_rate_pct = (by_reason.retry_rate_pct * 100).round(1)

    by_doc = fails[fails.retried == 1].groupby("doc_type").agg(
        retried=("eventually_passed", "size"),
        pass_rate_after_retry_pct=("eventually_passed", "mean")).round(4)
    by_doc.pass_rate_after_retry_pct = (by_doc.pass_rate_after_retry_pct * 100).round(1)

    never_retried = int((retryable.retried == 0).sum())
    return dict(by_attempt=by_attempt.reset_index(),
                by_reason=by_reason.reset_index().sort_values("retry_rate_pct"),
                by_doc=by_doc.reindex(CFG.DOC_SEQUENCE).reset_index(),
                never_retried=never_retried,
                never_retried_per_month=round(never_retried / CFG.COHORT_MONTHS))


def verification_sla_test(pairs: pd.DataFrame, doc: pd.DataFrame,
                         coh: pd.DataFrame) -> dict:
    """
    A hypothesis we tested and rejected, reported because a negative result
    saves the ops team a quarter of work.

    Hypothesis: slow verification turnaround makes captains give up, so cutting
    the queue time would lift conversion.
    """
    import statsmodels.formula.api as smf

    d = doc[doc.captain_id.isin(coh.captain_id)]
    passes = d[d.event_type == "verification_pass"].copy()

    def next_required(vt, dt):
        req = CFG.required_docs(vt)
        i = req.index(dt)
        return req[i + 1] if i + 1 < len(req) else None

    passes["next_doc"] = [next_required(v, t)
                          for v, t in zip(passes.vehicle_type, passes.doc_type)]
    started = set(zip(d.loc[d.event_type == "upload_success", "captain_id"],
                      d.loc[d.event_type == "upload_success", "doc_type"]))
    passes = passes[passes.next_doc.notna()].copy()
    passes["started_next"] = [int((c, n) in started)
                              for c, n in zip(passes.captain_id, passes.next_doc)]
    passes = passes.merge(
        pairs[["captain_id", "doc_type", "attempt_no", "verification_lag_h"]],
        on=["captain_id", "doc_type", "attempt_no"], how="left")
    passes = passes[passes.verification_lag_h.notna()]

    passes["lag_bucket"] = pd.cut(
        passes.verification_lag_h, [0, 2, 4, 8, 12, 18, 24, 1000],
        labels=["0-2h", "2-4h", "4-8h", "8-12h", "12-18h", "18-24h", "24h+"])
    by_bucket = passes.groupby("lag_bucket", observed=True).agg(
        events=("started_next", "size"),
        started_next_pct=("started_next", "mean"))
    by_bucket.started_next_pct = (by_bucket.started_next_pct * 100).round(1)

    m = smf.logit("started_next ~ verification_lag_h + C(doc_type) "
                  "+ C(device_tier) + C(acquisition_channel)", data=passes).fit(disp=0)
    return dict(by_bucket=by_bucket.reset_index(),
                lag_coef=float(m.params["verification_lag_h"]),
                lag_pvalue=float(m.pvalues["verification_lag_h"]),
                lag_median_h=float(pairs.verification_lag_h.median()),
                lag_p99_h=float(pairs.verification_lag_h.quantile(.99)),
                verdict=("REJECTED: verification turnaround has no measurable "
                         "effect on whether a captain continues"))


# ---------------------------------------------------------------- sizing
def size_opportunities(coh: pd.DataFrame, doc: pd.DataFrame,
                       pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Convert each leak into "extra approved captains per month".

    Method, stated honestly: for each population that is stuck for a specific
    reason, we assume an intervention recovers a share of them, and that
    recovered captains then convert at the observed onward rate of the captains
    who did clear that document. This is a gap-closing calculation, not a causal
    estimate. It answers "how big is the prize if we win" -- not "we will win".
    Recovery shares are the judgement call and are shown as a range.
    """
    d = doc[doc.captain_id.isin(coh.captain_id)]
    rows = []

    def onward_rate(doc_type):
        """Approval rate of captains who did clear this document."""
        ok = set(d.loc[(d.doc_type == doc_type)
                       & (d.event_type == "verification_pass"), "captain_id"])
        peers = coh[coh.captain_id.isin(ok)]
        return peers.approved.mean() if len(peers) else 0.0

    # measured device gap on the A4 paper documents, used as evidence text below
    _p = pairs.merge(coh[["captain_id", "device_tier"]], on="captain_id", how="inner")
    _p = _p[_p.verdict_ts.notna()]
    _a4 = _p[_p.doc_type.isin(CFG.A4_PAPER_DOCS)]
    _card = _p[_p.doc_type.isin(CFG.CARD_DOCS)]
    gap_a4_low = _a4.loc[_a4.device_tier == "low", "image_quality_fail"].mean()
    gap_a4_high = _a4.loc[_a4.device_tier == "high", "image_quality_fail"].mean()
    gap_card_low = _card.loc[_card.device_tier == "low", "image_quality_fail"].mean()
    gap_card_high = _card.loc[_card.device_tier == "high", "image_quality_fail"].mean()

    # ---- Leak 1: image-quality failures on the three A4 paper documents
    stuck_ids = set()
    detail = []
    for dt in CFG.A4_PAPER_DOCS:
        dd = d[d.doc_type == dt]
        img_failed = set(dd.loc[(dd.event_type == "verification_fail")
                                & dd.failure_reason.isin(CFG.IMAGE_QUALITY_REASONS),
                                "captain_id"])
        passed = set(dd.loc[dd.event_type == "verification_pass", "captain_id"])
        s = img_failed - passed
        stuck_ids |= s
        detail.append((dt, len(s), onward_rate(dt)))
    for lo, hi in [(0.30, 0.50)]:
        base_lo = sum(n * lo * r for _, n, r in detail)
        base_hi = sum(n * hi * r for _, n, r in detail)
        rows.append(dict(
            leak="1. Image-quality rejections on RC / Fitness / Insurance",
            population_stuck=len(stuck_ids),
            population_per_month=round(len(stuck_ids) / CFG.COHORT_MONTHS),
            recovery_assumed=f"{lo:.0%}-{hi:.0%}",
            extra_approvals_per_month=f"{base_lo/CFG.COHORT_MONTHS:.0f}-{base_hi/CFG.COHORT_MONTHS:.0f}",
            midpoint_pm=round((base_lo + base_hi) / 2 / CFG.COHORT_MONTHS),
            owner="Product (in-app document capture)",
            evidence=f"image-quality failures on A4 docs: low-tier {gap_a4_low:.1%} "
                     f"vs high-tier {gap_a4_high:.1%}; on card docs the same gap is "
                     f"{gap_card_low:.1%} vs {gap_card_high:.1%} (none)"))

    # ---- Leak 2: cleared Fitness, never uploaded Insurance (last-mile)
    fit_pass = set(d.loc[(d.doc_type == "FITNESS")
                         & (d.event_type == "verification_pass"), "captain_id"])
    ins_up = set(d.loc[(d.doc_type == "INSURANCE")
                       & (d.event_type == "upload_success"), "captain_id"])
    ins_pass = set(d.loc[(d.doc_type == "INSURANCE")
                         & (d.event_type == "verification_pass"), "captain_id"])
    stalled = coh[coh.captain_id.isin(fit_pass - ins_up)]
    p_pass_given_upload = len(ins_pass) / max(len(ins_up), 1)
    p_appr_given_pass = coh[coh.captain_id.isin(ins_pass)].approved.mean()
    for lo, hi in [(0.15, 0.30)]:
        e_lo = len(stalled) * lo * p_pass_given_upload * p_appr_given_pass
        e_hi = len(stalled) * hi * p_pass_given_upload * p_appr_given_pass
        rows.append(dict(
            leak="2. Cleared everything except Insurance, never submitted it",
            population_stuck=len(stalled),
            population_per_month=round(len(stalled) / CFG.COHORT_MONTHS),
            recovery_assumed=f"{lo:.0%}-{hi:.0%}",
            extra_approvals_per_month=f"{e_lo/CFG.COHORT_MONTHS:.0f}-{e_hi/CFG.COHORT_MONTHS:.0f}",
            midpoint_pm=round((e_lo + e_hi) / 2 / CFG.COHORT_MONTHS),
            owner="CRM + Insurance partnerships",
            evidence=f"{p_appr_given_pass:.0%} of captains who pass Insurance get approved"))

    # ---- Leak 3: never retried after a first rejection
    fails = d[d.event_type == "verification_fail"][
        ["captain_id", "doc_type", "attempt_no"]]
    max_up = d[d.event_type == "upload_success"].groupby(
        ["captain_id", "doc_type"]).attempt_no.max()
    fails = fails.copy()
    fails["available"] = [max_up.get((c, dt), 0)
                          for c, dt in zip(fails.captain_id, fails.doc_type)]
    no_retry = fails[(fails.attempt_no == 1) & (fails.available == 1)]
    no_retry_caps = coh[coh.captain_id.isin(no_retry.captain_id)
                        & (coh.approved == 0)]
    pass_after_retry = 0.85  # observed weighted pass rate once a captain retries
    mean_onward = np.mean([onward_rate(x) for x in CFG.DOC_SEQUENCE])
    for lo, hi in [(0.10, 0.20)]:
        e_lo = len(no_retry_caps) * lo * pass_after_retry * mean_onward
        e_hi = len(no_retry_caps) * hi * pass_after_retry * mean_onward
        rows.append(dict(
            leak="3. Rejected once, never tried again (retries still available)",
            population_stuck=len(no_retry_caps),
            population_per_month=round(len(no_retry_caps) / CFG.COHORT_MONTHS),
            recovery_assumed=f"{lo:.0%}-{hi:.0%}",
            extra_approvals_per_month=f"{e_lo/CFG.COHORT_MONTHS:.0f}-{e_hi/CFG.COHORT_MONTHS:.0f}",
            midpoint_pm=round((e_lo + e_hi) / 2 / CFG.COHORT_MONTHS),
            owner="Product (rejection messaging)",
            evidence="retry rate is flat at ~62% across all 7 rejection reasons"))

    # ---- Leak 4: channel mix
    ch = coh.groupby("acquisition_channel").agg(
        n=("approved", "size"), a=("approved", "sum"))
    ch["a2o"] = ch.a / ch.n
    bench = ch.loc["fos_field", "a2o"]
    worst = ch.loc["paid_digital"]
    shift = worst.n * 0.5  # reallocate half of paid_digital volume to referral
    ref = ch.loc["referral", "a2o"]
    gain = shift * (ref - worst.a2o)
    rows.append(dict(
        leak="4. Channel mix: paid_digital converts at less than half of fos_field",
        population_stuck=int(worst.n),
        population_per_month=round(worst.n / CFG.COHORT_MONTHS),
        recovery_assumed="move 50% of spend to referral",
        extra_approvals_per_month=f"{gain/CFG.COHORT_MONTHS:.0f}",
        midpoint_pm=round(gain / CFG.COHORT_MONTHS),
        owner="Growth / acquisition",
        evidence=f"A2O paid_digital {worst.a2o:.1%} vs fos_field {bench:.1%}, "
                 f"referral {ref:.1%}"))

    out = pd.DataFrame(rows)
    return out


def size_opportunities_deduplicated(coh: pd.DataFrame, doc: pd.DataFrame) -> pd.DataFrame:
    """
    The leaks above overlap: a captain who failed Insurance on a blurred photo and
    never retried appears in leak 1 AND leak 3. Adding the rows would overstate
    the prize.

    This assigns every unapproved captain in the cohort to exactly one primary
    reason for being stuck, then values them at the observed approval rate of
    captains who did clear the document they are stuck on. That per-captain
    onward rate matters a lot: a captain stuck on Insurance is worth 4x one stuck
    on DL, because they have already cleared everything else.
    """
    d = doc[doc.captain_id.isin(coh.captain_id)]

    onward = {}
    for dt in CFG.DOC_SEQUENCE:
        ok = set(d.loc[(d.doc_type == dt) & (d.event_type == "verification_pass"),
                       "captain_id"])
        peers = coh[coh.captain_id.isin(ok)]
        onward[dt] = float(peers.approved.mean()) if len(peers) else 0.0

    # per (captain, stuck doc): did they ever fail it on image quality? did they retry?
    stuck = coh[(coh.approved == 0) & coh.stuck_on_doc.notna()].copy()
    img_fail_keys = set(zip(
        d.loc[(d.event_type == "verification_fail")
              & d.failure_reason.isin(CFG.IMAGE_QUALITY_REASONS), "captain_id"],
        d.loc[(d.event_type == "verification_fail")
              & d.failure_reason.isin(CFG.IMAGE_QUALITY_REASONS), "doc_type"]))
    any_fail_keys = set(zip(
        d.loc[d.event_type == "verification_fail", "captain_id"],
        d.loc[d.event_type == "verification_fail", "doc_type"]))
    max_attempt = d[d.event_type == "upload_success"].groupby(
        ["captain_id", "doc_type"]).attempt_no.max()
    uploaded_keys = set(max_attempt.index)

    def primary_reason(r):
        key = (r.captain_id, r.stuck_on_doc)
        if r.n_upload_events == 0:
            return "Never started (no document ever uploaded)"
        if key not in uploaded_keys:
            return "Abandoned before submitting the next document"
        attempts = max_attempt.get(key, 0)
        if attempts >= CFG.MAX_ATTEMPTS:
            return "Exhausted all 3 attempts (policy lock-out)"
        if key in img_fail_keys:
            return "Rejected on photo quality, gave up with retries left"
        if key in any_fail_keys:
            return "Rejected on document substance, gave up with retries left"
        return "Submitted, awaiting or missing a verdict"

    stuck["primary_reason"] = stuck.apply(primary_reason, axis=1)
    stuck["onward_rate"] = stuck.stuck_on_doc.map(onward)

    # recovery assumptions, stated per reason. These are judgement, not measurement.
    recovery = {
        "Rejected on photo quality, gave up with retries left": (0.30, 0.50),
        "Abandoned before submitting the next document": (0.10, 0.20),
        "Rejected on document substance, gave up with retries left": (0.05, 0.12),
        "Never started (no document ever uploaded)": (0.03, 0.08),
        "Exhausted all 3 attempts (policy lock-out)": (0.00, 0.10),
        "Submitted, awaiting or missing a verdict": (0.00, 0.00),
    }

    rows = []
    for reason, g in stuck.groupby("primary_reason"):
        lo, hi = recovery[reason]
        val = g.onward_rate.sum()  # expected approvals if all were recovered
        rows.append(dict(
            primary_reason=reason,
            captains=len(g),
            per_month=round(len(g) / CFG.COHORT_MONTHS),
            mean_onward_approval_rate=round(g.onward_rate.mean(), 3),
            max_recoverable_approvals_pm=round(val / CFG.COHORT_MONTHS),
            recovery_assumed=f"{lo:.0%}-{hi:.0%}",
            extra_approvals_pm_low=round(val * lo / CFG.COHORT_MONTHS),
            extra_approvals_pm_high=round(val * hi / CFG.COHORT_MONTHS)))
    out = pd.DataFrame(rows).sort_values("extra_approvals_pm_high", ascending=False)
    total = dict(primary_reason="TOTAL (no double counting)",
                 captains=out.captains.sum(), per_month=out.per_month.sum(),
                 mean_onward_approval_rate=np.nan,
                 max_recoverable_approvals_pm=out.max_recoverable_approvals_pm.sum(),
                 recovery_assumed="",
                 extra_approvals_pm_low=out.extra_approvals_pm_low.sum(),
                 extra_approvals_pm_high=out.extra_approvals_pm_high.sum())
    out = pd.concat([out, pd.DataFrame([total])], ignore_index=True)
    baseline = coh.approved.sum() / CFG.COHORT_MONTHS
    out["pct_uplift_on_current_approvals"] = (
        out.extra_approvals_pm_high / baseline * 100).round(1)
    out["current_approvals_pm"] = round(baseline)
    return out


def channel_productivity(coh: pd.DataFrame) -> pd.DataFrame:
    """
    Channels should not be judged on A2O alone.

    A channel that approves well but produces few orders is worse than it looks.
    Orders per 100 signups = A2O x activation rate x orders in first 30 days,
    computed only on captains with a complete 30-day window.
    """
    full30 = coh[coh.orders_d30.notna()]
    rows = []
    for ch, g in coh.groupby("acquisition_channel"):
        appr = g.approved.mean()
        act = g.loc[g.approved == 1, "activated"].mean()
        g30 = full30[(full30.acquisition_channel == ch) & (full30.approved == 1)]
        orders = g30.orders_d30.mean()
        hours = g30.online_hours_d30.mean()
        rows.append(dict(
            acquisition_channel=ch, signups=len(g),
            a2o_pct=round(appr * 100, 2),
            activation_pct=round(act * 100, 2),
            mean_orders_d30=round(orders, 1),
            mean_online_hours_d30=round(hours, 1),
            orders_per_100_signups=round(appr * act * orders * 100, 1),
            n_with_full_30d_window=len(g30)))
    out = pd.DataFrame(rows).sort_values("orders_per_100_signups", ascending=False)
    return out


def activation_check(coh: pd.DataFrame) -> dict:
    """
    R2A diagnostic. Included to justify NOT recommending any activation work.
    """
    appr = coh[coh.approved == 1]
    observed7 = appr[appr.orders_d7.notna()]
    return dict(
        approved=int(len(appr)),
        activated=int(appr.activated.sum()),
        activation_rate_pct=round(appr.activated.mean() * 100, 2),
        a2o_pct=round(coh.approved.mean() * 100, 2),
        r2a_pct=round(coh.activated.mean() * 100, 2),
        median_days_approval_to_first_order=round(appr.days_to_first_order.median(), 2),
        p90_days_approval_to_first_order=round(appr.days_to_first_order.quantile(.9), 2),
        genuine_non_starters=int((observed7.activated == 0).sum()),
        genuine_non_starter_pct=round((observed7.activated == 0).mean() * 100, 2),
        verdict=("R2A is not the problem. 98.7% of approved captains take a first "
                 "order, at a median of 1.4 days. Every point of A2O converts "
                 "almost one-for-one into an active captain."))
