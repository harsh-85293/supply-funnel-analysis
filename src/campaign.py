"""
Part A3: assess the claim that CAMP_WA_002 is a big win and should be scaled 5x.

Short version of the method:

The campaign is not sent at random. It has a trigger: no captain receives it
until they have already cleared two documents (DL and RC). Zero of the 7,981
recipients in the cohort received it before that point. Clearing DL and RC is
the single hardest part of onboarding -- 36% of all funnel loss happens at or
before RC. So the recipient list is, by construction, a list of captains who
already survived the worst of the funnel.

Comparing recipients to all non-recipients therefore compares survivors to
everyone, which is selection on the outcome. This module walks the estimate down
from the naive number to a defensible one, and then runs two placebo tests to
check whether even the residual is real.
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import config as CFG

CAMP = CFG.CAMPAIGN_UNDER_REVIEW


# ---------------------------------------------------------------- trigger recovery
def recover_trigger(coh: pd.DataFrame, doc: pd.DataFrame) -> dict:
    """Reverse-engineer the eligibility rule from the data."""
    d = doc[doc.captain_id.isin(coh.captain_id)]
    passes = d[d.event_type == "verification_pass"]
    sent = coh[coh[f"{CAMP}_received"] == 1].copy()

    # documents already cleared at the moment the message was sent
    pj = passes.merge(coh[["captain_id", f"{CAMP}_sent_ts"]], on="captain_id")
    before = (pj[pj.event_ts < pj[f"{CAMP}_sent_ts"]]
              .groupby("captain_id").doc_type.nunique())
    sent["docs_cleared_before_send"] = sent.captain_id.map(before).fillna(0).astype(int)

    lag_h = ((sent[f"{CAMP}_sent_ts"] - sent.t_doc2_pass).dt.total_seconds() / 3600)

    eligible = coh[coh.n_docs_passed >= 2]
    coverage = coh[f"{CAMP}_received"].sum() / len(eligible)

    return dict(
        recipients=int(coh[f"{CAMP}_received"].sum()),
        recipients_with_fewer_than_2_docs_cleared=int(
            ((coh[f"{CAMP}_received"] == 1) & (coh.n_docs_passed < 2)).sum()),
        min_docs_cleared_before_send=int(sent.docs_cleared_before_send.min()),
        docs_cleared_before_send_distribution=dict(
            sent.docs_cleared_before_send.value_counts().sort_index()),
        median_lag_hours_after_2nd_doc=round(float(lag_h.median()), 2),
        p95_lag_hours_after_2nd_doc=round(float(lag_h.quantile(.95)), 2),
        max_lag_hours_after_2nd_doc=round(float(lag_h.max()), 2),
        negative_lags=int((lag_h < 0).sum()),
        eligible_population=int(len(eligible)),
        coverage_of_eligible_pct=round(coverage * 100, 1),
        conclusion=("CAMP_WA_002 is triggered by clearing the 2nd document. It is "
                    "sent a median of 6.7 hours later. It is structurally "
                    "impossible for it to influence the DL or RC stages, where "
                    "most of the funnel loss occurs."))


def receipt_by_progress(coh: pd.DataFrame) -> pd.DataFrame:
    """P(received) against documents eventually cleared -- shows the cliff at 2."""
    t = coh.groupby("n_docs_passed").agg(
        captains=(f"{CAMP}_received", "size"),
        received=(f"{CAMP}_received", "sum"))
    t["received_pct"] = (t.received / t.captains * 100).round(1)
    return t.reset_index()


# ---------------------------------------------------------------- estimate ladder
def _rate(g):
    return g.approved.mean()


def estimate_ladder(coh: pd.DataFrame) -> pd.DataFrame:
    """
    Walk from the naive comparison to the adjusted one, showing what each
    correction removes. This table is the argument.
    """
    import statsmodels.formula.api as smf
    rows = []

    # -- step 0: naive, recipients vs everyone else
    g = coh.groupby(f"{CAMP}_received")
    r1, r0 = _rate(g.get_group(1)), _rate(g.get_group(0))
    rows.append(dict(step="0. Naive: recipients vs all other signups",
                     treated_n=len(g.get_group(1)), control_n=len(g.get_group(0)),
                     treated_pct=round(r1 * 100, 2), control_pct=round(r0 * 100, 2),
                     lift_pp=round((r1 - r0) * 100, 2),
                     relative_lift_pct=round((r1 / r0 - 1) * 100, 1),
                     what_this_ignores="the campaign's eligibility trigger"))

    # -- step 1: restrict to the eligible population (cleared >= 2 docs)
    el = coh[coh.n_docs_passed >= 2]
    g = el.groupby(f"{CAMP}_received")
    r1, r0 = _rate(g.get_group(1)), _rate(g.get_group(0))
    rows.append(dict(step="1. Restrict both arms to captains who cleared 2+ documents",
                     treated_n=len(g.get_group(1)), control_n=len(g.get_group(0)),
                     treated_pct=round(r1 * 100, 2), control_pct=round(r0 * 100, 2),
                     lift_pp=round((r1 - r0) * 100, 2),
                     relative_lift_pct=round((r1 / r0 - 1) * 100, 1),
                     what_this_ignores="controls who quit before the send window"))

    # -- step 2: align on survival to the send point (removes immortal-time bias)
    el = el.copy()
    el["hours_alive_after_doc2"] = (
        (el.exit_ts - el.t_doc2_pass).dt.total_seconds() / 3600)
    LM = 7  # median send lag, rounded
    al = el[el.hours_alive_after_doc2 >= LM]
    g = al.groupby(f"{CAMP}_received")
    r1, r0 = _rate(g.get_group(1)), _rate(g.get_group(0))
    rows.append(dict(step=f"2. Also require both arms to be active {LM}h after doc 2",
                     treated_n=len(g.get_group(1)), control_n=len(g.get_group(0)),
                     treated_pct=round(r1 * 100, 2), control_pct=round(r0 * 100, 2),
                     lift_pp=round((r1 - r0) * 100, 2),
                     relative_lift_pct=round((r1 / r0 - 1) * 100, 1),
                     what_this_ignores="residual composition differences"))

    # -- step 3: covariate adjustment
    ame, ci_lo, ci_hi, model = _adjusted_effect(al)
    rows.append(dict(step="3. Plus covariate adjustment (logit, avg marginal effect)",
                     treated_n=len(g.get_group(1)), control_n=len(g.get_group(0)),
                     treated_pct=np.nan, control_pct=np.nan,
                     lift_pp=round(ame * 100, 2),
                     relative_lift_pct=np.nan,
                     what_this_ignores="unobserved reasons for being on the list"))
    return pd.DataFrame(rows), dict(ame_pp=round(ame * 100, 2),
                                    ci_pp=(round(ci_lo * 100, 2), round(ci_hi * 100, 2)),
                                    aligned_frame=al)


def _adjusted_effect(al: pd.DataFrame):
    """Logit with a bootstrap CI on the average marginal effect."""
    import statsmodels.formula.api as smf
    df = al.copy()
    df["doc2_month"] = df.t_doc2_pass.dt.to_period("M").astype(str)
    df["hours_signup_to_doc2"] = (
        (df.t_doc2_pass - df.signup_ts).dt.total_seconds() / 3600)
    df["treat"] = df[f"{CAMP}_received"]
    f = ("approved ~ treat + C(city) + C(vehicle_type) + C(acquisition_channel) "
         "+ C(device_tier) + C(age_band) + C(app_language) + C(doc2_month) "
         "+ hours_signup_to_doc2")
    m = smf.logit(f, data=df).fit(disp=0)
    j = list(m.params.index).index("treat")
    ex1, ex0 = m.model.exog.copy(), m.model.exog.copy()
    ex1[:, j], ex0[:, j] = 1, 0
    ame = float((m.model.predict(m.params.values, exog=ex1)
                 - m.model.predict(m.params.values, exog=ex0)).mean())
    rng = np.random.default_rng(CFG.RANDOM_SEED)
    draws = rng.multivariate_normal(m.params.values, m.cov_params().values, 600)
    sims = np.array([(m.model.predict(d, exog=ex1)
                      - m.model.predict(d, exog=ex0)).mean() for d in draws])
    return ame, float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5)), m


def covariate_balance(al: pd.DataFrame) -> pd.DataFrame:
    """
    Is receipt random among eligible captains? On everything we can observe, yes.
    That is what makes the residual gap interesting rather than obviously spurious.
    """
    rows = []
    for dim in ["city", "vehicle_type", "acquisition_channel", "device_tier",
                "app_language", "age_band"]:
        t = pd.crosstab(al[dim], al[f"{CAMP}_received"], normalize="columns")
        for lvl in t.index:
            rows.append(dict(dimension=dim, level=lvl,
                             control_share_pct=round(t.loc[lvl, 0] * 100, 2),
                             treated_share_pct=round(t.loc[lvl, 1] * 100, 2),
                             abs_diff_pp=round(abs(t.loc[lvl, 1] - t.loc[lvl, 0]) * 100, 2)))
    # pre-treatment continuous covariates
    al = al.copy()
    al["hours_signup_to_doc2"] = (
        (al.t_doc2_pass - al.signup_ts).dt.total_seconds() / 3600)
    for var in ["hours_signup_to_doc2", "n_failures"]:
        m = al.groupby(f"{CAMP}_received")[var].mean()
        rows.append(dict(dimension="pre-treatment mean", level=var,
                         control_share_pct=round(m[0], 3),
                         treated_share_pct=round(m[1], 3),
                         abs_diff_pp=round(abs(m[1] - m[0]), 3)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- placebo tests
def placebo_tests(al: pd.DataFrame) -> pd.DataFrame:
    """
    Two tests that a real messaging effect would have to pass.

    A WhatsApp message can only work if it arrives and is read. So:
      - captains whose message failed to deliver should behave like controls
      - captains who clicked should do better than those who did not

    Neither holds. But we also report each test's minimum detectable effect,
    because a null on an underpowered test is not proof of absence.
    """
    rows = []
    t = al[al[f"{CAMP}_received"] == 1]
    ctrl_rate = al.loc[al[f"{CAMP}_received"] == 0, "approved"].mean()

    def two_prop(a, b, label, expectation):
        pa, pb = a.mean(), b.mean()
        na, nb = len(a), len(b)
        pooled = (a.sum() + b.sum()) / (na + nb)
        se = np.sqrt(pooled * (1 - pooled) * (1 / na + 1 / nb))
        z = (pa - pb) / se if se > 0 else np.nan
        p = 2 * stats.norm.sf(abs(z))
        mde = (1.96 + 0.84) * np.sqrt(2 * pooled * (1 - pooled) / min(na, nb))
        rows.append(dict(test=label, group_a_n=na, group_a_pct=round(pa * 100, 2),
                         group_b_n=nb, group_b_pct=round(pb * 100, 2),
                         diff_pp=round((pa - pb) * 100, 2),
                         ci95_pp=f"[{(pa-pb)*100-1.96*se*100:.2f}, {(pa-pb)*100+1.96*se*100:.2f}]",
                         p_value=round(p, 3),
                         mde_80pct_power_pp=round(mde * 100, 2),
                         expectation_if_message_works=expectation))

    two_prop(t.loc[t[f"{CAMP}_delivered"] == 1, "approved"],
             t.loc[t[f"{CAMP}_delivered"] == 0, "approved"],
             "Delivered vs failed-to-deliver (both on the list)",
             "delivered should beat undelivered by the full effect size")
    d = t[t[f"{CAMP}_delivered"] == 1]
    two_prop(d.loc[d[f"{CAMP}_clicked"] == 1, "approved"],
             d.loc[d[f"{CAMP}_clicked"] == 0, "approved"],
             "Clicked vs not clicked (delivered only)",
             "clickers should beat non-clickers (dose-response)")

    out = pd.DataFrame(rows)
    out["control_arm_rate_pct"] = round(ctrl_rate * 100, 2)
    return out


# ---------------------------------------------------------------- scale-up test
def scale_up_headroom(coh: pd.DataFrame, effect_pp_range=(0.0, 3.8)) -> dict:
    """
    The claim is not just "it works", it is "give me 5x the budget".

    Even taking the adjusted effect at face value, 5x is arithmetically
    impossible inside the current trigger: the campaign already reaches 55% of
    every captain who becomes eligible.
    """
    eligible = int((coh.n_docs_passed >= 2).sum())
    sent = int(coh[f"{CAMP}_received"].sum())
    unserved = eligible - sent
    lo, hi = effect_pp_range
    return dict(
        eligible_population=eligible,
        currently_sent=sent,
        coverage_pct=round(sent / eligible * 100, 1),
        max_headroom_inside_trigger=f"{eligible/sent:.2f}x",
        requested_scale="5x",
        unserved_eligible=unserved,
        unserved_per_month=round(unserved / CFG.COHORT_MONTHS),
        extra_approvals_pm_at_zero_effect=0,
        extra_approvals_pm_at_upper_bound=round(unserved * hi / 100 / CFG.COHORT_MONTHS),
        conclusion=(
            "Full coverage of the current eligible pool is a 1.8x increase in "
            "sends, not 5x. Spending 5x requires sending to captains who have "
            "not cleared two documents -- a population in which this campaign "
            "has never been tested, and which is where most of the loss is."))


def holdout_design(coh: pd.DataFrame) -> pd.DataFrame:
    """What it would take to actually measure this. Cheap, so there is no excuse."""
    el = coh[coh.n_docs_passed >= 2]
    p = el.approved.mean()
    monthly_eligible = len(el) / CFG.COHORT_MONTHS
    rows = []
    for mde_pp in [2.0, 3.0, 4.0, 5.0]:
        mde = mde_pp / 100
        n_per_arm = ((1.96 + 0.84) ** 2) * 2 * p * (1 - p) / (mde ** 2)
        total = n_per_arm * 2
        # a 50/50 holdout consumes the whole eligible stream
        months_at_full_stream = total / monthly_eligible
        # a 20% holdout only slows the campaign for a fifth of captains
        months_at_20pct_holdout = n_per_arm / (monthly_eligible * 0.20)
        rows.append(dict(
            effect_to_detect_pp=mde_pp,
            n_per_arm=round(n_per_arm),
            total_n=round(total),
            months_with_5050_split=round(months_at_full_stream, 1),
            months_with_20pct_holdout=round(months_at_20pct_holdout, 1)))
    out = pd.DataFrame(rows)
    out["baseline_rate_pct"] = round(p * 100, 2)
    out["eligible_per_month"] = round(monthly_eligible)
    out["recommendation"] = np.where(
        out.effect_to_detect_pp >= 3.0,
        "feasible: run this before scaling", "too small to measure at this volume")
    return out


def other_campaigns(coh: pd.DataFrame, nudges: pd.DataFrame) -> pd.DataFrame:
    """Send timing and raw outcome for every campaign, so WA_002 is in context."""
    n = nudges.merge(coh[["captain_id", "signup_ts", "approved"]], on="captain_id")
    n["days_after_signup"] = (n.sent_ts - n.signup_ts).dt.total_seconds() / 86400
    t = n.groupby("campaign_id").agg(
        sends=("captain_id", "size"),
        captains=("captain_id", "nunique"),
        channel=("channel", "first"),
        median_send_day=("days_after_signup", "median"),
        p05_send_day=("days_after_signup", lambda s: s.quantile(.05)),
        delivered_pct=("delivered", "mean"),
        clicked_pct=("clicked", "mean"),
        raw_a2o_pct=("approved", "mean"))
    for c in ["median_send_day", "p05_send_day"]:
        t[c] = t[c].round(2)
    for c in ["delivered_pct", "clicked_pct", "raw_a2o_pct"]:
        t[c] = (t[c] * 100).round(2)
    t["cohort_a2o_pct"] = round(coh.approved.mean() * 100, 2)
    return t.reset_index().sort_values("median_send_day")
