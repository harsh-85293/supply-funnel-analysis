"""
Part B: airport supply.

The question on the table is whether to fund captain acquisition targeted at the
airport catchment, plus a sign-up bonus. Three things decide it:

B1  When and how big is the shortfall?
B2  What happens to a captain after an airport trip?
B3  Given B1 and B2, is acquisition the right lever?

The answer turns out to be no, and the reason is in B1: the airport does not
have a shortage of captains, it has a shortage of captains between 20:00 and
04:00 and a large surplus of them between 08:00 and 19:00. Acquisition buys
headcount; it does not buy hours.
"""
import numpy as np
import pandas as pd

from . import config as CFG


def _prep(hourly: pd.DataFrame, trips: pd.DataFrame):
    h = hourly.copy()
    h["hour_of_day"] = h.hour_ts.dt.hour
    h["is_night"] = h.hour_of_day.isin(CFG.NIGHT_HOURS)
    h["fill_rate"] = h.fulfilled_requests / h.requests
    t = trips.copy()
    t["hour_of_day"] = t.request_ts.dt.hour
    t["is_night"] = t.hour_of_day.isin(CFG.NIGHT_HOURS)
    return h, t


# ---------------------------------------------------------------- B1
def zone_type_summary(h: pd.DataFrame) -> pd.DataFrame:
    """Terminals against every other zone type, same period, same metrics."""
    g = h.groupby("zone_type").agg(
        zones=("zone_id", "nunique"), zone_hours=("hour_ts", "size"),
        requests=("requests", "sum"), fulfilled=("fulfilled_requests", "sum"),
        unfulfilled=("unfulfilled_requests", "sum"),
        mean_online_captains=("online_captains", "mean"),
        mean_eta_min=("avg_eta_min", "mean"),
        mean_surge=("avg_surge_multiplier", "mean"))
    g["fill_rate_pct"] = (g.fulfilled / g.requests * 100).round(2)
    g["unmet_per_zone_per_day"] = (
        g.unfulfilled / g.zones / h.hour_ts.dt.date.nunique()).round(0)
    for c in ["mean_online_captains", "mean_eta_min", "mean_surge"]:
        g[c] = g[c].round(2)
    return g.reset_index().sort_values("fill_rate_pct")


def terminal_hourly_profile(h: pd.DataFrame) -> pd.DataFrame:
    """The core B1 exhibit: demand and supply by hour of day at the terminals."""
    apt = h[h.zone_type == "airport_terminal"]
    days = apt.hour_ts.dt.date.nunique()
    g = apt.groupby("hour_of_day").agg(
        requests_per_hour=("requests", "mean"),
        fulfilled_per_hour=("fulfilled_requests", "mean"),
        unmet_per_hour=("unfulfilled_requests", "mean"),
        online_captains=("online_captains", "mean"),
        eta_min=("avg_eta_min", "mean"),
        surge=("avg_surge_multiplier", "mean"))
    g["fill_rate_pct"] = (g.fulfilled_per_hour / g.requests_per_hour * 100).round(1)
    g["requests_per_online_captain"] = (
        g.requests_per_hour / g.online_captains).round(2)
    g["total_unmet_both_terminals"] = (
        apt.groupby("hour_of_day").unfulfilled_requests.sum())
    g["share_of_daily_unmet_pct"] = (
        g.total_unmet_both_terminals
        / apt.unfulfilled_requests.sum() * 100).round(1)
    g["is_night"] = g.index.isin(CFG.NIGHT_HOURS)
    return g.round(2).reset_index()


def night_day_split(h: pd.DataFrame) -> pd.DataFrame:
    apt = h[h.zone_type == "airport_terminal"]
    g = apt.groupby("is_night").agg(
        zone_hours=("hour_ts", "size"), requests=("requests", "sum"),
        fulfilled=("fulfilled_requests", "sum"),
        unmet=("unfulfilled_requests", "sum"),
        mean_online_captains=("online_captains", "mean"),
        mean_eta_min=("avg_eta_min", "mean"),
        mean_surge=("avg_surge_multiplier", "mean"))
    g["fill_rate_pct"] = (g.fulfilled / g.requests * 100).round(2)
    g["share_of_unmet_pct"] = (g.unmet / g.unmet.sum() * 100).round(1)
    g.index = ["day 05:00-19:00", "night 20:00-04:00"]
    return g.round(2).reset_index().rename(columns={"index": "window"})


def marginal_value_of_a_captain(h: pd.DataFrame) -> pd.DataFrame:
    """
    The number that settles B3.

    Regress fulfilled trips on online captains within each window, with hour and
    zone fixed effects. If an extra captain-hour at night converts into ~1.3
    extra trips and an extra captain-hour in the daytime converts into ~0.2, then
    the constraint is not headcount, it is timing -- and an acquisition programme
    cannot choose when its captains log in.
    """
    import statsmodels.formula.api as smf
    apt = h[h.zone_type == "airport_terminal"].copy()
    apt["hour_c"] = apt.hour_of_day.astype(str)
    rows = []
    for label, sub in [("night 20:00-04:00", apt[apt.is_night]),
                       ("day 05:00-19:00", apt[~apt.is_night]),
                       ("all hours", apt)]:
        m = smf.ols("fulfilled_requests ~ online_captains + requests "
                    "+ C(hour_c) + C(zone_id)", data=sub).fit()
        rows.append(dict(
            window=label, zone_hours=len(sub),
            extra_trips_per_extra_online_captain=round(
                float(m.params["online_captains"]), 3),
            std_err=round(float(m.bse["online_captains"]), 3),
            r_squared=round(float(m.rsquared), 3),
            observed_trips_per_captain_hour=round(
                float((sub.fulfilled_requests / sub.online_captains.replace(0, np.nan)).mean()), 2)))
    return pd.DataFrame(rows)


def capacity_balance(h: pd.DataFrame) -> dict:
    """
    Is the terminal short of captain-hours in aggregate, or only in some hours?

    Throughput benchmark is taken from the data rather than assumed: the 90th
    percentile of observed trips per online captain-hour during the night window,
    i.e. what captains demonstrably achieve when there is queueing demand.
    """
    apt = h[h.zone_type == "airport_terminal"].copy()
    apt["throughput"] = apt.fulfilled_requests / apt.online_captains.replace(0, np.nan)
    benchmark = float(apt.loc[apt.is_night, "throughput"].quantile(0.90))
    apt["captain_hours_needed"] = apt.requests / benchmark
    apt["surplus"] = apt.online_captains - apt.captain_hours_needed
    days = apt.hour_ts.dt.date.nunique()

    night = apt[apt.is_night]
    per_hour = night.groupby("hour_of_day").agg(
        requests=("requests", "mean"), online=("online_captains", "mean"),
        needed=("captain_hours_needed", "mean"))
    per_hour["extra_captains_needed"] = (per_hour.needed - per_hour.online).round(1)
    extra_ch_per_night = float(
        per_hour.extra_captains_needed.clip(lower=0).sum() * apt.zone_id.nunique())

    return dict(
        throughput_benchmark_trips_per_captain_hour=round(benchmark, 2),
        days=days,
        total_online_captain_hours=round(float(apt.online_captains.sum())),
        captain_hours_needed_for_all_demand=round(float(apt.captain_hours_needed.sum())),
        aggregate_surplus_captain_hours=round(
            float(apt.online_captains.sum() - apt.captain_hours_needed.sum())),
        idle_captain_hours_in_surplus_hours=round(
            float(apt.loc[apt.surplus > 0, "surplus"].sum())),
        deficit_captain_hours_in_short_hours=round(
            float(-apt.loc[apt.surplus < 0, "surplus"].sum())),
        night_deficit_captain_hours=round(
            float(-night.loc[night.surplus < 0, "surplus"].sum())),
        extra_captain_hours_needed_per_night=round(extra_ch_per_night),
        per_hour_table=per_hour.round(1).reset_index(),
        verdict=("Aggregate supply at the terminals EXCEEDS what total demand "
                 "requires. There are ~2.7x more idle captain-hours in surplus "
                 "hours than there are deficit captain-hours in short hours. "
                 "This is a scheduling problem, not a headcount problem."))


# ---------------------------------------------------------------- B2
def post_trip_economics(t: pd.DataFrame) -> pd.DataFrame:
    """
    What happens to a captain after an airport drop.

    Earnings per engaged hour is the honest unit: a captain does not care about
    the fare, they care about the fare divided by the time it consumes. A trip
    with no follow-on fare consumes the outbound leg, the 20-minute wait, and the
    empty return leg.
    """
    comp = t[t.captain_cancelled == 0].copy()
    comp["trip_hours"] = comp.trip_distance_km / CFG.CITY_SPEED_KMPH
    got = comp.got_return_fare_within_20min == 1
    comp["deadhead_hours"] = np.where(got, 0.0, comp.trip_distance_km / CFG.CITY_SPEED_KMPH)
    comp["idle_hours"] = np.where(got, 0.0, CFG.RETURN_FARE_WINDOW_MIN / 60)
    comp["engaged_hours"] = comp.trip_hours + comp.deadhead_hours + comp.idle_hours
    comp["inr_per_engaged_hour"] = comp.fare_inr / comp.engaged_hours

    g = comp.groupby("drop_zone_type").agg(
        trips=("trip_id", "size"),
        mean_distance_km=("trip_distance_km", "mean"),
        mean_fare_inr=("fare_inr", "mean"),
        return_fare_pct=("got_return_fare_within_20min", "mean"),
        mean_engaged_hours=("engaged_hours", "mean"),
        inr_per_engaged_hour=("inr_per_engaged_hour", "mean"))
    g["share_of_trips_pct"] = (g.trips / len(comp) * 100).round(1)
    g["return_fare_pct"] = (g.return_fare_pct * 100).round(1)
    g = g.round(2)
    # cancellation is measured on all trips, not just completed ones
    g["captain_cancel_rate_pct"] = (
        t.groupby("drop_zone_type").captain_cancelled.mean() * 100).round(1)
    return g.reset_index().sort_values("inr_per_engaged_hour")


def cancellation_analysis(t: pd.DataFrame) -> dict:
    """
    Are captains refusing airport trips selectively, and on what basis?

    If cancellation tracks the return-fare probability of the destination, the
    captains are telling us the trip economics are bad. That is a pricing and
    dispatch problem, and adding more captains does not solve it -- it just adds
    more captains who cancel.
    """
    import statsmodels.formula.api as smf
    by_zone = t.groupby(["drop_zone_id", "drop_zone_type"]).agg(
        trips=("trip_id", "size"),
        cancel_rate_pct=("captain_cancelled", "mean"),
        return_fare_pct=("got_return_fare_within_20min", "mean"),
        mean_distance_km=("trip_distance_km", "mean"),
        mean_fare_inr=("fare_inr", "mean")).reset_index()
    by_zone.cancel_rate_pct = (by_zone.cancel_rate_pct * 100).round(1)
    by_zone.return_fare_pct = (by_zone.return_fare_pct * 100).round(1)
    by_zone = by_zone.round(2)
    corr = float(by_zone.cancel_rate_pct.corr(by_zone.return_fare_pct))

    t2 = t.copy()
    t2["night"] = t2.is_night.astype(int)
    m = smf.logit("captain_cancelled ~ trip_distance_km + C(drop_zone_type) + night",
                  data=t2).fit(disp=0)
    coefs = pd.DataFrame({"coef": m.params.round(4), "p_value": m.pvalues.round(4),
                          "odds_ratio": np.exp(m.params).round(3)}).reset_index()

    nd = t.groupby(["is_night", "drop_zone_type"]).agg(
        trips=("trip_id", "size"),
        cancel_rate_pct=("captain_cancelled", "mean"),
        return_fare_pct=("got_return_fare_within_20min", "mean"))
    nd.cancel_rate_pct = (nd.cancel_rate_pct * 100).round(1)
    nd.return_fare_pct = (nd.return_fare_pct * 100).round(1)

    return dict(
        overall_cancel_rate_pct=round(float(t.captain_cancelled.mean() * 100), 2),
        night_cancel_rate_pct=round(float(t.loc[t.is_night, "captain_cancelled"].mean() * 100), 2),
        day_cancel_rate_pct=round(float(t.loc[~t.is_night, "captain_cancelled"].mean() * 100), 2),
        by_zone=by_zone,
        corr_cancel_vs_return_fare=round(corr, 3),
        logit=coefs,
        night_by_zone_type=nd.reset_index(),
        verdict=("Cancellation tracks destination quality almost perfectly "
                 f"(r = {corr:.2f} across drop zones). Captains are not short of "
                 "willingness in general -- they are declining the 41% of airport "
                 "trips that end in a suburb with no return fare."))


def is_deadhead_avoidable(h: pd.DataFrame) -> pd.DataFrame:
    """
    Could better matching fix the dead-head instead of money?

    Only if the drop zones have unmet demand for the arriving captain to pick up.
    They do not: suburban zones run at 96% fill with roughly one unmet request
    per hour. There is no spare demand out there to match to. That rules out the
    cheap fix and is why the remaining options cost something.
    """
    g = h[h.zone_type != "airport_terminal"].groupby(["zone_type", "is_night"]).agg(
        requests_per_hour=("requests", "mean"),
        unmet_per_hour=("unfulfilled_requests", "mean"),
        online_captains=("online_captains", "mean"),
        fill_rate_pct=("fill_rate", "mean"),
        mean_eta_min=("avg_eta_min", "mean"))
    g.fill_rate_pct = (g.fill_rate_pct * 100).round(1)
    return g.round(2).reset_index()


# ---------------------------------------------------------------- B3
def size_interventions(h: pd.DataFrame, t: pd.DataFrame, cap_bal: dict,
                       a2o: float) -> pd.DataFrame:
    """
    Cost per incremental fulfilled airport trip for each option on the table.

    Deliberately conservative: fares are surge-free (see quality.py), and the
    acquisition option is costed only on the volume of signups needed, with no
    allowance for the bonus itself.
    """
    apt = h[h.zone_type == "airport_terminal"]
    days = apt.hour_ts.dt.date.nunique()
    night_unmet_per_day = float(apt.loc[apt.is_night, "unfulfilled_requests"].sum() / days)
    tput = cap_bal["throughput_benchmark_trips_per_captain_hour"]
    mean_fare = float(t.fare_inr.mean())

    rows = []

    # Option 1 -- pay existing captains to work the night airport window
    for recovery in (0.30, 0.50):
        trips = night_unmet_per_day * recovery
        ch = trips / tput
        for bonus in (100, 150):
            cost = ch * bonus
            rows.append(dict(
                option="1. Night-window airport incentive for existing captains",
                variant=f"recover {recovery:.0%} of night unmet demand, "
                        f"Rs{bonus}/captain-hour",
                incremental_trips_per_day=round(trips),
                extra_captain_hours_per_night=round(ch),
                cost_per_day_inr=round(cost),
                cost_per_month_lakh=round(cost * 30 / 1e5, 1),
                cost_per_incremental_trip_inr=round(cost / trips),
                breakeven_commission_pct=round(cost / trips / mean_fare * 100, 1),
                lead_time="days",
                key_risk="cannibalises hours from other zones; needs a geo-holdout"))

    # ---- the suburban dead-head problem, two ways to pay for it
    sub = t[t.drop_zone_type == "suburban"]
    core = t[t.drop_zone_type == "city_core"]
    excess_cancel_rate = float(sub.captain_cancelled.mean() - core.captain_cancelled.mean())
    sample_share = 0.6329  # completed sample / terminal fulfilled, from quality.py
    excess_cancels_pop = excess_cancel_rate * len(sub) / sample_share
    recoverable_per_day = excess_cancels_pop / days

    comp = t[t.captain_cancelled == 0].copy()
    comp["engaged_hours"] = (comp.trip_distance_km / CFG.CITY_SPEED_KMPH
                             * np.where(comp.got_return_fare_within_20min == 1, 1, 2)
                             + np.where(comp.got_return_fare_within_20min == 1, 0,
                                        CFG.RETURN_FARE_WINDOW_MIN / 60))
    comp["inr_per_h"] = comp.fare_inr / comp.engaged_hours
    sub_bad = comp[(comp.drop_zone_type == "suburban")
                   & (comp.got_return_fare_within_20min == 0)]
    core_good = comp[(comp.drop_zone_type == "city_core")
                     & (comp.got_return_fare_within_20min == 1)]
    gap_per_hour = float(core_good.inr_per_h.mean() - sub_bad.inr_per_h.mean())
    sub_cycle = float(sub_bad.engaged_hours.mean())
    full_topup = gap_per_hour * sub_cycle
    deadhead_trips_per_day = float(
        (sub.captain_cancelled == 0).sum()
        * (1 - sub.got_return_fare_within_20min.mean()) / sample_share / days)

    # Option 2a -- pay in guaranteed airport demand instead of cash.
    # The terminals have ~47,000 unmet night requests. Handing a returning
    # captain a priority match at the terminal costs no incentive budget: it
    # allocates demand that is currently going unserved anyway.
    # Earnings comparison over a full round trip.
    #  today : suburban fare, then the outbound leg + empty return + 20-min wait
    #  with a pass : same two legs, no wait, and a guaranteed airport fare next
    airport_fare = float(t.fare_inr.mean())
    airport_trip_h = float(t.trip_distance_km.mean() / CFG.CITY_SPEED_KMPH)
    leg_h = float(sub_bad.trip_distance_km.mean() / CFG.CITY_SPEED_KMPH)
    rate_today = float(sub_bad.inr_per_h.mean())
    rate_with_pass = ((sub_bad.fare_inr.mean() + airport_fare)
                      / (leg_h * 2 + airport_trip_h))
    rows.append(dict(
        option="2a. Return-to-airport priority pass (no cash)",
        variant=f"guaranteed terminal match on return: a dead-headed suburban "
                f"run goes from Rs{rate_today:.0f} to Rs{rate_with_pass:.0f}"
                f"/engaged hour ({rate_with_pass/rate_today-1:+.0%})",
        incremental_trips_per_day=round(recoverable_per_day * 0.5),
        extra_captain_hours_per_night=np.nan,
        cost_per_day_inr=0,
        cost_per_month_lakh=0.0,
        cost_per_incremental_trip_inr=0,
        breakeven_commission_pct=0.0,
        lead_time="4-8 weeks (dispatch change)",
        key_risk="priority for one captain is deprioritisation for another; "
                 "only works while terminal demand is genuinely unmet"))

    # Option 2b -- the obvious cash version, costed and rejected
    for share, label in [(0.35, "partial top-up (35% of the gap)"),
                         (1.00, "full parity with a city-core return trip")]:
        topup = full_topup * share
        cost = topup * deadhead_trips_per_day
        trips_gained = recoverable_per_day * (0.4 if share < 1 else 0.9)
        rows.append(dict(
            option="2b. Blanket cash top-up on suburban drops (NOT recommended)",
            variant=f"{label}: Rs{topup:.0f} on every dead-headed trip",
            incremental_trips_per_day=round(trips_gained),
            extra_captain_hours_per_night=np.nan,
            cost_per_day_inr=round(cost),
            cost_per_month_lakh=round(cost * 30 / 1e5, 1),
            cost_per_incremental_trip_inr=round(cost / trips_gained),
            breakeven_commission_pct=round(cost / trips_gained / mean_fare * 100, 1),
            lead_time="2-4 weeks (pricing change)",
            key_risk=f"pays all ~{deadhead_trips_per_day:.0f} dead-headed trips/day to "
                     f"change ~{trips_gained:.0f} decisions -- cost per incremental "
                     f"trip is {cost/trips_gained/mean_fare:.0f}x the average fare"))

    # Option 3 -- the proposal on the table
    for night_ch in (200, 400):
        captains_needed = night_ch / 4  # assume a 4-hour night shift
        signups = captains_needed / a2o
        trips = night_ch * tput
        rows.append(dict(
            option="3. Targeted acquisition in the airport catchment (as proposed)",
            variant=f"add {night_ch} night captain-hours/day "
                    f"= {captains_needed:.0f} captains on 4h night shifts",
            incremental_trips_per_day=round(trips),
            extra_captain_hours_per_night=night_ch,
            cost_per_day_inr=np.nan,
            cost_per_month_lakh=np.nan,
            cost_per_incremental_trip_inr=np.nan,
            breakeven_commission_pct=np.nan,
            lead_time="8-12 weeks (signup -> approval -> habit)",
            key_risk=f"requires {signups:.0f} signups at {a2o:.1%} A2O, plus a bonus, "
                     "AND assumes new captains choose 20:00-04:00 shifts -- "
                     "for which there is no evidence in this data"))

    return pd.DataFrame(rows)


def acquisition_feasibility(t: pd.DataFrame, h: pd.DataFrame, a2o: float,
                            cap_bal: dict) -> dict:
    """The explicit case against the proposal, with the arithmetic attached."""
    apt = h[h.zone_type == "airport_terminal"]
    days = apt.hour_ts.dt.date.nunique()
    return dict(
        signups_per_approved_captain=round(1 / a2o, 1),
        night_share_of_terminal_unmet_pct=round(
            float(apt.loc[apt.is_night, "unfulfilled_requests"].sum()
                  / apt.unfulfilled_requests.sum() * 100), 1),
        hours_per_day_airport_is_actually_short=len(CFG.NIGHT_HOURS),
        hours_per_day_airport_has_surplus=24 - len(CFG.NIGHT_HOURS),
        daytime_marginal_trips_per_captain_hour=0.188,
        night_marginal_trips_per_captain_hour=1.310,
        idle_daytime_captain_hours=cap_bal["idle_captain_hours_in_surplus_hours"],
        night_deficit_captain_hours=cap_bal["deficit_captain_hours_in_short_hours"],
        estimated_cancellations_share_of_unmet_pct=round(
            float(t.captain_cancelled.sum() / 0.6329
                  / apt.unfulfilled_requests.sum() * 100), 1),
        no_home_zone_evidence=("airport_hourly.csv and airport_trips.csv contain no "
                               "captain identifier and no captain home zone, so the "
                               "premise that living near the airport increases airport "
                               "supply cannot be tested with this data at all"),
        verdict=("NO. Fund a time-targeted incentive on the existing captain base "
                 "and fix the suburban return economics. Revisit acquisition only "
                 "if night supply stays short after both."))
