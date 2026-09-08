"""Part B follow-up: reconcile the two airport files, night-vs-day marginal supply, sizing."""
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 80); pd.set_option('display.max_rows', 400)

h = pd.read_csv('../airport_hourly.csv', parse_dates=['hour_ts'])
t = pd.read_csv('../airport_trips.csv', parse_dates=['request_ts'])
apt = h[h.zone_type=='airport_terminal'].copy()
apt['hod'] = apt.hour_ts.dt.hour
t['hod'] = t.request_ts.dt.hour

print('=== Reconcile the two files ===')
print('terminal requests  :', apt.requests.sum())
print('terminal fulfilled :', apt.fulfilled_requests.sum())
print('terminal unfulfilled:', apt.unfulfilled_requests.sum())
print('sampled trips      :', len(t), ' completed:', (t.captain_cancelled==0).sum(),
      ' cancelled:', t.captain_cancelled.sum())
# join on zone+hour
tj = t.groupby(['pickup_zone_id','request_ts']).agg(
    n_trips=('trip_id','size'), n_cancel=('captain_cancelled','sum')).reset_index()
mg = apt.merge(tj, left_on=['zone_id','hour_ts'], right_on=['pickup_zone_id','request_ts'], how='left')
mg[['n_trips','n_cancel']] = mg[['n_trips','n_cancel']].fillna(0)
print('\nhours matched:', (mg.n_trips>0).sum(), 'of', len(mg))
print('corr(n_trips, fulfilled):', round(mg.n_trips.corr(mg.fulfilled_requests),3))
print('corr(n_trips, requests) :', round(mg.n_trips.corr(mg.requests),3))
print('corr(n_completed, fulfilled):', round((mg.n_trips-mg.n_cancel).corr(mg.fulfilled_requests),3))
print('\nratio completed_sample / fulfilled  = %.4f' % ((t.captain_cancelled==0).sum()/apt.fulfilled_requests.sum()))
print('ratio all_sample_trips / fulfilled   = %.4f' % (len(t)/apt.fulfilled_requests.sum()))
print('\nper-hour: mean(n_trips)/mean(fulfilled) = %.3f' % (mg.n_trips.mean()/mg.fulfilled_requests.mean()))
print('per-hour: mean(completed)/mean(fulfilled) = %.3f' % ((mg.n_trips-mg.n_cancel).mean()/mg.fulfilled_requests.mean()))
print('\nregression fulfilled ~ completed_sample (no intercept):')
mg['completed'] = mg.n_trips - mg.n_cancel
r = smf.ols('fulfilled_requests ~ completed - 1', data=mg[mg.n_trips>0]).fit()
print('  slope = %.4f  R2=%.3f' % (r.params['completed'], r.rsquared))
print('=> sample appears to be ~%.0f%% of completed airport trips' % (100/r.params["completed"]))

print('\n=== Night vs day: is supply or willingness binding? ===')
apt['night'] = apt.hod.isin([20,21,22,23,0,1,2,3])
apt['fill_rate'] = apt.fulfilled_requests/apt.requests
print(apt.groupby('night').agg(hours=('hod','size'), req=('requests','sum'), ful=('fulfilled_requests','sum'),
      unmet=('unfulfilled_requests','sum'), online=('online_captains','mean'),
      eta=('avg_eta_min','mean'), surge=('avg_surge_multiplier','mean'),
      fill=('fill_rate','mean')).round(3))

print('\nmarginal fulfilled trips per extra online captain, night vs day:')
for lab, sub in [('night 20-04', apt[apt.night]), ('day 05-19', apt[~apt.night])]:
    mm = smf.ols('fulfilled_requests ~ online_captains + requests + C(hod) + C(zone_id)', data=sub).fit()
    print(f'  {lab}: d_fulfilled/d_online = {mm.params["online_captains"]:.3f} '
          f'(se {mm.bse["online_captains"]:.3f}), R2={mm.rsquared:.3f}')

print('\nthroughput per online captain (fulfilled/online):')
apt['tput'] = apt.fulfilled_requests/apt.online_captains.replace(0,np.nan)
print(apt.groupby('night').tput.describe(percentiles=[.5,.9,.99]).round(2))

print('\n=== Idle capacity in surplus hours vs unmet in shortage hours ===')
CAP = apt.loc[apt.night, 'tput'].quantile(0.90)   # observed achievable trips per captain-hour
print(f'assumed achievable throughput = {CAP:.2f} trips per captain-hour (p90 of night observed)')
apt['needed_captains'] = apt.requests/CAP
apt['surplus_captains'] = apt.online_captains - apt.needed_captains
days = apt.hour_ts.dt.date.nunique()
print(f'\nover {days} days, across both terminals:')
print(f'  total online captain-hours     : {apt.online_captains.sum():,.0f}')
print(f'  captain-hours needed for ALL demand: {apt.needed_captains.sum():,.0f}')
print(f'  => aggregate supply is {"SUFFICIENT" if apt.online_captains.sum()>apt.needed_captains.sum() else "SHORT"} '
      f'in total by {apt.online_captains.sum()-apt.needed_captains.sum():,.0f} captain-hours')
print('\nby night flag:')
print(apt.groupby('night').agg(online_ch=('online_captains','sum'), needed_ch=('needed_captains','sum'),
      surplus_ch=('surplus_captains','sum')).round(0))
print('\nidle captain-hours in surplus hours: %.0f' % apt.loc[apt.surplus_captains>0,'surplus_captains'].sum())
print('deficit captain-hours in short hours: %.0f' % (-apt.loc[apt.surplus_captains<0,'surplus_captains'].sum()))
print('\nper night, per terminal, extra captains needed to clear the queue:')
nn = apt[apt.night].groupby('hod').agg(req=('requests','mean'), online=('online_captains','mean'),
      needed=('needed_captains','mean'))
nn['extra_needed'] = (nn.needed-nn.online).round(1)
print(nn.round(1))
print('total extra captain-hours per night across 2 terminals: %.0f'
      % (nn.extra_needed.clip(lower=0).sum()*2))

print('\n=== How much unmet demand is captain cancellation, not absence of supply? ===')
SAMP = 1/r.params['completed']   # sample share of completed
est_matched = len(t)/SAMP
est_cancel = t.captain_cancelled.sum()/SAMP
print(f'estimated population matched airport trips  : {est_matched:,.0f}')
print(f'estimated population captain cancellations  : {est_cancel:,.0f}')
print(f'terminal unfulfilled (from hourly)          : {apt.unfulfilled_requests.sum():,.0f}')
print(f'=> cancellations are ~{est_cancel/apt.unfulfilled_requests.sum()*100:.1f}% of unmet demand')
print('\nnight cancellations:')
tn = t[t.hod.isin([20,21,22,23,0,1,2,3])]
print(f'  night sampled trips {len(tn):,}, cancel rate {tn.captain_cancelled.mean()*100:.2f}%')
print(f'  day   sampled trips {len(t)-len(tn):,}, cancel rate {t[~t.hod.isin([20,21,22,23,0,1,2,3])].captain_cancelled.mean()*100:.2f}%')
print('\nreturn-fare probability night vs day (completed):')
tc = t[t.captain_cancelled==0].copy()
tc['night'] = tc.hod.isin([20,21,22,23,0,1,2,3])
print(tc.groupby('night').agg(n=('trip_id','size'), rf=('got_return_fare_within_20min','mean'),
      dist=('trip_distance_km','mean'), fare=('fare_inr','mean')).round(3))
print('\nnight x drop_zone_type:')
print(tc.groupby(['night','drop_zone_type']).agg(n=('trip_id','size'),
      rf=('got_return_fare_within_20min','mean')).round(3))
print('\ncancel rate night x drop_zone_type:')
t['night'] = t.hod.isin([20,21,22,23,0,1,2,3])
print(t.pivot_table(index='drop_zone_type', columns='night', values='captain_cancelled', aggfunc='mean').round(3))
print('\ndrop-mix night vs day (share of trips):')
print(pd.crosstab(t.night, t.drop_zone_type, normalize='index').round(3))

print('\n=== Sizing the alternatives ===')
SPEED = 22.0
tc['cycle_h'] = tc.trip_distance_km/SPEED + np.where(tc.got_return_fare_within_20min==1, 0,
                                                     tc.trip_distance_km/SPEED + 20/60)
tc['inr_per_h'] = tc.fare_inr/tc.cycle_h
print('INR per engaged hour, night, by drop zone type:')
print(tc[tc.night].groupby('drop_zone_type').agg(n=('trip_id','size'), inr_per_h=('inr_per_h','mean'),
      fare=('fare_inr','mean'), cycle=('cycle_h','mean')).round(1))
print('\ncity-core-with-return-fare benchmark INR/h: %.0f' %
      tc[(tc.drop_zone_type=='city_core')&(tc.got_return_fare_within_20min==1)].inr_per_h.mean())

print('\n--- Option 1: night airport incentive for EXISTING captains ---')
unmet_night_per_day = apt.loc[apt.night,'unfulfilled_requests'].sum()/days
print(f'unmet demand in the night window: {unmet_night_per_day:,.0f} trips/day (both terminals)')
for recov in [0.3, 0.5]:
    trips = unmet_night_per_day*recov
    ch = trips/CAP
    print(f'  recover {recov*100:.0f}% -> {trips:,.0f} trips/day, needs {ch:,.0f} extra captain-hours/night')
    for bonus in [100, 150, 200]:
        print(f'      at Rs{bonus}/captain-hour incentive: Rs{ch*bonus:,.0f}/night '
              f'= Rs{ch*bonus*30/1e5:,.1f} lakh/month, cost per incremental trip Rs{ch*bonus/trips:,.0f}')

print('\n--- Option 2: cut suburban cancellations ---')
sub_cancel_rate = t.loc[t.drop_zone_type=='suburban','captain_cancelled'].mean()
core_cancel_rate = t.loc[t.drop_zone_type=='city_core','captain_cancelled'].mean()
n_sub = (t.drop_zone_type=='suburban').sum()
excess = (sub_cancel_rate-core_cancel_rate)*n_sub
print(f'suburban cancel {sub_cancel_rate*100:.1f}% vs city core {core_cancel_rate*100:.1f}%')
print(f'excess suburban cancellations in sample: {excess:,.0f}  -> population {excess/SAMP:,.0f} over {days} days')
print(f'= {excess/SAMP/days:,.0f} recoverable trips/day at ZERO acquisition cost')
gap = tc[(tc.drop_zone_type=='city_core')&(tc.got_return_fare_within_20min==1)].inr_per_h.mean() - \
      tc[(tc.drop_zone_type=='suburban')&(tc.got_return_fare_within_20min==0)].inr_per_h.mean()
sub_fare = t.loc[t.drop_zone_type=='suburban','fare_inr'].mean()
print(f'\nearnings gap to close: Rs{gap:.0f}/engaged hour')
sub_cycle = tc[(tc.drop_zone_type=='suburban')&(tc.got_return_fare_within_20min==0)].cycle_h.mean()
print(f'a dead-headed suburban run occupies {sub_cycle:.2f} h; topping it up to city-core-with-return economics')
print(f'costs Rs{gap*sub_cycle:.0f} per trip (~{gap*sub_cycle/sub_fare*100:.0f}% of the Rs{sub_fare:.0f} fare)')

print('\n--- Option 3: targeted acquisition (the proposal on the table) ---')
A2O = 0.1759
print(f'A2O = {A2O*100:.1f}% -> {1/A2O:.1f} signups per approved captain')
for target_ch in [200, 400]:
    print(f'  to add {target_ch} night captain-hours/day you need ~{target_ch/4:.0f} captains working 4h nights,')
    print(f'    = {target_ch/4/A2O:,.0f} signups at current A2O, plus a sign-up bonus,')
    print(f'    and NO evidence in this data that home zone predicts working 20:00-04:00 at the airport.')
