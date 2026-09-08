"""Part B: airport demand-supply mismatch, post-trip economics, and whether acquisition is the fix."""
import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 80); pd.set_option('display.max_rows', 400)

h = pd.read_csv('../airport_hourly.csv', parse_dates=['hour_ts'])
t = pd.read_csv('../airport_trips.csv', parse_dates=['request_ts'])

print('=== zones ===')
print(h.groupby(['zone_id','zone_type']).size())
print('hour_ts range', h.hour_ts.min(), '->', h.hour_ts.max(), '| hours:', h.hour_ts.nunique())
print('rows per zone:', h.groupby('zone_id').size().to_dict())

print('\n=== identity check: requests = fulfilled + unfulfilled ? ===')
h['resid'] = h.requests - h.fulfilled_requests - h.unfulfilled_requests
print(h.resid.describe())
print('rows violating identity:', (h.resid != 0).sum())
print(h[h.resid != 0].groupby('zone_id').size())

apt = h[h.zone_type == 'airport_terminal'].copy()
oth = h[h.zone_type != 'airport_terminal'].copy()
print('\n=== headline by zone_type ===')
g = h.groupby('zone_type').agg(requests=('requests','sum'), fulfilled=('fulfilled_requests','sum'),
                               unfulfilled=('unfulfilled_requests','sum'),
                               eta=('avg_eta_min','mean'), surge=('avg_surge_multiplier','mean'),
                               online=('online_captains','mean'))
g['fill_rate'] = (g.fulfilled/g.requests*100).round(2)
g['unmet_per_hr'] = (g.unfulfilled/h.groupby('zone_type').size()).round(1)
print(g.round(2))

print('\n=== B1: when is the airport short? hour-of-day profile (terminals) ===')
apt['hod'] = apt.hour_ts.dt.hour
apt['dow'] = apt.hour_ts.dt.dayofweek
prof = apt.groupby('hod').agg(requests=('requests','mean'), fulfilled=('fulfilled_requests','mean'),
                              unfulfilled=('unfulfilled_requests','mean'),
                              online=('online_captains','mean'), eta=('avg_eta_min','mean'),
                              surge=('avg_surge_multiplier','mean'))
prof['fill_rate'] = (prof.fulfilled/prof.requests*100).round(1)
prof['req_per_captain'] = (prof.requests/prof.online).round(2)
print(prof.round(2))

print('\n=== total unmet demand at terminals ===')
tot_unmet = apt.unfulfilled_requests.sum()
days = apt.hour_ts.dt.date.nunique()
print(f'unfulfilled at terminals: {tot_unmet:,} over {days} days = {tot_unmet/days:,.0f}/day')
print(f'fill rate terminals: {apt.fulfilled_requests.sum()/apt.requests.sum()*100:.2f}%')
print('\nshare of ALL terminal unmet demand by hour bucket:')
apt['bucket'] = pd.cut(apt.hod, [-1,4,7,11,15,19,23],
                       labels=['00-04','05-07','08-11','12-15','16-19','20-23'])
bb = apt.groupby('bucket', observed=True).agg(unmet=('unfulfilled_requests','sum'),
        req=('requests','sum'), online=('online_captains','mean'), eta=('avg_eta_min','mean'),
        surge=('avg_surge_multiplier','mean'))
bb['fill_rate'] = ((bb.req-bb.unmet)/bb.req*100).round(1)
bb['share_of_unmet'] = (bb.unmet/tot_unmet*100).round(1)
print(bb.round(2))
print('\nby zone:')
print(apt.groupby('zone_id').agg(req=('requests','sum'), unmet=('unfulfilled_requests','sum'),
      online=('online_captains','mean'), eta=('avg_eta_min','mean'), surge=('avg_surge_multiplier','mean')).round(2))

print('\n=== Is it really a SUPPLY shortage? requests per online captain ===')
apt['req_per_cap'] = apt.requests/apt.online_captains.replace(0,np.nan)
apt['fill_per_cap'] = apt.fulfilled_requests/apt.online_captains.replace(0,np.nan)
print(apt.groupby('bucket', observed=True)[['req_per_cap','fill_per_cap']].mean().round(2))
print('\ncorrelation of fill_rate with online_captains (terminals):')
apt['fill_rate'] = apt.fulfilled_requests/apt.requests
print(apt[['fill_rate','online_captains','requests','avg_eta_min','avg_surge_multiplier']].corr().round(3))
print('\nfill rate by online_captains decile (terminals):')
apt['cap_dec'] = pd.qcut(apt.online_captains, 10, duplicates='drop')
print(apt.groupby('cap_dec', observed=True).agg(fill=('fill_rate','mean'), req=('requests','mean'),
      unmet=('unfulfilled_requests','mean'), n=('fill_rate','size')).round(3))
print('\n>>> marginal fulfilment per extra online captain (terminals, OLS w/ hour FE):')
import statsmodels.formula.api as smf
apt['hod_c'] = apt.hod.astype(str)
m1 = smf.ols('fulfilled_requests ~ online_captains + requests + C(hod_c) + C(zone_id)', data=apt).fit()
print(m1.params[['online_captains','requests']].round(4))
print('  R2', round(m1.rsquared,3))
print('\n  captains appear supply-capped? fulfilled/online ratio distribution:')
print((apt.fulfilled_requests/apt.online_captains.replace(0,np.nan)).describe(percentiles=[.5,.9,.99]).round(2))

print('\n=== B2: what happens AFTER an airport trip ===')
print('trips:', len(t), '| range', t.request_ts.min(), '->', t.request_ts.max())
print('\nby drop_zone_type:')
tb = t.groupby('drop_zone_type').agg(n=('trip_id','size'), dist=('trip_distance_km','mean'),
      fare=('fare_inr','mean'), cancel=('captain_cancelled','mean'),
      return_fare=('got_return_fare_within_20min','mean'))
tb['share'] = (tb.n/len(t)*100).round(1)
print(tb.round(3))
print('\nby drop_zone_id:')
print(t.groupby(['drop_zone_id','drop_zone_type']).agg(n=('trip_id','size'),
      dist=('trip_distance_km','mean'), fare=('fare_inr','mean'),
      cancel=('captain_cancelled','mean'), rf=('got_return_fare_within_20min','mean')).round(3))

comp = t[t.captain_cancelled == 0].copy()
print('\n=== dead-heading economics (completed trips only) ===')
print('P(return fare within 20 min) overall: %.3f' % comp.got_return_fare_within_20min.mean())
print(comp.groupby('drop_zone_type').agg(n=('trip_id','size'), rf=('got_return_fare_within_20min','mean'),
      fare=('fare_inr','mean'), dist=('trip_distance_km','mean')).round(3))
# effective earnings per productive hour proxy
SPEED = 22.0  # km/h assumed city average
comp['trip_h'] = comp.trip_distance_km/SPEED
comp['deadhead_h'] = np.where(comp.got_return_fare_within_20min==1, 0, comp.trip_distance_km/SPEED)
comp['wait_h'] = 20/60
comp['cycle_h'] = comp.trip_h + comp.deadhead_h + np.where(comp.got_return_fare_within_20min==1, 0, comp.wait_h)
comp['inr_per_h'] = comp.fare_inr/comp.cycle_h
print('\nimplied INR per engaged hour, by drop zone type and return-fare status:')
print(comp.groupby(['drop_zone_type','got_return_fare_within_20min']).agg(
    n=('trip_id','size'), fare=('fare_inr','mean'), cycle_h=('cycle_h','mean'),
    inr_per_h=('inr_per_h','mean')).round(2))
print('\nweighted mean INR/engaged-hour by drop zone type:')
print(comp.groupby('drop_zone_type').inr_per_h.mean().round(1))

print('\n=== cancellation: are captains selecting against bad trips? ===')
print('overall cancel rate: %.4f' % t.captain_cancelled.mean())
print('\ncancel rate by drop_zone_type x distance quartile:')
t['dq'] = pd.qcut(t.trip_distance_km, 4, labels=['Q1 short','Q2','Q3','Q4 long'])
print(t.pivot_table(index='drop_zone_type', columns='dq', values='captain_cancelled',
                    aggfunc='mean', observed=True).round(3))
print('\ncancel rate by hour of day:')
t['hod'] = t.request_ts.dt.hour
print(t.groupby('hod').agg(n=('trip_id','size'), cancel=('captain_cancelled','mean')).round(3))
print('\ncancel rate by hour bucket + fare/dist:')
t['bucket'] = pd.cut(t.hod, [-1,4,7,11,15,19,23], labels=['00-04','05-07','08-11','12-15','16-19','20-23'])
print(t.groupby('bucket', observed=True).agg(n=('trip_id','size'), cancel=('captain_cancelled','mean'),
      dist=('trip_distance_km','mean'), fare=('fare_inr','mean'),
      rf=('got_return_fare_within_20min','mean')).round(3))
print('\n>>> cancel rate vs return-fare probability of the ROUTE (zone-level):')
z = t.groupby('drop_zone_id').agg(cancel=('captain_cancelled','mean'),
      rf=('got_return_fare_within_20min','mean'), dist=('trip_distance_km','mean'),
      fare=('fare_inr','mean'), n=('trip_id','size'))
print(z.round(3))
print('corr(cancel, return_fare) across zones: %.3f' % z.cancel.corr(z.rf))
print('\nlogit cancel ~ dist + drop_zone_type + hour bucket:')
mc = smf.logit('captain_cancelled ~ trip_distance_km + C(drop_zone_type) + C(bucket)', data=t).fit(disp=0)
print(mc.summary().tables[1])

print('\n=== How much supply does cancellation destroy? ===')
n_cancel = t.captain_cancelled.sum()
print(f'cancelled trips in sample: {n_cancel:,} of {len(t):,} ({t.captain_cancelled.mean()*100:.2f}%)')
print('sample vs population: airport_hourly terminal fulfilled =', apt.fulfilled_requests.sum(),
      '| trips sample =', len(t))
print('\nIF cancellations were eliminated, extra fulfilled trips at terminals (scaled):')
scale = apt.requests.sum()/len(t)
print(f'  scale factor requests/trip-sample = {scale:.2f}')

print('\n=== supply elasticity check: does high surge attract captains next hour? ===')
apt2 = apt.sort_values(['zone_id','hour_ts']).copy()
apt2['online_next'] = apt2.groupby('zone_id').online_captains.shift(-1)
apt2['d_online'] = apt2.online_next - apt2.online_captains
print(apt2[['avg_surge_multiplier','d_online','online_next','online_captains']].corr().round(3))
print('\nmean change in online captains next hour, by surge quintile:')
apt2['sq'] = pd.qcut(apt2.avg_surge_multiplier, 5)
print(apt2.groupby('sq', observed=True).agg(d_online=('d_online','mean'), n=('d_online','size'),
      surge=('avg_surge_multiplier','mean')).round(3))
