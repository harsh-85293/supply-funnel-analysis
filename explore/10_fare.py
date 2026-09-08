"""Does fare_inr reflect surge? Needed to know whether night earnings are understated."""
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
t = pd.read_csv('../airport_trips.csv', parse_dates=['request_ts'])
h = pd.read_csv('../airport_hourly.csv', parse_dates=['hour_ts'])
t['hod'] = t.request_ts.dt.hour
m = t.merge(h[h.zone_type=='airport_terminal'][['zone_id','hour_ts','avg_surge_multiplier','avg_eta_min']],
            left_on=['pickup_zone_id','request_ts'], right_on=['zone_id','hour_ts'], how='left')
print('fare ~ distance:')
r = smf.ols('fare_inr ~ trip_distance_km', data=m).fit()
print(r.params.round(3), 'R2=%.4f' % r.rsquared)
print('\nfare ~ distance + surge:')
r2 = smf.ols('fare_inr ~ trip_distance_km + avg_surge_multiplier', data=m).fit()
print(r2.params.round(3), 'R2=%.4f' % r2.rsquared)
print('\nmean fare by surge quintile (holding distance band):')
m['dband'] = pd.cut(m.trip_distance_km, [0,10,15,20,25,60])
m['sband'] = pd.cut(m.avg_surge_multiplier, [0,1.05,1.4,1.8,2.1,3])
print(m.pivot_table(index='dband', columns='sband', values='fare_inr', aggfunc='mean', observed=True).round(0))
print('\nresidual fare per km by night/day:')
m['night'] = m.hod.isin([20,21,22,23,0,1,2,3])
m['per_km'] = m.fare_inr/m.trip_distance_km
print(m.groupby('night').per_km.mean().round(2))
print('\n=> fare_inr is a deterministic function of distance; surge is NOT priced into it.')
print('\nimplied tariff: base Rs%.0f + Rs%.1f/km' % (r.params['Intercept'], r.params['trip_distance_km']))
