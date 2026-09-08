"""Data-quality and internal-consistency audit. Exploration only."""
import pandas as pd, numpy as np
pd.set_option('display.width', 240); pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 300)

EXTRACT = pd.Timestamp('2026-06-30 23:59:00')

cap = pd.read_csv('../captains.csv', parse_dates=['signup_ts'])
doc = pd.read_csv('../doc_events.csv', parse_dates=['event_ts'])
apr = pd.read_csv('../approvals.csv', parse_dates=['decision_ts'])
act = pd.read_csv('../activation.csv', parse_dates=['first_order_ts'])
nud = pd.read_csv('../nudges.csv', parse_dates=['sent_ts'])

m = cap.merge(apr, on='captain_id')

print('=== 1. Vehicle type vs required docs (PERMIT only Auto/Cab) ===')
print(pd.crosstab(m.vehicle_type, m.docs_cleared))
print('\napproved docs_cleared by vehicle:')
print(pd.crosstab(m.loc[m.final_status == 'approved', 'vehicle_type'],
                  m.loc[m.final_status == 'approved', 'docs_cleared']))
print('\nPERMIT events by vehicle type (should be 0 for ERickshaw):')
dv = doc.merge(cap[['captain_id', 'vehicle_type']], on='captain_id')
print(pd.crosstab(dv.vehicle_type, dv.doc_type))

print('\n=== 2. Time to decision (maturity / censoring) ===')
m['days_to_decision'] = (m.decision_ts - m.signup_ts).dt.total_seconds() / 86400
print(m.days_to_decision.describe(percentiles=[.5, .75, .9, .95, .99]))
print('\nnegative days_to_decision:', (m.days_to_decision < 0).sum())
print('\nlast doc event age for in_progress:')
lastev = doc.groupby('captain_id').event_ts.max().rename('last_event_ts')
m2 = m.merge(lastev, on='captain_id', how='left')
ip = m2[m2.final_status == 'in_progress']
print('signup age (days) of in_progress:')
print(((EXTRACT - ip.signup_ts).dt.total_seconds() / 86400).describe(percentiles=[.1, .25, .5, .75, .9]))
print('\nsignup age of dropped_in_docs:')
dr = m2[m2.final_status == 'dropped_in_docs']
print(((EXTRACT - dr.signup_ts).dt.total_seconds() / 86400).describe(percentiles=[.1, .25, .5, .75, .9]))
print('\nstatus mix by signup month:')
print(pd.crosstab(m.signup_ts.dt.to_period('M'), m.final_status, normalize='index').round(4))
print('\nstatus counts by signup week (last 10 weeks):')
ct = pd.crosstab(m.signup_ts.dt.to_period('W'), m.final_status)
print(ct.tail(12))

print('\n=== 3. docs_cleared vs actual doc_events passes ===')
passes = (doc[doc.event_type == 'verification_pass']
          .groupby('captain_id').doc_type.nunique().rename('n_pass'))
chk = m.merge(passes, on='captain_id', how='left')
chk['n_pass'] = chk.n_pass.fillna(0).astype(int)
print('mismatch docs_cleared vs distinct passed doc_types:',
      (chk.docs_cleared != chk.n_pass).sum())
print(pd.crosstab(chk.docs_cleared, chk.n_pass))

print('\n=== 4. Event ordering sanity ===')
d = doc.sort_values(['captain_id', 'event_ts'])
# any verification without a preceding upload for same captain/doc/attempt
up = doc[doc.event_type == 'upload_success'][['captain_id', 'doc_type', 'attempt_no', 'event_ts']]
up = up.rename(columns={'event_ts': 'up_ts'})
ver = doc[doc.event_type != 'upload_success'][['captain_id', 'doc_type', 'attempt_no', 'event_ts', 'event_type']]
j = ver.merge(up, on=['captain_id', 'doc_type', 'attempt_no'], how='left')
print('verification events with NO matching upload:', j.up_ts.isna().sum())
print('verification BEFORE its upload:', (j.event_ts < j.up_ts).sum())
j['verif_lag_h'] = (j.event_ts - j.up_ts).dt.total_seconds() / 3600
print('verification lag hours:'); print(j.verif_lag_h.describe(percentiles=[.5, .9, .95, .99]))
print('\nboth pass and fail on same captain/doc/attempt:')
dupver = ver.groupby(['captain_id', 'doc_type', 'attempt_no']).event_type.nunique()
print((dupver > 1).sum())
print('\nmultiple uploads on same captain/doc/attempt:')
print((up.groupby(['captain_id', 'doc_type', 'attempt_no']).size() > 1).sum())
print('\nevents before signup:')
ds = doc.merge(cap[['captain_id', 'signup_ts']], on='captain_id')
print((ds.event_ts < ds.signup_ts).sum())
print('\nevents after decision_ts (for approved):')
da = doc.merge(apr[['captain_id', 'decision_ts', 'final_status']], on='captain_id')
print((da.event_ts > da.decision_ts).sum(), 'of', da.decision_ts.notna().sum())

print('\n=== 5. Doc sequence violations (out-of-order clears) ===')
SEQ = ['DL', 'RC', 'AADHAAR', 'PERMIT', 'FITNESS', 'INSURANCE']
p = doc[doc.event_type == 'verification_pass'].merge(cap[['captain_id', 'vehicle_type']], on='captain_id')
p['ord'] = p.doc_type.map({d: i for i, d in enumerate(SEQ)})
p = p.sort_values(['captain_id', 'event_ts'])
viol = p.groupby('captain_id')['ord'].apply(lambda s: (s.diff() < 0).any())
print('captains clearing docs out of stated order:', viol.sum(), '/', len(viol))

print('\n=== 6. activation contradictions ===')
a = act.copy()
print('first_order_ts null but orders_d7>0:', ((a.first_order_ts.isna()) & (a.orders_d7 > 0)).sum())
print('first_order_ts null rows, orders_d7 values:')
print(a.loc[a.first_order_ts.isna(), 'orders_d7'].value_counts(dropna=False))
print('\norders_d7 > orders_d30:', (a.orders_d7 > a.orders_d30).sum())
print('orders_d30>0 but online_hours_d30==0:', ((a.orders_d30 > 0) & (a.online_hours_d30 == 0)).sum())
print('online_hours_d30==0 but orders_d30>0 rows:', ((a.online_hours_d30 == 0)).sum())
aa = a.merge(apr[['captain_id', 'decision_ts']], on='captain_id')
print('first_order BEFORE approval decision:', (aa.first_order_ts < aa.decision_ts).sum())
aa['days_to_first_order'] = (aa.first_order_ts - aa.decision_ts).dt.total_seconds() / 86400
print(aa.days_to_first_order.describe(percentiles=[.25, .5, .75, .9, .95]))
aa['days_since_decision'] = (EXTRACT - aa.decision_ts).dt.total_seconds() / 86400
print('\nnull-orders_d30 rows: days_since_decision describe (expect <30):')
print(aa.loc[aa.orders_d30.isna(), 'days_since_decision'].describe())
print('non-null orders_d30 rows: days_since_decision describe (expect >=30):')
print(aa.loc[aa.orders_d30.notna(), 'days_since_decision'].describe())
print('\nnull-orders_d7: days_since_decision describe (expect <7):')
print(aa.loc[aa.orders_d7.isna(), 'days_since_decision'].describe())

print('\n=== 7. nudges anomalies ===')
print('clicked=1 & delivered=0:', ((nud.clicked == 1) & (nud.delivered == 0)).sum())
print('by campaign:')
print(nud[(nud.clicked == 1) & (nud.delivered == 0)].campaign_id.value_counts())
ns = nud.merge(cap[['captain_id', 'signup_ts']], on='captain_id')
ns['days_after_signup'] = (ns.sent_ts - ns.signup_ts).dt.total_seconds() / 86400
print('\nnudge sent before signup:', (ns.days_after_signup < 0).sum())
print(ns.groupby('campaign_id').days_after_signup.describe(percentiles=[.1, .5, .9]))
nsa = nud.merge(apr[['captain_id', 'decision_ts', 'final_status']], on='captain_id')
print('\nnudge sent AFTER approval decision:', (nsa.sent_ts > nsa.decision_ts).sum(),
      'of', nsa.decision_ts.notna().sum())
print(nsa[nsa.sent_ts > nsa.decision_ts].groupby(['campaign_id', 'final_status']).size())

print('\n=== 8. duplicate/near-duplicate captains ===')
print('exact dup on (city,vehicle,zone,signup_ts):',
      cap.duplicated(subset=['city', 'vehicle_type', 'signup_zone_id', 'signup_ts']).sum())
print('signup_zone_id prefix vs city consistency:')
cap['zpref'] = cap.signup_zone_id.str.split('-').str[0]
print(pd.crosstab(cap.city, cap.zpref))
