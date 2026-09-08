"""A3: is CAMP_WA_002 a real effect or a selection artefact?"""
import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 300)

SEQ_ALL = ['DL', 'RC', 'AADHAAR', 'PERMIT', 'FITNESS', 'INSURANCE']
cap = pd.read_csv('../captains.csv', parse_dates=['signup_ts'])
doc = pd.read_csv('../doc_events.csv', parse_dates=['event_ts'])
apr = pd.read_csv('../approvals.csv', parse_dates=['decision_ts'])
nud = pd.read_csv('../nudges.csv', parse_dates=['sent_ts'])

cap['n_required'] = np.where(cap.vehicle_type == 'ERickshaw', 5, 6)
base = cap.merge(apr, on='captain_id')
base = base[base.signup_ts < pd.Timestamp('2026-06-16')].copy()   # mature cohort
base['approved'] = (base.final_status == 'approved').astype(int)

passes = doc[doc.event_type == 'verification_pass']
npass = passes.groupby('captain_id').doc_type.nunique()
base['n_passed'] = base.captain_id.map(npass).fillna(0).astype(int)

# last activity time in funnel (proxy for when they gave up)
lastev = doc.groupby('captain_id').event_ts.max()
base['last_event_ts'] = base.captain_id.map(lastev)
base['exit_ts'] = base.decision_ts.fillna(base.last_event_ts).fillna(base.signup_ts)
base['exit_day'] = (base.exit_ts - base.signup_ts).dt.total_seconds()/86400

wa2 = nud[nud.campaign_id == 'CAMP_WA_002']
print('=== Who gets CAMP_WA_002? ===')
print('recipients total:', wa2.captain_id.nunique(), '| in mature cohort:',
      wa2.captain_id.isin(base.captain_id).sum())
base['wa2'] = base.captain_id.isin(wa2.captain_id).astype(int)
base['any_nudge'] = base.captain_id.isin(nud.captain_id).astype(int)
base['n_nudges'] = base.captain_id.map(nud.groupby('captain_id').size()).fillna(0).astype(int)
print(base.groupby('wa2').size())

print('\n--- NAIVE comparison (what growth is probably quoting) ---')
naive = base.groupby('wa2').approved.agg(['mean', 'sum', 'count'])
print(naive.assign(pct=lambda d: (d['mean']*100).round(2)))
lift_pp = (naive.loc[1, 'mean'] - naive.loc[0, 'mean'])*100
print(f'NAIVE lift = {lift_pp:.2f} pp  (relative {(naive.loc[1,"mean"]/naive.loc[0,"mean"]-1)*100:.1f}%)')

print('\n--- vs other-campaign recipients only (removes "any comms" selection) ---')
other = nud[(nud.campaign_id != 'CAMP_WA_002')].captain_id.unique()
base['grp'] = np.where(base.wa2 == 1, 'WA_002',
              np.where(base.captain_id.isin(other), 'other_campaign', 'no_nudge'))
print(base.groupby('grp').approved.agg(['mean', 'sum', 'count']).assign(
    pct=lambda d: (d['mean']*100).round(2)))

print('\n=== WHY: send-time distribution ===')
sends = nud.merge(cap[['captain_id', 'signup_ts']], on='captain_id')
sends['day'] = (sends.sent_ts - sends.signup_ts).dt.total_seconds()/86400
print(sends.groupby('campaign_id').day.describe(percentiles=[.05, .25, .5, .75, .95]).round(3))

print('\n=== IMMORTAL TIME: were WA_002 recipients still alive when it was sent? ===')
w = wa2.groupby('captain_id').sent_ts.min().rename('wa2_ts')
b = base.merge(w, on='captain_id', how='left')
b['wa2_day'] = (b.wa2_ts - b.signup_ts).dt.total_seconds()/86400
print('among recipients, exit_day vs wa2_day:')
r = b[b.wa2 == 1]
print('  received AFTER they had already exited the funnel:', (r.exit_day < r.wa2_day).sum())
print('  wa2_day describe:'); print(r.wa2_day.describe(percentiles=[.05,.5,.95]).round(3))
print('\nexit_day distribution by group:')
print(b.groupby('wa2').exit_day.describe(percentiles=[.1,.25,.5,.75,.9]).round(2))
LM = r.wa2_day.quantile(0.95)
print(f'\n95th pct of send day = {LM:.2f} -> use as landmark')

print('\n=== LANDMARK ANALYSIS: condition on still being in funnel at the landmark ===')
for L in [1.0, 1.5, 2.0, 2.5, 3.0, 3.6]:
    alive = b[(b.exit_day >= L)].copy()
    g = alive.groupby('wa2').approved.agg(['mean','sum','count'])
    if 0 in g.index and 1 in g.index:
        print(f'  landmark day {L}: n={len(alive)}  WA2 {g.loc[1,"mean"]*100:.2f}% (n={g.loc[1,"count"]})'
              f'  vs ctrl {g.loc[0,"mean"]*100:.2f}% (n={g.loc[0,"count"]})'
              f'  -> lift {(g.loc[1,"mean"]-g.loc[0,"mean"])*100:+.2f} pp')

L = 2.0
alive = b[b.exit_day >= L].copy()
print(f'\n=== Covariate balance at landmark day {L} (n={len(alive)}) ===')
alive['n_passed_at_L'] = [
    passes[(passes.captain_id == c) ].pipe(lambda d: (d.event_ts - s).dt.total_seconds().le(L*86400).sum())
    for c, s in zip(alive.captain_id.head(0), alive.signup_ts.head(0))] if False else np.nan
# vectorised: docs passed by landmark
p2 = passes.merge(cap[['captain_id','signup_ts']], on='captain_id')
p2['day'] = (p2.event_ts - p2.signup_ts).dt.total_seconds()/86400
npass_L = p2[p2.day <= L].groupby('captain_id').doc_type.nunique()
alive['n_passed_at_L'] = alive.captain_id.map(npass_L).fillna(0).astype(int)
for c in ['city','vehicle_type','acquisition_channel','device_tier','n_passed_at_L']:
    print(f'\n-- {c}')
    print(pd.crosstab(alive[c], alive.wa2, normalize='columns').round(4))
print('\napproval rate by (n_passed_at_L, wa2):')
print(alive.groupby(['n_passed_at_L','wa2']).approved.agg(['mean','count']).round(4))

print('\n=== Stratified / regression-adjusted estimate ===')
import statsmodels.formula.api as smf
alive['n_passed_at_L_c'] = alive.n_passed_at_L.astype(str)
mdl = smf.logit('approved ~ wa2 + C(city) + C(vehicle_type) + C(acquisition_channel)'
                ' + C(device_tier) + C(n_passed_at_L_c) + C(age_band) + n_nudges',
                data=alive).fit(disp=0)
print(mdl.summary().tables[1])
# marginal effect of wa2
a0 = alive.assign(wa2=0); a1 = alive.assign(wa2=1)
me = (mdl.predict(a1) - mdl.predict(a0)).mean()
print(f'\nAdjusted average marginal effect of WA_002 = {me*100:+.2f} pp')
se_beta = mdl.bse['wa2']; b_wa2 = mdl.params['wa2']
lo, hi = b_wa2 - 1.96*se_beta, b_wa2 + 1.96*se_beta
print(f'log-odds {b_wa2:.4f} [{lo:.4f},{hi:.4f}]  OR {np.exp(b_wa2):.3f} [{np.exp(lo):.3f},{np.exp(hi):.3f}]')
me_lo = (mdl.predict(a1.assign()) * 0).mean()  # placeholder
# delta-method-free CI on pp scale via param bootstrap
rng = np.random.default_rng(7)
draws = rng.multivariate_normal(mdl.params.values, mdl.cov_params().values, 400)
mes = []
for d in draws:
    pr = mdl.model.predict(d, exog=mdl.model.exog)
    ex1 = mdl.model.exog.copy(); ex0 = mdl.model.exog.copy()
    j = list(mdl.params.index).index('wa2')
    ex1[:, j] = 1; ex0[:, j] = 0
    mes.append((mdl.model.predict(d, exog=ex1) - mdl.model.predict(d, exog=ex0)).mean())
mes = np.array(mes)*100
print(f'bootstrap CI on AME: [{np.percentile(mes,2.5):+.2f}, {np.percentile(mes,97.5):+.2f}] pp')

print('\n=== Same-stage, same-day head-to-head: WA_002 vs WA_001 ===')
w1 = nud[nud.campaign_id=='CAMP_WA_001'].groupby('captain_id').sent_ts.min()
alive['wa1'] = alive.captain_id.isin(w1.index).astype(int)
print(pd.crosstab(alive.wa1, alive.wa2))
hh = alive[(alive.wa1+alive.wa2)==1]
print(hh.groupby('wa2').approved.agg(['mean','sum','count']).round(4))

print('\n=== Placebo: does WA_002 also "predict" things it cannot cause? ===')
# docs already passed BEFORE the nudge was sent
b['npass_before'] = np.nan
pw = p2.merge(w, on='captain_id', how='inner')
before = pw[pw.event_ts < pw.wa2_ts].groupby('captain_id').doc_type.nunique()
r2 = b[b.wa2==1].copy(); r2['npass_before'] = r2.captain_id.map(before).fillna(0)
print('docs already passed before WA_002 was sent:')
print(r2.npass_before.value_counts().sort_index())
print('\n=> recipients had already cleared a median of',
      r2.npass_before.median(), 'documents before the message arrived.')

print('\n=== Click-through as an outcome (self-selection check) ===')
w2c = wa2.groupby('captain_id').clicked.max()
alive['wa2_clicked'] = alive.captain_id.map(w2c)
print(alive[alive.wa2==1].groupby('wa2_clicked').approved.agg(['mean','count']).round(4))
print('\ndelivered=0 but clicked=1 rows in WA_002:', ((wa2.delivered==0)&(wa2.clicked==1)).sum())
print('\n--- undelivered WA_002 as a natural control ---')
w2d = wa2.groupby('captain_id').delivered.max()
alive['wa2_delivered'] = alive.captain_id.map(w2d)
print(alive[alive.wa2==1].groupby('wa2_delivered').approved.agg(['mean','count']).round(4))
