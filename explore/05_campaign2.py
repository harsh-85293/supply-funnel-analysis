"""A3 part 2: recover the assignment rule for CAMP_WA_002, then estimate a defensible effect."""
import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 80); pd.set_option('display.max_rows', 300)

cap = pd.read_csv('../captains.csv', parse_dates=['signup_ts'])
doc = pd.read_csv('../doc_events.csv', parse_dates=['event_ts'])
apr = pd.read_csv('../approvals.csv', parse_dates=['decision_ts'])
nud = pd.read_csv('../nudges.csv', parse_dates=['sent_ts'])

base = cap.merge(apr, on='captain_id')
base = base[base.signup_ts < pd.Timestamp('2026-06-16')].copy()
base['approved'] = (base.final_status == 'approved').astype(int)
passes = doc[doc.event_type == 'verification_pass'].copy()
npass = passes.groupby('captain_id').doc_type.nunique()
base['n_passed'] = base.captain_id.map(npass).fillna(0).astype(int)
lastev = doc.groupby('captain_id').event_ts.max()
base['exit_ts'] = base.decision_ts.fillna(base.captain_id.map(lastev)).fillna(base.signup_ts)

# time of 2nd doc clearance
p_ord = passes.sort_values(['captain_id', 'event_ts']).copy()
p_ord['rk'] = p_ord.groupby('captain_id').cumcount() + 1
second_pass = (p_ord[p_ord.rk == 2].set_index('captain_id').event_ts.rename('t_doc2'))
base['t_doc2'] = base.captain_id.map(second_pass)

wa2 = nud[nud.campaign_id == 'CAMP_WA_002'].groupby('captain_id').agg(
    wa2_ts=('sent_ts','min'), wa2_delivered=('delivered','max'), wa2_clicked=('clicked','max'))
base = base.merge(wa2, on='captain_id', how='left')
base['wa2'] = base.wa2_ts.notna().astype(int)

print('=== Assignment rule diagnostics ===')
print('mature cohort with >=2 docs passed:', (base.n_passed >= 2).sum())
print('WA_002 recipients:', base.wa2.sum())
print('recipients with <2 docs passed ever:', ((base.wa2==1) & (base.n_passed < 2)).sum())
print('\nP(receive WA_002) by n_passed:')
print(base.groupby('n_passed').wa2.agg(['mean','sum','count']).round(4))

el = base[base.n_passed >= 2].copy()
el['lag_h'] = (el.wa2_ts - el.t_doc2).dt.total_seconds()/3600
print('\nlag from 2nd doc clearance to WA_002 send (hours):')
print(el.lag_h.describe(percentiles=[.01,.05,.5,.95,.99]).round(3))
print('negative lags (sent before clearing doc 2):', (el.lag_h < 0).sum())

print('\n=== Is non-receipt a calendar/launch effect? (natural experiment?) ===')
print('WA_002 sent_ts by calendar month:')
print(base.loc[base.wa2==1, 'wa2_ts'].dt.to_period('M').value_counts().sort_index())
print('\nP(receive | cleared>=2 docs) by 2nd-doc-clearance month:')
print(el.groupby(el.t_doc2.dt.to_period('M')).wa2.agg(['mean','count']).round(4))
print('\nP(receive | cleared>=2 docs) by 2nd-doc-clearance WEEK:')
print(el.groupby(el.t_doc2.dt.to_period('W')).wa2.agg(['mean','count']).round(4).to_string())
print('\nP(receive) by day-of-week and hour of doc2 clearance:')
print(el.groupby(el.t_doc2.dt.dayofweek).wa2.mean().round(4))
print(el.groupby(el.t_doc2.dt.hour).wa2.mean().round(4).to_string())

print('\n=== ELIGIBLE-ONLY comparison, aligned on doc-2 clearance ===')
# survival requirement: still in funnel `lag` hours after doc2 (median lag)
med_lag = el.lag_h.median()
p95_lag = el.lag_h.quantile(0.95)
print(f'median lag {med_lag:.2f}h, p95 {p95_lag:.2f}h')
el['alive_h_after_doc2'] = (el.exit_ts - el.t_doc2).dt.total_seconds()/3600
for L in [0, 6, 12, 18, 24, 30, 36]:
    s = el[el.alive_h_after_doc2 >= L]
    g = s.groupby('wa2').approved.agg(['mean','sum','count'])
    if len(g) == 2:
        print(f'  require alive >= {L:>2}h after doc2: n={len(s):>6}  '
              f'WA2 {g.loc[1,"mean"]*100:5.2f}% (n={g.loc[1,"count"]:>5})  '
              f'ctrl {g.loc[0,"mean"]*100:5.2f}% (n={g.loc[0,"count"]:>5})  '
              f'lift {(g.loc[1,"mean"]-g.loc[0,"mean"])*100:+.2f} pp')

L = round(med_lag)
s = el[el.alive_h_after_doc2 >= L].copy()
print(f'\n=== Balance in eligible+aligned set (alive >= {L}h after doc2), n={len(s)} ===')
for c in ['city','vehicle_type','acquisition_channel','device_tier','n_passed','app_language','age_band']:
    t = pd.crosstab(s[c], s.wa2, normalize='columns').round(4)
    t['abs_diff'] = (t[1]-t[0]).abs().round(4)
    print(f'\n-- {c}'); print(t)
s['doc2_month'] = s.t_doc2.dt.to_period('M').astype(str)
print('\n-- doc2_month'); print(pd.crosstab(s.doc2_month, s.wa2, normalize='columns').round(4))

print('\n=== PLACEBO TESTS (the decisive evidence) ===')
print('1) Delivered vs NOT delivered, among recipients:')
print(s[s.wa2==1].groupby('wa2_delivered').approved.agg(['mean','sum','count']).round(4))
from scipy import stats
d0 = s[(s.wa2==1)&(s.wa2_delivered==0)].approved; d1 = s[(s.wa2==1)&(s.wa2_delivered==1)].approved
print('   diff = %+.2f pp, two-prop z-test p = %.3f' % ((d1.mean()-d0.mean())*100,
      stats.norm.sf(abs((d1.mean()-d0.mean())/np.sqrt(d1.var()/len(d1)+d0.var()/len(d0))))*2))
print('\n2) Clicked vs not clicked, among delivered recipients:')
sd = s[(s.wa2==1)&(s.wa2_delivered==1)]
print(sd.groupby('wa2_clicked').approved.agg(['mean','sum','count']).round(4))
c0 = sd[sd.wa2_clicked==0].approved; c1 = sd[sd.wa2_clicked==1].approved
print('   diff = %+.2f pp, p = %.3f' % ((c1.mean()-c0.mean())*100,
      stats.norm.sf(abs((c1.mean()-c0.mean())/np.sqrt(c1.var()/len(c1)+c0.var()/len(c0))))*2))
print('\n3) Placebo outcome: did WA_002 "cause" the RC pass that preceded it?')
#   compare pre-treatment outcome: attempts needed on RC (fully determined before send)
rc_att = doc[(doc.doc_type=='RC')&(doc.event_type=='verification_pass')].set_index('captain_id').attempt_no
s['rc_attempts'] = s.captain_id.map(rc_att)
print(s.groupby('wa2').rc_attempts.agg(['mean','count']).round(4))
print('   -> pre-treatment covariate; any gap here is pure selection, not effect.')
print('\n4) Placebo: time from signup to doc2 (entirely pre-treatment)')
s['h_to_doc2'] = (s.t_doc2 - s.signup_ts).dt.total_seconds()/3600
print(s.groupby('wa2').h_to_doc2.agg(['mean','median','count']).round(3))

print('\n=== Regression-adjusted effect within eligible+aligned set ===')
import statsmodels.formula.api as smf
s['n_passed_c'] = s.n_passed.clip(upper=4).astype(str)
mdl = smf.logit('approved ~ wa2 + C(city)+C(vehicle_type)+C(acquisition_channel)+C(device_tier)'
                '+C(age_band)+C(doc2_month)+rc_attempts+h_to_doc2', data=s).fit(disp=0)
print(mdl.summary().tables[1])
j = list(mdl.params.index).index('wa2')
ex1 = mdl.model.exog.copy(); ex0 = mdl.model.exog.copy(); ex1[:,j]=1; ex0[:,j]=0
ame = (mdl.model.predict(mdl.params.values, exog=ex1)-mdl.model.predict(mdl.params.values, exog=ex0)).mean()
rng = np.random.default_rng(11)
draws = rng.multivariate_normal(mdl.params.values, mdl.cov_params().values, 600)
ames = np.array([(mdl.model.predict(d, exog=ex1)-mdl.model.predict(d, exog=ex0)).mean() for d in draws])*100
print(f'\nAME of WA_002 within eligible+aligned set = {ame*100:+.2f} pp  '
      f'95% CI [{np.percentile(ames,2.5):+.2f}, {np.percentile(ames,97.5):+.2f}]')

print('\n=== What the deck number should be: decomposition ===')
naive = base.groupby('wa2').approved.mean()
print(f'  naive raw gap                      : {(naive[1]-naive[0])*100:+.2f} pp')
el_g = el.groupby('wa2').approved.mean()
print(f'  restrict to eligible (>=2 docs)    : {(el_g[1]-el_g[0])*100:+.2f} pp')
s_g = s.groupby('wa2').approved.mean()
print(f'  + align on survival to send time   : {(s_g[1]-s_g[0])*100:+.2f} pp')
print(f'  + covariate adjustment             : {ame*100:+.2f} pp')
print(f'  placebo (undelivered vs delivered) : {(d1.mean()-d0.mean())*100:+.2f} pp  <- should be ~0 if real')
