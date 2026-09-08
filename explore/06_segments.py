"""A2: segment the leak, test ops levers (verification SLA, device, channel), size the fixes."""
import pandas as pd, numpy as np
from scipy import stats
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 80); pd.set_option('display.max_rows', 400)

SEQ = ['DL','RC','AADHAAR','PERMIT','FITNESS','INSURANCE']
cap = pd.read_csv('../captains.csv', parse_dates=['signup_ts'])
doc = pd.read_csv('../doc_events.csv', parse_dates=['event_ts'])
apr = pd.read_csv('../approvals.csv', parse_dates=['decision_ts'])
act = pd.read_csv('../activation.csv', parse_dates=['first_order_ts'])

CUT = pd.Timestamp('2026-06-16')
base = cap.merge(apr, on='captain_id')
base = base[base.signup_ts < CUT].copy()
base['approved'] = (base.final_status == 'approved').astype(int)
MONTHS = (CUT - pd.Timestamp('2026-01-01')).days / 30.4375
print(f'mature cohort n={len(base)}, span={MONTHS:.2f} months, '
      f'signups/mo={len(base)/MONTHS:.0f}, approvals/mo={base.approved.sum()/MONTHS:.0f}')
print(f'A2O = {base.approved.mean()*100:.2f}%')

npass = doc[doc.event_type=='verification_pass'].groupby('captain_id').doc_type.nunique()
base['n_passed'] = base.captain_id.map(npass).fillna(0).astype(int)
base['n_required'] = np.where(base.vehicle_type=='ERickshaw', 5, 6)

print('\n=== A2O by segment (mature cohort) ===')
for c in ['city','vehicle_type','acquisition_channel','device_tier','app_language','age_band']:
    t = base.groupby(c).agg(n=('approved','size'), approved=('approved','sum'))
    t['a2o_pct'] = (t.approved/t.n*100).round(2)
    t['share_of_signups'] = (t.n/len(base)*100).round(1)
    print(f'\n-- {c}'); print(t.sort_values('a2o_pct'))

print('\n=== Two-way: channel x device_tier A2O ===')
print(pd.crosstab(base.acquisition_channel, base.device_tier, values=base.approved,
                  aggfunc='mean').round(4)*100)
print('\ncounts:'); print(pd.crosstab(base.acquisition_channel, base.device_tier))

print('\n=== LEVER 1: verification turnaround time -> abandonment ===')
up = doc[doc.event_type=='upload_success'][['captain_id','doc_type','attempt_no','event_ts']].rename(columns={'event_ts':'up_ts'})
ver = doc[doc.event_type!='upload_success'][['captain_id','doc_type','attempt_no','event_ts','event_type']].rename(columns={'event_ts':'ver_ts'})
uv = up.merge(ver, on=['captain_id','doc_type','attempt_no'], how='left')
uv['lag_h'] = (uv.ver_ts - uv.up_ts).dt.total_seconds()/3600
uv = uv[uv.captain_id.isin(base.captain_id)]
print('verification lag by doc:'); print(uv.groupby('doc_type').lag_h.describe(percentiles=[.5,.9,.99]).round(2))

# does a slow PASS make the captain less likely to start the next doc?
p = doc[doc.event_type=='verification_pass'].copy()
p = p.merge(cap[['captain_id','vehicle_type']], on='captain_id')
p['seq'] = p.doc_type.map({d:i for i,d in enumerate(SEQ)})
p = p.sort_values(['captain_id','event_ts'])
p['rk'] = p.groupby('captain_id').cumcount()+1
# next required doc after this pass
def nxt(vt, d):
    req = [x for x in SEQ if not (x=='PERMIT' and vt=='ERickshaw')]
    i = req.index(d)
    return req[i+1] if i+1 < len(req) else None
p['next_doc'] = [nxt(v,d) for v,d in zip(p.vehicle_type, p.doc_type)]
pl = p.merge(uv[['captain_id','doc_type','attempt_no','lag_h']], on=['captain_id','doc_type','attempt_no'], how='left')
up_types = doc[doc.event_type=='upload_success'].groupby(['captain_id','doc_type']).size()
pl['started_next'] = [1 if (n is not None and (c,n) in up_types.index) else (np.nan if n is None else 0)
                      for c,n in zip(pl.captain_id, pl.next_doc)]
pl = pl[pl.started_next.notna() & pl.captain_id.isin(base.captain_id)]
pl['lag_bucket'] = pd.cut(pl.lag_h, [0,2,4,8,12,18,24,100],
                          labels=['0-2h','2-4h','4-8h','8-12h','12-18h','18-24h','24h+'])
print('\nP(start next doc) by verification lag on the doc just passed:')
print(pl.groupby('lag_bucket', observed=True).started_next.agg(['mean','count']).round(4))
print('\nsame, controlling for stage (P(start next) by doc x lag bucket):')
print(pl.pivot_table(index='doc_type', columns='lag_bucket', values='started_next',
                     aggfunc='mean', observed=True).round(3))
print('\nlogit: started_next ~ lag_h + stage + device + channel')
import statsmodels.formula.api as smf
pl2 = pl.merge(base[['captain_id','device_tier','acquisition_channel','city']], on='captain_id')
mm = smf.logit('started_next ~ lag_h + C(doc_type) + C(device_tier) + C(acquisition_channel)', data=pl2).fit(disp=0)
print(mm.summary().tables[1])

print('\n=== LEVER 2: RC — the worst document ===')
rc = doc[doc.doc_type=='RC']
rc_up = rc[rc.event_type=='upload_success'].captain_id.nunique()
rc_pass = rc[rc.event_type=='verification_pass'].captain_id.nunique()
print(f'RC uploaders {rc_up}, passers {rc_pass}, never passed {rc_up-rc_pass}')
rcf = rc[rc.event_type=='verification_fail']
print('\nRC failure reasons:'); 
t = rcf.failure_reason.value_counts(); print(pd.DataFrame({'n':t,'pct':(t/t.sum()*100).round(1)}))
IMG = ['image_blurred','ocr_low_confidence','details_not_legible']
print(f'\nimage-quality cluster = {rcf.failure_reason.isin(IMG).sum()} / {len(rcf)} '
      f'= {rcf.failure_reason.isin(IMG).mean()*100:.1f}% of RC fails')
allf = doc[doc.event_type=='verification_fail']
print(f'image-quality cluster across ALL docs = {allf.failure_reason.isin(IMG).sum()} / {len(allf)} '
      f'= {allf.failure_reason.isin(IMG).mean()*100:.1f}%')
print('\nfail-rate per upload, by doc x device_tier:')
du = doc.merge(cap[['captain_id','device_tier']], on='captain_id')
duv = du[du.event_type!='upload_success'].assign(fail=lambda d: (d.event_type=='verification_fail').astype(int))
print(duv.pivot_table(index='doc_type', columns='device_tier', values='fail', aggfunc='mean').round(4))
print('\nimage-quality fail RATE per upload (share of uploads failing for image reasons):')
duv['imgfail'] = duv.fail * duv.failure_reason.isin(IMG).astype(int)
print(duv.pivot_table(index='doc_type', columns='device_tier', values='imgfail', aggfunc='mean').round(4))

print('\n=== LEVER 3: retry behaviour after a failure ===')
f_ev = doc[doc.event_type=='verification_fail'][['captain_id','doc_type','attempt_no','event_ts']]
f_ev = f_ev[f_ev.captain_id.isin(base.captain_id)]
nxt_up = doc[doc.event_type=='upload_success'].groupby(['captain_id','doc_type']).attempt_no.max()
f_ev['max_attempt'] = [nxt_up.get((c,d),0) for c,d in zip(f_ev.captain_id, f_ev.doc_type)]
f_ev['retried'] = (f_ev.max_attempt > f_ev.attempt_no).astype(int)
print('P(retry) after a failure, by attempt_no:')
print(f_ev.groupby('attempt_no').retried.agg(['mean','count']).round(4))
print('\nP(retry) by failure_reason:')
fr = doc[doc.event_type=='verification_fail'][['captain_id','doc_type','attempt_no','failure_reason']]
f_ev = f_ev.merge(fr, on=['captain_id','doc_type','attempt_no'])
print(f_ev[f_ev.attempt_no<3].groupby('failure_reason').retried.agg(['mean','count']).round(4).sort_values('mean'))
print('\nP(eventually pass | retried) by doc:')
rp = f_ev[f_ev.retried==1].copy()
passed_doc = set(zip(doc[doc.event_type=='verification_pass'].captain_id, doc[doc.event_type=='verification_pass'].doc_type))
rp['later_pass'] = [(c,d) in passed_doc for c,d in zip(rp.captain_id, rp.doc_type)]
print(rp.groupby('doc_type').later_pass.agg(['mean','count']).round(4))

print('\n=== SIZING: gap-closing scenarios (approvals per month) ===')
tot_signup_mo = len(base)/MONTHS
appr_mo = base.approved.sum()/MONTHS
print(f'baseline: {appr_mo:.0f} approvals/mo from {tot_signup_mo:.0f} signups/mo (A2O {base.approved.mean()*100:.2f}%)')

# S1: image-quality failures on RC+FITNESS+INSURANCE removed at rate r
def sim_pass_uplift(target_docs, share_recovered):
    """crude: recover a share of captains who ever failed an image-quality reason on target docs
       and never passed that doc; assume they then convert at the rate of peers who passed that doc."""
    out = {}
    for d in target_docs:
        dd = doc[(doc.doc_type==d)]
        failed_img = set(dd[(dd.event_type=='verification_fail') & dd.failure_reason.isin(IMG)].captain_id)
        passed = set(dd[dd.event_type=='verification_pass'].captain_id)
        stuck = (failed_img - passed) & set(base.captain_id)
        # onward conversion of peers who passed this doc
        peers = base[base.captain_id.isin(passed)]
        onward = peers.approved.mean()
        out[d] = dict(stuck=len(stuck), onward_rate=round(onward,4),
                      extra_approvals=len(stuck)*share_recovered*onward)
    return pd.DataFrame(out).T

print('\n-- S1 image-quality capture gate (recover 40% of image-quality-stuck captains) --')
s1 = sim_pass_uplift(['DL','RC','AADHAAR','PERMIT','FITNESS','INSURANCE'], 0.40)
s1['extra_per_month'] = (s1.extra_approvals/MONTHS).round(1)
print(s1)
print('TOTAL extra approvals/mo:', round(s1.extra_per_month.sum(),1))

print('\n-- S2 last-mile: captains who cleared all but INSURANCE and never uploaded it --')
ins_up = set(doc[(doc.doc_type=='INSURANCE')&(doc.event_type=='upload_success')].captain_id)
ins_pass = set(doc[(doc.doc_type=='INSURANCE')&(doc.event_type=='verification_pass')].captain_id)
fit_pass = set(doc[(doc.doc_type=='FITNESS')&(doc.event_type=='verification_pass')].captain_id)
stalled = base[base.captain_id.isin(fit_pass - ins_up)]
print('cleared FITNESS, never uploaded INSURANCE:', len(stalled), f'({len(stalled)/MONTHS:.0f}/mo)')
p_pass_ins = len(ins_pass)/max(len(ins_up),1)
p_appr_given_ins = base[base.captain_id.isin(ins_pass)].approved.mean()
print(f'P(pass INSURANCE | upload) = {p_pass_ins:.3f}; P(approved | passed INSURANCE) = {p_appr_given_ins:.3f}')
for r in [0.15,0.25,0.35]:
    print(f'  recover {r*100:.0f}% -> {len(stalled)*r*p_pass_ins*p_appr_given_ins/MONTHS:.1f} approvals/mo')

print('\n-- S3 channel mix: paid_digital & organic_app vs fos_field --')
ch = base.groupby('acquisition_channel').agg(n=('approved','size'), a=('approved','sum'))
ch['a2o'] = ch.a/ch.n
print(ch.round(4))
bench = ch.loc['fos_field','a2o']
for c in ch.index:
    if ch.loc[c,'a2o'] < bench:
        gap = (bench - ch.loc[c,'a2o'])*ch.loc[c,'n']
        print(f'  {c}: closing HALF the gap to fos_field -> {gap*0.5/MONTHS:.1f} approvals/mo')

print('\n-- S4 verification SLA: cut lag to <4h for all uploads --')
fast = pl2[pl2.lag_h<=4].started_next.mean(); slow_n = (pl2.lag_h>4).sum()
print(f'P(start next|lag<=4h)={fast:.4f}; uploads with lag>4h={slow_n}')
byb = pl2.groupby(pd.cut(pl2.lag_h,[0,4,8,12,18,24,100]), observed=True).started_next.agg(['mean','count'])
print(byb.round(4))
extra_starts = sum((fast - r['mean'])*r['count'] for _,r in byb.iloc[1:].iterrows())
print(f'extra "next doc started" events if all <=4h: {extra_starts:.0f}')
onward = base[base.n_passed>=1].approved.mean()
print(f'  x P(approved | in funnel) {onward:.3f} -> {extra_starts*onward/MONTHS:.1f} approvals/mo (upper bound, correlational)')

print('\n=== R2A (signup -> first order) ===')
a = base.merge(act, on='captain_id', how='left')
r2a_pool = base[base.signup_ts < pd.Timestamp('2026-06-30') - pd.Timedelta(days=14)]
print('approved:', base.approved.sum())
print('approved with first_order_ts:', a.first_order_ts.notna().sum())
print(f'R2A (mature cohort) = {a.first_order_ts.notna().sum()/len(base)*100:.2f}%')
print(f'activation rate among approved = {a.first_order_ts.notna().sum()/base.approved.sum()*100:.2f}%')
appr = a[a.approved==1]
print('\napproved but never ordered:', appr.first_order_ts.isna().sum())
print('  of which censored (<7d since decision):',
      (appr.first_order_ts.isna() & appr.orders_d7.isna()).sum())
print('  genuine non-starters (orders_d7==0 observed):',
      (appr.first_order_ts.isna() & (appr.orders_d7==0)).sum())
print('\ndays approval -> first order:')
print(((appr.first_order_ts - appr.decision_ts).dt.total_seconds()/86400).describe(percentiles=[.5,.9]).round(2))
print('\nR2A by segment:')
appr2 = appr[appr.orders_d7.notna()]
for c in ['city','vehicle_type','acquisition_channel','device_tier']:
    t = appr2.groupby(c).apply(lambda d: pd.Series({
        'approved': len(d), 'activated': d.first_order_ts.notna().sum(),
        'act_rate': round(d.first_order_ts.notna().mean()*100,2),
        'mean_orders_d30': round(d.orders_d30.mean(),1)}), include_groups=False)
    print(f'\n-- {c}'); print(t)
