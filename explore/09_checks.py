"""Final checks: suburban night demand (is dead-heading avoidable?), placebo power, sizing inputs."""
import pandas as pd, numpy as np
from scipy import stats
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 80); pd.set_option('display.max_rows', 400)

h = pd.read_csv('../airport_hourly.csv', parse_dates=['hour_ts'])
t = pd.read_csv('../airport_trips.csv', parse_dates=['request_ts'])
h['hod'] = h.hour_ts.dt.hour
h['night'] = h.hod.isin([20,21,22,23,0,1,2,3])
h['fill'] = h.fulfilled_requests/h.requests

print('=== Is there demand in the drop zones at night? (avoidable dead-head?) ===')
print(h.groupby(['zone_type','night']).agg(hours=('hod','size'), req=('requests','sum'),
      req_per_hr=('requests','mean'), unmet=('unfulfilled_requests','sum'),
      online=('online_captains','mean'), fill=('fill','mean'),
      eta=('avg_eta_min','mean'), surge=('avg_surge_multiplier','mean')).round(3))
print('\nnon-terminal zones, requests per hour by hour of day:')
print(h[h.zone_type!='airport_terminal'].pivot_table(index='hod', columns='zone_type',
      values='requests', aggfunc='mean').round(1))
print('\nnon-terminal online captains by hour:')
print(h[h.zone_type!='airport_terminal'].pivot_table(index='hod', columns='zone_type',
      values='online_captains', aggfunc='mean').round(1))
print('\n>>> suburban zones at night: requests/hr vs online captains')
sn = h[(h.zone_type=='suburban')]
print(sn.groupby('hod').agg(req=('requests','mean'), online=('online_captains','mean'),
      fill=('fill','mean'), unmet=('unfulfilled_requests','mean')).round(2))

print('\n=== Value of an incremental airport trip ===')
print('avg fare, airport-origin trips: Rs%.0f' % t.fare_inr.mean())
tn = t[t.request_ts.dt.hour.isin([20,21,22,23,0,1,2,3])]
print('avg fare, night airport trips : Rs%.0f' % tn.fare_inr.mean())
print('night surge at terminals      : %.2fx' % h[(h.zone_type=='airport_terminal')&h.night].avg_surge_multiplier.mean())
for tr in [0.15,0.20,0.25]:
    print(f'  at {tr*100:.0f}% commission, contribution per night trip = Rs{tn.fare_inr.mean()*tr:.0f}')

print('\n=== Placebo power check for CAMP_WA_002 ===')
n1, n0, p = 6888, 510, 0.3146
se = np.sqrt(p*(1-p)*(1/n1+1/n0))
print(f'delivered vs undelivered: n={n1}/{n0}, SE of diff = {se*100:.2f} pp')
print(f'  95% CI half-width = {1.96*se*100:.2f} pp -> can only rule out effects larger than ~{1.96*se*100:.1f} pp')
print(f'  observed diff = -0.30 pp, 95% CI [{-0.30-1.96*se*100:.2f}, {-0.30+1.96*se*100:.2f}] pp')
n1c, n0c = 2839, 4049
sec = np.sqrt(p*(1-p)*(1/n1c+1/n0c))
print(f'clicked vs not: SE = {sec*100:.2f} pp, 95% CI on observed -1.21 pp: '
      f'[{-1.21-1.96*sec*100:.2f}, {-1.21+1.96*sec*100:.2f}] pp')
print('\nMDE (80% power, alpha .05) for a holdout test on eligible captains:')
for nper in [1000, 2000, 5000, 10000]:
    mde = (1.96+0.84)*np.sqrt(2*p*(1-p)/nper)
    print(f'  n={nper} per arm -> MDE = {mde*100:.2f} pp')

print('\n=== 5x scale-up headroom for CAMP_WA_002 ===')
cap = pd.read_csv('../captains.csv', parse_dates=['signup_ts'])
doc = pd.read_csv('../doc_events.csv', parse_dates=['event_ts'])
nud = pd.read_csv('../nudges.csv', parse_dates=['sent_ts'])
apr = pd.read_csv('../approvals.csv', parse_dates=['decision_ts'])
base = cap.merge(apr, on='captain_id'); base = base[base.signup_ts < pd.Timestamp('2026-06-16')]
npass = doc[doc.event_type=='verification_pass'].groupby('captain_id').doc_type.nunique()
base = base.assign(n_passed=base.captain_id.map(npass).fillna(0).astype(int))
elig = (base.n_passed>=2).sum()
sent = base.captain_id.isin(nud[nud.campaign_id=='CAMP_WA_002'].captain_id).sum()
print(f'eligible (cleared >=2 docs) in mature cohort: {elig:,}')
print(f'currently receive WA_002                    : {sent:,}  ({sent/elig*100:.1f}% coverage)')
print(f'MAX headroom inside current trigger         : {elig/sent:.2f}x  (not 5x)')
print(f'unserved eligible captains                  : {elig-sent:,}')
MONTHS = 5.45
print(f'= {(elig-sent)/MONTHS:.0f} captains/month currently eligible but not sent')
for eff in [0.0, 0.02, 0.038]:
    print(f'  at a true effect of {eff*100:.1f} pp -> {(elig-sent)*eff/MONTHS:.0f} extra approvals/month')

print('\n=== Final funnel numbers for the memo ===')
base = base.assign(approved=(base.final_status=='approved').astype(int))
print(f'mature cohort            : {len(base):,} signups (1 Jan - 15 Jun 2026)')
print(f'approved                 : {base.approved.sum():,}  A2O = {base.approved.mean()*100:.2f}%')
print(f'signups/month            : {len(base)/MONTHS:.0f}')
print(f'approvals/month          : {base.approved.sum()/MONTHS:.0f}')
act = pd.read_csv('../activation.csv', parse_dates=['first_order_ts'])
b2 = base.merge(act, on='captain_id', how='left')
print(f'first order              : {b2.first_order_ts.notna().sum():,}  R2A = {b2.first_order_ts.notna().sum()/len(base)*100:.2f}%')
print(f'activation | approved    : {b2.first_order_ts.notna().sum()/base.approved.sum()*100:.2f}%')

print('\n=== Low-tier device sizing (RC/FITNESS/INSURANCE image-quality) ===')
IMG = ['image_blurred','ocr_low_confidence','details_not_legible']
du = doc.merge(cap[['captain_id','device_tier']], on='captain_id')
du = du[du.captain_id.isin(base.captain_id)]
A4 = ['RC','FITNESS','INSURANCE']
CARD = ['DL','AADHAAR','PERMIT']
ver = du[du.event_type!='upload_success'].copy()
ver['fail'] = (ver.event_type=='verification_fail').astype(int)
ver['imgfail'] = ver.fail*ver.failure_reason.isin(IMG).astype(int)
print('image-quality fail rate per upload:')
piv = ver.pivot_table(index='doc_type', columns='device_tier', values='imgfail', aggfunc='mean')
print(piv.round(4).loc[A4+CARD])
print('\nA4 paper docs (RC/FITNESS/INSURANCE) vs card docs (DL/AADHAAR/PERMIT):')
ver['form'] = np.where(ver.doc_type.isin(A4),'A4 paper','card')
print(ver.pivot_table(index='form', columns='device_tier', values=['imgfail','fail'], aggfunc='mean').round(4))
print('\nuploads affected: low-tier uploads of A4 docs')
lowA4 = ver[(ver.device_tier=='low') & ver.doc_type.isin(A4)]
print(f'  low-tier A4 verifications: {len(lowA4):,}, image-quality failures: {lowA4.imgfail.sum():,}')
midrate = ver[(ver.device_tier=='mid') & ver.doc_type.isin(A4)].imgfail.mean()
print(f'  if low-tier matched MID-tier image-fail rate ({midrate*100:.2f}%), '
      f'failures would be {len(lowA4)*midrate:,.0f} -> {lowA4.imgfail.sum()-len(lowA4)*midrate:,.0f} fewer failures')
# translate avoided failures into approvals
stuck_ids = set()
for d in A4:
    dd = doc[doc.doc_type==d]
    fi = set(dd[(dd.event_type=='verification_fail') & dd.failure_reason.isin(IMG)].captain_id)
    ps = set(dd[dd.event_type=='verification_pass'].captain_id)
    stuck_ids |= (fi-ps)
stuck = base[base.captain_id.isin(stuck_ids)]
low_stuck = stuck[stuck.device_tier=='low']
print(f'\ncaptains permanently stuck on an A4 doc for image reasons: {len(stuck):,} '
      f'(low-tier {len(low_stuck):,} = {len(low_stuck)/len(stuck)*100:.0f}%)')
print(f'= {len(stuck)/MONTHS:.0f}/month')
# onward conversion if they had passed: use peers who passed the same doc
for r in [0.3,0.4,0.5]:
    tot = 0
    for d in A4:
        dd = doc[doc.doc_type==d]
        fi = set(dd[(dd.event_type=='verification_fail') & dd.failure_reason.isin(IMG)].captain_id)
        ps = set(dd[dd.event_type=='verification_pass'].captain_id)
        s = base[base.captain_id.isin((fi-ps))]
        onward = base[base.captain_id.isin(ps)].approved.mean()
        tot += len(s)*r*onward
    print(f'  recover {r*100:.0f}% -> {tot/MONTHS:.0f} extra approvals/month')
