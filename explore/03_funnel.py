"""Funnel mechanics: where volume is lost, drop type (fail vs abandon), segment cuts."""
import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 300)

EXTRACT = pd.Timestamp('2026-06-30 23:59:00')
SEQ_ALL = ['DL', 'RC', 'AADHAAR', 'PERMIT', 'FITNESS', 'INSURANCE']

cap = pd.read_csv('../captains.csv', parse_dates=['signup_ts'])
doc = pd.read_csv('../doc_events.csv', parse_dates=['event_ts'])
apr = pd.read_csv('../approvals.csv', parse_dates=['decision_ts'])

def required(vt):
    return [d for d in SEQ_ALL if not (d == 'PERMIT' and vt == 'ERickshaw')]

cap['n_required'] = np.where(cap.vehicle_type == 'ERickshaw', 5, 6)
m = cap.merge(apr, on='captain_id')

print('=== Exact censoring cutoff ===')
ip = m[m.final_status == 'in_progress']
print('in_progress signup_ts min/max:', ip.signup_ts.min(), ip.signup_ts.max())
print('max signup_ts of TERMINAL captains:', m.loc[m.final_status != 'in_progress', 'signup_ts'].max())
for cutoff in ['2026-06-14', '2026-06-15', '2026-06-16', '2026-06-17', '2026-06-18']:
    c = pd.Timestamp(cutoff)
    sub = m[m.signup_ts < c]
    print(f'  cutoff {cutoff}: n={len(sub)}, in_progress={(sub.final_status=="in_progress").sum()}',
          f'({(sub.final_status=="in_progress").mean()*100:.2f}%)')
print('\nmax days_to_decision observed:',
      ((m.decision_ts - m.signup_ts).dt.total_seconds()/86400).max())

print('\n=== Rebuild per-captain doc progress from events ===')
doc = doc.merge(cap[['captain_id', 'vehicle_type', 'signup_ts', 'n_required']], on='captain_id')
doc['seq'] = doc.doc_type.map({d: i for i, d in enumerate(SEQ_ALL)})
# effective stage index within the captain's required sequence
doc['req_idx'] = doc.apply(lambda r: required(r.vehicle_type).index(r.doc_type), axis=1)

passed = doc[doc.event_type == 'verification_pass'].groupby('captain_id').agg(
    n_passed=('doc_type', 'nunique'), last_pass_ts=('event_ts', 'max'))
uploaded = doc[doc.event_type == 'upload_success'].groupby('captain_id').agg(
    n_uploaded_types=('doc_type', 'nunique'), n_uploads=('event_id', 'count'),
    last_upload_ts=('event_ts', 'max'))
failed = doc[doc.event_type == 'verification_fail'].groupby('captain_id').agg(
    n_fails=('event_id', 'count'))
lastev = doc.sort_values('event_ts').groupby('captain_id').agg(
    last_event_type=('event_type', 'last'), last_event_doc=('doc_type', 'last'),
    last_event_attempt=('attempt_no', 'last'), last_event_ts=('event_ts', 'last'))

f = m.merge(passed, on='captain_id', how='left').merge(uploaded, on='captain_id', how='left') \
     .merge(failed, on='captain_id', how='left').merge(lastev, on='captain_id', how='left')
for c in ['n_passed', 'n_uploaded_types', 'n_uploads', 'n_fails']:
    f[c] = f[c].fillna(0).astype(int)
f['completed_docs'] = f.n_passed >= f.n_required

print('completed all required docs (from events):', f.completed_docs.sum())
print('final_status x completed_docs:')
print(pd.crosstab(f.final_status, f.completed_docs))
print('\nMISMATCH: approvals.docs_cleared vs events n_passed:',
      (f.docs_cleared != f.n_passed).sum(), f'({(f.docs_cleared != f.n_passed).mean()*100:.2f}%)')
_mm = f.loc[f.docs_cleared != f.n_passed]
print((_mm.docs_cleared - _mm.n_passed).value_counts().sort_index())
# does last_stage_reached agree with the furthest doc actually reached?
furthest = (doc.sort_values('event_ts').groupby('captain_id').doc_type.last())
_chk = f[f.final_status == 'dropped_in_docs'].copy()
_chk['furthest_event_doc'] = _chk.captain_id.map(furthest)
print('\nlast_stage_reached vs furthest doc in events (droppers): mismatch =',
      (_chk.last_stage_reached != _chk.furthest_event_doc).sum(), 'of', len(_chk))
print(pd.crosstab(_chk.last_stage_reached, _chk.furthest_event_doc))

MAT = f[f.signup_ts < pd.Timestamp('2026-06-16')].copy()
print(f'\n=== MATURE COHORT n={len(MAT)} (signup < 2026-06-16), in_progress={(MAT.final_status=="in_progress").sum()} ===')

print('\n=== Headline funnel (mature cohort) ===')
n = len(MAT)
stages = {
    'signed up': n,
    'uploaded >=1 doc': (MAT.n_uploads > 0).sum(),
    'passed >=1 doc': (MAT.n_passed >= 1).sum(),
    'cleared all required docs': MAT.completed_docs.sum(),
    'approved': (MAT.final_status == 'approved').sum(),
}
sf = pd.DataFrame({'n': stages})
sf['pct_of_signup'] = (sf.n / n * 100).round(2)
sf['step_conv'] = (sf.n / sf.n.shift()).round(4)
sf['lost_here'] = (sf.n.shift() - sf.n).fillna(0).astype(int)
print(sf)

print('\n=== Stage-by-stage document funnel (mature cohort) ===')
rows = []
prev = n
for i, d in enumerate(SEQ_ALL):
    elig = MAT[MAT.vehicle_type.isin(['Auto', 'Cab'])] if d == 'PERMIT' else MAT
    n_elig = len(elig)
    ids = set(elig.captain_id)
    dd = doc[(doc.doc_type == d) & (doc.captain_id.isin(ids))]
    n_up = dd[dd.event_type == 'upload_success'].captain_id.nunique()
    n_pass = dd[dd.event_type == 'verification_pass'].captain_id.nunique()
    # reached this stage = cleared all prior required docs
    rows.append(dict(doc=d, eligible=n_elig, uploaded=n_up, passed=n_pass,
                     never_uploaded=n_elig - n_up))
fd = pd.DataFrame(rows)
print(fd)

print('\n=== Reached-stage funnel (properly conditioned) ===')
# reached stage k = passed all required docs before k
pass_by = doc[doc.event_type == 'verification_pass'].pivot_table(
    index='captain_id', columns='doc_type', values='event_ts', aggfunc='min')
pass_by = pass_by.reindex(MAT.captain_id)
up_by = doc[doc.event_type == 'upload_success'].pivot_table(
    index='captain_id', columns='doc_type', values='event_ts', aggfunc='min').reindex(MAT.captain_id)
vt = MAT.set_index('captain_id').vehicle_type
rows = []
reached = pd.Series(True, index=MAT.captain_id)
for d in SEQ_ALL:
    elig = reached & ((vt != 'ERickshaw') if d == 'PERMIT' else True)
    up = elig & up_by.get(d, pd.Series(np.nan, index=MAT.captain_id)).notna()
    ps = elig & pass_by.get(d, pd.Series(np.nan, index=MAT.captain_id)).notna()
    rows.append(dict(stage=d, reached=int(elig.sum()), uploaded=int(up.sum()), passed=int(ps.sum()),
                     drop_before_upload=int((elig & ~up).sum()),
                     drop_at_verification=int((up & ~ps).sum()),
                     upload_rate=round(up.sum()/max(elig.sum(),1), 4),
                     pass_given_upload=round(ps.sum()/max(up.sum(),1), 4)))
    # next stage reached: passed this doc, OR skipped because ineligible
    reached = ps | (reached & ~elig)
rf = pd.DataFrame(rows)
print(rf)
print('\ncleared-all after INSURANCE:', int(reached.sum()))

print('\n=== Drop type for dropped_in_docs (mature) ===')
dr = MAT[MAT.final_status == 'dropped_in_docs']
print('last_event_type of droppers:')
print(dr.last_event_type.value_counts(dropna=False))
print('\nDrop taxonomy:')
dr = dr.copy()
attempts_at_last = doc.groupby(['captain_id', 'doc_type']).attempt_no.max()
dr['max_attempt_last_doc'] = [attempts_at_last.get((c, d), 0)
                              for c, d in zip(dr.captain_id, dr.last_event_doc)]
def taxo(r):
    if r.n_uploads == 0:
        return 'A. never uploaded anything'
    if r.last_event_type == 'verification_fail' and r.max_attempt_last_doc >= 3:
        return 'B. locked out (3 attempts exhausted)'
    if r.last_event_type == 'verification_fail':
        return 'C. abandoned after a failure (attempts left)'
    if r.last_event_type == 'verification_pass':
        return 'D. abandoned after passing (never started next doc)'
    return 'E. uploaded, no verdict recorded'
dr['drop_type'] = dr.apply(taxo, axis=1)
print(dr.drop_type.value_counts())
print((dr.drop_type.value_counts(normalize=True)*100).round(2))
print('\ndrop_type x last_event_doc:')
print(pd.crosstab(dr.drop_type, dr.last_event_doc))

print('\n=== Failure reasons ===')
fl = doc[doc.event_type == 'verification_fail']
print(pd.crosstab(fl.doc_type, fl.failure_reason, margins=True))
print('\nfailure reason share by device_tier:')
fl2 = fl.merge(cap[['captain_id', 'device_tier', 'city', 'acquisition_channel', 'app_language']], on='captain_id')
print(pd.crosstab(fl2.device_tier, fl2.failure_reason, normalize='index').round(3))
print('\nfail rate per upload by doc x attempt:')
tot_up = doc[doc.event_type=='upload_success'].groupby(['doc_type','attempt_no']).size()
tot_fail = fl.groupby(['doc_type','attempt_no']).size()
print((tot_fail/tot_up).unstack().round(3))
print('\noverall fail rate per upload by device_tier:')
du = doc.merge(cap[['captain_id','device_tier']], on='captain_id')
print(du[du.event_type!='upload_success'].assign(fail=lambda d: d.event_type=='verification_fail')
      .groupby('device_tier').fail.agg(['mean','count']).round(4))
