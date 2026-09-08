"""Quick structural peek at all 7 raw files. Exploration only, not part of the pipeline."""
import pandas as pd
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 50)

FILES = ['captains', 'doc_events', 'approvals', 'activation', 'nudges',
         'airport_hourly', 'airport_trips']

for f in FILES:
    df = pd.read_csv(f'../{f}.csv')
    print('=' * 100)
    print(f'{f}.csv   shape={df.shape}')
    print('-- dtypes / nulls / nunique --')
    info = pd.DataFrame({
        'dtype': df.dtypes.astype(str),
        'nulls': df.isna().sum(),
        'null_pct': (df.isna().mean() * 100).round(2),
        'nunique': df.nunique(),
    })
    print(info)
    print('-- head --')
    print(df.head(4))
    print('-- dup rows:', df.duplicated().sum())
