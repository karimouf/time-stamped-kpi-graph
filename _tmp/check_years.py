import json
from pathlib import Path

vdir = Path(r'c:\Users\karim\dev\time-stamped-kpi-graph\data\output\trial-27-validation')
years = {}
for f in sorted(vdir.glob('*_kpis.json')):
    d = json.load(open(f))
    stats = d.get('statistics', {})
    yr = d.get('year')
    total = stats.get('total_kpis', 0)
    missing = stats.get('missing_tables', 0)
    vk = d.get('valid_kpis', [])
    good = [v for v in vk if v['kpi'].get('year') is not None and v['kpi'].get('value') is not None]
    if yr not in years:
        years[yr] = {'tables': 0, 'fully': 0, 'good_kpis': 0}
    years[yr]['tables'] += 1
    if missing == 0 and total > 0:
        years[yr]['fully'] += 1
    years[yr]['good_kpis'] += len(good)

for yr in sorted(years):
    info = years[yr]
    print(f"  {yr}: {info['tables']} tables total, {info['fully']} fully validated (missing_tables==0), {info['good_kpis']} valid non-null KPIs")
