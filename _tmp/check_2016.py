import json
from pathlib import Path

f = Path(r'c:\Users\karim\dev\time-stamped-kpi-graph\data\output\trial-27-validation\management_report_vw_ar16_page_007_table_00_kpis.json')
d = json.load(open(f))
vk = d.get('valid_kpis', [])
print(f"valid_kpis count: {len(vk)}")
for entry in vk[:3]:
    kpi = entry.get('kpi', {})
    print(f"  year={kpi.get('year')!r}  value={kpi.get('value')!r}  name={kpi.get('name')!r}")
