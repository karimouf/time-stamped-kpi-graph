import json
from pathlib import Path

f = Path(r'c:\Users\karim\dev\time-stamped-kpi-graph\data\output\trial-27-validation\management_report_vw_ar16_page_007_table_00_kpis.json')
d = json.load(open(f))
print("keys:", list(d.keys()))
print("stats:", d.get('statistics'))
ik = d.get('invalid_kpis', [])
print(f"invalid_kpis count: {len(ik)}")
if ik:
    print("sample invalid:", json.dumps(ik[0], indent=2)[:400])
