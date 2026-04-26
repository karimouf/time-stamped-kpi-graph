"""
Fix page numbers for VW2015 divisions tables in pack_context.db.
Applies +1 to all divisions tables from VW2015_T370dd7 onward.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "pack_context.db"

TABLE_IDS = [
    "VW2015_T370dd7",  # Bentley – Production:          14 → 15
    "VW2015_T7b0ff5",  # Bentley – Key Figures:         14 → 15
    "VW2015_T704982",  # Porsche – Production:          16 → 17
    "VW2015_Ta58de3",  # Porsche – Key Figures:         16 → 17
    "VW2015_Tb49406",  # VW Commercial Vehicles – Prod: 18 → 19
    "VW2015_Tb99596",  # VW Commercial Vehicles – KF:   18 → 19
    "VW2015_Tb1224f",  # Scania – Production:           20 → 21
    "VW2015_Te9a16c",  # Scania – Key Figures:          20 → 21
    "VW2015_T9330cf",  # MAN – Production:              22 → 23
    "VW2015_Tdb039f",  # MAN – Key Figures:             22 → 23
    "VW2015_T9327b6",  # VW China – Local Production:   24 → 25
    "VW2015_T737670",  # VW China – Key Figures:        24 → 25
    "VW2015_T71eee3",  # VW China – Earnings:           24 → 25
    "VW2015_Tac1626",  # VW Financial Services – KF:    27 → 28
]

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

print("Before:")
placeholders = ",".join("?" * len(TABLE_IDS))
cur.execute(
    f"SELECT table_id, page, title FROM context_packs WHERE table_id IN ({placeholders}) ORDER BY page",
    TABLE_IDS,
)
for row in cur.fetchall():
    print(f"  {row[0]}  page={row[1]}  {row[2]}")

cur.execute(
    f"UPDATE context_packs SET page = page + 1 WHERE table_id IN ({placeholders})",
    TABLE_IDS,
)
conn.commit()

print(f"\nUpdated {cur.rowcount} rows.")

print("\nAfter:")
cur.execute(
    f"SELECT table_id, page, title FROM context_packs WHERE table_id IN ({placeholders}) ORDER BY page",
    TABLE_IDS,
)
for row in cur.fetchall():
    print(f"  {row[0]}  page={row[1]}  {row[2]}")

conn.close()
