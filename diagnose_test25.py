"""
Read-only diagnostic for test_25 (lag_valid is None). Changes NO project files.
It repeats test_25's exact inserts with your real, unmodified database.py into a
TEMPORARY database, then reports how many rows were actually stored.

Run from the project root:   python diagnose_test25.py
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import database  # noqa: E402

info = time.get_clock_info("time")
print(f"Python {sys.version.split()[0]} | system clock resolution: {info.resolution * 1000:.3f} ms ({info.implementation})")

with tempfile.TemporaryDirectory() as d:
    db = Path(d) / "diag.db"
    database.init_db(db)
    now = time.time()
    database.log_reading({"soil1_raw": 1500, "temperature_c": 22.0}, db_path=db, recorded_at=now - 3500)
    database.log_reading({"soil1_raw": 1550, "temperature_c": 22.5}, db_path=db, recorded_at=now - 100)
    database.log_reading({"soil1_raw": 1560, "temperature_c": 22.6}, db_path=db, recorded_at=now)
    offsets = list(range(200, 3100, 60))
    for o in offsets:
        database.log_reading({"soil1_raw": 1600, "temperature_c": 23.0}, db_path=db, recorded_at=now - o)
    conn = database.get_connection(db)
    ages = sorted(round(now - r[0]) for r in conn.execute("SELECT recorded_at FROM sensor_readings"))
    dup_ts = conn.execute("SELECT COUNT(*) - COUNT(DISTINCT timestamp) FROM sensor_readings").fetchone()[0]
    conn.close()
    gaps = [b - a for a, b in zip(ages, ages[1:])]
    lag = database.get_lag_reading(3600, 900, current_time=now, db_path=db, require_continuous=True)

expected = 3 + len(offsets)
print(f"Inserts attempted: {expected} | rows actually stored: {len(ages)} | silently dropped: {expected - len(ages)}")
print(f"Largest gap between stored readings: {max(gaps)} s "
      "(420 s is expected - the deliberate gap before the old reading; anything larger means rows were lost)")
print(f"get_lag_reading(require_continuous=True) -> {'None' if lag is None else 'a reading'}")
if len(ages) < expected:
    print("DIAGNOSIS: rows were dropped because log_reading() uses the wall clock (datetime.now()) for the UNIQUE "
          "'timestamp' column; inserts within one clock tick collide and INSERT OR IGNORE discards them.")
else:
    print("DIAGNOSIS: no rows dropped on this run - the failure has a different cause; send me this output.")
