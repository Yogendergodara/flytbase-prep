"""#25 (partial): alerts leaving the process.

Deliberately NOT a message queue, a dashboard backend, or an auth system -
building those is a separate, much bigger project this repo does not take on.
This is the minimal honest version of "an alert reaches something outside
this CLI": a webhook POST (retried) and/or a local SQLite incident store
(persistent, queryable, no server to run). Both are optional and off by
default; enabling neither costs nothing.
"""
import json
import sqlite3
import time


def send_webhook(url, alert, timeout=3, retries=2):
    if not url:
        return
    import urllib.request
    body = json.dumps(alert).encode("utf-8")
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"},
                method="POST")
            urllib.request.urlopen(req, timeout=timeout)
            return
        except Exception as e:
            if attempt == retries:
                print(f"[alert_sink] webhook failed after {retries + 1} tries: {e}")
            else:
                time.sleep(0.5 * (attempt + 1))


def init_db(path):
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE IF NOT EXISTS incidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at REAL, kind TEXT, track_id INTEGER, cls INTEGER,
        t_start REAL, t_end REAL, score REAL, why TEXT, facts TEXT,
        acknowledged INTEGER DEFAULT 0)""")
    conn.commit()
    return conn


def store_incident(conn, alert):
    conn.execute(
        "INSERT INTO incidents (created_at,kind,track_id,cls,t_start,t_end,"
        "score,why,facts) VALUES (?,?,?,?,?,?,?,?,?)",
        (time.time(), alert["kind"], alert["track_id"], alert["cls"],
         alert["t_start"], alert["t_end"], alert["score"], alert.get("why"),
         json.dumps(alert.get("facts", {}))))
    conn.commit()


def make_sink(cfg):
    """Returns an on_alert(alert) callable, or None if nothing is enabled."""
    t = cfg.get("alerts", {}).get("transport", {}) or {}
    webhook_url = t.get("webhook_url")
    db_path = t.get("sqlite_path")
    if not webhook_url and not db_path:
        return None

    conn = init_db(db_path) if db_path else None

    def on_alert(alert):
        if webhook_url:
            send_webhook(webhook_url, alert)
        if conn:
            store_incident(conn, alert)

    return on_alert
