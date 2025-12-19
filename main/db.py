import mysql.connector
import uuid
import json
from contextlib import contextmanager

with open("/home/asadel/ASADEL PROJECTS/Fall_Detection/main/config.json", "r") as f:
    config = json.load(f)

DB_CONFIG = config['database']

@contextmanager
def get_connection():
    conn = mysql.connector.connect(
        host= DB_CONFIG["host"],
        user= DB_CONFIG["username"],
        password= DB_CONFIG["password"],
        database= DB_CONFIG["db"],
        port= DB_CONFIG["port"]
    )
    try:
        yield conn
    finally:
        conn.close()

# Fetch active cameras

def get_active_cameras():
    with get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT CameraId, CameraName, RTSPUrl
                FROM Cameras
                WHERE Status = 'true'
                """)
            cameras = cursor.fetchall()
        finally:
            cursor.close()
    return cameras

#Insert fall alert 

def insert_fall_alert(alert_id, camera_id, snapshot_64, clip_path):

    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO Alerts(
                    AlertId,
                    CameraId,
                    Analytics,
                    AlertType,
                    Image1,
                    Image2
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                    (
                    alert_id,
                    camera_id,
                    "Fall Detection",
                    "FALL DETECTED",
                    snapshot_64,
                    clip_path
            ))
            conn.commit()
        finally:
            cursor.close()
        
    return alert_id