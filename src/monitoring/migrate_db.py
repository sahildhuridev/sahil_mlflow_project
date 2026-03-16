import sqlite3
import yaml
import os

def migrate():
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    db_path = config["paths"]["predictions_db"]
    print(f"Migrating database at: {db_path}")
    
    if not os.path.exists(db_path):
        print("Database does not exist. It will be created with the new schema by the app.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if column already exists
        cursor.execute("PRAGMA table_info(predictions)")
        columns = [i[1] for i in cursor.fetchall()]
        
        if "predicted_time" not in columns:
            print("Adding 'predicted_time' column...")
            cursor.execute("ALTER TABLE predictions ADD COLUMN predicted_time TEXT")
            conn.commit()
            print("Migration successful.")
        else:
            print("Column 'predicted_time' already exists.")
            
    except Exception as e:
        print(f"Migration failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
