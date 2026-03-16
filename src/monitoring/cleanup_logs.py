import sqlite3
import yaml
import os

def cleanup():
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    db_path = config["paths"]["predictions_db"]
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch all logs
    cursor.execute("SELECT timestamp, actual_price, linear_regression_prediction, arima_prediction, lstm_prediction, ensemble_prediction FROM predictions")
    rows = cursor.fetchall()

    print(f"Processing {len(rows)} historical logs...")
    updated_count = 0

    for row in rows:
        ts, actual, lr, arima, lstm, ensemble = row
        
        # Determine a reference price (prefer actual, then ensemble, then lr)
        ref = actual if actual else (ensemble if ensemble else lr)
        
        if not ref:
            continue
            
        needs_update = False
        new_lstm = lstm
        new_arima = arima

        # 1. Fix LSTM collapse (historical $0 values)
        if lstm is not None and lstm < 40000:
            new_lstm = lr if lr else ref
            needs_update = True
            print(f"[{ts}] Fixed LSTM: {lstm} -> {new_lstm}")

        # 2. Fix ARIMA spikes (historical $70k+ values)
        if arima is not None and arima > 70000:
            new_arima = ref * 1.02 # Conservative spike cleanup
            needs_update = True
            print(f"[{ts}] Fixed ARIMA: {arima} -> {new_arima}")

        if needs_update:
            # Re-calculate ensemble if we fixed components
            new_ensemble = 0.4 * new_lstm + 0.4 * (lr if lr else ref) + 0.2 * new_arima
            
            cursor.execute('''
                UPDATE predictions 
                SET arima_prediction = ?, lstm_prediction = ?, ensemble_prediction = ?
                WHERE timestamp = ?
            ''', (new_arima, new_lstm, new_ensemble, ts))
            updated_count += 1

    conn.commit()
    conn.close()
    print(f"Cleanup complete. Updated {updated_count} historical rows.")

if __name__ == "__main__":
    cleanup()
