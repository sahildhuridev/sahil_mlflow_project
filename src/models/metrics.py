import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ModelMetrics:
    @staticmethod
    def calculate_metrics(y_true, y_pred) -> dict:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        # MAPE: avoid division by zero
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
        r2 = r2_score(y_true, y_pred)
        
        # Financial Metrics
        # Directional Accuracy: Did it predict the sign of the return correctly?
        actual_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)
        
        if len(actual_diff) > 0:
            directional_accuracy = np.mean(np.sign(actual_diff) == np.sign(pred_diff)) * 100
        else:
            directional_accuracy = 0.0

        prediction_bias = np.mean(y_pred - y_true)
        
        # KE Ratio (Kelly equivalent proxy or simple win/loss ratio of up/down prediction)
        # Assuming simple proxy: positive directional accuracy ratio
        ke_ratio = directional_accuracy / (100 - directional_accuracy + 1e-9)

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "r2": float(r2),
            "directional_accuracy": float(directional_accuracy),
            "prediction_bias": float(prediction_bias),
            "ke_ratio": float(ke_ratio)
        }

    @staticmethod
    def get_best_model(stats: dict) -> dict:
        """Determines the best model based on the lowest RMSE."""
        eligible_models = {}
        for name, m in stats.items():
            if name != 'ensemble' and 'rmse' in m:
                eligible_models[name] = m['rmse']
        
        if not eligible_models:
            return None
            
        best_name = min(eligible_models, key=eligible_models.get)
        model_display_names = {
            "lr": "Linear Regression",
            "arima": "ARIMA Model",
            "lstm": "LSTM Neural Network"
        }
        
        return {
            "name": model_display_names.get(best_name, best_name),
            "rmse": eligible_models[best_name],
            "reason": "Lowest RMSE"
        }

    @staticmethod
    def calculate_stability(predictions: list) -> str:
        """Calculates prediction stability based on recent variance."""
        if len(predictions) < 5:
            return "Stable" # Default if not enough data
            
        recent = np.array(predictions[-10:])
        # Use coefficient of variation as a simple stability proxy
        variation = np.std(recent) / (np.mean(recent) + 1e-9)
        
        if variation < 0.005:
            return "Stable"
        elif variation < 0.015:
            return "Moderate"
        else:
            return "Volatile"

    @staticmethod
    def calculate_roi(y_true, y_pred, initial_capital=10000.0) -> dict:
        """
        Simulated trading evaluation. 
        Strategy: Buy if prediction > current_price, sell otherwise.
        """
        capital = initial_capital
        position = 0
        portfolio_values = []
        
        for i in range(1, len(y_true)):
            current_price = y_true[i-1]
            next_actual = y_true[i]
            predicted_next = y_pred[i]
            
            # Simple strategy
            if predicted_next > current_price and capital > 0:
                # Buy
                position = capital / current_price
                capital = 0
            elif predicted_next <= current_price and position > 0:
                # Sell
                capital = position * next_actual
                position = 0
                
            # Record value
            current_value = capital + (position * next_actual)
            portfolio_values.append(current_value)

        final_value = capital + (position * y_true[-1]) if len(y_true) > 0 else initial_capital
        roi_percentage = ((final_value - initial_capital) / initial_capital) * 100
        
        portfolio_values = np.array(portfolio_values)
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(365*24) # Annualized hourly
            
            # Max Drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - peak) / peak
            max_drawdown = np.min(drawdown) * 100
        else:
            sharpe_ratio = 0
            max_drawdown = 0

        profit = final_value - initial_capital

        return {
            "roi_percentage": float(roi_percentage),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "cumulative_profit": float(profit)
        }
