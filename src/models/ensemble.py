import numpy as np

class EnsembleModel:
    def __init__(self, weights=None):
        """
        weights: dict containing weights for each model.
        e.g., {'lr': 0.2, 'arima': 0.2, 'lstm': 0.6}
        """
        if weights is None:
            self.weights = {'lr': 0.33, 'arima': 0.33, 'lstm': 0.34}
        else:
            self.weights = weights

    def predict(self, lr_pred: float, arima_pred: float, lstm_pred: float) -> float:
        """Combine predictions based on fixed weights for stability."""
        # LSTM and LR are more stable (40% each), ARIMA is more volatile (20%)
        ensemble_pred = (
            lr_pred * 0.4 +
            lstm_pred * 0.4 +
            arima_pred * 0.2
        )
        return ensemble_pred

    def update_weights(self, metrics_dict: dict):
        """
        Update weights inversely proportional to their validation MSE 
        (lower error = higher weight).
        metrics_dict: {'lr': mse_lr, 'arima': mse_arima, 'lstm': mse_lstm}
        """
        inv_errors = {k: 1.0 / (v + 1e-9) for k, v in metrics_dict.items()}
        total_inv_error = sum(inv_errors.values())
        
        for k in self.weights.keys():
            if k in inv_errors:
                self.weights[k] = inv_errors[k] / total_inv_error
        
        return self.weights
