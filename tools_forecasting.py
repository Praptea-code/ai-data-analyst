import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from langchain_core.tools import tool
try:
    from prophet import Prophet
except ImportError:
    Prophet = None


@tool
def forecast_revenue(historical_data: str, periods: int = 12) -> dict:
    """Forecast future revenue using time-series analysis.

    Args:
        historical_data: JSON string of list with 'date' and 'revenue'
                       e.g., '[{"date": "2026-01", "revenue": 100000}, ...]'
        periods: Number of months to forecast (default 12)

    Returns:
        dict with forecast values, confidence intervals, and trend
    """
    try:
        # Parse input
        data = json.loads(historical_data)
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])

        # If Prophet available, use it (more accurate)
        if Prophet is not None:
            prophet_df = df.rename(columns={'date': 'ds', 'revenue': 'y'})
            model = Prophet(yearly_seasonality=True, daily_seasonality=False)
            model.fit(prophet_df)

            future = model.make_future_dataframe(periods=periods, freq='MS')
            forecast = model.predict(future)

            # Extract forecast for future periods only
            future_forecast = forecast[forecast['ds'] > df['date'].max()]

            return {
                "status": "success",
                "method": "Prophet",
                "forecast": future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records'),
                "trend": "increasing" if forecast['trend'].iloc[-1] > forecast['trend'].iloc[0] else "decreasing"
            }

        # Fallback: exponential smoothing
        else:
            from scipy.signal import detrend

            values = df['revenue'].values
            trend = np.polyfit(np.arange(len(values)), values, 1)[0]

            # Exponential smoothing for forecast
            alpha = 0.3
            forecast_values = []
            last_value = values[-1]

            for i in range(periods):
                next_val = alpha * last_value + (1 - alpha) * values[-1]
                forecast_values.append({
                    'date': (df['date'].max() + timedelta(days=30*(i+1))).strftime('%Y-%m'),
                    'forecast': round(next_val, 2),
                    'upper': round(next_val * 1.15, 2),
                    'lower': round(next_val * 0.85, 2)
                })
                last_value = next_val

            return {
                "status": "success",
                "method": "Exponential Smoothing (Prophet unavailable)",
                "forecast": forecast_values,
                "trend": "increasing" if trend > 0 else "decreasing"
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Forecasting failed: {str(e)}"
        }
