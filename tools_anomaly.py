import pandas as pd
import numpy as np
import json
from langchain_core.tools import tool


@tool
def detect_anomalies(historical_data: str, sensitivity: float = 2.0) -> dict:
    """Detect unusual revenue patterns using statistical methods.

    Args:
        historical_data: JSON string of list with 'date' and 'revenue'
        sensitivity: Standard deviation threshold (default 2.0 = 95% confidence)

    Returns:
        dict with detected anomalies, their severity, and explanations
    """

    try:
        data = json.loads(historical_data)
        df = pd.DataFrame(data)
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')

        # Calculate mean and std dev
        mean_revenue = df['revenue'].mean()
        std_revenue = df['revenue'].std()

        # Identify outliers (Z-score method)
        df['z_score'] = np.abs((df['revenue'] - mean_revenue) / std_revenue)
        anomalies = df[df['z_score'] > sensitivity]

        # Detect trends (slope)
        if len(df) > 1:
            x = np.arange(len(df))
            slope = np.polyfit(x, df['revenue'].values, 1)[0]
            trend = "increasing" if slope > 0 else "decreasing"
        else:
            trend = "unknown"

        return {
            "status": "success",
            "mean_revenue": round(mean_revenue, 2),
            "std_dev": round(std_revenue, 2),
            "anomalies_detected": len(anomalies),
            "anomaly_details": anomalies[['date', 'revenue', 'z_score']].to_dict('records'),
            "trend": trend,
            "severity": "high" if len(anomalies) > 2 else "low" if len(anomalies) == 0 else "medium"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Anomaly detection failed: {str(e)}"
        }
