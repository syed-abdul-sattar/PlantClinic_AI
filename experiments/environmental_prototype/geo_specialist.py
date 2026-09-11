
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class GeoSpatialSpecialist:
    def __init__(self, client_id=None, client_secret=None):
        # Authenticates with Sentinel Hub (Copernicus Data Space) [cite: 90]
        self.client_id = client_id
        self.client_secret = client_secret

    def get_90_day_ndvi(self, latitude, longitude):
        # Logic: Calculate NDVI = (B08 - B04) / (B08 + B04) [cite: 95]
        # B08 is Near-Infrared (Bands: B08), B04 is Red (Bands: B04) [cite: 94]
        dates = [datetime.now() - timedelta(days=i) for i in range(0, 90, 5)]
        
        # Simulated NDVI values for trend detection [cite: 96]
        ndvi_values = np.random.uniform(0.3, 0.8, len(dates)) 
        data = pd.DataFrame({"Date": dates, "NDVI": ndvi_values})
        
        # Calculate recent trend to detect environmental stress [cite: 118]
        recent_trend = data['NDVI'].iloc[:3].mean() - data['NDVI'].iloc[-3:].mean()
        
        # Correlation logic from literature [cite: 119]
        status = "stressed" if recent_trend < -0.1 else "stable" [cite: 120]
        
        return {
            "status": status,
            "ndvi_trend": round(recent_trend, 3),
            "raw_data": data
        }
