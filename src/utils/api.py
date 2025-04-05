import requests
import pandas as pd

def get_historical_weather_data(latitude, longitude, start_date, end_date, variables=None):
    if variables is None:
        variables = ["temperature_2m"]
    if isinstance(variables, str):
        variables = [variables]

    url = "https://archive-api.open-meteo.com/v1/archive"
    hourly_param = ",".join(variables)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly_param,
        "timezone": "auto"
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame({"time": data["hourly"]["time"]})

        for var in variables:
            df[var] = data["hourly"].get(var, [None]*len(df))
            if df[var].isna().sum() > 0:
                df[var] = df[var].interpolate(method="linear")

        df["time"] = pd.to_datetime(df["time"])
        return df
    else:
        print(f"Erreur API : {response.status_code}")
        print(response.text)
        return None
