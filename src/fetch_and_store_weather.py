from db import WeatherDB
from utils.api import get_historical_weather_data

# Paramètres de récupération
latitude = 48.85
longitude = 2.35
start_date = "2024-01-01"
end_date = "2025-04-04"
variables = ["temperature_2m", "relativehumidity_2m"]

# Récupération depuis l’API
print("Récupération des données météo...")
df = get_historical_weather_data(latitude, longitude, start_date, end_date, variables)

# Sauvegarde dans la base
if df is not None and not df.empty:
    db = WeatherDB()
    if db.save_weather_data(df):
        print(f" {len(df)} lignes enregistrées dans la base.")
    else:
        print("Problème lors de l’insertion.")
else:
    print("Aucune donnée météo récupérée.")
