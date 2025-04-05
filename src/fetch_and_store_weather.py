import logging
from db import WeatherDB
from utils.api import get_historical_weather_data

logging.basicConfig(
    filename='logs/fetch_and_store_weather.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fetch_and_store_weather(latitude, longitude, start_date, end_date, variables):
    """
    Récupère les données météo depuis une API et les sauvegarde dans la base de données.
    """
    logging.info(f"Début de la récupération des données météo pour {start_date} à {end_date}")
    try:
        # Récupération des données depuis l'API
        df = get_historical_weather_data(latitude, longitude, start_date, end_date, variables)

        if df is not None and not df.empty:
            db = WeatherDB()
            if db.save_weather_data(df):
                logging.info(f"{len(df)} lignes enregistrées dans la base.")
                print(f"{len(df)} lignes enregistrées dans la base.")
            else:
                logging.error("Problème lors de l'insertion des données dans la base.")
                print("Problème lors de l'insertion.")
        else:
            logging.warning("Aucune donnée météo récupérée.")
            print("Aucune donnée météo récupérée.")
    except Exception as e:
        logging.error(f"Erreur lors de la récupération ou de la sauvegarde des données : {e}")
        print(f"Erreur : {e}")

if __name__ == "__main__":
    # Paramètres de récupération
    latitude = 48.85
    longitude = 2.35
    start_date = "2024-01-01"
    end_date = "2025-04-04"
    variables = ["temperature_2m", "relativehumidity_2m"]

    # Exécution de la fonction
    fetch_and_store_weather(latitude, longitude, start_date, end_date, variables)