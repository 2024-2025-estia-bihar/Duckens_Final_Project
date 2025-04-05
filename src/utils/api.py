import requests
import pandas as pd
import logging

# Configuration des logs
logging.basicConfig(
    filename='logs/api.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_historical_weather_data(latitude, longitude, start_date, end_date, variables=None):
    """
    Récupère les données météo historiques depuis l'API Open-Meteo.

    Args:
        latitude (float): Latitude de la localisation.
        longitude (float): Longitude de la localisation.
        start_date (str): Date de début au format 'YYYY-MM-DD'.
        end_date (str): Date de fin au format 'YYYY-MM-DD'.
        variables (list or str): Liste des variables météo à récupérer (par défaut : ["temperature_2m"]).

    Returns:
        pd.DataFrame: DataFrame contenant les données météo ou None en cas d'erreur.
    """
    logging.info(f"Récupération des données météo pour {latitude}, {longitude} du {start_date} au {end_date}")

    # Validation des variables
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

    try:
        # Appel API
        response = requests.get(url, params=params)
        response.raise_for_status()  # Lève une exception pour les erreurs HTTP

        data = response.json()

        # Vérification des données reçues
        if "hourly" not in data or "time" not in data["hourly"]:
            logging.error("Les données reçues ne contiennent pas les informations horaires attendues.")
            return None

        # Création du DataFrame
        df = pd.DataFrame({"time": data["hourly"]["time"]})
        for var in variables:
            if var in data["hourly"]:
                df[var] = data["hourly"][var]
            else:
                logging.warning(f"La variable '{var}' est absente des données reçues.")
                df[var] = None

        # Interpolation des valeurs manquantes
        for var in variables:
            if df[var].isna().sum() > 0:
                logging.info(f"Interpolation des valeurs manquantes pour la variable '{var}'.")
                df[var] = df[var].interpolate(method="linear")

        # Conversion de la colonne 'time' en datetime
        df["time"] = pd.to_datetime(df["time"])

        logging.info(f"Données récupérées avec succès : {len(df)} lignes.")
        return df

    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur lors de l'appel API : {e}")
        return None
    except Exception as e:
        logging.error(f"Erreur inattendue : {e}")
        return None