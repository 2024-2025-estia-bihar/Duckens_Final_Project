# src/prediction.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import requests
from io import StringIO

from src.model_training import load_model, create_features
from src.database import WeatherDB

logging.basicConfig(
    filename='logs/prediction.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_weather_data(start_date, end_date=None, location="47.262626,-1.5343945"):
    """
    Télécharge les données météo récentes depuis une API.
    Utilisez l'API de votre choix (exemple avec Open-Meteo).
    """
    logging.info(f"Téléchargement des données météo du {start_date} au {end_date}")
    
    # Format des dates pour l'API
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d") if end_date else datetime.now().strftime("%Y-%m-%d")
    
    # Exemple avec l'API Open-Meteo (gratuite)
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={location.split(',')[0]}&longitude={location.split(',')[1]}&start_date={start_date_str}&end_date={end_date_str}&hourly=temperature_2m,relativehumidity_2m"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Gérer les erreurs HTTP
        
        data = response.json()
        
        # Convertir en DataFrame
        df = pd.DataFrame({
            'time': pd.to_datetime(data['hourly']['time']),
            'temperature_2m': data['hourly']['temperature_2m'],
            'relativehumidity_2m': data['hourly']['relativehumidity_2m']
        })
        
        logging.info(f"Données téléchargées avec succès: {len(df)} enregistrements")
        return df
    
    except Exception as e:
        logging.error(f"Erreur lors du téléchargement des données: {e}")
        return pd.DataFrame()  # Retourner un DataFrame vide en cas d'erreur

def generate_predictions(model_info, data, forecast_horizon=24):
    """
    Génère des prédictions pour une période donnée.
    """
    logging.info(f"Génération des prédictions pour {forecast_horizon} heures")
    
    # Préparer les données avec les features
    df = create_features(data.copy())
    
    # Obtenir les données les plus récentes
    latest_data = df.iloc[-1:]
    
    # Liste pour stocker les prédictions
    predictions = []
    current_time = latest_data.index[0]
    
    # Obtenir les scaler, modèle et features
    scaler = model_info['scaler']
    model = model_info['model']
    features = model_info['features']
    
    # Pour chaque heure à prédire
    for i in range(1, forecast_horizon + 1):
        next_time = current_time + timedelta(hours=i)
        
        # Créer une nouvelle ligne pour la prédiction
        new_row = latest_data.copy()
        new_row.index = [next_time]
        
        # Mettre à jour les variables cycliques pour la nouvelle heure
        new_row['hour'] = next_time.hour
        new_row['day_of_week'] = next_time.dayofweek
        new_row['month'] = next_time.month
        
        new_row['hour_sin'] = np.sin(2 * np.pi * new_row['hour']/24)
        new_row['hour_cos'] = np.cos(2 * np.pi * new_row['hour']/24)
        new_row['day_sin'] = np.sin(2 * np.pi * new_row['day_of_week']/7)
        new_row['day_cos'] = np.cos(2 * np.pi * new_row['day_of_week']/7)
        new_row['month_sin'] = np.sin(2 * np.pi * new_row['month']/12)
        new_row['month_cos'] = np.cos(2 * np.pi * new_row['month']/12)
        
        # Prédire avec le modèle
        X = new_row[features]
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        
        # Stocker la prédiction
        predictions.append({
            'time': next_time,
            'prediction': prediction
        })
        
        # Mettre à jour les données pour la prochaine itération
        # Si on prédit la température, on l'utilise pour la prochaine prédiction
        if model_info['target'] == 'temperature_2m':
            new_row['temperature_2m'] = prediction
        else:
            new_row['relativehumidity_2m'] = prediction
        
        # Mettre à jour les variables retardées et moyennes mobiles
        # (Simplifié pour l'exemple)
        latest_data = new_row
    
    # Convertir en DataFrame
    predictions_df = pd.DataFrame(predictions)
    
    logging.info(f"Prédictions générées: {len(predictions_df)} enregistrements")
    return predictions_df

def run_prediction_pipeline(forecast_horizon=24):
    """
    Execute le pipeline complet de prédiction:
    1. Télécharger les données récentes
    2. Charger le modèle le plus récent
    3. Générer les prédictions
    4. Stocker les prédictions dans la base de données
    """
    logging.info("Démarrage du pipeline de prédiction")
    db = WeatherDB()
    
    # 1. Télécharger les données récentes (30 jours)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    new_data = download_weather_data(start_date, end_date)
    if not new_data.empty:
        # Sauvegarder les nouvelles données
        db.save_weather_data(new_data)
    
    # 2. Obtenir toutes les données nécessaires pour les prédictions
    data = db.get_weather_data(start_date=(end_date - timedelta(days=45)).strftime('%Y-%m-%d'))
    
    # 3. Charger le modèle le plus récent pour la température
    temp_model_info = db.get_latest_model(target='temperature_2m')
    if temp_model_info:
        temp_model = load_model(temp_model_info['model_path'])
        
        # 4. Générer les prédictions de température
        temp_predictions = generate_predictions(temp_model, data, forecast_horizon)
        
        # 5. Sauvegarder les prédictions
        db.save_predictions(temp_model_info['id'], temp_predictions)
    
    # 6. Répéter pour le modèle d'humidité si nécessaire
    humidity_model_info = db.get_latest_model(target='relativehumidity_2m')
    if humidity_model_info:
        humidity_model = load_model(humidity_model_info['model_path'])
        humidity_predictions = generate_predictions(humidity_model, data, forecast_horizon)
        db.save_predictions(humidity_model_info['id'], humidity_predictions)
    
    logging.info("Pipeline de prédiction terminé")
    return True

if __name__ == "__main__":
    # Exécuter le pipeline de prédiction
    run_prediction_pipeline(forecast_horizon=48)  # Prédire pour les 48 prochaines heures