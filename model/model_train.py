import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.db import WeatherDB
from src.data_processing import transform_to_3h_interval

# === Configuration du log ===
logging.basicConfig(
    filename='logs/model_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_features(df):
    """Crée des variables explicatives (lags, moyennes, cycliques)."""
    logging.info("Création des variables explicatives...")

    # Convertir la colonne time en datetime si ce n'est pas déjà fait
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')

    # Création de variables retardées (lag)
    for lag in range(1, 4):
        df[f'temp_lag_{lag}'] = df['temperature_2m'].shift(lag)
        df[f'humidity_lag_{lag}'] = df['relativehumidity_2m'].shift(lag)

    # Création de variables agrégées (moyennes mobiles)
    for window in [3, 6, 12]:
        df[f'temp_rolling_mean_{window}'] = df['temperature_2m'].rolling(window=window).mean()
        df[f'humidity_rolling_mean_{window}'] = df['relativehumidity_2m'].rolling(window=window).mean()

    # Variables cycliques pour capturer les effets saisonniers
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    # Transformation des variables cycliques en coordonnées circulaires
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)

    # Supprimer les lignes avec des valeurs manquantes
    df = df.dropna()
    return df

def train_model(df, target='temperature_2m'):
    """Entraîne un modèle de régression linéaire avec split temporel."""
    logging.info(f"Début de l'entraînement pour la cible : {target}")
    
    # Créer les features
    data = create_features(df.copy())

    # Définir les variables explicatives
    features = [
        'relativehumidity_2m',
        'temp_lag_1', 'temp_lag_2', 'temp_lag_3',
        'humidity_lag_1', 'humidity_lag_2', 'humidity_lag_3',
        'temp_rolling_mean_3', 'temp_rolling_mean_6', 'temp_rolling_mean_12',
        'humidity_rolling_mean_3', 'humidity_rolling_mean_6', 'humidity_rolling_mean_12',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
    ]

    # Supprimer les colonnes cibles des features si on prédit la température
    if target != 'temperature_2m' and 'relativehumidity_2m' in features:
        features.remove('relativehumidity_2m')
        features.append('temperature_2m')

    split_idx = int(len(data) * 0.8)
    X_train = data[features].iloc[:split_idx]
    y_train = data[target].iloc[:split_idx]
    X_test = data[features].iloc[split_idx:]
    y_test = data[target].iloc[split_idx:]

    # Normaliser les données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entraîner le modèle (LinearRegression comme meilleur modèle d'après l'expérimentation)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Évaluer le modèle
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logging.info(f"Evaluation du modèle: RMSE = {rmse:.4f}, R2 = {r2:.4f}")

    # Créer un dictionnaire avec le modèle et ses métadonnées
    model_info = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'target': target,
        'metrics': {'rmse': rmse, 'r2': r2},
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_data_start': data.index.min().strftime('%Y-%m-%d'),
        'training_data_end': data.index.max().strftime('%Y-%m-%d')
    }

    return model_info

def save_model(model_info, model_path='model/registry'):
    """Sauvegarde le modèle et ses métadonnées."""
    os.makedirs(model_path, exist_ok=True)
    
    version = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{model_path}/{model_info['target']}_model_{version}.joblib"
    
    joblib.dump(model_info, filename)
    logging.info(f"Modèle sauvegardé: {filename}")
    return filename

def load_model(path):
    model = joblib.load(path)
    logging.info(f"Modèle chargé: {path}")
    return model


if __name__ == "__main__":
    
    db = WeatherDB()
    #Charger les données météo
    df = db.get_weather_data()

    if df.empty:
        print("Aucune donnée disponible.")
    else:
        # Transformer les données pour les intervalles de 3 heures
        df_transformed = transform_to_3h_interval(df)
        
        # Entraîner le modèle
        model_info = train_model(df_transformed, target='temperature_2m')
        
        # Sauvegarder le modèle
        path = save_model(model_info)
        
        # Enregistrer les métadonnées du modèle dans la base de données
        model_id = db.save_model_metadata(model_info, path)
        print(f"Modèle enregistré : {path} | ID : {model_id}")
