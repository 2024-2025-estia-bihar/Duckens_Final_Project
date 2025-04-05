# src/model_training.py
import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(
    filename='logs/model_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_features(df):
    """Crée des variables explicatives à partir des données brutes."""
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
    
    # Supprimer les lignes avec des valeurs NaN (créées par les décalages)
    df = df.dropna()
    
    return df

def train_model(df, target='temperature_2m'):
    """Entraîne le modèle de prédiction météo."""
    logging.info(f"Entraînement du modèle pour la cible: {target}")
    
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
    if target == 'temperature_2m' and 'relativehumidity_2m' in features:
        X = data[features]
        y = data[target]
    else:  # Si on prédit l'humidité, on utilise la température comme feature
        features.remove('relativehumidity_2m')
        features.append('temperature_2m')
        X = data[features]
        y = data[target]
    
    # Normaliser les données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Entraîner le modèle (RandomForest comme meilleur modèle d'après l'expérimentation)
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42
    )
    
    model.fit(X_scaled, y)
    
    # Évaluer le modèle
    y_pred = model.predict(X_scaled)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    logging.info(f"Modèle entraîné avec RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    # Créer un dictionnaire avec le modèle et ses métadonnées
    model_info = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'target': target,
        'metrics': {
            'rmse': rmse,
            'r2': r2
        },
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_data_start': data.index.min().strftime('%Y-%m-%d'),
        'training_data_end': data.index.max().strftime('%Y-%m-%d')
    }
    
    return model_info

def save_model(model_info, model_path='models'):
    """Sauvegarde le modèle et ses métadonnées."""
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    model_version = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{model_path}/{model_info['target']}_model_{model_version}.joblib"
    
    joblib.dump(model_info, filename)
    logging.info(f"Modèle sauvegardé: {filename}")
    
    return filename

def load_model(model_path):
    """Charge un modèle sauvegardé."""
    model_info = joblib.load(model_path)
    logging.info(f"Modèle chargé: {model_path}")
    return model_info

if __name__ == "__main__":
    # Test du pipeline d'entraînement
    data_path = "data/weather_data.csv"
    df = pd.read_csv(data_path)
    
    # Entraîner le modèle
    model_info = train_model(df, target='temperature_2m')
    
    # Sauvegarder le modèle
    save_model(model_info)