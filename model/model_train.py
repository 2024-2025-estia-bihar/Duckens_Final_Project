import pandas as pd
import numpy as np
import joblib
import os
import logging
import json
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.db import WeatherDB
from src.data_processing import transform_to_3h_interval

# === Configuration du log ===
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/model_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_features(df):
    """
    Crée des variables explicatives enrichies pour le modèle de prédiction météo.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données météo avec au moins 'time', 'temperature_2m', 'relativehumidity_2m'
        
    Returns:
        pd.DataFrame: DataFrame avec les features créées
    """
    logging.info("Création des variables explicatives avancées...")

    # Vérification des colonnes nécessaires
    required_columns = ['time', 'temperature_2m', 'relativehumidity_2m']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Colonnes manquantes dans les données : {required_columns}")

    # Convertir en datetime si nécessaire
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])

    # Définir time comme index
    df = df.set_index('time')
    
    # Extraction des composantes temporelles
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    
    # Variables cycliques
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # Création de variables retardées (lags)
    for lag in range(1, 7):  # Augmenté à 6 lags
        df[f'temp_lag_{lag}'] = df['temperature_2m'].shift(lag)
        df[f'humidity_lag_{lag}'] = df['relativehumidity_2m'].shift(lag)

    # Création de différences (tendances)
    df['temp_diff_1'] = df['temperature_2m'].diff(1)
    df['temp_diff_3'] = df['temperature_2m'].diff(3)
    df['humidity_diff_1'] = df['relativehumidity_2m'].diff(1)
    df['humidity_diff_3'] = df['relativehumidity_2m'].diff(3)

    # Création de moyennes mobiles
    for window in [3, 6, 12, 24, 48]:  # Ajout de fenêtres plus larges
        df[f'temp_rolling_mean_{window}'] = df['temperature_2m'].rolling(window=window).mean()
        df[f'humidity_rolling_mean_{window}'] = df['relativehumidity_2m'].rolling(window=window).mean()
        # Ajout de l'écart-type mobile (volatilité)
        df[f'temp_rolling_std_{window}'] = df['temperature_2m'].rolling(window=window).std()
        df[f'humidity_rolling_std_{window}'] = df['relativehumidity_2m'].rolling(window=window).std()

    # Variables d'interaction
    df['temp_humidity_interaction'] = df['temperature_2m'] * df['relativehumidity_2m'] / 100
    df['temp_sq'] = df['temperature_2m'] ** 2  # Relation quadratique
    
    # Détection des changements de tendance
    df['temp_trend_change'] = ((df['temp_diff_1'] > 0) & (df['temp_diff_1'].shift(1) < 0)) | ((df['temp_diff_1'] < 0) & (df['temp_diff_1'].shift(1) > 0))
    df['temp_trend_change'] = df['temp_trend_change'].astype(int)

    # Supprimer les lignes avec des valeurs manquantes
    df = df.dropna()

    logging.info(f"Variables explicatives créées avec succès : {df.shape[1]} colonnes, {df.shape[0]} lignes")
    return df


def select_best_features(X_train, y_train, threshold='median'):
    """
    Sélectionne les features les plus importantes.
    
    Args:
        X_train: Features d'entraînement
        y_train: Variable cible
        threshold: Seuil pour la sélection des features
        
    Returns:
        list: Liste des features sélectionnées
    """
    logging.info("Sélection des meilleures features...")
    
    # Utiliser RandomForest pour évaluer l'importance des features
    selector = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    selector.fit(X_train, y_train)
    
    # Obtenir l'importance des features
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': selector.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Sauvegarder le graphique d'importance des features
    plt.figure(figsize=(12, 8))
    plt.barh(importances['feature'].head(20), importances['importance'].head(20))
    plt.xlabel('Importance')
    plt.title('Top 20 Features les plus importantes')
    os.makedirs('logs/plots', exist_ok=True)
    plt.savefig(f'logs/plots/feature_importance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    
    # Sélectionner les features importantes
    select_model = SelectFromModel(selector, threshold=threshold)
    select_model.fit(X_train, y_train)
    
    # Obtenir les indices des features sélectionnées
    selected_features = X_train.columns[select_model.get_support()]
    logging.info(f"Features sélectionnées: {list(selected_features)}")
    
    return list(selected_features)


def optimize_hyperparameters(X_train, y_train, model_type, cv=5):
    """
    Optimise les hyperparamètres du modèle.
    
    Args:
        X_train: Features d'entraînement
        y_train: Variable cible
        model_type: Type de modèle ('linear', 'random_forest', ou 'gradient_boosting')
        cv: Nombre de plis pour la validation croisée
        
    Returns:
        model: Modèle optimisé
    """
    logging.info(f"Optimisation des hyperparamètres pour le modèle {model_type}...")
    
    # Configuration de la validation croisée temporelle
    tscv = TimeSeriesSplit(n_splits=cv)
    
    # Définir les paramètres de recherche selon le type de modèle
    if model_type == 'linear':
        model = LinearRegression()
        param_grid = {'fit_intercept': [True, False]}
    
    elif model_type == 'random_forest':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
    
    else:
        raise ValueError(f"Type de modèle non supporté : {model_type}")
    
    # Recherche des meilleurs hyperparamètres
    grid_search = GridSearchCV(
        model, param_grid, 
        cv=tscv, 
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,  # Utilise tous les processeurs disponibles
        verbose=1
    )
    
    # Exécution de la recherche
    grid_search.fit(X_train, y_train)
    
    logging.info(f"Meilleurs hyperparamètres : {grid_search.best_params_}")
    logging.info(f"Meilleur score : {-grid_search.best_score_:.4f} RMSE")
    
    return grid_search.best_estimator_


def analyze_residuals(y_test, y_pred, target, save_plots=True):
    """
    Analyse les résidus pour évaluer la qualité du modèle.
    
    Args:
        y_test: Valeurs réelles
        y_pred: Prédictions
        target: Nom de la variable cible
        save_plots: Sauvegarder les graphiques
        
    Returns:
        dict: Métriques d'évaluation
    """
    logging.info("Analyse des résidus...")
    
    # Calcul des métriques
    residuals = y_test - y_pred
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs(residuals / y_test)) * 100
    
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"R²: {r2:.4f}")
    logging.info(f"MAE: {mae:.4f}")
    logging.info(f"MAPE: {mape:.4f}%")
    
    # Test de normalité des résidus
    from scipy import stats
    _, p_value = stats.normaltest(residuals)
    logging.info(f"Test de normalité des résidus: p-value = {p_value:.4f}")
    
    # Autocorrélation des résidus
    autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    logging.info(f"Autocorrélation des résidus: {autocorr:.4f}")
    
    if save_plots:
        # Créer le dossier pour les graphiques
        os.makedirs('logs/plots', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Résidus vs valeurs prédites
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Prédictions')
        plt.ylabel('Résidus')
        plt.title('Résidus vs Prédictions')
        plt.savefig(f'logs/plots/residuals_vs_predictions_{target}_{timestamp}.png')
        
        # 2. Distribution des résidus
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='-')
        plt.xlabel('Résidus')
        plt.ylabel('Fréquence')
        plt.title('Distribution des résidus')
        plt.savefig(f'logs/plots/residuals_distribution_{target}_{timestamp}.png')
        
        # 3. Valeurs réelles vs prédites
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        plt.xlabel('Valeurs réelles')
        plt.ylabel('Prédictions')
        plt.title('Valeurs réelles vs Prédictions')
        plt.savefig(f'logs/plots/actual_vs_predicted_{target}_{timestamp}.png')
    
    # Retourner les métriques
    return {
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'mape': mape,
        'normality_p_value': p_value,
        'autocorrelation': autocorr
    }


def train_model(df, target='temperature_2m', features=None, model_type='random_forest', optimize=True):
    """
    Entraîne un modèle de prédiction météo avec optimisation des hyperparamètres.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données météo
        target (str): Variable cible à prédire ('temperature_2m' ou 'relativehumidity_2m')
        features (list): Liste des features à utiliser (si None, toutes sont utilisées)
        model_type (str): Type de modèle ('linear', 'random_forest', ou 'gradient_boosting')
        optimize (bool): Activer l'optimisation des hyperparamètres
        
    Returns:
        dict: Informations sur le modèle entraîné
    """
    logging.info(f"Début de l'entraînement pour la cible : {target} avec modèle {model_type}")
    start_time = datetime.now()

    # Créer les features
    data = create_features(df.copy())

    # Vérification de la cible
    if target not in data.columns:
        raise ValueError(f"La colonne cible '{target}' est manquante dans les données.")

    # Utiliser toutes les features disponibles si aucune n'est spécifiée
    if features is None or len(features) == 0:
        features = [col for col in data.columns if col != target]
        logging.info(f"Utilisation de toutes les features disponibles: {len(features)} features")
    
    # Séparer les données en ensembles d'entraînement et de test (80% - 20%)
    split_idx = int(len(data) * 0.8)
    X_train = data[features].iloc[:split_idx]
    y_train = data[target].iloc[:split_idx]
    X_test = data[features].iloc[split_idx:]
    y_test = data[target].iloc[split_idx:]
    
    logging.info(f"Dimensions du jeu d'entraînement: {X_train.shape}")
    logging.info(f"Dimensions du jeu de test: {X_test.shape}")
    
    # Sélection des meilleures features
    if len(features) > 10:  # Seulement si beaucoup de features
        selected_features = select_best_features(X_train, y_train)
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        features = selected_features
        logging.info(f"Après sélection: {len(features)} features")

    # Normaliser les données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Optimiser les hyperparamètres si demandé
    if optimize:
        model = optimize_hyperparameters(X_train_scaled, y_train, model_type)
    else:
        # Modèle par défaut selon le type
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(n_estimators=200, random_state=42)
        else:
            raise ValueError(f"Type de modèle non supporté : {model_type}")
        
        # Entraîner le modèle
        model.fit(X_train_scaled, y_train)

    # Évaluer sur l'ensemble de test
    y_pred = model.predict(X_test_scaled)
    
    # Analyser les résidus et obtenir les métriques
    metrics = analyze_residuals(y_test, y_pred, target)
    
    # Temps d'exécution
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logging.info(f"Durée d'entraînement: {duration:.2f} secondes")
    
    # Entraîner sur l'ensemble complet pour le modèle final
    X_full = data[features]
    y_full = data[target]
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X_full)
    
    if optimize:
        # Si les hyperparamètres ont été optimisés, créer un nouveau modèle avec les mêmes paramètres
        params = model.get_params()
        if model_type == 'linear':
            final_model = LinearRegression(**params)
        elif model_type == 'random_forest':
            # Supprimer n_jobs des paramètres s'il existe déjà
            if 'n_jobs' in params:
                del params['n_jobs']
            final_model = RandomForestRegressor(**params, n_jobs=-1)
        elif model_type == 'gradient_boosting':
            final_model = GradientBoostingRegressor(**params)
    
    # Entraîner le modèle final
    final_model.fit(X_full_scaled, y_full)
    
    # Créer un dictionnaire avec le modèle et ses métadonnées
    model_info = {
        'model': final_model,
        'scaler': scaler_full,
        'features': features,
        'target': target,
        'metrics': metrics,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_data_start': data.index.min().strftime('%Y-%m-%d'),
        'training_data_end': data.index.max().strftime('%Y-%m-%d'),
        'hyperparameters': final_model.get_params(),
        'model_type': model_type,
        'data_shape': data.shape,
        'training_duration': duration
    }

    return model_info


def save_model(model_info, model_path='model/registry'):
    """
    Sauvegarde le modèle et ses métadonnées.
    
    Args:
        model_info (dict): Informations sur le modèle
        model_path (str): Chemin pour sauvegarder le modèle
        
    Returns:
        str: Chemin du fichier sauvegardé
    """
    os.makedirs(model_path, exist_ok=True)

    # Création d'un identifiant unique pour le modèle
    version = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_id = f"{model_info['target']}_{model_info['model_type']}_{version}"
    
    # Chemin du fichier
    filename = f"{model_path}/{model_id}.joblib"

    # Sauvegarder le modèle
    joblib.dump(model_info, filename)
    logging.info(f"Modèle sauvegardé: {filename}")
    
    # Créer un fichier de métadonnées JSON pour faciliter la gestion
    meta_filename = f"{model_path}/{model_id}_meta.json"
    metadata = {
        'target': model_info['target'],
        'model_type': model_info['model_type'],
        'features': model_info['features'],
        'metrics': model_info['metrics'],
        'training_date': model_info['training_date'],
        'training_data_start': model_info['training_data_start'],
        'training_data_end': model_info['training_data_end'],
        'hyperparameters': str(model_info['hyperparameters']),
        'file_path': filename
    }
    
    with open(meta_filename, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return filename


def load_model(path):
    """
    Charge un modèle sauvegardé.
    
    Args:
        path (str): Chemin du fichier modèle
        
    Returns:
        dict: Informations sur le modèle chargé
    """
    try:
        model_info = joblib.load(path)
        logging.info(f"Modèle chargé: {path}")
        
        # Vérifier l'intégrité du modèle
        required_keys = ['model', 'scaler', 'features', 'target']
        if not all(key in model_info for key in required_keys):
            logging.warning(f"Le modèle chargé ne contient pas toutes les clés requises: {required_keys}")
        
        return model_info
    except FileNotFoundError:
        logging.error(f"Fichier introuvable : {path}")
        raise
    except Exception as e:
        logging.error(f"Erreur lors du chargement du modèle : {e}")
        raise

if __name__ == "__main__":
    # Création du dossier pour les logs
    os.makedirs('logs', exist_ok=True)
    os.makedirs('logs/plots', exist_ok=True)
    
    db = WeatherDB()

    # Charger les données météo
    logging.info("Récupération des données depuis la base...")
    df = db.get_weather_data()

    if df.empty:
        logging.error("Aucune donnée disponible pour l'entraînement.")
        print("❌ Aucune donnée disponible pour l'entraînement.")
    else:
        # Transformer les données pour les intervalles de 3 heures
        logging.info("Transformation des données en intervalles de 3 heures...")
        df_transformed = transform_to_3h_interval(df)
        
        print(f"🔍 Données chargées : {len(df_transformed)} enregistrements")
        
        # Entraînement d'un seul modèle pour la température
        target = 'temperature_2m'
        model_type = 'random_forest'  # Vous pouvez choisir 'linear' ou 'gradient_boosting' également
        
        print(f"🚀 Entraînement d'un modèle {model_type} pour {target}...")
        
        # Définir des features plus complètes pour la température
        base_features = [
            'relativehumidity_2m',
            'hour_sin', 'hour_cos',
            'day_of_year_sin', 'day_of_year_cos',
            'month_sin', 'month_cos'
        ]
        
        # Entraîner le modèle avec optimisation
        model_info = train_model(
            df_transformed, 
            target=target, 
            features=base_features,
            model_type=model_type,
            optimize=True
        )
        
        # Sauvegarder le modèle
        path = save_model(model_info)
        
        # Enregistrer les métadonnées du modèle dans la base de données
        model_id = db.save_model_metadata(model_info, path)
        
        # Afficher les résultats
        rmse = model_info['metrics']['rmse']
        r2 = model_info['metrics']['r2']
        print(f"✅ Modèle enregistré : {path} | ID : {model_id}")
        print(f"📊 Performances : RMSE = {rmse:.2f}, R² = {r2:.4f}")
        
        print("\n✅ Modèle de température entraîné avec succès!")