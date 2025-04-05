import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import requests
from io import StringIO
import os
import time
import json
import os.path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_train import load_model, create_features
from src.db import WeatherDB
from src.data_processing import transform_to_3h_interval
from sklearn.preprocessing import StandardScaler

# Création du dossier logs s'il n'existe pas
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    filename='logs/prediction.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_weather_data(start_date, end_date=None, location="47.262626,-1.5343945", max_retries=3):
    """Version améliorée avec retry, validation et gestion d'erreurs"""
    logging.info(f"Téléchargement des données météo du {start_date} au {end_date}")
    
    # Format des dates
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d") if end_date else datetime.now().strftime("%Y-%m-%d")
    
    # URL de l'API
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={location.split(',')[0]}&longitude={location.split(',')[1]}&start_date={start_date_str}&end_date={end_date_str}&hourly=temperature_2m,relativehumidity_2m"
    
    # Implémentation du retry
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)  # Timeout augmenté pour les connexions lentes
            response.raise_for_status()
            
            data = response.json()
            
            # Valider la structure des données
            if 'hourly' not in data or 'time' not in data['hourly']:
                logging.error(f"Structure de données invalide reçue de l'API: {data.keys()}")
                time.sleep(2)
                continue
                
            # Créer le DataFrame
            df = pd.DataFrame({
                'time': pd.to_datetime(data['hourly']['time']),
                'temperature_2m': data['hourly']['temperature_2m'],
                'relativehumidity_2m': data['hourly']['relativehumidity_2m']
            })
            
            # Valider les données
            if df.empty:
                logging.warning("L'API a retourné un DataFrame vide")
                time.sleep(2)
                continue
                
            # Vérifier les valeurs manquantes et les remplacer par interpolation
            missing_count = df.isna().sum().sum()
            if missing_count > 0:
                logging.warning(f"Remplacement de {missing_count} valeurs manquantes par interpolation")
                df = df.interpolate(method='linear')
                
            logging.info(f"Données téléchargées avec succès: {len(df)} enregistrements")
            return df
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Erreur de requête (tentative {attempt+1}/{max_retries}): {e}")
            time.sleep(3)  # Attendre plus longtemps avant de réessayer
        except json.JSONDecodeError:
            logging.error(f"Erreur de décodage JSON (tentative {attempt+1}/{max_retries})")
            time.sleep(3)
        except Exception as e:
            logging.error(f"Erreur inattendue (tentative {attempt+1}/{max_retries}): {e}")
            time.sleep(3)
    
    # Si toutes les tentatives échouent
    logging.error(f"Échec du téléchargement des données après {max_retries} tentatives")
    return pd.DataFrame()


def generate_predictions(model_info, data, forecast_horizon=24):
    """Version corrigée pour générer des prédictions diversifiées"""
    logging.info(f"Génération des prédictions pour {forecast_horizon} heures")
    
    # Extraire les informations du modèle
    model = model_info['model']
    scaler = model_info['scaler']
    features = model_info['features']
    target = model_info['target']
    
    # Utiliser une copie de l'historique avec seulement les colonnes essentielles
    history = data[['time', 'temperature_2m', 'relativehumidity_2m']].copy()
    history['time'] = pd.to_datetime(history['time'])
    history = history.sort_values('time')
    
    # Tableau pour collecter les prédictions
    predictions = []
    
    # Journaliser les premières valeurs pour débogage
    logging.info(f"Valeurs initiales : temp={history['temperature_2m'].iloc[-1]:.2f}, humidity={history['relativehumidity_2m'].iloc[-1]:.2f}")
    
    for i in range(forecast_horizon):
        try:
            # Calculer le prochain timestamp
            next_time = history["time"].iloc[-1] + timedelta(hours=3)
            
            # Convertir l'historique en features
            temp_df = create_features(history.copy())
            
            # Vérifier si toutes les features sont disponibles
            missing_features = [f for f in features if f not in temp_df.columns]
            if missing_features:
                logging.error(f"Features manquantes: {missing_features}")
                break
                
            # Extraire les features pour la prédiction
            X = temp_df[features].iloc[[-1]]
            
            # Normaliser
            X_scaled = scaler.transform(X)
            
            # Prédire
            prediction = float(model.predict(X_scaled)[0])
            
            # Borner la prédiction
            if target == 'temperature_2m':
                prediction = max(min(prediction, 50), -30)
            elif target == 'relativehumidity_2m':
                prediction = max(min(prediction, 100), 0)
                
            # Journaliser la prédiction pour débogage
            logging.info(f"Prédiction {i+1}: {next_time.strftime('%Y-%m-%d %H:%M')} = {prediction:.2f}")
            
            # Stocker la prédiction et les features
            pred_data = {
                "time": next_time,
                "prediction": prediction
            }
            
            # Inclure les features pour analyse
            for feat in features:
                pred_data[feat] = float(X[feat].iloc[0])
            
            predictions.append(pred_data)
            
            # Important: Mettre à jour l'historique avec la nouvelle prédiction
            new_row = pd.DataFrame({
                "time": [next_time],
                "temperature_2m": [prediction if target == 'temperature_2m' else history['temperature_2m'].iloc[-1]],
                "relativehumidity_2m": [prediction if target == 'relativehumidity_2m' else history['relativehumidity_2m'].iloc[-1]]
            })
            
            # Concatener la nouvelle ligne à l'historique
            history = pd.concat([history, new_row], ignore_index=True)
        
        except Exception as e:
            logging.error(f"Erreur lors de la génération de la prédiction {i+1}: {e}")
            # Continuer avec les prédictions suivantes même si une échoue
            continue
    
    # Vérifier la diversité des prédictions
    predictions_df = pd.DataFrame(predictions)
    
    if len(predictions_df) > 1:
        std = predictions_df['prediction'].std()
        logging.info(f"Écart-type des prédictions: {std:.4f}")
        if std < 0.1:
            logging.warning("Attention: prédictions très similaires!")
    
    return predictions_df


def run_prediction_pipeline(forecast_horizon=24):
    """Pipeline optimisé avec meilleure gestion des erreurs et verrouillage"""
    # Créer un fichier de verrouillage pour éviter les exécutions simultanées
    lock_file = "prediction_pipeline.lock"
    
    if os.path.exists(lock_file):
        lock_time = datetime.fromtimestamp(os.path.getmtime(lock_file))
        if datetime.now() - lock_time < timedelta(hours=1):
            logging.warning("Une autre instance du pipeline est en cours d'exécution")
            print("Pipeline déjà en cours d'exécution")
            return False
    
    # Créer le verrou
    with open(lock_file, 'w') as f:
        f.write(f"Started at {datetime.now()}")
    
    try:
        logging.info("Démarrage du pipeline de prédiction")
        print("📡 Lancement du pipeline de prédiction...")
        
        # Création du répertoire de la base de données si nécessaire
        os.makedirs('data/db', exist_ok=True)
        db = WeatherDB()
        
        # Télécharger uniquement les nouvelles données
        try:
            # Vérifier la date de la dernière donnée en base
            last_data = db.get_weather_data(limit=1)
            
            if not last_data.empty:
                last_date = last_data['time'].max()
                start_date = last_date - timedelta(days=1)  # Chevauchement d'un jour pour cohérence
            else:
                # Si pas de données, télécharger sur 30 jours
                start_date = datetime.now() - timedelta(days=30)
            
            end_date = datetime.now()
            
            print(f"Téléchargement des données météo du {start_date.strftime('%Y-%m-%d')} au {end_date.strftime('%Y-%m-%d')}...")
            new_data = download_weather_data(start_date, end_date)
            
            if not new_data.empty:
                # Filtrer les données déjà existantes pour éviter les doublons
                new_data_saved = db.save_weather_data(new_data)
                print(f"{len(new_data)} nouvelles lignes téléchargées et sauvegardées")
            else:
                print("Aucune nouvelle donnée météo récupérée")
        except Exception as e:
            logging.error(f"Erreur lors du téléchargement des données: {e}")
            print(f"Erreur lors du téléchargement des données: {str(e)}")
            # Continuer avec les données existantes
                
        # Récupérer les données pour la prédiction (45 derniers jours)
        data_start_date = (datetime.now() - timedelta(days=45)).strftime('%Y-%m-%d')
        print(f"🔍 Récupération des données depuis {data_start_date}...")
        data = db.get_weather_data(start_date=data_start_date)
        
        if data.empty:
            logging.error("Aucune donnée disponible pour les prédictions")
            print("Impossible de générer des prédictions : aucune donnée disponible")
            return False
            
        # Transformation en tranches de 3 heures
        try:
            data = transform_to_3h_interval(data)
        except Exception as e:
            logging.error(f"Erreur lors de la transformation des données: {e}")
            print(f"Erreur lors de la transformation des données: {str(e)}")
            return False
        
        # Génération des prédictions de température
        success = False
        
        # Température
        temp_model_info = db.get_latest_model(target='temperature_2m')
        if temp_model_info:
            try:
                temp_model = load_model(temp_model_info['model_path'])
                print(f"Modèle température chargé : {os.path.basename(temp_model_info['model_path'])}")
                
                # Générer et sauvegarder les prédictions
                temp_predictions = generate_predictions(temp_model, data, forecast_horizon)
                
                if not temp_predictions.empty:
                    min_pred = temp_predictions['prediction'].min()
                    max_pred = temp_predictions['prediction'].max()
                    print(f"Diversité des prédictions: Min={min_pred:.2f}, Max={max_pred:.2f}, Plage={max_pred-min_pred:.2f}")
                    if max_pred - min_pred < 0.5:
                        print("ATTENTION: Les prédictions sont presque identiques!")
                    
                    # IMPORTANT: Sauvegarder les prédictions de température
                    db.save_predictions(temp_model_info['id'], temp_predictions)
                    print(f"{len(temp_predictions)} prédictions de température générées et sauvegardées")
                    success = True
                else:
                    print("Échec de génération des prédictions de température")
            except FileNotFoundError:
                logging.error(f"Modèle non trouvé : {temp_model_info['model_path']}")
                print(f"Modèle introuvable : {os.path.basename(temp_model_info['model_path'])}")
            except Exception as e:
                logging.error(f"Erreur lors du chargement du modèle température : {e}")
                print(f"Erreur lors du chargement du modèle : {str(e)}")
        else:
            print("Aucun modèle température trouvé")
        
        # Humidité (si besoin)
        humidity_model_info = db.get_latest_model(target='relativehumidity_2m')
        if humidity_model_info:
            try:
                humidity_model = load_model(humidity_model_info['model_path'])
                humidity_predictions = generate_predictions(humidity_model, data, forecast_horizon)
                
                if not humidity_predictions.empty:
                    db.save_predictions(humidity_model_info['id'], humidity_predictions)
                    print(f" {len(humidity_predictions)} prédictions d'humidité générées et sauvegardées")
                    success = True
            except Exception as e:
                logging.error(f"Erreur lors des prédictions d'humidité : {e}")
                print(f" Erreur lors des prédictions d'humidité: {str(e)}")
        
        if not success:
            logging.warning("Aucune prédiction n'a été sauvegardée")
        
        logging.info("Pipeline de prédiction terminé")
        print("Pipeline de prédiction terminé" if success else " Pipeline terminé avec des erreurs")
        return success
        
    except Exception as e:
        logging.error(f"Erreur non gérée dans le pipeline : {e}")
        print(f"Erreur grave : {str(e)}")
        return False
    finally:
        # Supprimer le verrou à la fin
        if os.path.exists(lock_file):
            os.remove(lock_file)
            
if __name__ == "__main__":
    # Exécuter le pipeline de prédiction
    run_prediction_pipeline(forecast_horizon=48)  # Prédire pour les 48 prochaines heures