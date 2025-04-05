from fastapi import FastAPI, Query, HTTPException, Depends, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import os
import sys
import logging
from datetime import datetime, timedelta
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.db import WeatherDB
from src.prediction import run_prediction_pipeline

# Configuration des logs
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/api.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Modèles de données Pydantic
class WeatherData(BaseModel):
    time: str
    temperature_2m: float
    relativehumidity_2m: float

class PredictionResult(BaseModel):
    id: int
    target: str
    target_date: str
    predicted_value: float

class ModelInfo(BaseModel):
    id: int
    model_path: str
    target: str
    training_date: str
    metrics: Dict[str, Any]
    features: List[str]

class ApiResponse(BaseModel):
    status: str = "success"
    message: str
    data: Optional[Dict[str, Any]] = None
    count: Optional[int] = None

# Application FastAPI avec documentation enrichie
app = FastAPI(
    title="Weather Prediction API",
    description="""
    API de prédiction météorologique développée dans le cadre du projet de fin d'études.
    
    Cette API permet de :
    - Consulter les données météorologiques historiques
    - Gérer les modèles de prédiction
    - Générer des prédictions météorologiques
    - Visualiser les performances des modèles
    
    Développé par: [Jean Duckens SANNON]
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Ajouter la prise en charge CORS pour faciliter l'intégration avec des frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Dans un environnement de production, spécifiez les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fonction de dépendance pour obtenir une instance de la base de données
def get_db():
    db = WeatherDB()
    try:
        yield db
    finally:
        pass  # La connexion est fermée automatiquement grâce au gestionnaire de contexte

@app.get("/", response_model=ApiResponse, tags=["Général"])
def root():
    """
    Point d'entrée principal de l'API.
    Fournit des informations générales sur l'API et ses fonctionnalités.
    """
    return {
        "status": "success",
        "message": "Bienvenue sur l'API de prédiction météorologique",
        "data": {
            "endpoints": {
                "/weather/historical": "Données météorologiques historiques",
                "/weather/stats": "Statistiques météorologiques",
                "/models": "Modèles de prédiction",
                "/predictions/run": "Exécution des prédictions",
                "/predictions": "Récupération des prédictions"
            },
            "documentation": "/docs",
            "version": "1.0.0"
        }
    }

@app.get("/weather/historical/", response_model=ApiResponse, tags=["Données Météo"])
def get_historical_data(
    start_date: Optional[str] = Query(None, description="Date de début (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Date de fin (YYYY-MM-DD)"),
    limit: int = Query(100, description="Nombre maximum de résultats", ge=1, le=1000),
    db: WeatherDB = Depends(get_db)
):
    """
    Récupère les données météorologiques historiques depuis la base de données.
    
    Les données peuvent être filtrées par période et limitées en nombre.
    """
    try:
        data = db.get_weather_data(start_date=start_date, end_date=end_date, limit=limit)
        if data.empty:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": "Aucune donnée météo trouvée."}
            )
        
        # Convertir les timestamps en format ISO standard
        data['time'] = data['time'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        
        return {
            "status": "success",
            "message": "Données météorologiques récupérées avec succès",
            "data": {"historical_data": data.to_dict(orient="records")},
            "count": len(data)
        }
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des données météo : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")

@app.get("/weather/stats/", response_model=ApiResponse, tags=["Données Météo"])
def get_weather_stats(
    start_date: Optional[str] = Query(None, description="Date de début (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Date de fin (YYYY-MM-DD)"),
    db: WeatherDB = Depends(get_db)
):
    """
    Calcule et renvoie des statistiques descriptives sur les données météorologiques.
    
    Les statistiques incluent minimum, maximum, moyenne, écart-type et quartiles.
    """
    try:
        data = db.get_weather_data(start_date=start_date, end_date=end_date)
        if data.empty:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": "Aucune donnée météo trouvée pour calculer les statistiques."}
            )
        
        # Calculer les statistiques
        stats = {
            "temperature_2m": {
                "min": float(data['temperature_2m'].min()),
                "max": float(data['temperature_2m'].max()),
                "mean": float(data['temperature_2m'].mean()),
                "median": float(data['temperature_2m'].median()),
                "std": float(data['temperature_2m'].std())
            },
            "relativehumidity_2m": {
                "min": float(data['relativehumidity_2m'].min()),
                "max": float(data['relativehumidity_2m'].max()),
                "mean": float(data['relativehumidity_2m'].mean()),
                "median": float(data['relativehumidity_2m'].median()),
                "std": float(data['relativehumidity_2m'].std())
            },
            "period": {
                "start_date": data['time'].min().strftime('%Y-%m-%d'),
                "end_date": data['time'].max().strftime('%Y-%m-%d'),
                "total_records": len(data)
            }
        }
        
        return {
            "status": "success",
            "message": "Statistiques météorologiques calculées avec succès",
            "data": stats
        }
    except Exception as e:
        logging.error(f"Erreur lors du calcul des statistiques : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")

@app.get("/models/", response_model=ApiResponse, tags=["Modèles"])
def get_models(db: WeatherDB = Depends(get_db)):
    """
    Récupère la liste des modèles de prédiction enregistrés dans la base de données.
    """
    try:
        with db.connect() as conn:
            models_df = pd.read_sql_query(
                """
                SELECT id, model_path, target, training_date, training_start_date, 
                       training_end_date, metrics, features
                FROM models ORDER BY training_date DESC
                """,
                conn
            )
            
        if models_df.empty:
            return {
                "status": "success",
                "message": "Aucun modèle n'a été trouvé",
                "data": {"models": []},
                "count": 0
            }
        
        # Transformer les colonnes JSON
        models_df['metrics'] = models_df['metrics'].apply(json.loads)
        models_df['features'] = models_df['features'].apply(json.loads)
        
        return {
            "status": "success",
            "message": "Modèles récupérés avec succès",
            "data": {"models": models_df.to_dict(orient="records")},
            "count": len(models_df)
        }
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des modèles : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")

@app.get("/models/{model_id}", response_model=ApiResponse, tags=["Modèles"])
def get_model_details(
    model_id: int = Path(..., description="ID du modèle à consulter", ge=1),
    db: WeatherDB = Depends(get_db)
):
    """
    Récupère les détails complets d'un modèle spécifique.
    """
    try:
        with db.connect() as conn:
            model_df = pd.read_sql_query(
                """
                SELECT * FROM models WHERE id = ?
                """,
                conn,
                params=(model_id,)
            )
        
        if model_df.empty:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": f"Modèle avec ID {model_id} non trouvé."}
            )
        
        # Transformer les colonnes JSON
        model_data = model_df.iloc[0].to_dict()
        model_data['metrics'] = json.loads(model_data['metrics'])
        model_data['features'] = json.loads(model_data['features'])
        model_data['hyperparameters'] = json.loads(model_data['hyperparameters'].replace("'", '"')) if isinstance(model_data['hyperparameters'], str) else model_data['hyperparameters']
        
        return {
            "status": "success",
            "message": f"Détails du modèle {model_id} récupérés avec succès",
            "data": {"model": model_data}
        }
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des détails du modèle {model_id}: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")

@app.post("/predictions/run/", response_model=ApiResponse, tags=["Prédictions"])
def run_predictions(
    forecast_horizon: int = Query(24, description="Horizon de prédiction en heures", ge=1, le=168),
    force: bool = Query(False, description="Forcer l'exécution même si des prédictions récentes existent")
):
    """
    Exécute le pipeline de prédiction pour générer des prédictions météorologiques.
    
    Ce processus télécharge les nouvelles données météorologiques,
    effectue des prédictions basées sur le modèle le plus récent,
    et stocke les résultats dans la base de données.
    """
    try:
        if not force:
            # Vérifier si des prédictions récentes existent déjà
            db = WeatherDB()
            recent_predictions = db.get_predictions(limit=1)
            
            if not recent_predictions.empty:
                last_prediction_time = recent_predictions['target_date'].max()
                if last_prediction_time > datetime.now():
                    return {
                        "status": "success",
                        "message": "Des prédictions récentes existent déjà. Utilisez 'force=True' pour regénérer.",
                        "data": {
                            "last_prediction": last_prediction_time.strftime('%Y-%m-%dT%H:%M:%S'),
                            "forecast_available_until": last_prediction_time.strftime('%Y-%m-%dT%H:%M:%S')
                        }
                    }
        
        logging.info(f"Lancement du pipeline de prédiction via l'API avec horizon de {forecast_horizon}h.")
        result = run_prediction_pipeline(forecast_horizon=forecast_horizon)
        
        if result:
            return {
                "status": "success",
                "message": "Prédictions générées avec succès",
                "data": {
                    "forecast_horizon": forecast_horizon,
                    "execution_time": datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                }
            }
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Erreur lors de l'exécution du pipeline de prédiction."
                }
            )
    except Exception as e:
        logging.error(f"Erreur lors de l'exécution du pipeline de prédiction : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur: {str(e)}")

@app.get("/predictions/", response_model=ApiResponse, tags=["Prédictions"])
def get_predictions(
    start_date: Optional[str] = Query(None, description="Date de début des prédictions (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Date de fin des prédictions (YYYY-MM-DD)"),
    target: Optional[str] = Query(None, description="Cible des prédictions (temperature_2m, relativehumidity_2m)"),
    limit: int = Query(100, description="Nombre maximum de résultats", ge=1, le=1000),
    db: WeatherDB = Depends(get_db)
):
    """
    Récupère les prédictions météorologiques stockées dans la base de données.
    
    Les prédictions peuvent être filtrées par période et par cible.
    """
    try:
        # Modifier la fonction get_predictions dans db.py pour supporter le filtrage par target
        # Pour l'instant, gestion côté API
        predictions = db.get_predictions(start_date=start_date, end_date=end_date, limit=limit)
        
        if predictions.empty:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": "Aucune prédiction trouvée."}
            )
            
        # Filtrer par target si spécifié
        if target:
            predictions = predictions[predictions['target'] == target]
            if predictions.empty:
                return JSONResponse(
                    status_code=404,
                    content={"status": "error", "message": f"Aucune prédiction trouvée pour la cible {target}."}
                )
        
        # Convertir les timestamps en format ISO standard
        predictions['target_date'] = predictions['target_date'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        
        # Organiser les prédictions par date croissante pour faciliter l'affichage
        predictions = predictions.sort_values('target_date')
        
        return {
            "status": "success",
            "message": "Prédictions récupérées avec succès",
            "data": {"predictions": predictions.to_dict(orient="records")},
            "count": len(predictions)
        }
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des prédictions : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")

@app.get("/predictions/latest/", response_model=ApiResponse, tags=["Prédictions"])
def get_latest_predictions(
    hours: int = Query(24, description="Nombre d'heures de prédiction à retourner", ge=1, le=168),
    target: Optional[str] = Query(None, description="Cible des prédictions (temperature_2m, relativehumidity_2m)"),
    db: WeatherDB = Depends(get_db)
):
    """
    Récupère les prédictions les plus récentes pour les prochaines heures.
    
    Endpoint pratique pour les applications d'affichage de la météo.
    """
    try:
        # Obtenir les prédictions récentes
        now = datetime.now()
        start_date = now.strftime('%Y-%m-%d')
        end_date = (now + timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')
        
        predictions = db.get_predictions(start_date=start_date, end_date=end_date, limit=hours*2)  # *2 pour gérer les deux cibles
        
        if predictions.empty:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": "Aucune prédiction récente n'est disponible."}
            )
            
        # Filtrer par target si spécifié
        if target:
            predictions = predictions[predictions['target'] == target]
            if predictions.empty:
                return JSONResponse(
                    status_code=404, 
                    content={"status": "error", "message": f"Aucune prédiction récente pour {target} n'est disponible."}
                )
        
        # Filtrer uniquement les prédictions futures
        predictions = predictions[predictions['target_date'] >= now.strftime('%Y-%m-%d %H:%M:%S')]
        if predictions.empty:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": "Aucune prédiction future n'est disponible."}
            )
            
        # Convertir les timestamps en format ISO standard
        predictions['target_date'] = predictions['target_date'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        
        # Organiser par date
        predictions = predictions.sort_values('target_date')
        
        # Restructurer les données pour un affichage plus pratique
        restructured_data = []
        for target_date, group in predictions.groupby('target_date'):
            entry = {"date": target_date}
            for _, row in group.iterrows():
                entry[row['target']] = float(row['predicted_value'])
            restructured_data.append(entry)
        
        return {
            "status": "success",
            "message": "Prédictions récentes récupérées avec succès",
            "data": {"forecast": restructured_data},
            "count": len(restructured_data)
        }
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des prédictions récentes : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")

@app.get("/health", response_model=ApiResponse, tags=["Système"])
def health_check(db: WeatherDB = Depends(get_db)):
    """
    Vérifie l'état de santé de l'API et de ses dépendances.
    """
    health = {
        "api": "ok",
        "database": "ok",
        "models": None,
        "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    }
    
    # Vérifier la base de données
    try:
        with db.connect() as conn:
            conn.execute("SELECT 1")
    except Exception as e:
        health["database"] = "error"
        health["database_error"] = str(e)
    
    # Vérifier les modèles
    try:
        temp_model = db.get_latest_model(target='temperature_2m')
        health["models"] = {
            "temperature_model": "available" if temp_model else "missing"
        }
    except Exception as e:
        health["models"] = "error"
        health["models_error"] = str(e)
    
    return {
        "status": "success",
        "message": "Contrôle de santé effectué",
        "data": health
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)