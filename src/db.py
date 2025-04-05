
import sqlite3
import pandas as pd
import json
import logging
from datetime import datetime

logging.basicConfig(
    filename='logs/database.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class WeatherDB:
    def __init__(self, db_path='data/db/weather.db'):
        """Initialise la connexion à la base de données."""
        self.db_path = db_path
        self.conn = None
        self.create_tables()
    
    def connect(self):
        """Établit la connexion à la base de données."""
        self.conn = sqlite3.connect(self.db_path)
        return self.conn
    
    def close(self):
        """Ferme la connexion à la base de données."""
        if self.conn:
            self.conn.close()
    
    def create_tables(self):
        """Crée les tables nécessaires si elles n'existent pas."""
        conn = self.connect()
        cursor = conn.cursor()
        
        # Table pour les données météo historiques
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS weather_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            temperature_2m REAL,
            relativehumidity_2m REAL
        )
        ''')
        
        # Table pour les métadonnées des modèles
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_path TEXT NOT NULL,
            target TEXT NOT NULL,
            training_start_date DATE,
            training_end_date DATE,
            training_date DATETIME,
            metrics TEXT,
            features TEXT,
            hyperparameters TEXT
        )
        ''')
        
        # Table pour les prédictions
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER,
            prediction_date DATETIME NOT NULL,
            target_date DATETIME NOT NULL,
            features TEXT,
            predicted_value REAL,
            FOREIGN KEY (model_id) REFERENCES models (id)
        )
        ''')
        
        conn.commit()
        self.close()
        logging.info("Tables créées avec succès")
    
    def save_weather_data(self, df):
        """Sauvegarde les données météo dans la base de données."""
        conn = self.connect()
        
        # Assurons-nous que les colonnes nécessaires sont présentes
        required_cols = ['time', 'temperature_2m', 'relativehumidity_2m']
        if not all(col in df.columns for col in required_cols):
            logging.error("Colonnes manquantes dans le DataFrame")
            return False
        
        # Convertir la colonne time en datetime si nécessaire
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])
        
        # Préparer les données pour l'insertion
        data_to_insert = []
        for _, row in df.iterrows():
            data_to_insert.append((
                row['time'].strftime('%Y-%m-%d %H:%M:%S'),
                row['temperature_2m'],
                row['relativehumidity_2m']
            ))
        
        # Insérer les données
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT INTO weather_data (timestamp, temperature_2m, relativehumidity_2m) VALUES (?, ?, ?)",
            data_to_insert
        )
        
        conn.commit()
        self.close()
        logging.info(f"Sauvegarde de {len(data_to_insert)} enregistrements météo")
        return True
    
    def get_weather_data(self, start_date=None, end_date=None):
        """Récupère les données météo de la base de données."""
        conn = self.connect()
        
        query = "SELECT timestamp, temperature_2m, relativehumidity_2m FROM weather_data"
        params = []
        
        if start_date:
            query += " WHERE timestamp >= ?"
            params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
        elif end_date:
            query += " WHERE timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp"
        
        df = pd.read_sql_query(query, conn, params=params)
        df['time'] = pd.to_datetime(df['timestamp'])
        df = df.drop('timestamp', axis=1)
        
        self.close()
        return df
    
    def save_model_metadata(self, model_info, model_path):
        """Sauvegarde les métadonnées du modèle dans la base de données."""
        conn = self.connect()
        cursor = conn.cursor()
        
        # Extraire les métadonnées
        target = model_info['target']
        training_start = model_info['training_data_start']
        training_end = model_info['training_data_end']
        training_date = model_info['training_date']
        metrics = json.dumps(model_info['metrics'])
        features = json.dumps(model_info['features'])
        
        # Extraire les hyperparamètres
        if hasattr(model_info['model'], 'get_params'):
            hyperparameters = json.dumps(model_info['model'].get_params())
        else:
            hyperparameters = '{}'
        
        # Insérer dans la base de données
        cursor.execute('''
        INSERT INTO models (
            model_path, target, training_start_date, training_end_date, 
            training_date, metrics, features, hyperparameters
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_path, target, training_start, training_end,
            training_date, metrics, features, hyperparameters
        ))
        
        model_id = cursor.lastrowid
        conn.commit()
        self.close()
        
        logging.info(f"Métadonnées du modèle sauvegardées avec ID: {model_id}")
        return model_id
    
    def get_latest_model(self, target='temperature_2m'):
        """Récupère les métadonnées du modèle le plus récent pour une cible donnée."""
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, model_path FROM models 
        WHERE target = ? 
        ORDER BY training_date DESC 
        LIMIT 1
        ''', (target,))
        
        result = cursor.fetchone()
        self.close()
        
        if result:
            return {'id': result[0], 'model_path': result[1]}
        else:
            return None
    
    def save_predictions(self, model_id, predictions_df):
        """Sauvegarde les prédictions dans la base de données."""
        conn = self.connect()
        cursor = conn.cursor()
        
        # Préparer les données pour l'insertion
        data_to_insert = []
        for _, row in predictions_df.iterrows():
            # Convertir les features en JSON
            features = {col: row[col] for col in predictions_df.columns 
                       if col not in ['time', 'prediction']}
            
            data_to_insert.append((
                model_id,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                row['time'].strftime('%Y-%m-%d %H:%M:%S'),
                json.dumps(features),
                row['prediction']
            ))
        
        # Insérer les données
        cursor.executemany('''
        INSERT INTO predictions (
            model_id, prediction_date, target_date, features, predicted_value
        ) VALUES (?, ?, ?, ?, ?)
        ''', data_to_insert)
        
        conn.commit()
        self.close()
        
        logging.info(f"Sauvegarde de {len(data_to_insert)} prédictions pour le modèle {model_id}")
        return True
    
    def get_predictions(self, start_date=None, end_date=None, limit=100):
        """Récupère les prédictions de la base de données."""
        conn = self.connect()
        
        query = '''
        SELECT p.id, m.target, p.target_date, p.predicted_value
        FROM predictions p
        JOIN models m ON p.model_id = m.id
        '''
        
        params = []
        if start_date:
            query += " WHERE p.target_date >= ?"
            params.append(start_date)
            
            if end_date:
                query += " AND p.target_date <= ?"
                params.append(end_date)
        elif end_date:
            query += " WHERE p.target_date <= ?"
            params.append(end_date)
        
        query += " ORDER BY p.target_date DESC LIMIT ?"
        params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        df['target_date'] = pd.to_datetime(df['target_date'])
        
        self.close()
        return df