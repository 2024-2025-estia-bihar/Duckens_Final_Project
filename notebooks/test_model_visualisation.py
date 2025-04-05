import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model_train import load_model, create_features
from src.db import WeatherDB
from src.data_processing import transform_to_3h_interval
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging

# === Étapes ===

# 1. Charger les données météo depuis la base
print("Chargement des données depuis la base...")
db = WeatherDB()
df = db.get_weather_data()
df = transform_to_3h_interval(df)

print("Taille du DataFrame :", df.shape)
print("Dates min/max :", df.index.min(), " -> ", df.index.max())

# 2. Créer les features
features_df = create_features(df)

# 3. Charger le dernier modèle sauvegardé
print("Chargement du modèle...")
model_meta = db.get_latest_model(target="temperature_2m")
model_info = joblib.load(model_meta['model_path'])

# 4. Reprendre les features et scaler du modèle
features = model_info['features']
target = model_info['target']
scaler = model_info['scaler']
model = model_info['model']

# 5. Créer jeu de test (20% des données les plus récentes)
split_idx = int(len(features_df) * 0.8)
X_test = features_df[features].iloc[split_idx:]
y_test = features_df[target].iloc[split_idx:]
X_test_scaled = scaler.transform(X_test)

print("Taille X_test :", X_test.shape)
print("Taille X_train :", split_idx)
print("Dates min/max dans X_test :", X_test.index.min(), " -> ", X_test.index.max())


# 6. Prédictions
print("Prédictions sur les données de test...")
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE : {rmse:.3f} | R2 : {r2:.3f}")

# 7. Visualisation
plt.figure(figsize=(14, 6))
plt.plot(y_test.index, y_test, label="Valeurs réelles", color='blue')
plt.plot(y_test.index, y_pred, label="Prédictions", color='orange', linestyle='--')
plt.title(f"Évaluation du modèle sur les données de test\nRMSE = {rmse:.2f}, R² = {r2:.2f}")
plt.xlabel("Temps")
plt.ylabel("Température (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/evaluation_test_set.png")
plt.show()
