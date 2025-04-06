"""
Script python pour générer un graphique de comparaison des prédictions
avec les données réelles observées.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys

# Ajouter le chemin racine du projet au PYTHONPATH
# Il faut remonter de deux niveaux depuis le répertoire actuel (visualizations/monitoring)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Créer le dossier de sortie pour les graphiques
output_dir = "visualizations/monitoring/output"
os.makedirs(output_dir, exist_ok=True)

from src.db import WeatherDB

def plot_predictions_vs_actuals(target='temperature_2m', start_date=None, end_date=None):
    """
    Génère un graphique comparant les prédictions aux données réelles observées.

    Args:
        target (str): La variable cible ('temperature_2m' ou 'relativehumidity_2m').
        start_date (str): Date de début au format 'YYYY-MM-DD'.
        end_date (str): Date de fin au format 'YYYY-MM-DD'.
    """
    # Connexion à la base de données
    db = WeatherDB()

    # Récupérer les données réelles
    actual_data = db.get_weather_data(start_date=start_date, end_date=end_date)
    if actual_data.empty:
        print(f"Aucune donnée réelle disponible pour la période spécifiée ({start_date} à {end_date}).")
        return

    # Récupérer les prédictions
    predictions = db.get_predictions(start_date=start_date, end_date=end_date)
    if predictions.empty:
        print(f"Aucune prédiction disponible pour la période spécifiée ({start_date} à {end_date}).")
        return

    # Filtrer les données pour la cible spécifiée
    actual_data = actual_data[['time', target]].rename(columns={target: 'actual'})
    predictions = predictions.rename(columns={'target_date': 'time', 'predicted_value': 'prediction'})

    # Fusionner les données réelles et les prédictions
    merged_data = pd.merge(actual_data, predictions, on='time', how='inner')

    if merged_data.empty:
        print("Aucune correspondance entre les données réelles et les prédictions.")
        return

    # Générer le graphique
    plt.figure(figsize=(12, 6))
    plt.plot(merged_data['time'], merged_data['actual'], label='Données réelles', color='blue', linewidth=2)
    plt.plot(merged_data['time'], merged_data['prediction'], label='Prédictions', color='orange', linestyle='--', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel(target.replace('_', ' ').capitalize())
    plt.title(f"Comparaison des prédictions et des données réelles ({target})")
    plt.legend()
    plt.grid(True)

    # Sauvegarder le graphique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{target}_predictions_vs_actuals_{timestamp}.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Graphique généré et sauvegardé : {output_path}")

if __name__ == "__main__":
    plot_predictions_vs_actuals(target='temperature_2m', start_date='2025-01-01', end_date='2025-04-01')
    plot_predictions_vs_actuals(target='relativehumidity_2m', start_date='2025-01-01', end_date='2025-04-01')