import pandas as pd
import logging
from datetime import datetime

# Configuration des logs
logging.basicConfig(
    filename='logs/data_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def transform_to_3h_interval(df):
    """
    Regroupe les données horaires en tranches de 3 heures.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes 'time' et les variables à regrouper.
    
    Returns:
        pd.DataFrame: DataFrame regroupé en intervalles de 3 heures.
    """
    logging.info("Début de la transformation des données en intervalles de 3 heures.")

    # Vérification des colonnes nécessaires
    if 'time' not in df.columns:
        logging.error("La colonne 'time' est manquante dans le DataFrame.")
        raise ValueError("La colonne 'time' est requise pour la transformation.")

    # Vérification du type de la colonne 'time'
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        logging.info("Conversion de la colonne 'time' en type datetime.")
        df['time'] = pd.to_datetime(df['time'])

    # Ajout des colonnes nécessaires pour le regroupement
    df['group_3h'] = (df['time'].dt.hour // 3) * 3
    df['time_3h'] = df['time'].dt.floor('3H')

    # Regroupement par tranches de 3 heures
    grouped_df = df.groupby('time_3h').mean().reset_index()

    # Arrondir les valeurs numériques à une décimale
    numeric_columns = grouped_df.select_dtypes(include=['float', 'int']).columns
    grouped_df[numeric_columns] = grouped_df[numeric_columns].round(1)

    logging.info(f"Transformation terminée : {len(grouped_df)} lignes générées.")
    return grouped_df