#!/bin/bash
#Script d'initialisation pour s'assurer que tout fonctionne
# Créer les dossiers nécessaires
mkdir -p data/db logs/plots model/registry

# Vérifier si les modèles existent déjà
if [ ! "$(ls -A model/registry)" ]; then
    echo "Aucun modèle trouvé. Téléchargement des données et entraînement du modèle..."
    python model/model_train.py
else
    echo "Modèles existants détectés."
fi

# Démarrer l'API
uvicorn api.main:app --host 0.0.0.0 --port 8000