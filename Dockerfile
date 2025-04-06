# Dockerfile
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier uniquement les fichiers nécessaires pour l'API
COPY api /app/api
COPY src /app/src
COPY model/model_train.py /app/model/


# Créer les dossiers nécessaires
RUN mkdir -p data/db logs/plots model/registry

# Variables d'environnement
ENV PYTHONPATH=/app

# Exposer le port pour l'API
EXPOSE 8000

# Commande pour démarrer l'application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]