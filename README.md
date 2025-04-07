# Projet de Système de Prédiction Météorologique

Un système complet de prédiction météo qui utilise l'apprentissage automatique pour prédire la température et l'humidité relative, avec une API RESTful pour accéder aux prédictions.

## Table des matières
- Architecture
- Installation
- Utilisation
- API Endpoints
- Tests
- Docker
- CI/CD
- Structure du projet

## Architecture

Ce projet utilise une architecture modulaire avec plusieurs composants clés :

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Pipeline  │────>│  Model Training │────>│ Model Registry  │
└────────┬────────┘     └─────────────────┘     └────────┬────────┘
         │                                               │
         v                                               v
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Database (SQLite) │<────│  API (FastAPI)  │<────│ Prediction Engine │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

- **Data Pipeline**: Récupération et prétraitement des données météorologiques
- **Model Training**: Entraînement périodique des modèles ML avec optimisation 
- **Model Registry**: Stockage et versionnement des modèles entraînés
- **Database**: Stockage persistant des données et prédictions (SQLite)
- **Prediction Engine**: Génération des prédictions météorologiques 
- **API**: Interface RESTful pour accéder aux données et prédictions

### Flux de données

1. Les données météorologiques sont récupérées depuis l'API Open-Meteo
2. Les données sont prétraitées et stockées dans la base de données
3. Les modèles sont entraînés sur les données historiques
4. Le moteur de prédiction génère des prévisions basées sur les modèles
5. L'API expose les données et prédictions aux consommateurs

## Installation

### Prérequis
- Python 3.10+
- Docker et Docker Compose (pour l'installation conteneurisée)

### Installation locale
```bash
# Cloner le dépôt
git clone https://github.com/votre-utilisateur/weather-prediction.git
cd weather-prediction

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Créer les dossiers nécessaires
mkdir -p data/db logs/plots model/registry
```

### Installation avec Docker
```bash
# Construction de l'image
docker build -t weather-api .

# Démarrage du conteneur
docker run -p 8000:8000 -v ./data:/app/data -v ./logs:/app/logs -v ./model/registry:/app/model/registry weather-api
```

## Utilisation

### Entraîner un nouveau modèle
```bash
python model/model_train.py
```

### Générer des prédictions
```bash
python src/prediction.py
```

### Démarrer l'API
```bash
uvicorn api.main:app --reload
```

## API Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Page d'accueil |
| `/weather/historical/` | GET | Récupère les données météo historiques |
| `/weather/stats/` | GET | Statistiques sur les données météo |
| `/models/` | GET | Liste des modèles disponibles |
| `/predictions/run/` | POST | Exécute le pipeline de prédiction |
| `/predictions/` | GET | Récupère les prédictions existantes |
| `/predictions/latest/` | GET | Dernières prédictions disponibles |
| `/health` | GET | Vérifie l'état du système |
| `/version` | GET | Retourne la version actuelle de l'API |

Une documentation interactive est disponible à l'adresse `http://localhost:8000/docs`.

## Tests

```bash
# Installation des dépendances de test
pip install pytest httpx

# Exécution de tous les tests
python -m pytest tests/ -v

# Exécution d'un test spécifique
python -m pytest tests/test_api.py::test_version_endpoint
```

## Docker

Voir le fichier DOCKER.md pour plus d'informations sur la conteneurisation du projet.

## CI/CD

Ce projet utilise GitHub Actions pour l'intégration et le déploiement continus:

- **test**: Exécute tous les tests automatisés
- **build-and-push**: Construit et pousse l'image Docker vers GitHub Container Registry
- **test-api**: Vérifie que l'API fonctionne correctement

Voir le fichier ci.yaml pour plus de détails.
