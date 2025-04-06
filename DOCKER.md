# Guide Docker pour l'application de prédiction météo

## Prérequis
- Docker
- Docker Compose

## Démarrage rapide

### Construction de l'image Docker
```bash
docker build -t weather-api .

Démarrer l'API avec Docker
docker run -p 8000:8000 -v ./data:/app/data -v ./logs:/app/logs -v ./model/registry:/app/model/registry weather-api

Utiliser Docker Compose 
docker-compose up

Architecture Docker
Le Dockerfile est optimisé pour permettre une construction rapide lorsque seul le code source change :

Installation des dépendances (couche rarement modifiée)
Copie des fichiers source (couche fréquemment modifiée)
Configuration des volumes pour la persistance des données

Cette structure assure que les dépendances ne sont pas réinstallées à chaque modification du code, accélérant ainsi le processus de build.

Volumes
Les volumes suivants sont configurés pour préserver les données entre les exécutions :

./data/db : Base de données SQLite
./model/registry : Modèles ML entraînés
./logs : Fichiers de logs


Maintenance
Vérifier les logs du conteneur
docker logs weather-api

Reconstruire l'image après modifications
docker-compose build

Reconstruire l'image après modifications
docker-compose build

Nettoyer les volumes inutilisés
docker volume prune

CI/CD avec GitHub Actions
Notre pipeline CI/CD automatise les étapes suivantes :

Exécution des tests unitaires
Construction de l'image Docker
Publication de l'image sur GitHub Container Registry
Test de l'image publiée

Dépannage
Problème : "Port déjà utilisé"
Si le port 8000 est déjà utilisé, vous pouvez spécifier un autre port :
docker run -p 8080:8000 weather-api
ccédez ensuite à l'API via http://localhost:8080.

Problème : "Permission denied" lors de l'accès aux volumes
Assurez-vous que les permissions des dossiers locaux sont correctement configurées :
sudo chmod -R 777 ./data ./logs ./model/registry


Problème : "Image not found" lors de l'exécution
Vérifiez que l'image a bien été construite :
docker images

Si l'image n'apparaît pas, reconstruisez-la :
docker build -t weather-api .