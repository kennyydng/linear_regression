# Linear Regression - Prédiction de prix de voitures

Projet d'introduction au machine learning : implémentation d'une régression linéaire depuis zéro pour prédire le prix d'une voiture en fonction de son kilométrage.

## 📖 Description

Ce projet implémente un modèle de régression linéaire simple utilisant la descente de gradient. L'objectif est de prédire le prix d'un véhicule basé sur son kilométrage selon l'hypothèse :

```
estimatePrice(mileage) = θ0 + θ1 × mileage
```

Le projet comprend :
(Obligatoire)
- **Entraînement** : Calcul des paramètres θ0 et θ1 par descente de gradient
- **Prédiction** : Estimation du prix pour un kilométrage donné
(Bonus)
- **Menu interactif** qui donne accès aux differents programmes
- **Visualisation** : Graphiques des données et de la droite de régression
- **Évaluation** : Calcul des métriques de précision (R², MAE, RMSE, MSE)

## 🚀 Installation

### Prérequis
- Python 3.7 ou supérieur
- pip

### Étapes d'installation

1. **Cloner le repository**
```bash
git clone https://github.com/kennyydng/linear_regression.git
cd linear_regression
```

2. **Créer l'environnement virtuel**
```bash
python3 -m venv .venv
```

3. **Activer l'environnement virtuel**
```bash
source .venv/bin/activate
```

4. **Installer les dépendances**
```bash
pip install matplotlib
```

## 🎯 Utilisation

### Menu interactif (recommandé)

Lancer le menu principal qui regroupe tous les outils :

```bash
python3 main.py
```

Le menu permet d'accéder à toutes les fonctionnalités via une interface simple.

### Utilisation manuelle

**1. Entraîner le modèle**
```bash
python3 src/train.py
```

**2. Prédire un prix**
```bash
python3 src/predict.py
```

**3. Visualiser les données**
```bash
python3 src/visualize.py
```

**4. Évaluer la précision**
```bash
python3 src/evaluate.py
```

## 📁 Structure du projet

```
linear_regression/
├── main.py              # Menu interactif principal
├── inputs/              # Données d'entrée/sortie
│   ├── data.csv        # Dataset (kilométrage, prix)
│   └── theta.csv       # Paramètres entraînés (θ0, θ1)
├── src/                 # Code source
│   ├── train.py        # Entraînement du modèle
│   ├── predict.py      # Prédiction de prix
│   ├── visualize.py    # Visualisation graphique
│   └── evaluate.py     # Évaluation de la précision
└── README.md
```

## 🔬 Métriques de précision

Le modèle est évalué avec plusieurs métriques :
- **R²** (coefficient de détermination) : Métrique principale, indique le pourcentage de variance expliquée
- **MAE** (Mean Absolute Error) : Erreur absolue moyenne en euros
- **RMSE** (Root Mean Squared Error) : Racine de l'erreur quadratique moyenne
- **MSE** (Mean Squared Error) : Erreur quadratique moyenne

---

*Projet réalisé dans le cadre de l'apprentissage du machine learning*
