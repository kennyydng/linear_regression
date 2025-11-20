"""train.py

Lit un dataset CSV (colonnes: km, price) et effectue une régression linéaire
pour trouver θ0 et θ1 de la formule : prix = θ0 + θ1 * km

Les paramètres entraînés sont sauvegardés dans 'theta.csv' pour être utilisés
par predict.py.
"""

import csv
import os
import sys


def load_data(path):
    """Charge le dataset depuis un CSV avec colonnes km,price."""
    mileages = []
    prices = []
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    km = float(row["km"])
                    price = float(row["price"])
                    mileages.append(km)
                    prices.append(price)
                except (ValueError, KeyError) as e:
                    print(f"Ligne ignorée (erreur de format): {row} — {e}")
                    continue
    except FileNotFoundError:
        print(f"Erreur : fichier '{path}' introuvable.")
        sys.exit(1)
    except Exception as e:
        print(f"Erreur lors de la lecture du dataset : {e}")
        sys.exit(1)
    
    if not mileages:
        print("Aucune donnée valide trouvée dans le dataset.")
        sys.exit(1)
    
    return mileages, prices


def normalize(values):
    """Normalise une liste de valeurs : (x - mean) / std."""
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std = variance ** 0.5
    if std == 0:
        return [0.0] * len(values), mean, 1.0
    normalized = [(x - mean) / std for x in values]
    return normalized, mean, std


def gradient_descent(mileages_norm, prices_norm, learning_rate=0.01, iterations=1000):
    """Effectue la descente de gradient pour trouver θ0 et θ1 (sur données normalisées)."""
    theta0 = 0.0
    theta1 = 0.0
    m = len(mileages_norm)
    
    for i in range(iterations):
        # Calcul des prédictions
        predictions = [theta0 + theta1 * km for km in mileages_norm]
        
        # Calcul des erreurs
        errors = [pred - actual for pred, actual in zip(predictions, prices_norm)]
        
        # Calcul des gradients
        grad0 = sum(errors) / m
        grad1 = sum(err * km for err, km in zip(errors, mileages_norm)) / m
        
        # Mise à jour des paramètres
        theta0 -= learning_rate * grad0
        theta1 -= learning_rate * grad1
        
        # Affichage occasionnel de la progression
        if (i + 1) % 100 == 0:
            mse = sum(e ** 2 for e in errors) / m
            print(f"Itération {i + 1}/{iterations} — MSE: {mse:.4f}")
    
    return theta0, theta1


def denormalize_theta(theta0_norm, theta1_norm, km_mean, km_std, price_mean, price_std):
    """Convertit θ0 et θ1 des valeurs normalisées vers l'échelle originale."""
    # price_normalized = theta0_norm + theta1_norm * km_normalized
    # price_normalized = (price - price_mean) / price_std
    # km_normalized = (km - km_mean) / km_std
    #
    # Donc: (price - price_mean) / price_std = theta0_norm + theta1_norm * (km - km_mean) / km_std
    # => price = price_mean + price_std * [theta0_norm + theta1_norm * (km - km_mean) / km_std]
    # => price = price_mean + price_std * theta0_norm + (price_std * theta1_norm / km_std) * (km - km_mean)
    # => price = [price_mean + price_std * theta0_norm - (price_std * theta1_norm / km_std) * km_mean] + (price_std * theta1_norm / km_std) * km
    
    theta1 = (price_std * theta1_norm) / km_std
    theta0 = price_mean + price_std * theta0_norm - theta1 * km_mean
    
    return theta0, theta1


def save_theta(path, theta0, theta1):
    """Sauvegarde θ0 et θ1 dans un fichier CSV."""
    try:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([theta0, theta1])
        print(f"\nParamètres sauvegardés dans '{path}':")
        print(f"  θ0 = {theta0:.6f}")
        print(f"  θ1 = {theta1:.6f}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de theta.csv : {e}")
        sys.exit(1)


def main():
    # Chemin vers le dataset
    project_root = os.path.dirname(os.path.dirname(__file__)) or os.getcwd()
    data_path = os.path.join(project_root, "inputs", "data.csv")
    theta_path = os.path.join(project_root, "inputs", "theta.csv")
    
    print(f"Chargement du dataset depuis '{data_path}'...")
    mileages, prices = load_data(data_path)
    print(f"  {len(mileages)} échantillons chargés.\n")
    
    # Normalisation des données
    print("Normalisation des données...")
    mileages_norm, km_mean, km_std = normalize(mileages)
    prices_norm, price_mean, price_std = normalize(prices)
    print(f"  Kilométrage: moyenne={km_mean:.2f}, écart-type={km_std:.2f}")
    print(f"  Prix: moyenne={price_mean:.2f}, écart-type={price_std:.2f}\n")
    
    # Descente de gradient
    print("Entraînement du modèle par descente de gradient...")
    theta0_norm, theta1_norm = gradient_descent(mileages_norm, prices_norm, 
                                                  learning_rate=0.1, iterations=1000)
    
    # Dénormalisation des paramètres
    theta0, theta1 = denormalize_theta(theta0_norm, theta1_norm, 
                                        km_mean, km_std, price_mean, price_std)
    
    # Sauvegarde
    save_theta(theta_path, theta0, theta1)
    
    print("\nEntraînement terminé !")
    print("Utilisez 'python3 predict.py' pour prédire des prix.")


if __name__ == '__main__':
    main()
