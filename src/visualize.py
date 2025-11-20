"""visualize.py

Affiche un graphique avec :
- Les points du dataset (km, price)
- La ligne de régression linéaire calculée par train.py
"""

import csv
import os
import sys

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Erreur : matplotlib n'est pas installé.")
    print("Installez-le avec : pip3 install matplotlib")
    sys.exit(1)


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
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        print(f"Erreur : fichier '{path}' introuvable.")
        sys.exit(1)
    
    if not mileages:
        print("Aucune donnée valide trouvée dans le dataset.")
        sys.exit(1)
    
    return mileages, prices


def load_theta(path):
    """Charge theta0 et theta1 depuis theta.csv."""
    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            row = next(reader, None)
            if row and len(row) >= 2:
                return float(row[0]), float(row[1])
    except (FileNotFoundError, ValueError):
        pass
    return 0.0, 0.0


def main():
    project_root = os.path.dirname(os.path.dirname(__file__)) or os.getcwd()
    data_path = os.path.join(project_root, "inputs", "data.csv")
    theta_path = os.path.join(project_root, "inputs", "theta.csv")
    
    # Chargement des données
    mileages, prices = load_data(data_path)
    theta0, theta1 = load_theta(theta_path)
    
    # Création du graphique
    plt.figure(figsize=(10, 6))
    
    # Points du dataset
    plt.scatter(mileages, prices, color='blue', alpha=0.6, s=50, label='Données réelles')
    
    # Ligne de régression
    km_min, km_max = min(mileages), max(mileages)
    km_range = [km_min, km_max]
    price_range = [theta0 + theta1 * km for km in km_range]
    plt.plot(km_range, price_range, color='red', linewidth=2, 
             label=f'Régression: y = {theta0:.2f} + {theta1:.6f}×km')
    
    # Mise en forme
    plt.xlabel('Kilométrage (km)', fontsize=12)
    plt.ylabel('Prix (€)', fontsize=12)
    plt.title('Régression linéaire : Prix en fonction du kilométrage', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print(f"Paramètres utilisés : θ0 = {theta0:.6f}, θ1 = {theta1:.6f}")
    print("Affichage du graphique...")
    plt.show()


if __name__ == '__main__':
    main()
