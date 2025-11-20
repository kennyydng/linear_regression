"""evaluate.py

Calcule la précision de l'algorithme de régression linéaire.

Métriques calculées :
- R² (coefficient de détermination) : mesure la qualité de l'ajustement (0-1, 1 = parfait)
- MSE (Mean Squared Error) : erreur quadratique moyenne
- RMSE (Root Mean Squared Error) : racine de l'erreur quadratique moyenne
- MAE (Mean Absolute Error) : erreur absolue moyenne
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


def calculate_metrics(actual, predicted):
    """Calcule les métriques de précision."""
    n = len(actual)
    
    # Mean Squared Error (MSE)
    mse = sum((a - p) ** 2 for a, p in zip(actual, predicted)) / n
    
    # Root Mean Squared Error (RMSE)
    rmse = mse ** 0.5
    
    # Mean Absolute Error (MAE)
    mae = sum(abs(a - p) for a, p in zip(actual, predicted)) / n
    
    # R² (coefficient de détermination)
    mean_actual = sum(actual) / n
    ss_tot = sum((a - mean_actual) ** 2 for a in actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def main():
    project_root = os.path.dirname(os.path.dirname(__file__)) or os.getcwd()
    data_path = os.path.join(project_root, "inputs", "data.csv")
    theta_path = os.path.join(project_root, "inputs", "theta.csv")
    
    # Chargement des données et paramètres
    mileages, actual_prices = load_data(data_path)
    theta0, theta1 = load_theta(theta_path)
    
    # Calcul des prédictions
    predicted_prices = [theta0 + theta1 * km for km in mileages]
    
    # Calcul des métriques
    metrics = calculate_metrics(actual_prices, predicted_prices)
    
    # Affichage
    print("\n" + "╔" + "=" * 58 + "╗")
    print("║" + " " * 12 + "ÉVALUATION DE LA PRÉCISION DU MODÈLE" + " " * 10 + "║")
    print("╚" + "=" * 58 + "╝")
    
    print(f"\n📋 Paramètres du modèle : θ0 = {theta0:.6f} | θ1 = {theta1:.6f}")
    print(f"📊 Échantillons analysés : {len(actual_prices)}")
    
    # Métrique principale
    print("\n" + "┌" + "─" * 58 + "┐")
    print("│" + " " * 15 + "🎯 PRÉCISION DU MODÈLE" + " " * 21 + "│")
    print("└" + "─" * 58 + "┘")
    print(f"\n  {emoji} R² (Coefficient de détermination) = {metrics['r2']:.4f}")
    print(f"     → Le modèle explique {metrics['r2']*100:.1f}% de la variance des prix")
    
    # Métriques complémentaires
    print("\n" + "┌" + "─" * 58 + "┐")
    print("│" + " " * 13 + "📈 MÉTRIQUES COMPLÉMENTAIRES" + " " * 17 + "│")
    print("└" + "─" * 58 + "┘")
    print(f"\n  💰 MAE (Erreur Absolue Moyenne): {metrics['mae']:.2f}€")
    print(f"     → Interprétation : Le modèle se trompe d'environ {metrics['mae']:.0f}€ par prédiction")
    
    print(f"\n  📊 RMSE (Racine de l'Erreur Quadratique): {metrics['rmse']:.2f}€")
    print(f"     → Écart-type des erreurs (pénalise les grandes erreurs)")
    
    print(f"\n  📉 MSE (Erreur Quadratique Moyenne): {metrics['mse']:.2f}")
    print(f"     → Métrique technique (unité = €²)")
    
    print("\n" + "═" * 60)
    
    # Comparaison RMSE vs MAE
    ratio = metrics['rmse'] / metrics['mae'] if metrics['mae'] > 0 else 1
    if ratio > 1.2:
        print("⚠️  RMSE >> MAE : Présence de quelques erreurs importantes")
    else:
        print("✓  RMSE ≈ MAE : Les erreurs sont homogènes")
    
    print("═" * 60 + "\n")


if __name__ == '__main__':
    main()
