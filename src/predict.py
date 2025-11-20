"""predict.py

Demande le kilométrage à l'utilisateur et renvoie le prix estimé
selon la hypothèse : estimatePrice(mileage) = theta0 + theta1 * mileage

Les paramètres θ sont lus depuis `theta.csv` au format :
  theta0,theta1
Si le fichier n'existe pas ou est invalide, on utilise 0.0 pour les deux.
"""

import csv
import os
import sys


def load_theta(path):
	"""Charge theta0 et theta1 depuis theta.csv. Retourne (0.0, 0.0) par défaut."""
	try:
		with open(path, newline="") as f:
			reader = csv.reader(f)
			row = next(reader, None)
			if row and len(row) >= 2:
				try:
					t0 = float(row[0])
					t1 = float(row[1])
					return t0, t1
				except ValueError:
					pass
	except FileNotFoundError:
		pass
	except Exception:
		pass
	return 0.0, 0.0


def parse_mileage(raw):
	if raw is None:
		return None
	s = raw.strip().replace(",", ".")
	try:
		return float(s)
	except ValueError:
		return None


def main():
    project_root = os.path.dirname(os.path.dirname(__file__)) or os.getcwd()
    theta_path = os.path.join(project_root, "inputs", "theta.csv")
    theta0, theta1 = load_theta(theta_path)

    # Accept mileage as command-line argument or prompt the user
    raw = None
    if len(sys.argv) >= 2:
        raw = sys.argv[1]
    else:
        try:
            raw = input("Entrez le kilométrage du véhicule: ")
        except EOFError:
            print("Aucune entrée fournie. Fin.")
            return

    mileage = parse_mileage(raw)
    if mileage is None:
        print("Kilométrage invalide. Entrez un nombre (ex: 120000).")
        return

    estimated = theta0 + (theta1 * mileage)
    print(f"Estimation du prix pour {mileage:.2f} km : {estimated:.2f} €")


if __name__ == '__main__':
    main()

