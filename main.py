"""menu.py

Menu interactif pour utiliser tous les outils du projet de régression linéaire.

Permet d'accéder facilement à :
- Entraînement du modèle (train.py)
- Prédiction de prix (predict.py)
- Visualisation des données (visualize.py)
- Évaluation de la précision (evaluate.py)
"""

import os
import sys
import subprocess


def clear_screen():
    """Efface l'écran du terminal."""
    os.system('clear' if os.name != 'nt' else 'cls')


def get_python_command():
    """Retourne la commande Python à utiliser."""
    venv_python = os.path.join(os.path.dirname(__file__), ".venv", "bin", "python")
    if os.path.exists(venv_python):
        return venv_python
    return "python3"


def run_script(script_name, wait_after=True):
    """Execute un script Python et attend une confirmation de l'utilisateur."""
    python_cmd = get_python_command()
    script_path = os.path.join(os.path.dirname(__file__), "src", script_name)
    
    if not os.path.exists(script_path):
        print(f"\n❌ Erreur : Le fichier '{script_name}' n'existe pas.")
        input("\nAppuyez sur Entrée pour continuer...")
        return
    
    print(f"\n{'='*60}")
    print(f"Exécution de {script_name}...")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run([python_cmd, script_path], cwd=os.path.dirname(__file__))
        if result.returncode != 0:
            print(f"\n⚠️  Le programme s'est terminé avec le code {result.returncode}")
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption par l'utilisateur.")
    except Exception as e:
        print(f"\n❌ Erreur lors de l'exécution : {e}")
    
    if wait_after:
        input("\nAppuyez sur Entrée pour revenir au menu...")


def predict_interactive():
    """Lance une prédiction interactive."""
    python_cmd = get_python_command()
    script_path = os.path.join(os.path.dirname(__file__), "src", "predict.py")
    
    print(f"\n{'='*60}")
    print("PRÉDICTION DE PRIX")
    print(f"{'='*60}\n")
    
    try:
        mileage = input("Entrez le kilométrage du véhicule (ou 'q' pour annuler) : ").strip()
        if mileage.lower() == 'q':
            return
        
        result = subprocess.run(
            [python_cmd, script_path, mileage],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Annulation.")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
    
    input("\nAppuyez sur Entrée pour revenir au menu...")


def show_menu():
    """Affiche le menu principal."""
    clear_screen()
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "RÉGRESSION LINÉAIRE" + " "*24 + "║")
    print("║" + " "*17 + "Menu Principal" + " "*27 + "║")
    print("╚" + "="*58 + "╝")
    print()
    print("  1. 🚀 Entraîner le modèle")
    print("     └─ Analyse le dataset et calcule θ0 et θ1")
    print()
    print("  2. 💰 Prédire le prix d'un véhicule")
    print("     └─ Estime le prix pour un kilométrage donné")
    print()
    print("  3. 📊 Visualiser les données et la régression")
    print("     └─ Affiche le graphique avec la droite de régression")
    print()
    print("  4. 📈 Évaluer la précision du modèle")
    print("     └─ Calcule R², MSE, RMSE et MAE")
    print()
    print("  5. ❌ Quitter")
    print()
    print("─" * 60)


def main():
    """Boucle principale du menu."""
    while True:
        show_menu()
        
        try:
            choice = input("Choisissez une option (1-5) : ").strip()
            
            if choice == '1':
                run_script("train.py")
            
            elif choice == '2':
                predict_interactive()
            
            elif choice == '3':
                run_script("visualize.py", wait_after=False)
                print("\n✓ Graphique affiché (fermez la fenêtre pour continuer)")
                input("Appuyez sur Entrée pour revenir au menu...")
            
            elif choice == '4':
                run_script("evaluate.py")
            
            elif choice == '5':
                clear_screen()
                print("\n👋 Au revoir !\n")
                sys.exit(0)
            
            else:
                print("\n❌ Option invalide. Veuillez choisir entre 1 et 5.")
                input("Appuyez sur Entrée pour continuer...")
        
        except KeyboardInterrupt:
            clear_screen()
            print("\n\n👋 Programme interrompu. Au revoir !\n")
            sys.exit(0)
        except EOFError:
            clear_screen()
            print("\n\n👋 Au revoir !\n")
            sys.exit(0)


if __name__ == '__main__':
    main()
