import numpy as np
import os
from skyjo_engine import SkyjoGame
from skyjo_brain import SkyjoBrain

def get_human_action(game, phase, mask):

    legal_actions = np.where(mask == 1)[0]

    if phase == 1:
        print("0: Ablagestapel ziehen; 1: Nachziehstapel ziehen")
    else:
        print("0-11: Handkarte tauschen (0-11 entsprechen den Slots)")
        print("12-23: Handkarte abwerfen und Slot aufdecken")

    while True:
        try:
            action = int(input("Geben Sie die Aktionsnummer ein: "))
            if action in legal_actions:
                return action
            else:
                print("Ungültige Aktion. Bitte wählen Sie eine legale Aktion.")
        except ValueError:
            print("Ungültige Eingabe. Bitte geben Sie eine Zahl ein.")


def play():
    HIDDEN_SIZE = 128
    num_players = 3
    human_idx = 0

    # Champion laden
    champion = SkyjoBrain(hidden_size=HIDDEN_SIZE)
    
    path = "Champions/overall_best_brain.npz"

    if not os.path.exists(path):
        print(f"Fehler: Champion-Datei '{path}' nicht gefunden. Bitte trainieren Sie zuerst eine KI und speichern Sie sie als 'overall_best_brain.npz' im Champions-Ordner.")
        return
    champion.load(path)
    print("Champion geladen. Viel Spaß beim Spielen gegen die KI!")

    # Spiel initialisieren
    game = SkyjoGame(num_players=num_players)
    game.reset()
    done = False

    while not done:
        game.render()
        current_player = game.current_player
        mask = game.get_legal_mask()

        if current_player == human_idx:
            action = get_human_action(game, game.phase, mask)
        else:
            print(f"KI Spieler {current_player} ist am Zug...")
            action = champion.predict(game.get_state(), game.phase, mask, epsilon=0.0)
            print(f"KI Spieler {current_player} wählt Aktion: {champion.translate_action(action, game.phase)}")

        state, done = game.step(action)

    game.render()
    scores = game.get_score()   
    print("\n--- Spielende ---")
    for i, s in enumerate(scores):
        if i == human_idx:
            print(f"Deine Punktzahl: {s}")
        else:
            print(f"KI Spieler {i} Punktzahl: {s}")

    winner = np.argmin(scores)
    if winner == human_idx:
        print("\nGratulation! Du hast den Champion geschlagen!")
    else:
        print(f"\nDer Champion (Spieler {winner}) hat gewonnen!")


if __name__ == "__main__":
    play()