from skyjo_engine import SkyjoGame
import numpy as np
import random

def run_test():
    print("Running tests for SkyjoGame...")
    game = SkyjoGame(num_players=3)
    game.reset()
    done = False
    steps = 0

    while not done:
        # 1. Hole erlaubte Züge
        mask = game.get_legal_mask()

        # 2 Suche Indizes, wo Maske 1 ist
        legal_actions = np.where(mask == 1)[0]

        if len(legal_actions) == 0:
            print("No legal actions available! This should not happen.")
            break
        
        # 3. Wähle zufällig eine legale Aktion
        action = random.choice(legal_actions)

        # 4. Führe Aktion aus
        state, done = game.step(action)
        steps += 1

        if steps > 1000:
            print("Too many steps, something might be wrong.")
            break
    print(f"Spiel beenden nach {steps} Schritten")
    print(f"Punktestände: {game.get_score()}")
    print("Finaler Spielstand:")
    game.render()


def test_last_round():
    print("--- TEST: Letzte Runde ---")
    game = SkyjoGame(num_players=3)
    game.reset()

    # Wir manipulieren den Spielstand so, dass Spieler 0 am Zug ist und die letzte Runde beginnt
    game.current_player = 0

    # Alle Slots von Spieler 0 sind offen (egal welche Werte)
    p = 0
    for i in range(11):
        game.visible[p, i] = 1

    print('Spielstand vor dem Zug:')
    game.render()

    # Wir führen einen beliebigen Zug aus (z.B. Karte vom Deck ziehen und auf Slot 0 legen)
    print("Spieler 0 zieht eine 5 und legt sie auf Slot 11.")
    game.held_card = 5
    game.source = 'deck'
    action = 11# Tauschen mit Slot 11 (letzter Slot)
    game.phase = 2
    state, done = game.step(action)
    game.render()
    print(f"Done: {done} (sollte True sein, da letzte Runde vorbei)")

if __name__ == "__main__":
    run_test()