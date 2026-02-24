import numpy as np
from skyjo_brain import SkyjoBrain
from skyjo_engine import SkyjoGame

def run_brain_test():
    print("--- TEST: SKYJO BRAIN ---")

    # 1. Initialisierung
    brain = SkyjoBrain()
    game = SkyjoGame(num_players=3)
    game.reset()
    print("✅ Brain und Spiel initialisiert")
    game.render()
    # 2. Karte ziehen
    action = brain.predict(game.get_state(), game.phase, game.get_legal_mask(), epsilon=0.0)
    print('Aktion:', brain.translate_action(action, game.phase))
    game.step(action)
    game.render()
    # 3. Karte tauschen/aufdecken
    action = brain.predict(game.get_state(), game.phase, game.get_legal_mask(), epsilon=0.0)
    print('Aktion:', brain.translate_action(action, game.phase))
    game.step(action)
    game.render()


if __name__ == "__main__":
    run_brain_test()