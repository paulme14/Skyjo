import numpy as np
from skyjo_engine import SkyjoGame
from skyjo_brain import SkyjoBrain
import time
import os
# --------------------------
# Ansatz: Evolutionäre Strategie mit Selektion, Mutation und Elitismus
# normalisierung der Punktzahlen zwischen 0 und 1
# Die fitness ergibt sich aus der Differenz zwischen eigener Punktzahl und durchschnittlicher Punktzahl der Gegner
# wenn das Spiel nicht regulär beendet wird (z.B. zu viele Züge), gibt es eine schlechte Fitness
# --- Hyperparameter ---
POPULATION_SIZE = 100 # Anzahl der Gehirne in der Population
GENERATIONS = 100 # Anzahl der Trainingsgenerationen
GAME_MULTIPLIER = 50 # Anzahl der Spiele pro Gehirn und Generation (je mehr, desto stabilere Bewertung, aber auch länger dauerndes Training)
TOP_K_SURVIVORS = 15 # Anzahl der besten Gehirne, die in die nächste Generation übernommen werden
ELITISM_COUNT = 5 # Anzahl der besten Gehirne, die unverändert in die nächste Generation übernommen werden
HIDDEN_SIZE = 128 # Anzahl der Neuronen im versteckten Layer
MUTATION_RATE = 0.15 # Wahrscheinlichkeit, mit der ein Gewicht mutiert
NOISE_SCALE = 0.1 # Standardabweichung des Rauschens, das bei der Mutation hinzugefügt wird
EPSILON_START = 0.3 # Startwert für Epsilon in der epsilon-greedy Strategie
EPSILON_DECAY = 0.995 # Faktor, um den Epsilon nach jeder Generation zu multiplizieren (für weniger Exploration im Laufe der Zeit)

def evaluate_population(population, gen_idx=0, num_players=3, epsilon=None):
    """ Lässt die Population gegeneinander antreten und bewertet sie basierend auf der erreichten Punktzahl. 
    Die Bewertug ist abhängig von der eigennen Punktzahl und der Punktzahl der Gegner (großer Abstand ist gut). """

    total_fitness = np.zeros(len(population))
    games_played = np.zeros(len(population))
    #print(f"Evaluating population with epsilon={epsilon:.4f} for generation {gen_idx+1}...")
    env = SkyjoGame(num_players=num_players)

    all_indices = np.arange(len(population))
    games_not_finished = 0

    for _ in range(GAME_MULTIPLIER):
        for i in range(len(population)):
            # Wähle den i-ten Spieler als Hauptspieler und zwei weitere zufällig aus der Population als Gegner
            opp_indices = all_indices[all_indices != i] # Alle Indizes außer i
            opp = np.random.choice(opp_indices, num_players-1, replace=False)
            player_idx = np.concatenate(([i], opp))
            np.random.shuffle(player_idx) # Zufällige Reihenfolge der Spieler im Spiel
            players = [population[idx] for idx in player_idx]
            state = env.reset()
            done = False
            turns = 0

            while not done and turns < 200: # Maximal 200 Züge pro Spiel, um Endlosschleifen zu vermeiden
                brain = players[env.current_player]
                # PHASE 1: Ziehen
                mask1 = env.get_legal_mask()
                action1 = brain.predict(state, phase=env.phase, mask=mask1, epsilon=epsilon)
                state, _ = env.step(action1)

                # PHASE 2: Ausspielen (KI sieht jetzt held_card!)
                mask2 = env.get_legal_mask()
                action2 = brain.predict(state, phase=env.phase, mask=mask2, epsilon=epsilon)
                state, done =env.step(action2)

                turns += 1

            # Am Ende des Spiels Fitness berechenen:
            if done:
                raw_scores = env.get_score() # Punktzahlen aller Spieler
                for j in range(num_players):
                    p_idx = player_idx[j] # Index des j-ten Spielers in der Population
                    own_score = raw_scores[j]
                    opp_scores = [raw_scores[k] for k in range(num_players) if k != j]
                    fitness = own_score - np.mean(opp_scores) # Je niedriger die eigene Punktzahl und je höher die Gegnerpunktzahlen, desto besser (großer Abstand ist gut)
                    # Bestrafen von hohen Punktzahlen
                    fitness += own_score * 1.5 # Strafe für hohe Punktzahlen, um die KI zu motivieren, nicht nur besser als die Gegner zu sein, sondern auch insgesamt niedrigere Punktzahlen zu erreichen
                    # Sieg-Bonus
                    if own_score < np.min(opp_scores):
                        fitness -= 50 # Bonus für den Sieg, um die KI zu motivieren, nicht nur besser als die Gegner zu sein, sondern auch zu gewinnen
                    total_fitness[p_idx] += fitness
                    games_played[p_idx] += 1
            else:
                # Wenn das Spiel nicht regulär beendet wurde (z.B. zu viele Züge), geben wir eine schlechte Fitness
                for j in range(num_players):
                    p_idx = player_idx[j]
                    total_fitness[p_idx] += 500 # Strafe für unvollständiges Spiel
                    games_played[p_idx] += 1
                games_not_finished += 1
    print(f"Evaluation abgeschlossen. {games_not_finished} von {GAME_MULTIPLIER * len(population)} Spielen wurden nicht regulär beendet.")
    avg_fitness = total_fitness / games_played
    timeout_rate = games_not_finished / (GAME_MULTIPLIER * len(population))
    return avg_fitness, timeout_rate


def run_evolution(initial_population=None, start_gen=0):
    # Initialisierung der Population
    if initial_population is not None:
        print("Initialisiere Population mit vorgegebenen Gehirnen...")
        population = initial_population
    else:
        print("Starte neue Evolution")
        population = [SkyjoBrain(hidden_size=HIDDEN_SIZE) for _ in range(POPULATION_SIZE)]

    best_global_score = float('inf')
    epsilon_boost = 0.0
    history = []
    previous_best = SkyjoBrain(HIDDEN_SIZE)
    overall_best = SkyjoBrain(HIDDEN_SIZE)
    path = r"Champions\overall_best_brain.npz"
    if os.path.exists(path):
        overall_best.load(path)
    else: 
        overall_best = population[0].copy()

    for gen in range(start_gen, GENERATIONS + start_gen):
        # ------------------------------------------------------------------
        # Spielen und Epsilon dynamisch anpassen
        print(f"Starte Training Generation {gen+1}/{GENERATIONS + start_gen}...")
        base_epsilon = max(EPSILON_START * (EPSILON_DECAY ** gen), 0.01) # Epsilon decay über die Generationen
        current_epsilon = base_epsilon + epsilon_boost # Aktuelles Epsilon mit möglichem Boost
        # Evaluation der Population
        fitness_scores, timeout_rate = evaluate_population(population, gen_idx=gen, num_players=3, epsilon=current_epsilon)
        # Timeout-Policy: Wenn zu viele Spiele nicht regulär beendet werden, erhöhen wir vorübergehend Epsilon, um mehr Exploration zu fördern
        if timeout_rate > 0.1: # Wenn mehr als 20% der Spiele nicht regulär beendet werden, erhöhen wir Epsilon
            epsilon_boost = 0.2 # Erhöhe Epsilon um 0.1 für die nächste Generation
            current_mutation_rate = 0.2 # Erhöhe auch die Mutationsrate, um mehr Vielfalt zu fördern
        else:
            epsilon_boost = 0.0 # Keine Erhöhung, wenn keine Timeout-Probleme auftreten
            current_mutation_rate = MUTATION_RATE # Normale Mutationsrate
        # -------------------------------------------------------------------
        # Ranking der Population basierend auf Fitness
        sorted_indices = np.argsort(fitness_scores)
        best_fitness = fitness_scores[sorted_indices[0]]

        if best_fitness < best_global_score:
            best_global_score = best_fitness
            # Speichere das beste Gehirn
            population[sorted_indices[0]].save("Champions/overall_best_brain.npz")
        history.append((gen, best_fitness))

        # Testen des besten gegen bestern aus letzer Generation und Gegener der zufällig spielt:
        bench = _run_benchmark(current_champ=population[sorted_indices[0]], previous_champ=previous_best, overall_champ=overall_best)

        if bench:
            wr = bench['winrates']
            print(f"Siege -> Aktuell: {wr[0]:.1%} | Letzte: {wr[1]:.1%} | Overall: {wr[2]:.1%}")
        # Evolution ----------------------------------------
        # Selektion der besten Gehirne
        new_population = []
        # Elitismus: Die besten Gehirne unverändert übernehmen
        for i in range(ELITISM_COUNT):
            elite_idx = sorted_indices[i]
            new_population.append(population[elite_idx].copy())
        
        # Restliche Plätze in der neuen Population mit mutierten Kopien der besten Gehirne füllen
        while len(new_population) < POPULATION_SIZE:
            parent_idx = np.random.choice(sorted_indices[:TOP_K_SURVIVORS]) # Auswahl eines Elternteils aus den Top K
            child = population[parent_idx].copy()
            child.mutate(rate=current_mutation_rate, noise=NOISE_SCALE)
            new_population.append(child)
        population = new_population

        print(f"Generation {gen+1 + start_gen} abgeschlossen. Beste Fitness: {best_fitness:.4f}, Durchschnittliche Fitness: {np.mean(fitness_scores):.4f}, Best Global Score: {best_global_score:.4f}")
    
    # Fortschritt speichern am Ende der Evolution
    for i, brain in enumerate(population):
        brain.save(f"Brains/brain_{i}.npz")
    
    with open("Brains/last_generation.txt", "w") as f:
        f.write(f"{gen+1}")

    return population, history


def _run_benchmark(current_champ, previous_champ, overall_champ, num_games=50):
    """
    Testet den aktuellen Champion gegen den Vorgänger und einen Zufallsspieler.
    """
    env = SkyjoGame(num_players=3)
    results = {"current": 0, "previous": 0, "random": 0}
    wins = {"current": 0, "previous": 0, "random": 0}
    timeouts = 0

    for _ in range(num_games):
        state = env.reset()
        done = False
        turns = 0
        
        # Zuordnung: 0: Current, 1: Previous, 2: Random
        # Wir mischen die Positionen für faire Bedingungen
        roles = [0, 1, 2]
        np.random.shuffle(roles)

        while not done and turns < 500:
            p_idx = env.current_player
            role = roles[p_idx]
            mask = env.get_legal_mask()

            if role == 0: # Current Champ
                action = current_champ.predict(state, env.phase, mask, epsilon=0.0)
            elif role == 1: # Previous Champ
                action = previous_champ.predict(state, env.phase, mask, epsilon=0.0)
            else: # Overall Champ
                action = overall_champ.predict(state, env.phase, mask, epsilon=0.0)

            state, done = env.step(action)
            turns += 1

        if done:
            scores = env.get_score()
            # Ergebnisse zuordnen
            for i, role in enumerate(roles):
                if role == 0: results["current"] += scores[i]
                elif role == 1: results["previous"] += scores[i]
                else: results["random"] += scores[i]
            
            winner_idx = np.argmin(scores)
            winner_role = roles[winner_idx]
            if winner_role == 0: wins["current"] += 1
            elif winner_role == 1: wins["previous"] += 1
            else: wins["random"] += 1
        else:
            timeouts += 1

    # Durchschnittsberechnung
    n = num_games - timeouts
    if n == 0: 
        print('Alle Benchmarkspiele im Timeout!')
        return None
    
    return {
        "avg_scores": {k: v / n for k, v in results.items()},
        "win_rates": {k: v / n for k, v in wins.items()},
        "timeout_rate": timeouts / num_games
    }


if __name__ == "__main__":
    if os.path.exists("Brains/last_generation.txt"):
        with open("Brains/last_generation.txt", "r") as f:
            last_gen = int(f.read().strip())
        print(f"Fortsetzen von Generation {last_gen}...")
        
        loaded_population = []
        for i in range(POPULATION_SIZE):
            brain = SkyjoBrain(hidden_size=HIDDEN_SIZE)
            brain.load(f"Brains/brain_{i}.npz")
            loaded_population.append(brain)
        run_evolution(initial_population=loaded_population, start_gen=last_gen)
    else:
        run_evolution()
    

















