import numpy as np
from skyjo_engine_for_one_hot import SkyjoGame
from skyjo_brain_two_hidden import SkyjoBrain_TwoHidden
import os





# HYPERPARAMETER
# Evolutionäre Parameter
POPULATION_SIZE = 50
GENERATIONS = 200
GAMES_PER_MODEL_PER_GEN = 20
INPUT_SIZE = 644 # 36 Karten * 17 One-Hot-Encoded Werte + 15 Ablagestapel + 15 gehaltene Karte + 1 Letzte Runde Flag + 1 Phase Flag
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 64

# Selektions- und Mutationsparameter
ELITISM_RATE = 0.1  # Prozentsatz der besten Modelle, die direkt in die nächste Generation übernommen werden
#CROSSOVER_RATE = 0.5  # Prozentsatz der Kinder, die durch Crossover erzeugt werden (der Rest wird durch Mutation erzeugt)
TOP_K_SURVIVORS = 0.2  # Prozentsatz der besten Modelle, die als Eltern für die nächste Generation ausgewählt werden
MUTATION_RATE = 0.02  # Wahrscheinlichkeit, mit der ein Kind mutiert wird
MUTATION_NOISE = 0.02  # Stärke des Rauschens, das

# Fitness-Parameter
ILLEGAL_CHOICE_PENALTY = 10  # Strafpunkte für illegale Entscheidungen
FINISHER_BONUS = 100  # Bonus für den Spieler, der das Spiel gewinnt
CARDS_OPENED_BONUS = 20  # Bonus pro geöffneter Karte, um das Öffnen von Karten zu fördern
TIMEOUT_PENALTY = 300  # Strafpunkte für Spiele, die nicht regulär beendet wurden (z.B. durch Zeitüberschreitung)
# EPSILON DECAY PARAMETER
EPSILON_START = 1.0  # Anfangswert von Epsilon für die Epsilon-Greedy-Strategie
EPSILON_DECAY = 0.96  # Faktor, um den Epsilon nach jeder Generation zu multiplizieren
EPSILON_MIN = 0.05

def card_to_one_hot(card, mask):
    """ Wandelt eine Karte in ein One-Hot-Encoded Array um mit 17 Elementen:
        -2, -1, 0, 1, ..., 12, nicht sichtbar oder entfernt."""
    one_hot = np.zeros(17)  # 14 mögliche Werte: -2 bis 12, nicht sichtbar oder entfernt
    if mask == 0:  # verdeckt
        one_hot[15] = 1
    elif mask == 2:  # entfernt
        one_hot[16] = 1
    else:  # sichtbare Karte
        index = card + 2  # Karte -2 wird zu Index 0, Karte 12 wird zu Index 14
        one_hot[int(index)] = 1

    return one_hot

def translate_state(state):
    """ Übersetzt den Zustand in ein One-Hot-Encoded Array."""
    # 0-611: 17 One-Hot-Encoded Werte für jede der 36 Karten (12 pro Spieler) basierend auf Wert und Sichtbarkeit
    # 612-626: One-Hot-Encoded Werte für Ablagestapel (15 Werte) und gehaltene Karte (15 Werte)
    # 627: Letzte Runde Flag
    # 628: Aktuelle Phase (1 oder 2, normalisiert auf 0 oder 1)
    # Annahme: state ist ein Array mit Kartenwerten und Masken
    one_hot_state = np.array([])
    for player in range(3):  # eigener Spieler + 2 Gegner
        for card in range(12):
            card_value = state[player * 24 + card]  # Kartenwert
            card_mask = state[player * 24 + 12 + card]  # Sichtbarkeitsmaske
            one_hot_card = card_to_one_hot(card_value, card_mask)
            one_hot_state = np.concatenate((one_hot_state, one_hot_card))
    # Ablagestapel
    top_card_discard = np.zeros(15) 
    if state[72] is not None and state[75] == 1:  # Ablagestapel ist immer sichtbar, aber nur relevant in Phase 1
        top_card_discard[int(state[72] + 2)] = 1  # Karte -2 wird zu Index 0, Karte 12 wird zu Index 14
        one_hot_state = np.concatenate((one_hot_state, top_card_discard))  # Ablagestapel ist immer sichtbar
    else:
        one_hot_state = np.concatenate((one_hot_state, top_card_discard))  # Kein Ablagestapel oder irrelevant
    # Gehaltene Karte (nur relevant in Phase 2)
    if state[73] is not None and state[75] == 2:  # Gehaltene Karte ist nur relevant in Phase 2
        held_card_one_hot = np.zeros(15)
        held_card_one_hot[int(state[73]) + 2] = 1  # Karte -2 wird zu Index 0, Karte 12 wird zu Index 14
        one_hot_state = np.concatenate((one_hot_state, held_card_one_hot))
    else:
        one_hot_state = np.concatenate((one_hot_state, np.zeros(15)))  # Keine gehaltene Karte oder irrelevant
    # Letzte Runde Flag
    one_hot_state = np.concatenate((one_hot_state, np.array([state[74]])))
    # Aktuelle Phase
    one_hot_state = np.concatenate((one_hot_state, np.array([state[75] - 1])))  # Phase 1 oder 2, normalisiert auf 0 oder 1

    return one_hot_state


def choose_action(logits, mask, phase, epsilon=0.1):
    """ Wählt eine Aktion basierend auf den Logits, der Maske, der Phase und einem Epsilon-Wert für Exploration."""
    chosen_ation_was_legal = True
    
    if phase == 1:
        action = np.argmax(logits[:2])  # Ziehen: 2 Aktionen
        if np.argmax(logits) != action:
            chosen_ation_was_legal = False
        return action, chosen_ation_was_legal
    elif phase == 2:

        # Zufallige Aktion
        if np.random.rand() < epsilon:
            legal_actions = np.where(mask == 1)[0] 
            action = np.random.choice(legal_actions) - 2 
        else:
            masked_logits = np.copy(logits)
            masked_logits[mask == 0] = -1e10  # Setze Logits illegaler Aktionen auf einen sehr niedrigen Wert
            action = np.argmax(masked_logits)  # Karte tauschen oder ablegen und verdeckte Karte aufdecken -> 24 Aktionen
            
            if np.argmax(logits) != action:
                chosen_ation_was_legal = False
        return action - 2, chosen_ation_was_legal  # Aktionen 2-25 (Phase 2) werden auf 0-23 abgebildet
    else:
        raise ValueError("Ungültige Phase in choose_action: " + str(phase))



def play_game(players: list[SkyjoBrain_TwoHidden], env: SkyjoGame, epsilon=0.1):
    env.reset()
    done = False
    turns = 0
    track_illegal_choice = np.zeros(len(players))  # Array, um illegale Entscheidungen zu verfolgen
    # Zufälligen Spieler auswählen, der beginnt
    start_player = np.random.randint(0, len(players))
    env.current_player = start_player
    while not done and turns < 200:  # Sicherheitsabfrage, um endlose Spiele zu vermeiden
        current_player = env.current_player
        brain = players[current_player]
        raw_state = env.get_state()
        state = translate_state(raw_state) 
        logits = brain.predict(state) 
        mask = env.get_legal_mask()
        action, chose_legal = choose_action(logits, mask, env.phase, epsilon)

        if not chose_legal:
            track_illegal_choice[current_player] += 1
        _, done = env.step(action)
        turns += 1

    # Zähle geöffnete Karten für jeden Spieler
    cards_opened = [np.sum(env.visible[p] >= 1) for p in range(len(players))]
    if done:
        return env.get_score(), track_illegal_choice, env.finisher_id, cards_opened
    else: 
        return None, track_illegal_choice, None, cards_opened

def evaluate_fitness(population: list[SkyjoBrain_TwoHidden], scores, illegal_choices, finisher_id, cards_opened):
    # Wenn das Spiel nicht regulär beendet wurde, erhalten alle Spieler eine Fitness von 0, um zu verhindern, dass unvollständige Spiele die Evolution verzerren.
    fitness_scores = np.zeros(len(population))
    if scores is None:
            fitness_scores = fitness_scores - TIMEOUT_PENALTY - (illegal_choices * ILLEGAL_CHOICE_PENALTY)  # Strafpunkte für illegale Entscheidungen, aber keine Belohnung für das Spiel, da es nicht regulär beendet wurde
            fitness_scores += np.array(cards_opened) * CARDS_OPENED_BONUS  # Bonus für geöffnete Karten, um das Öffnen von Karten zu fördern, auch wenn das Spiel nicht regulär beendet wurde
    else:
        for i in range(len(population)):
            own_score = scores[i]
            opponents_scores = [scores[j] for j in range(len(population)) if j != i]
            # Fitness, je höher, desto besser. 
            # Wir wollen einen niedrigen eigenen Score und hohe Scores der Gegner.
            fitness = 100 - own_score + np.mean(opponents_scores)
            # Auswahl illealer Entscheidungen in die Fitness einbeziehen
            fitness -= illegal_choices[i] * ILLEGAL_CHOICE_PENALTY  # Strafpunkte für illegale Entscheidungen
            # Siegerbonus hinzufügen, um das Gewinnen zu belohnen
            if finisher_id == i:
                fitness += FINISHER_BONUS  # Bonus für den Sieger
            fitness_scores[i] = fitness
    return fitness_scores
    
def evaluate_population(population: list[SkyjoBrain_TwoHidden], epsilon=0.1):
    
    total_fitness = np.zeros(POPULATION_SIZE)
    games_played = np.zeros(POPULATION_SIZE)
    env = SkyjoGame()
    games_not_finfished = 0
    
    for _ in range(GAMES_PER_MODEL_PER_GEN):
        for i in range(POPULATION_SIZE):
            # Jeder Brain spielt gegen 2 zufällige Gegner aus der Population
            opponents = np.random.choice([j for j in range(POPULATION_SIZE) if j != i], size=2, replace=False)
            players = [population[i]] + [population[j] for j in opponents]
            # Spiel spielen
            raw_scores, illegal_choices, finisher_id, cards_opened = play_game(players, env, epsilon)
            if raw_scores is None:
                games_not_finfished += 1
            # Fitness bewerten
            fitness_scores = evaluate_fitness(players, raw_scores, illegal_choices, finisher_id, cards_opened)
            total_fitness[i] += fitness_scores[0]  # Fitness des eigenen Brain hinzufügen
            games_played[i] += 1
            for opponent_index, opponent in enumerate(opponents):
                total_fitness[opponent] += fitness_scores[opponent_index + 1]  # Fitness der Gegner hinzufügen   
                games_played[opponent] += 1
    
    average_fitness = total_fitness / games_played
    return average_fitness, games_not_finfished


def uniform_crossover(parent1: SkyjoBrain_TwoHidden, parent2: SkyjoBrain_TwoHidden):
    # Implement later
    pass

def create_children(population: list[SkyjoBrain_TwoHidden], fitness_scores):
    # Elitism
    num_elites = int(ELITISM_RATE * POPULATION_SIZE)
    new_population = population[:num_elites]  # Behalte die besten Modelle direkt bei
    # Mutation 
    select_range = int(TOP_K_SURVIVORS * POPULATION_SIZE)
    for _ in range(num_elites, POPULATION_SIZE):
        parent = np.random.choice(population[:select_range])  # Wähle einen Elternteil aus den Top-K-Modellen
        child = parent.copy()  # Erstelle eine Kopie des Elternteils
        child.mutate(rate=MUTATION_RATE, noise=MUTATION_NOISE)  # Mutieren des Kindes
        new_population.append(child)
    return new_population



def run_evolution(initial_population= None, gen_start = 0):
    # Initialisierung der Population
    # Wenn eine initiale Population übergeben wird, verwenden wir diese.
    # Ansonsten erstellen wir eine neue zufällige Population
    if initial_population is not  None:
        population = initial_population
    else:
        population = [SkyjoBrain_TwoHidden(input_size=INPUT_SIZE, hidden_size_1=HIDDEN_SIZE_1, hidden_size_2=HIDDEN_SIZE_2, output_size=26) for _ in range(POPULATION_SIZE)]

    best_global_fitness = -float('inf')
    best_global_brain = None
    best_brain_previous_gen = None
    best_brain_current_gen = None

    tracking_dict = {
        "games_not_finished_per_gen": [],
        "best_fitness_per_gen": [],
        "average_fitness_per_gen": []}
    path = r"Brains" 

    global_path = os.path.join(path, "best_global_brain.npz")
    if os.path.exists(global_path):
        # Lade den besten globalen Brain, um ihn als Referenz für die Fitness zu verwenden
        best_global_brain = SkyjoBrain_TwoHidden(input_size=INPUT_SIZE, hidden_size_1=HIDDEN_SIZE_1, hidden_size_2=HIDDEN_SIZE_2, output_size=26)
        best_global_brain.load(global_path)
    
    prev_path = os.path.join(path, "best_brain_previous_gen.npz")
    if os.path.exists(prev_path):
        best_brain_previous_gen = SkyjoBrain_TwoHidden(input_size=INPUT_SIZE, hidden_size_1=HIDDEN_SIZE_1, hidden_size_2=HIDDEN_SIZE_2, output_size=26)
        best_brain_previous_gen.load(prev_path)

    # Epsilon-Greedy

    for gen in range(gen_start, gen_start + GENERATIONS):
        # current epsilon
        epsilon = max(EPSILON_MIN, EPSILON_START * (EPSILON_DECAY ** gen))
        
        fitness_scores, games_not_finished = evaluate_population(population, epsilon)
        print(f"Generation {gen}: {games_not_finished} games not finished, best fitness: {np.max(fitness_scores)}, average fitness: {np.mean(fitness_scores)}")
        tracking_dict["games_not_finished_per_gen"].append(games_not_finished)
        tracking_dict["best_fitness_per_gen"].append(np.max(fitness_scores))
        tracking_dict["average_fitness_per_gen"].append(np.mean(fitness_scores))
    # Ranking der Population basierend auf der Fitness 
        ranked_indices = np.argsort(fitness_scores)[::-1]
        # Sortieren der Population entsprechend der Fitness
        population = [population[i] for i in ranked_indices]
        # Sortieren der Fitness Scores entsprechend der Rangfolge
        fitness_scores = fitness_scores[ranked_indices]
        # neue Population erstellen
        population = create_children(population, fitness_scores)
        # Speichern des besten Brains der aktuellen Generation
        best_brain_current_gen = population[0]
        path = r"Brains\best_per_gen"
        best_brain_current_gen.save(os.path.join(path, f"best_brain_gen_{gen}.npz"))
    
        # Aktualisieren des besten globalen Brains, wenn der beste Brain der aktuellen Generation besser ist als der bisherige beste globale Brain
        if best_global_brain is None or fitness_scores[0] > best_global_fitness:
            best_global_brain = best_brain_current_gen.copy()
            best_global_fitness = fitness_scores[0]
            best_global_brain.save(global_path)  # Speichern des besten globalen Brains
            print(f"New best global brain found in generation {gen} with fitness {best_global_fitness}")

        # Speichern des tracking dicts
        np.savez(os.path.join(r"Tracking", "training_tracking.npz"), **tracking_dict)
        
    # Speichern des nach dem Training besten Brains
    best_global_brain.save(global_path)

if __name__ == "__main__":
    run_evolution()
