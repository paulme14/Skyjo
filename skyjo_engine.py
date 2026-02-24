import numpy as np
import random

# Aktionen:
# 0: Karte von Ablage ziehen 
# 1 : Karte von Deck ziehen
# 2-13: Handkarte auf Board ersetzen
# 14-25: Verdeckte Karte aufdecken und Handkarte abwerfen
class SkyjoGame:
    def __init__(self, num_players=3):
        self.num_players = num_players
        self.grid = np.zeros((num_players, 12), dtype=int)
        self.visible = np.zeros((num_players, 12), dtype=float) # 0:zu, 1 :offen, 2:entfernt
        self.deck = []
        self.discard_pile = []
        self.current_player = 0
        self.held_card = None
        self.source = None # 'deck' oder 'discard'
        self.phase = 1 # 1: Ziehen, 2: Ausspielen
        self.round_finished = False
        self.finisher_id = -1
        self.turns_left = -1
        

    def reset(self):
        # Karten erstellen und mischen
        cards = [-2]*5 + [-1]*10 + [0]*15 + [i for i in range(1, 13) for _ in range(10)]
        random.shuffle(cards)
        self.deck = cards
        
        # Startkarten auf die Spieler verteilen
        total = self.num_players * 12
        # obersten Karten austeilen
        self.grid = np.array(self.deck[:total]).reshape(self.num_players, 12)
        # und aus Deck entfernen
        self.deck = self.deck[total:]
        # Sichtbarkeit der Karten festlegen (initial alle verdeckt)
        self.visible = np.zeros((self.num_players, 12), dtype=int)
        
        # Zwei zufällige Karten pro Spieler aufdecken
        for p in range(self.num_players):
            indices = random.sample(range(12), 2)
            self.visible[p, indices] = 1

        # Oberste Karte auf den Ablagestapel legen    
        self.discard_pile = [self.deck.pop()]
        self.current_player = 0
        self.phase = 1
        self.held_card = None
        
        
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """ gibt den state des spiels als 1D Array zurück, das alle relevanten Informationen enthält:
        0-11: eigenes Board (nicht maskiert)
        12-23: Sichtbarkeit eigenes Board (0=verdeckt, 1=offen, 2=entfernt)
        24-35: Gegner 1 Board nicht maskiert
        36-47: Gegner 1 Sichtbarkeit
        48-59: Gegner 2 Board nicht maskiert
        60-71: Gegner 2 Sichtbarkeit 
        72: oberste Karte Ablagestapel
        73: gehaltene Karte (nach Phase 1)
        74: letzte Runde Flag (1 wenn letzte Runde, sonst 0)
        75: aktuelle Phase (1 oder 2)
        76: gehaltene Karte Quelle (0: Ablage, 1: Deck, -1: keine Karte)"""
        # Aktuelles Board des Spielers
        my_board = self.grid[self.current_player].copy()
        my_vis = self.visible[self.current_player].copy()

        # Gegnerboards erstellen
        opp_data = []
        for i in range(1, self.num_players):
            idx = (self.current_player + i) % self.num_players
            o_board = self.grid[idx].copy()
            o_vis = self.visible[idx].copy()
            opp_data.extend([o_board, o_vis]) # Gegnerboard und Sichtbarkeit


        # oberste Karte des Ablagestapels
        top_discard = self.discard_pile[-1] if len(self.discard_pile) > 0 else None
        # gehaltene Karte
        held_card = self.held_card 

        # letzter Zug markieren
        last_turn_flag = 1 if self.round_finished else 0
        # gibt alle infos al 1D Array zurück: 
        # eigenes Board (maskiert), Sichtbarkeit eigenes Board, Gegnerboards (maskiert), Sichtbarkeit Gegnerboards, oberste Ablagekarte, gehaltene Karte
        state =  np.concatenate([
            my_board, my_vis, 
            *opp_data, 
            [top_discard, held_card, last_turn_flag, self.phase]
        ])
        # normalisieren auf 0-1 Bereich
        return state.astype(np.float32)
    
    def get_legal_mask(self) -> np.ndarray:
        # Gibt einen Maskenvektor zurück, der anzeigt, welche Aktionen legal sind basierend auf der aktuellen Phase und Spielregeln
        # 1 = legal, 0 = illegal
        if self.phase == 1:
            # Ziehen: immer beide Optionen legal
            return np.concatenate((np.array([1, 1], dtype=int), np.zeros(24, dtype=int))) # 0: Ablage, 1: Deck
        elif self.phase == 2:
            # Ausspielen: 24 mögliche Aktionen
            # 0-11: Handkarte auf Board ersetzen -> mit allen Slots legal, außer wenn Slot bereits entfernt (sichtbar=2)
            # 12-23: Handkarte abwerfen und verdeckte Karte aufdecken -> nur legal, wenn Karte von Deck gezogen wurde und Slot verdeckt ist
            mask = np.ones(24, dtype=int)
            # Fall Karte auf Board bereits entfernt: Aktion nicht legal
            for i in range(12):
                if self.visible[self.current_player, i] == 2: # entfernt
                    mask[i] = 0 # Karte tauschen nicht möglich
                    mask[12 + i] = 0 # Karte aufdecken nicht möglich
            
            # Fall Karte von Ablage gezogen: Handkarte abwerfen nicht legal
            if self.source == 'discard':
                for i in range(12,24):
                    mask[i] = 0 # Karte aufdecken nicht möglich
            # Fall Karte von Deck gezoge: abwerfen legal und verdeckt aufdecken legal und tauschen legal
            elif self.source == 'deck':
                for i in range(12,24):
                    if self.visible[self.current_player, i-12] != 0: # nicht verdeckt
                        mask[i] = 0 # Karte aufdecken nicht möglich
            else:
                raise ValueError("Ungültige Quelle in get_legal_mask: " + str(self.source))
            return np.concatenate((np.zeros(2, dtype=int), mask)) # 0-11 tauschen, 12-23 abwerfen/aufdecken
        else:
            raise ValueError("Ungültige Phase in get_legal_mask: " + str(self.phase))
        
    
    def step(self, action) -> tuple[np.ndarray, int, bool]:
        # Phase 1: Ziehen (0: Ablage, 1: Deck)
        if self.phase == 1:
            # Ablage ziehen
            if action == 0: 
                self.held_card = self.discard_pile.pop()
                self.source = 'discard'
            # Deck ziehen 
            elif action == 1:
                self.held_card = self.deck.pop() if self.deck else self._reshuffle()
                self.source = 'deck'
            else:
                raise ValueError("Ungültige Aktion in Phase 1: " + str(action))
            self.phase = 2
            # gibt neuen Zustand zurück, Spielstate = False
            return self.get_state(), False
        # Phase 2: Ausspielen
        elif self.phase == 2:
            p = self.current_player
            # Karte auf Board ersetzen
            if action < 12:
                idx = action
                self.discard_pile.append(self.grid[p, idx]) # alte Karte auf Ablage
                self.grid[p, idx] = self.held_card # neue Karte auf Board
                self.visible[p, idx] = 1 # Karte ist jetzt offen
            # Verdeckte Karte aufdecken und Handkarte abwerfen
            else:
                idx = action - 12
                self.discard_pile.append(self.held_card) # Handkarte auf Ablage
                self.visible[p, idx] = 1 # Karte aufdecken
            
            # Spalte entfernen, wenn alle Karten gleich
            self._check_columns(p)
            # Handkarte zurücksetzen, Phase zurücksetzen
            self.held_card = None
            self.phase = 1  
            # Rundenende prüfen
            done = self._check_round_end()
            # Wenn nicht vorbei, nächsten Spieler setzen
            if not done:
                self.current_player = (self.current_player + 1) % self.num_players
            # neuen Zustand, Spielstate zurückgeben
            return self.get_state(), done
        else:
            raise ValueError("Ungültige Phase in step: " + str(self.phase))

    def _check_columns(self, p):
        for col in range(4):
            idxs = [col, col+4, col+8]
            if np.all(self.visible[p, idxs] == 1):
                vals = self.grid[p, idxs]
                if vals[0] == vals[1] == vals[2]:
                    self.visible[p, idxs] = 2
                    self.discard_pile.extend(vals)

    def _check_round_end(self):
        if not self.round_finished:
            if np.all(self.visible[self.current_player] != 0):
                self.round_finished = True
                self.finisher_id = self.current_player
                self.turns_left = self.num_players - 1
                return False
        else:
            self.turns_left -= 1
            if self.turns_left <= 0:
                self.visible[self.visible == 0] = 1
                return True
        return False

    def get_score(self):
        scores = []
        for p in range(self.num_players):
            s = np.sum(self.grid[p][self.visible[p] != 2])
            scores.append(s)
        if self.finisher_id != -1 and scores[self.finisher_id] > min(scores):
            scores[self.finisher_id] *= 2
        return scores

    def _reshuffle(self):
        top = self.discard_pile.pop()
        self.deck = self.discard_pile
        random.shuffle(self.deck)
        self.discard_pile = [top]
        return self.deck.pop()
    
    def render(self):
        print("\n" + "═"*100)
        
        if self.phase == 1:
            # --- ANZEIGE FÜR PHASE 1: ENTSCHEIDUNG WO MAN ZIEHT ---
            print(f" Ablagestapel: [{self.discard_pile[-1] if self.discard_pile else 'Leer'}]", end="")
            print(f" Deck: [ ? ]")
            if self.round_finished:
                print(f" !!! LETZTE RUNDE: Noch {self.turns_left} Züge !!!")
            print("─"*100)
            #self._render_board() # Zeigt das Spielfeld
            
        else:
            # --- ANZEIGE FÜR PHASE 2: AKTION MIT DER GEZOGENEN KARTE ---
            source_str = "Nachziehstapel" if self.source == "deck" else "Ablagestapel"
            print(f" SPIELER {self.current_player} | PHASE: AUSPIELEN")
            print(f" Du hast gezogen von: {source_str}")
            print(f"\n >>> DEINE KARTE: [[ {self.held_card} ]] <<<")
            print("\n (Entscheide nun: Auf Feld 0-11 tauschen oder abwerfen/aufdecken)")
            print("─"*100)
            # Optional: Hier zeigen wir das Board trotzdem, damit der Mensch weiß, wo er tauschen will
            self._render_board() 

    def _render_board(self):
        """Interne Hilfsfunktion, um nur das Spielfeld auszugeben."""
        header = ""
        for p in range(self.num_players):
            name = f"SPIELER {p}"
            if p == self.current_player:
                name = f"> {name} <"
            header += f"{name:^30}   "
        print(header)

        for row in range(3):
            line_str = ""
            for p in range(self.num_players):
                row_indices = range(row * 4, (row + 1) * 4)
                row_content = ""
                for i in row_indices:
                    vis = self.visible[p, i]
                    val = self.grid[p, i]
                    if vis == 1:    # Offen
                        row_content += f"[{val:>3}] "
                    elif vis == 2:  # Entfernt
                        row_content += "  .   "
                    else:           # Verdeckt
                        row_content += "[ ? ] "
                line_str += f"  {row_content}      "
            print(line_str)
        print("═"*100 + "\n")