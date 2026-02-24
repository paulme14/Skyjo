import numpy as np 

class SkyjoBrain:

    def __init__(self, hidden_size=64):
        # Board, Sichtbarkeit, Gegnerinfos, Ablagekarte, gehaltene Karte, letzte Runde Flag
        self.input_size = 12*2 + 2*2*12 + 3 
        self.hidden_size = hidden_size
        # Gemeinsames Verständnis des Boards
        # Input -> Hidden 
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.b1 = np.zeros(self.hidden_size)

        # Head 1: Ziehen (2 Aktionen)
        self.W_p1 = np.random.randn(self.hidden_size, 2) * 0.1
        self.b_p1 = np.zeros(2)

        # Head 2: Ausspielen (24 Aktionen)
        self.W_p2 = np.random.randn(self.hidden_size, 24) * 0.1
        self.b_p2 = np.zeros(24)

    def predict(self, state, phase, mask, epsilon=0.0):
        # Epsilon-greedy Aktion auswählen um Exploration zu ermöglichen
        if np.random.rand() < epsilon:
            return np.random.choice(np.where(mask == 1)[0])
        # Vorwärtsdurchlauf durch das Netzwerk mit ReLU Aktivierung
        hidden = np.maximum(0, np.dot(state, self.W1) + self.b1)
        # Ausgabe der entsprechenden Head basierend auf der Phase
        # Nachziehstaopel oder Ablagestapel ziehen -> 2 Aktionen
        if phase == 1:
            logits = np.dot(hidden, self.W_p1) + self.b_p1
        # Karte tauschen oder ablegen und verdeckte Karte aufdecken -> 24 Aktionen
        else:
            logits = np.dot(hidden, self.W_p2) + self.b_p2

        # Maskierung der illegalen Aktionen, damit sie nicht gewählt werden
        logits[mask == 0] = -1e10
        # Aktion mit dem höchsten Logit-Wert auswählen
        return np.argmax(logits)
    
    def mutate(self, rate=0.05, noise=0.05):
        # Mutation der Gewichte und Biases mit einer bestimmten Rate und Rauschstärke
        for param in [self.W1, self.b1, self.W_p1, self.b_p1, self.W_p2, self.b_p2]:
            # Erstellen einer Maske, um zu entscheiden, welche Parameter mutiert werden
            mutation_mask = np.random.rand(*param.shape) < rate
            # Mutation hinzufügen: Rauschen nur zu den ausgewählten Parametern hinzufügen
            param += mutation_mask * np.random.randn(*param.shape) * noise

    def translate_action(self, action, phase):

        if phase == 1:
            return "Ablagestapel ziehen" if action == 0 else "Nachziehstapel ziehen"
        else:
            if action < 12:
                # 0-11: Handkarte auf Board ersetzen
                return f"Handkarte tauschen mit Slot {action}"
            else:
                # 12-23: Verdeckte Karte aufdecken und Handkarte abwerfen
                return f"handkarte abwerfen und verdeckte Karte {action-12} aufdecken"
            

    def save(self, path):
        # Speichern der Gewichte und Biases in einer .npz Datei
        np.savez(path, W1=self.W1, b1=self.b1, Wp1=self.W_p1, bp1=self.b_p1, Wp2=self.W_p2, bp2=self.b_p2)

    def load(self, path):
        # Laden der Gewichte und Biases aus einer .npz Datei
        data = np.load(path)
        self.W1, self.b1 = data['W1'], data['b1']
        self.W_p1, self.b_p1 = data['Wp1'], data['bp1']
        self.W_p2, self.b_p2 = data['Wp2'], data['bp2']

    def copy(self):
        # Erstellen einer Kopie des Gehirns mit den gleichen Gewichten und Biases
        new_brain = SkyjoBrain(self.hidden_size)
        new_brain.W1, new_brain.b1 = self.W1.copy(), self.b1.copy()
        new_brain.W_p1, new_brain.b_p1 = self.W_p1.copy(), self.b_p1.copy()
        new_brain.W_p2, new_brain.b_p2 = self.W_p2.copy(), self.b_p2.copy()
        return new_brain