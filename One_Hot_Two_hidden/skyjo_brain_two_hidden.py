import numpy as np 

class SkyjoBrain_TwoHidden:

    def __init__(self, input_size=32, hidden_size_1=64, hidden_size_2=32, output_size=26):
        # Board, Sichtbarkeit, Gegnerinfos, Ablagekarte, gehaltene Karte, letzte Runde Flag
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size  
        # Gemeinsames Verständnis des Boards
        # Input -> Hidden 1
        self.W1 = np.random.randn(self.input_size, self.hidden_size_1) * 0.1
        self.b1 = np.zeros(self.hidden_size_1)

        # Hidden 1 -> Hidden 2
        self.W2 = np.random.randn(self.hidden_size_1, self.hidden_size_2) * 0.1
        self.b2 = np.zeros(self.hidden_size_2)

        # Hidden 2 -> Output
        self.W_out = np.random.randn(self.hidden_size_2, self.output_size) * 0.1
        self.b_out = np.zeros(self.output_size)

    def predict(self, state):
        """ Returns logits for all 26 Aktionen
        """
        # Input -> Hidden 1 Relu
        hidden_1 = np.maximum(0, np.dot(state, self.W1) + self.b1)
        # Hidden 1 -> Hidden 2 Relu
        hidden_2 = np.maximum(0, np.dot(hidden_1, self.W2) + self.b2)
        # Hidden 2 -> Output
        logits = np.dot(hidden_2, self.W_out) + self.b_out
        return logits

    

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
            

    def mutate(self, rate=0.05, noise=0.05):
        # Mutation der Gewichte und Biases mit einer bestimmten Rate und Rauschstärke
        for param in [self.W1, self.b1, self.W2, self.b2, self.W_out, self.b_out]:
            # Erstellen einer Maske, um zu entscheiden, welche Parameter mutiert werden
            mutation_mask = np.random.rand(*param.shape) < rate
            # Mutation hinzufügen: Rauschen nur zu den ausgewählten Parametern hinzufügen
            param += mutation_mask * np.random.randn(*param.shape) * noise


    def save(self, path):
        # Speichern der Gewichte und Biases in einer .npz Datei
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W_out=self.W_out, b_out=self.b_out)

    def load(self, path):
        # Laden der Gewichte und Biases aus einer .npz Datei
        data = np.load(path)
        self.W1, self.b1 = data['W1'], data['b1']
        self.W2, self.b2 = data['W2'], data['b2']
        self.W_out, self.b_out = data['W_out'], data['b_out']

    def copy(self):
        # Erstellen einer Kopie des Gehirns mit den gleichen Gewichten und Biases
        new_brain = SkyjoBrain_TwoHidden(self.input_size, self.hidden_size_1, self.hidden_size_2, self.output_size)
        new_brain.W1, new_brain.b1 = self.W1.copy(), self.b1.copy()
        new_brain.W2, new_brain.b2 = self.W2.copy(), self.b2.copy()
        new_brain.W_out, new_brain.b_out = self.W_out.copy(), self.b_out.copy()
        return new_brain