import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

class LipColorClassifier:
    def __init__(self, model_path='lip_model.pkl'):
        self.model_path = model_path
        self.model = None
        
        # Automatically load the model if it exists, otherwise build the synthetic datset and train it.
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self._train_and_save_synthetic_model()

    def _generate_synthetic_data(self, n_samples=2000):
        """
        Generates simulated HSV lip variations for training.
        """
        X = []
        y = []
        
        # Normal (Label 0): Pink/Red hues indicating healthy oxygen circulation.
        # Cyanosis (Label 1): Blue/Purple hues indicating poor oxygen processing.
        for _ in range(n_samples // 2):
            # Normal Cases
            h1 = np.random.randint(0, 15)      # Red hue range part 1
            h2 = np.random.randint(165, 180)   # Red hue range part 2
            h = h1 if np.random.random() > 0.5 else h2
            s = np.random.randint(50, 160)
            v = np.random.randint(120, 240)
            X.append([h, s, v])
            y.append(0)
            
            # Cyanosis Risk Cases (Bluish/Purplish)
            h_c = np.random.randint(110, 150)
            s_c = np.random.randint(40, 150)
            v_c = np.random.randint(80, 180)
            X.append([h_c, s_c, v_c])
            y.append(1)
            
        return np.array(X), np.array(y)

    def _train_and_save_synthetic_model(self):
        print("Model file not found. Training synthetic lip color model for prototype...")
        X, y = self._generate_synthetic_data(2000)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # We use a robust Random Forest classifier for stability on this generated dataset
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Synthetic model trained with an accuracy of: {acc*100:.2f}%")
        
        # Ensure directory structure exists (just in case model_path contains directories)
        os.makedirs(os.path.dirname(os.path.abspath(self.model_path)), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def predict(self, mean_hsv):
        """
        Predicts based on a given mean HSV tuple.
        Returns prediction category (0: Normal, 1: Risk) and array of probabilities.
        """
        if self.model is None:
            raise ValueError("Model is not initialized.")
            
        X_input = np.array([mean_hsv])
        prediction = self.model.predict(X_input)[0]
        probabilities = self.model.predict_proba(X_input)[0]
        
        return prediction, probabilities
