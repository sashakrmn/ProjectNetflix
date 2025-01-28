import pickle

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


class ModelTrainer:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestClassifier(random_state=42)
        self.X = self.data.drop(columns=['main_genre'])
        self.y = self.data['main_genre']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    #Trenowanie modelu na danych
    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    #Ocena modelu
    def evaluate_model(self, sns=None):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))
        print("Confusion Matrix:")
        sns.heatmap(confusion_matrix(self.y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=self.data['main_genre'].unique(), yticklabels=self.data['main_genre'].unique())
        plt.show()

    #Zapis
    def save_model(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.model, f)
