import unittest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        #Dane dla testów
        self.data = load_iris()
        self.model = RandomForestClassifier(random_state=42, n_jobs=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data.data, self.data.target, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(random_state=42)

    def test_model_training(self):
        #Testuje model i sprawdza, czy dokładność jest większa niż 90%
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        self.assertGreater(accuracy, 0.9, "Dokładność powinna być większa niż 0,9")

    def test_model_prediction(self):
        #Sprawdza, czy liczba przewidywań zgadza się z liczbą próbek testowych
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        self.assertEqual(len(y_pred), len(self.y_test), "Liczba przewidywań powinna odpowiadać liczbie próbek testowych")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
