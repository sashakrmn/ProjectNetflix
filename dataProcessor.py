import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

class DataProcessor:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    #Funkcja uzupełnia brakujące wartości w kolumnach
    def fill_missing_values(self):
        self.data['director'] = self.data['director'].fillna('Unknown')
        self.data['country'] = self.data['country'].fillna('Unknown')
        self.data['cast'] = self.data['cast'].fillna('Unknown')
        self.data['rating'] = self.data['rating'].fillna('Not Rated')
        self.data['duration'] = self.data['duration'].fillna('0 min')

    #Konwertacja czasu trwania
    def convert_duration(self):
        self.data['duration'] = self.data['duration'].str.replace(' min', '').str.replace('Season', '').str.strip()
        self.data['duration'] = pd.to_numeric(self.data['duration'], errors='coerce').fillna(0)

    #Przekstałcanie danych
    def encode_categories(self):
        label_encoder = LabelEncoder()
        self.data['type'] = label_encoder.fit_transform(self.data['type'])
        self.data['country'] = label_encoder.fit_transform(self.data['country'])
        self.data['director'] = label_encoder.fit_transform(self.data['director'])
        self.data['cast'] = label_encoder.fit_transform(self.data['cast'])
        self.data['rating'] = label_encoder.fit_transform(self.data['rating'])

    #Przekstałcanie gatunku
    def preprocess_genres(self):
        #Wyodrębnianie pierwszego gatunku
        self.data['main_genre'] = self.data['listed_in'].str.split(',').str[0]
        self.data.drop(columns=['listed_in'], inplace=True)


        #Kodowanie głównego gatunku
        label_encoder = LabelEncoder()
        self.data['main_genre'] = label_encoder.fit_transform(self.data['main_genre'])
        self.genre_classes = label_encoder.classes_

    #Normalizacja danych
    def normalize_data(self):
        scaler = MinMaxScaler()
        self.data[['release_year', 'duration']] = scaler.fit_transform(self.data[['release_year', 'duration']])

    def get_features_and_target(self):
        #Cechy
        X = self.data.drop(columns=['main_genre', 'title', 'description', 'date_added'])
        #Wartość docelowa
        y = self.data['main_genre']
        return X, y


class ModelTrainer:
    #Przyjmuje dane treningowe
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def train_random_forest(self):
        #Trenowanie modelu
        param_grid = { #Zestaw parametrów do przetestowania
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(self.X_train, self.y_train)
        print(f"Najlepszy model: {grid_search.best_params_}")
        return grid_search.best_estimator_

    #Ocena modelu
    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)

        print("Dokłądność:", accuracy_score(y_test, y_pred))
        print("Raport:\n", classification_report(y_test, y_pred, zero_division=1))

        conf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens')
        plt.title("Confusion Matrix")
        plt.show()

    def save_model(self, model, filename):
        #Zapis modelu
        with open(filename, 'wb') as f:
            pickle.dump(model, f)


def main():
    #Scieżka
    file_path = '/Users/mariana.liakh/Documents/Python/netflix_titles_nov_2019.csv'

    #Przetwarzanie danych
    processor = DataProcessor(file_path)
    processor.fill_missing_values()
    processor.convert_duration()
    processor.encode_categories()
    processor.preprocess_genres()
    processor.normalize_data()
    X, y = processor.get_features_and_target()

    #Rozdzielenie danych
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Trenowanie modelu
    trainer = ModelTrainer(X_train, y_train)
    rf_model = trainer.train_random_forest()

    #Ocena modelu
    print("\nEvaluation of Random Forest:")
    trainer.evaluate_model(rf_model, X_test, y_test)

    #Zapis modelu
    trainer.save_model(rf_model, 'random_forest_model.pkl')
    print("Model saved as random_forest_model.pkl")


if __name__ == "__main__":
    main()