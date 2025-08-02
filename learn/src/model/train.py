import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import mlflow

def main(data_file, learning_rate):
    # Загрузка данных
    data = pd.read_csv(data_file)
    
    # Подготовка данных
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Обучение модели
    model = CatBoostClassifier(learning_rate=learning_rate, random_state=42)
    model.fit(X_train, y_train)
    
    # Логирование метрик
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_metric("accuracy", model.score(X_test, y_test))
    
    # Сохранение модели
    mlflow.catboost.log_model(model, "model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to the data file")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    args = parser.parse_args()
    
    main(args.path, args.lr)