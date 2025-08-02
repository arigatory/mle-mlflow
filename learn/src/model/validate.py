import argparse
import pandas as pd
import mlflow

def main(model_path, data_source_uri):
    # Загрузка данных
    data = pd.read_csv(data_source_uri)
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Загрузка модели
    model = mlflow.catboost.load_model(model_path)
    
    # Валидация модели
    accuracy = model.score(X, y)
    
    # Логирование метрик
    mlflow.log_metric("validation_accuracy", accuracy)
    
    print(f"Validation accuracy: {accuracy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to the saved model")
    parser.add_argument("--ds", type=str, help="Path to the validation data")
    args = parser.parse_args()
    
    main(args.model_path, args.ds)