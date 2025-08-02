import argparse
import pandas as pd
import mlflow
from mlflow.catboost import load_model

def main(input_data_path):
    # Загрузка модели
    model = load_model("model")

    # Загрузка данных для прогнозирования
    data = pd.read_csv(input_data_path)

    # Выполнение прогноза
    predictions = model.predict(data)

    # Вывод результатов
    print("Predictions:")
    print(predictions)

    # Сохранение прогнозов в файл
    output_path = "predictions.csv"
    pd.DataFrame(predictions, columns=["Prediction"]).to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions using the trained model")
    parser.add_argument("--input", type=str, required=True, help="Path to the input data file")
    args = parser.parse_args()

    main(args.input)