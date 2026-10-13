import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data():
    while True:
        file_path = input("CSV 파일 경로를 입력하세요: ").strip().strip('"')
        path = Path(file_path)

        if not path.exists():
            print("파일을 찾을 수 없습니다. 경로를 다시 확인하세요.")
            continue

        try:
            data = None
            last_error = None
            # 서울시 공공데이터 CSV에서 자주 쓰이는 인코딩을 순서대로 시도합니다.
            for encoding in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
                try:
                    data = pd.read_csv(path, encoding=encoding)
                    break
                except Exception as error:
                    last_error = error
            if data is None:
                raise last_error
            if data.empty:
                print("파일에 데이터가 없습니다.")
                continue
            data.columns = [str(column).strip() for column in data.columns]
            data = data.replace([-999, -9999, -99, "-999", "-9999", "-99"], np.nan)
            # 숫자로 저장된 값이 문자열로 읽힌 경우 숫자형으로 변환합니다.
            for column in data.columns:
                if data[column].dtype == "object":
                    converted = pd.to_numeric(
                        data[column].astype(str).str.replace(",", "", regex=False),
                        errors="coerce"
                    )
                    non_missing = data[column].notna().sum()
                    if converted.notna().sum() > 0 and converted.notna().sum() >= max(3, int(0.5 * non_missing)):
                        data[column] = converted
            return data
        except Exception as error:
            print(f"파일을 읽는 중 오류가 발생했습니다: {error}")


def select_target(data):
    columns = list(data.columns)

    print("\n[데이터 열 목록]")
    for index, column in enumerate(columns, start=1):
        print(f"{index}. {column}")

    while True:
        answer = input("\n예측 대상 열 번호를 입력하세요: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(columns):
            return columns[int(answer) - 1]
        print("목록에 있는 번호를 입력하세요.")


def select_features(data, target):
    columns = [column for column in data.columns if column != target]

    print("\n[입력변수 후보]")
    for index, column in enumerate(columns, start=1):
        print(f"{index}. {column}")

    while True:
        answer = input(
            "\n사용할 입력변수 번호를 쉼표로 구분해 입력하세요 "
            "(예: 1,2,4): "
        ).strip()

        try:
            indexes = [int(value.strip()) for value in answer.split(",")]
            if not indexes or any(index < 1 or index > len(columns) for index in indexes):
                raise ValueError
            return list(dict.fromkeys(columns[index - 1] for index in indexes))
        except ValueError:
            print("올바른 열 번호를 쉼표로 구분해 입력하세요.")


def main():
    print("=" * 50)
    print("인공신경망 기반 데이터 예측 프로그램")
    print("=" * 50)

    data = load_data()

    print(f"\n데이터 크기: {data.shape[0]}행 × {data.shape[1]}열")
    print("\n[데이터 미리보기]")
    print(data.head().to_string(index=False))

    target = select_target(data)
    features = select_features(data, target)

    dataset = data[features + [target]].copy()
    dataset[target] = pd.to_numeric(dataset[target], errors="coerce")
    dataset = dataset.dropna(subset=[target])

    if len(dataset) < 20:
        print("예측 대상의 유효한 데이터가 부족합니다. 최소 20개 이상을 권장합니다.")
        sys.exit(1)

    X = dataset[features].copy()
    y = dataset[target].astype(float)

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    try:
        test_ratio_text = input("\n테스트 데이터 비율을 입력하세요 (기본값 0.2): ").strip()
        test_ratio = float(test_ratio_text) if test_ratio_text else 0.2
        if not 0.1 <= test_ratio <= 0.4:
            raise ValueError
    except ValueError:
        print("비율이 올바르지 않아 기본값 0.2를 사용합니다.")
        test_ratio = 0.2

    try:
        neuron_text = input("은닉층 뉴런 수를 입력하세요 (기본값 32): ").strip()
        neuron_count = int(neuron_text) if neuron_text else 32
        if neuron_count < 1:
            raise ValueError
    except ValueError:
        print("값이 올바르지 않아 기본값 32를 사용합니다.")
        neuron_count = 32

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    preprocessing = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("neural_network", MLPRegressor(
            hidden_layer_sizes=(neuron_count,),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_ratio,
        random_state=42
    )

    print("\n모델을 학습하고 있습니다...")
    try:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    except Exception as error:
        print(f"모델 학습 중 오류가 발생했습니다: {error}")
        sys.exit(1)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results = pd.DataFrame({
        "Actual": y_test.to_numpy(),
        "Predicted": predictions
    })
    results["Error"] = results["Actual"] - results["Predicted"]

    print("\n[예측 성능 평가]")
    print(f"예측 대상: {target}")
    print(f"입력변수: {', '.join(features)}")
    print(f"학습 데이터 수: {len(X_train)}")
    print(f"테스트 데이터 수: {len(X_test)}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    print("\n[예측 결과 일부]")
    print(results.head(10).round(4).to_string(index=False))

    results.to_csv("prediction_results.csv", index=False, encoding="utf-8-sig")
    print("\n예측 결과를 prediction_results.csv 파일로 저장했습니다.")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.6)

    lower = min(float(y_test.min()), float(np.min(predictions)))
    upper = max(float(y_test.max()), float(np.max(predictions)))
    plt.plot([lower, upper], [lower, upper], linestyle="--")

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig("prediction_graph.png", dpi=150)
    print("예측 그래프를 prediction_graph.png 파일로 저장했습니다.")

    plt.show()


if __name__ == "__main__":
    main()
