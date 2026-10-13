
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. 웹 페이지 설정
st.set_page_config(
    page_title="인공신경망 데이터 예측",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 인공신경망 기반 데이터 예측 프로그램")
st.write(
    "공개 데이터를 업로드하고 인공신경망을 학습하여 "
    "예측 결과와 모델 성능을 확인할 수 있습니다."
)

st.divider()


# 2. CSV 파일 업로드
st.header("1. 데이터 업로드")

uploaded_file = st.file_uploader(
    "분석할 CSV 파일을 업로드하세요.",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="cp949")
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        st.stop()

    if df.empty or len(df.columns) < 2:
        st.error("데이터에 최소 2개 이상의 열이 필요합니다.")
        st.stop()

    st.success("데이터를 성공적으로 불러왔습니다.")

    col1, col2, col3 = st.columns(3)

    col1.metric("데이터 행 수", len(df))
    col2.metric("데이터 열 수", len(df.columns))
    col3.metric("결측값 수", int(df.isna().sum().sum()))

    st.subheader("데이터 미리보기")
    st.dataframe(df.head(10), use_container_width=True)

    with st.expander("기초 통계 확인"):
        st.write(df.describe(include="all").T)


    # 3. 입력 변수와 예측 대상 선택
    st.divider()
    st.header("2. 예측 변수 설정")

    target_col = st.selectbox(
        "예측 대상(종속변수)을 선택하세요.",
        options=list(df.columns)
    )

    feature_options = [
        col for col in df.columns if col != target_col
    ]

    selected_features = st.multiselect(
        "예측에 사용할 입력 변수(독립변수)를 선택하세요.",
        options=feature_options,
        default=feature_options
    )

    test_size = st.slider(
        "테스트 데이터 비율",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.1
    )

    hidden_neurons = st.select_slider(
        "은닉층 뉴런 수",
        options=[16, 32, 64, 128],
        value=64
    )

    max_iter = st.select_slider(
        "최대 학습 반복 횟수",
        options=[200, 500, 1000],
        value=500
    )


    # 4. 인공신경망 학습 및 예측
    st.divider()
    st.header("3. 인공신경망 학습 및 예측")

    if st.button("인공신경망 학습 및 예측 실행", type="primary"):

        if not selected_features:
            st.error("입력 변수를 하나 이상 선택하세요.")
            st.stop()

        try:
            X = df[selected_features].copy()
            y = pd.to_numeric(df[target_col], errors="coerce")

            # 예측 대상이 없는 행 제거
            valid_rows = y.notna()
            X = X.loc[valid_rows]
            y = y.loc[valid_rows]

            if len(X) < 5:
                st.error("유효한 데이터가 너무 적습니다. 최소 5개 이상의 행이 필요합니다.")
                st.stop()

            if y.nunique() < 2:
                st.error("예측 대상에 서로 다른 값이 2개 이상 필요합니다.")
                st.stop()

            if len(X.select_dtypes(exclude=np.number).columns) == len(X.columns):
                # 모든 입력 변수가 범주형이어도 처리 가능
                pass

            numeric_features = X.select_dtypes(
                include=np.number
            ).columns.tolist()

            categorical_features = X.select_dtypes(
                exclude=np.number
            ).columns.tolist()

            transformers = []

            if numeric_features:
                numeric_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ])

                transformers.append((
                    "numeric",
                    numeric_pipeline,
                    numeric_features
                ))

            if categorical_features:
                categorical_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ])

                transformers.append((
                    "categorical",
                    categorical_pipeline,
                    categorical_features
                ))

            preprocessor = ColumnTransformer(
                transformers=transformers
            )

            model = Pipeline([
                ("preprocessor", preprocessor),
                ("neural_network", MLPRegressor(
                    hidden_layer_sizes=(hidden_neurons,),
                    activation="relu",
                    solver="adam",
                    max_iter=max_iter,
                    early_stopping=True,
                    random_state=42
                ))
            ])

            # 학습 데이터와 테스트 데이터 분리
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42
            )

            progress = st.progress(0)
            status = st.empty()

            status.write("인공신경망을 학습하고 있습니다...")

            progress.progress(30)

            model.fit(X_train, y_train)

            progress.progress(75)

            # 테스트 데이터 예측
            y_pred = model.predict(X_test)

            progress.progress(100)
            status.success("학습 및 예측이 완료되었습니다.")

            # 5. 성능 평가
            st.divider()
            st.header("4. 예측 성능 평가")

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric("MAE", f"{mae:.4f}")
            metric2.metric("RMSE", f"{rmse:.4f}")
            metric3.metric("R²", f"{r2:.4f}")

            st.caption(
                "MAE와 RMSE는 낮을수록 일반적으로 좋고, "
                "R²는 1에 가까울수록 모델이 데이터를 잘 설명합니다. "
                "단, R²는 음수가 될 수도 있습니다."
            )

            # 6. 실제값과 예측값 비교
            st.divider()
            st.header("5. 실제값과 예측값 비교")

            results = pd.DataFrame({
                "실제값": y_test.to_numpy(),
                "예측값": y_pred
            })

            results["오차"] = (
                results["실제값"] - results["예측값"]
            )

            st.dataframe(
                results.reset_index(drop=True),
                use_container_width=True
            )

            # 결과 다운로드
            csv_data = results.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                label="예측 결과 CSV 다운로드",
                data=csv_data,
                file_name="prediction_results.csv",
                mime="text/csv"
            )

            # 7. 그래프
            st.divider()
            st.header("6. 예측 결과 그래프")

            fig, ax = plt.subplots(figsize=(9, 5))

            ax.plot(
                range(len(y_test)),
                y_test.to_numpy(),
                marker="o",
                label="Actual"
            )

            ax.plot(
                range(len(y_pred)),
                y_pred,
                marker="x",
                label="Predicted"
            )

            ax.set_xlabel("Test Sample")
            ax.set_ylabel(target_col)
            ax.set_title("Actual vs Predicted")
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)
            plt.close(fig)

            # 실제값과 예측값 산점도
            fig2, ax2 = plt.subplots(figsize=(7, 5))

            ax2.scatter(y_test, y_pred)

            min_value = min(y_test.min(), np.min(y_pred))
            max_value = max(y_test.max(), np.max(y_pred))

            ax2.plot(
                [min_value, max_value],
                [min_value, max_value],
                linestyle="--"
            )

            ax2.set_xlabel("Actual")
            ax2.set_ylabel("Predicted")
            ax2.set_title("Actual vs Predicted Scatter Plot")
            ax2.grid(True, alpha=0.3)

            st.pyplot(fig2)
            plt.close(fig2)

            # 학습 정보
            st.divider()
            st.header("7. 학습 정보")

            st.write(f"- 예측 대상: {target_col}")
            st.write(f"- 입력 변수: {', '.join(selected_features)}")
            st.write(f"- 학습 데이터 수: {len(X_train)}")
            st.write(f"- 테스트 데이터 수: {len(X_test)}")
            st.write(f"- 은닉층 뉴런 수: {hidden_neurons}")
            st.write(f"- 최대 학습 반복 횟수: {max_iter}")

        except Exception as e:
            st.error(f"예측 과정에서 오류가 발생했습니다: {e}")

else:
    st.info("먼저 CSV 파일을 업로드하세요.")

    st.markdown("""
    **사용 순서**
    1. CSV 데이터 파일 업로드
    2. 예측 대상 선택
    3. 입력 변수 선택
    4. 인공신경망 학습 및 예측 실행
    5. 성능 지표와 그래프 확인
    """)
