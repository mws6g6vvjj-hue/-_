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


st.set_page_config(page_title="인공신경망 데이터 예측", layout="wide")
st.title("인공신경망 기반 데이터 예측 프로그램")
st.write(
    "CSV 파일을 업로드하고 입력 변수와 예측 대상을 선택하면 "
    "인공신경망을 학습하여 예측 결과와 평가 지표를 보여줍니다."
)


# 1. CSV 파일 업로드
uploaded_file = st.file_uploader(
    "공개 데이터 CSV 파일 업로드",
    type=["csv"],
    help="예: 서울시_기간별_시간평균_대기환경_정보_2025.csv"
)

if uploaded_file is None:
    st.info("먼저 CSV 파일을 업로드하세요.")
    st.stop()


# 2. CSV 인코딩 처리
@st.cache_data
def load_csv(file_bytes):
    errors = []
    for encoding in ("cp949", "utf-8-sig", "utf-8"):
        try:
            from io import BytesIO
            return pd.read_csv(BytesIO(file_bytes), encoding=encoding)
        except Exception as e:
            errors.append(f"{encoding}: {e}")
    raise ValueError("CSV 파일을 읽지 못했습니다. " + " / ".join(errors))


try:
    df = load_csv(uploaded_file.getvalue())
except Exception as e:
    st.error(f"CSV 파일을 읽을 수 없습니다: {e}")
    st.stop()

if df.empty:
    st.error("CSV 파일에 데이터가 없습니다.")
    st.stop()

# 열 이름 앞뒤 공백 제거
df.columns = [str(c).strip() for c in df.columns]

st.success(f"파일 업로드 완료: {len(df):,}행, {len(df.columns)}개 열")
with st.expander("업로드한 데이터 미리보기", expanded=True):
    st.dataframe(df.head(10), use_container_width=True)
    st.caption("열 이름")
    st.write(df.columns.tolist())


# 3. 수치형/범주형 열 분류
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# 서울시 데이터의 결측 표기값 등을 결측으로 처리
df = df.replace([-999, -9999, -99], np.nan)

if not numeric_cols:
    st.error("예측에 사용할 숫자형 열이 없습니다.")
    st.stop()

# 4. 예측 대상 자동 추천
target_candidates = [
    c for c in df.columns
    if c.strip().lower().replace(" ", "") in
    ["pm2.5", "pm25", "pm2_5", "초미세먼지(㎍/㎥)".lower().replace(" ", "")]
]
default_target = target_candidates[0] if target_candidates else numeric_cols[-1]

st.header("1. 예측 변수 설정")
target = st.selectbox(
    "예측할 대상 변수(Y)",
    options=numeric_cols,
    index=numeric_cols.index(default_target) if default_target in numeric_cols else 0
)

# 타깃 열 및 사실상 식별자/날짜 열은 기본 입력 후보에서 제외
excluded_names = {"측정일시", "측정소코드", "권역코드"}
feature_candidates = [
    c for c in df.columns
    if c != target and c not in excluded_names
    and (c in numeric_cols or c in categorical_cols)
]

# 서울시 CSV에 대한 입력 변수 추천
seoul_feature_names = [
    "미세먼지 1시간(㎍/㎥)",
    "미세먼지 24시간(㎍/㎥)",
    "오존(ppm)",
    "이산화질소농도(ppm)",
    "일산화탄소농도(ppm)",
    "아황산가스농도(ppm)",
]
suggested_features = [
    c for c in seoul_feature_names
    if c in feature_candidates and c != target
]
if not suggested_features:
    suggested_features = [c for c in feature_candidates if c in numeric_cols][:min(6, len(numeric_cols))]

features = st.multiselect(
    "입력 변수(X) — 예측에 사용할 열을 선택하세요",
    options=feature_candidates,
    default=suggested_features
)

if target in features:
    st.error("예측 대상 변수는 입력 변수에 포함할 수 없습니다. 입력 변수 선택을 수정하세요.")
    st.stop()

if not features:
    st.warning("입력 변수를 하나 이상 선택하세요.")
    st.stop()

# 5. 학습 설정
st.header("2. 인공신경망 설정")
c1, c2, c3 = st.columns(3)
with c1:
    test_size_percent = st.slider("테스트 데이터 비율 (%)", 10, 40, 20, 5)
with c2:
    hidden_neurons = st.selectbox("은닉층 뉴런 구성", ["32", "64", "64, 32", "100, 50"], index=2)
with c3:
    max_iter = st.slider("최대 학습 반복 횟수", 50, 500, 200, 50)

hidden_layers = tuple(int(x.strip()) for x in hidden_neurons.split(","))

st.caption(
    f"예측 대상: {target} | 입력 변수 {len(features)}개 | "
    f"테스트 데이터: {test_size_percent}% | 은닉층: {hidden_layers}"
)

# 6. 데이터 준비 및 학습
if st.button("인공신경망 학습 및 예측", type="primary", use_container_width=True):
    work = df[features + [target]].copy()

    # 타깃은 숫자로 변환하고 값이 없는 행 제거
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(subset=[target])
    work = work[work[target] >= 0]

    if len(work) < 10:
        st.error("학습할 수 있는 데이터가 너무 적습니다. CSV 파일과 예측 대상을 확인하세요.")
        st.stop()

    X = work[features].copy()
    y = work[target].copy()

    # 범주형/숫자형 입력 변수에 각각 전처리 적용
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    transformers = []
    if numeric_features:
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_percent / 100, random_state=42
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("ann", MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True
        ))
    ])

    with st.spinner("인공신경망을 학습하고 예측하는 중입니다..."):
        try:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
        except Exception as e:
            st.error(f"학습 중 오류가 발생했습니다: {e}")
            st.stop()

    # 7. 평가 지표
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions) if len(y_test) >= 2 else float("nan")

    st.header("3. 예측 성능 평가")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae:.3f}")
    m2.metric("RMSE", f"{rmse:.3f}")
    m3.metric("R²", f"{r2:.3f}")

    st.caption(
        "MAE와 RMSE는 작을수록 일반적으로 오차가 작고, "
        "R²는 1에 가까울수록 테스트 데이터의 변동을 잘 설명합니다. "
        "평가값은 업로드한 데이터와 선택한 변수 및 설정에 따라 달라집니다."
    )

    # 8. 결과 표 및 다운로드
    result = pd.DataFrame({
        "실제값": y_test.to_numpy(),
        "예측값": predictions,
        "절대 오차": np.abs(y_test.to_numpy() - predictions)
    })

    st.header("4. 예측 결과")
    st.dataframe(result.head(100), use_container_width=True)

    csv_result = result.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "예측 결과 CSV 다운로드",
        data=csv_result,
        file_name="인공신경망_예측결과.csv",
        mime="text/csv",
        use_container_width=True
    )

    # 9. 그래프: 실제값 vs 예측값 산점도
    st.header("5. 예측 결과 그래프")

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.scatter(y_test, predictions, alpha=0.4)
    low = min(float(y_test.min()), float(np.min(predictions)))
    high = max(float(y_test.max()), float(np.max(predictions)))
    ax1.plot([low, high], [low, high])
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Actual vs Predicted")
    ax1.grid(True, alpha=0.25)
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    # 10. 그래프: 테스트 표본 중 최대 100개 비교
    plot_n = min(100, len(y_test))
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(np.arange(1, plot_n + 1), y_test.iloc[:plot_n].to_numpy(), label="Actual")
    ax2.plot(np.arange(1, plot_n + 1), predictions[:plot_n], label="Predicted")
    ax2.set_xlabel("Test sample")
    ax2.set_ylabel(str(target))
    ax2.set_title("Actual vs Predicted (First 100 Test Samples)")
    ax2.legend()
    ax2.grid(True, alpha=0.25)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.success("학습과 예측이 완료되었습니다.")
