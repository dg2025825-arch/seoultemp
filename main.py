import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

# -----------------------------
# 기본 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="서울 기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균 기온 예측기")
st.write("서울 기온 관측 데이터를 바탕으로 연도별 평균기온의 추세를 분석하고, "
         "회귀 직선을 이용해 특정 연도의 예상 기온을 확인할 수 있습니다.")

# -----------------------------
# 데이터 불러오기 & 전처리
# -----------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year
    return df

df = load_data()

# 기준 기간: 2025년까지만 사용
df = df[df["연도"] <= 2025]

# 연도별 관측일 수와 평균기온 계산
yearly = df.groupby("연도").agg(
    평균기온=("평균기온", "mean"),
    관측일수=("평균기온", "count")
).reset_index()

# 관측일이 300일 미만인 해는 제외
yearly = yearly[yearly["관측일수"] >= 300].sort_values("연도").reset_index(drop=True)

if len(yearly) < 2:
    st.error("회귀 분석을 하기에 데이터가 충분하지 않습니다.")
    st.stop()

# -----------------------------
# 회귀 분석 (선형 회귀)
# -----------------------------
x = yearly["연도"].values
y = yearly["평균기온"].values

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
r_squared = r_value ** 2

start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
n_years = len(yearly)

def predict_temp(year):
    return slope * year + intercept

# -----------------------------
# 화면 레이아웃: 정보 요약
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("사용된 연도 개수", f"{n_years}개")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("상관계수 (r)", f"{r_value:.4f}")

st.divider()

# -----------------------------
# 슬라이더로 연도 선택 & 예측
# -----------------------------
st.subheader("📅 연도를 선택해서 예상 기온을 확인해 보세요")

selected_year = st.slider(
    "연도 선택",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

predicted_temp = predict_temp(selected_year)

# 실제 데이터가 있는 해라면 실제값도 함께 보여주기
actual_row = yearly[yearly["연도"] == selected_year]

st.markdown(
    f"""
    <div style="text-align:center; padding: 30px; background-color:#f0f8ff; border-radius:15px;">
        <h2 style="color:#1f77b4;">📌 {selected_year}년 예상 평균기온</h2>
        <h1 style="font-size:60px; color:#d62728;">{predicted_temp:.2f} °C</h1>
    </div>
    """,
    unsafe_allow_html=True
)

if not actual_row.empty:
    actual_temp = actual_row["평균기온"].values[0]
    st.info(f"✅ {selected_year}년의 실제 관측 평균기온은 **{actual_temp:.2f}°C** 입니다.")
else:
    st.warning("⚠️ 이 연도는 실제 관측 데이터 범위 밖이거나, 자료가 부족해 제외된 해입니다. "
               "회귀 직선을 이용한 예측값만 표시됩니다.")

st.divider()

# -----------------------------
# Plotly 산점도 + 회귀 직선
# -----------------------------
st.subheader("📈 연도별 평균기온 산점도와 회귀 직선")

fig = go.Figure()

# 산점도 (실제 데이터)
fig.add_trace(go.Scatter(
    x=yearly["연도"],
    y=yearly["평균기온"],
    mode="markers",
    name="연도별 평균기온",
    marker=dict(size=8, color="#1f77b4", opacity=0.7)
))

# 회귀 직선 (관측 기간 + 슬라이더로 선택한 연도까지 확장)
line_x_min = min(start_year, selected_year)
line_x_max = max(end_year, selected_year)
line_x = np.linspace(line_x_min, line_x_max, 100)
line_y = predict_temp(line_x)

fig.add_trace(go.Scatter(
    x=line_x,
    y=line_y,
    mode="lines",
    name="회귀 직선",
    line=dict(color="#d62728", width=3, dash="dash")
))

# 선택한 연도의 예측값을 별표로 강조
fig.add_trace(go.Scatter(
    x=[selected_year],
    y=[predicted_temp],
    mode="markers",
    name=f"{selected_year}년 예측값",
    marker=dict(size=18, color="gold", symbol="star",
                line=dict(color="black", width=1))
))

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="closest",
    template="plotly_white",
    height=550,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 회귀식 및 통계 정보
# -----------------------------
st.subheader("🧮 회귀 분석 결과")

eq_col, stat_col = st.columns(2)

with eq_col:
    sign = "+" if intercept >= 0 else "-"
    st.markdown(f"**회귀식:**  \n평균기온 = {slope:.5f} × 연도 {sign} {abs(intercept):.3f}")
    st.markdown(f"**결정계수 (R²):** {r_squared:.4f}")

with stat_col:
    st.markdown(f"**상관계수 (r):** {r_value:.4f}")
    st.markdown(f"**p-value:** {p_value:.4e}")

st.caption(
    f"※ 이 회귀 직선은 {start_year}년부터 {end_year}년까지 총 {n_years}개 연도의 데이터"
    f"(연간 관측일 300일 이상, 2025년까지의 자료만 사용)를 바탕으로 만들어졌습니다."
)
