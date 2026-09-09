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
# 회귀 분석 함수 (재사용을 위해 함수로 분리)
# -----------------------------
def run_regression(data):
    """연도(x)와 평균기온(y)으로 선형 회귀를 수행하고 결과를 반환하는 함수"""
    x = data["연도"].values
    y = data["평균기온"].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return {
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "p_value": p_value,
        "start_year": int(data["연도"].min()),
        "end_year": int(data["연도"].max()),
        "n_years": len(data),
    }

# 전체 기간 회귀
full_result = run_regression(yearly)

# 최근 20년 회귀 (끝 연도 기준 -19년부터)
recent_end = full_result["end_year"]
recent_start = recent_end - 19
recent_yearly = yearly[yearly["연도"] >= recent_start]

# 최근 20년 데이터도 최소 연도 개수 조건 확인 (2개 미만이면 회귀 불가)
has_recent_data = len(recent_yearly) >= 2
if has_recent_data:
    recent_result = run_regression(recent_yearly)

def predict_temp(result, year):
    return result["slope"] * year + result["intercept"]

def slope_per_100years(slope):
    """연간 기울기를 100년 단위 기온 변화로 변환"""
    return slope * 100

# -----------------------------
# 화면 레이아웃: 전체 기간 정보 요약
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("사용된 연도 개수", f"{full_result['n_years']}개")
col2.metric("시작 연도", f"{full_result['start_year']}년")
col3.metric("끝 연도", f"{full_result['end_year']}년")
col4.metric("상관계수 (r)", f"{full_result['r_value']:.4f}")

st.divider()

# -----------------------------
# 100년당 기온 상승폭 크게 표시 (전체 vs 최근 20년 비교)
# -----------------------------
st.subheader("🌡️ 100년당 기온 상승폭 비교")

full_100 = slope_per_100years(full_result["slope"])
recent_100 = slope_per_100years(recent_result["slope"]) if has_recent_data else None

compare_col1, compare_col2 = st.columns(2)

with compare_col1:
    st.markdown(
        f"""
        <div style="text-align:center; padding: 25px; background-color:#eef6ff; border-radius:15px; border: 2px solid #1f77b4;">
            <h3 style="color:#1f77b4;">📊 전체 기간</h3>
            <p style="font-size:16px; color:gray;">{full_result['start_year']}년 ~ {full_result['end_year']}년 ({full_result['n_years']}개 연도)</p>
            <h1 style="font-size:55px; color:#1f77b4;">{full_100:+.2f} °C</h1>
            <p style="font-size:16px;">100년당 기온 변화</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with compare_col2:
    if has_recent_data:
        st.markdown(
            f"""
            <div style="text-align:center; padding: 25px; background-color:#fff0f0; border-radius:15px; border: 2px solid #d62728;">
                <h3 style="color:#d62728;">🔥 최근 20년</h3>
                <p style="font-size:16px; color:gray;">{recent_result['start_year']}년 ~ {recent_result['end_year']}년 ({recent_result['n_years']}개 연도)</p>
                <h1 style="font-size:55px; color:#d62728;">{recent_100:+.2f} °C</h1>
                <p style="font-size:16px;">100년당 기온 변화</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("최근 20년 구간에는 회귀 분석을 하기에 데이터가 충분하지 않습니다.")

# 두 기울기 비교 코멘트
if has_recent_data:
    diff = recent_100 - full_100
    if diff > 0:
        st.success(f"➡️ 최근 20년의 기온 상승 속도가 전체 기간 평균보다 **{diff:.2f}°C/100년 더 빠릅니다.**")
    elif diff < 0:
        st.info(f"➡️ 최근 20년의 기온 상승 속도가 전체 기간 평균보다 **{abs(diff):.2f}°C/100년 더 느립니다.**")
    else:
        st.info("➡️ 최근 20년과 전체 기간의 기온 상승 속도가 거의 같습니다.")

st.divider()

# -----------------------------
# 슬라이더로 연도 선택 & 예측 (전체 기간 회귀식 기준)
# -----------------------------
st.subheader("📅 연도를 선택해서 예상 기온을 확인해 보세요")

selected_year = st.slider(
    "연도 선택",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

predicted_temp = predict_temp(full_result, selected_year)

# 실제 데이터가 있는 해라면 실제값도 함께 보여주기
actual_row = yearly[yearly["연도"] == selected_year]

st.markdown(
    f"""
    <div style="text-align:center; padding: 30px; background-color:#f0f8ff; border-radius:15px;">
        <h2 style="color:#1f77b4;">📌 {selected_year}년 예상 평균기온 (전체 기간 회귀식 기준)</h2>
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
# Plotly 산점도 + 회귀 직선 (전체 기간 + 최근 20년 함께 표시)
# -----------------------------
st.subheader("📈 연도별 평균기온 산점도와 회귀 직선 비교")

fig = go.Figure()

# 산점도 (전체 데이터)
fig.add_trace(go.Scatter(
    x=yearly["연도"],
    y=yearly["평균기온"],
    mode="markers",
    name="연도별 평균기온",
    marker=dict(size=8, color="#7f7f7f", opacity=0.6)
))

# 최근 20년 데이터는 다른 색으로 강조
if has_recent_data:
    fig.add_trace(go.Scatter(
        x=recent_yearly["연도"],
        y=recent_yearly["평균기온"],
        mode="markers",
        name="최근 20년 데이터",
        marker=dict(size=9, color="#d62728", opacity=0.8)
    ))

# 전체 기간 회귀 직선 (슬라이더 선택 연도까지 확장)
line_x_min = min(full_result["start_year"], selected_year)
line_x_max = max(full_result["end_year"], selected_year)
line_x_full = np.linspace(line_x_min, line_x_max, 100)
line_y_full = predict_temp(full_result, line_x_full)

fig.add_trace(go.Scatter(
    x=line_x_full,
    y=line_y_full,
    mode="lines",
    name="회귀 직선 (전체 기간)",
    line=dict(color="#1f77b4", width=3, dash="dash")
))

# 최근 20년 회귀 직선
if has_recent_data:
    line_x_recent = np.linspace(recent_result["start_year"], recent_result["end_year"], 50)
    line_y_recent = predict_temp(recent_result, line_x_recent)

    fig.add_trace(go.Scatter(
        x=line_x_recent,
        y=line_y_recent,
        mode="lines",
        name="회귀 직선 (최근 20년)",
        line=dict(color="#d62728", width=4)
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
# 회귀식 및 통계 정보 (전체 vs 최근 20년)
# -----------------------------
st.subheader("🧮 회귀 분석 결과 상세 비교")

stat_col1, stat_col2 = st.columns(2)

with stat_col1:
    st.markdown("### 📊 전체 기간")
    sign = "+" if full_result["intercept"] >= 0 else "-"
    st.markdown(f"**회귀식:**  \n평균기온 = {full_result['slope']:.5f} × 연도 {sign} {abs(full_result['intercept']):.3f}")
    st.markdown(f"**상관계수 (r):** {full_result['r_value']:.4f}")
    st.markdown(f"**결정계수 (R²):** {full_result['r_value']**2:.4f}")
    st.markdown(f"**p-value:** {full_result['p_value']:.4e}")
    st.markdown(f"**100년당 기온 변화:** {full_100:+.2f}°C")

with stat_col2:
    st.markdown("### 🔥 최근 20년")
    if has_recent_data:
        sign2 = "+" if recent_result["intercept"] >= 0 else "-"
        st.markdown(f"**회귀식:**  \n평균기온 = {recent_result['slope']:.5f} × 연도 {sign2} {abs(recent_result['intercept']):.3f}")
        st.markdown(f"**상관계수 (r):** {recent_result['r_value']:.4f}")
        st.markdown(f"**결정계수 (R²):** {recent_result['r_value']**2:.4f}")
        st.markdown(f"**p-value:** {recent_result['p_value']:.4e}")
        st.markdown(f"**100년당 기온 변화:** {recent_100:+.2f}°C")
    else:
        st.warning("데이터가 충분하지 않습니다.")

st.caption(
    f"※ 전체 기간 회귀 직선은 {full_result['start_year']}년부터 {full_result['end_year']}년까지 "
    f"총 {full_result['n_years']}개 연도의 데이터(연간 관측일 300일 이상, 2025년까지 자료)를 사용했습니다. "
    f"최근 20년 회귀 직선은 {recent_start}년부터 {recent_end}년까지의 데이터를 사용했습니다."
)
