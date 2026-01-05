import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📈 Efficient Frontier & Portfolio Optimization")

# --- 1. จัดการรายชื่อหุ้น (Session State) ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = ["AAPL", "MSFT", "GOOGL", "AMZN"] # Default หุ้นเทคฯ

# Input รับชื่อหุ้น
with st.sidebar:
    st.header("1. เลือกหุ้นเข้าพอร์ต")
    new_stock = st.text_input("พิมพ์ชื่อหุ้น (เช่น TSLA, NVDA):", key="input_stock").upper()
    if st.button("เพิ่มหุ้น"):
        if new_stock and new_stock not in st.session_state.portfolio:
            st.session_state.portfolio.append(new_stock)
            st.rerun()
    
    # แสดงรายการหุ้นและปุ่มลบ
    st.write("---")
    st.write("📋 รายชื่อหุ้น:")
    for stock in st.session_state.portfolio:
        col_a, col_b = st.columns([4, 1])
        col_a.text(stock)
        if col_b.button("❌", key=f"del_{stock}"):
            st.session_state.portfolio.remove(stock)
            st.rerun()

# --- 2. ฟังก์ชันดึงข้อมูลและคำนวณ (Cached) ---
@st.cache_data
def get_stock_data(tickers):
    if not tickers:
        return pd.DataFrame()
    # ดึงข้อมูลย้อนหลัง 1 ปี
    data = yf.download(tickers, period="1y")['Close']
    return data

# ดึงข้อมูลจริง
if len(st.session_state.portfolio) > 1:
    with st.spinner('กำลังดึงข้อมูลราคาหุ้น...'):
        df = get_stock_data(st.session_state.portfolio)
        
        # คำนวณ Daily Returns
        daily_returns = df.pct_change().dropna()
        # Covariance Matrix (รายปี = 252 วันทำการ)
        cov_matrix = daily_returns.cov() * 252

    # --- 3. ส่วนปรับสัดส่วน (Sliders) ---
    st.subheader("2. กำหนดน้ำหนักการลงทุน (Weight)")
    col_input, col_graph = st.columns([1, 2])
    user_weights = {}
    total_score = 0
    
    with col_input:
        for stock in st.session_state.portfolio:
            score = st.slider(f"น้ำหนัก {stock}", 0, 10, 5, key=f"w_{stock}")
            user_weights[stock] = score
            total_score += score
        
        # แปลง Score เป็น % จริง
        if total_score == 0: total_score = 1 # กัน error หาร 0
        final_weights = np.array([user_weights[s]/total_score for s in st.session_state.portfolio])
        
        # แสดงตารางสรุป
        st.write("---")
        st.write("📊 **สัดส่วนพอร์ตของคุณ:**")
        alloc_df = pd.DataFrame({
            "Stock": st.session_state.portfolio,
            "Weight": [f"{w*100:.2f}%" for w in final_weights]
        })
        st.dataframe(alloc_df, hide_index=True)

    # --- 4. คำนวณและจำลอง Efficient Frontier ---
    with col_graph:
        # คำนวณ Return และ Volatility ของพอร์ตผู้ใช้ (User Portfolio)
        # Expected Return = sum(weight * mean_daily_return * 252)
        user_return = np.sum(daily_returns.mean() * final_weights) * 252
        # Volatility = sqrt(w.T * Cov * w)
        user_volatility = np.sqrt(np.dot(final_weights.T, np.dot(cov_matrix, final_weights)))
        
        # Simulation (Monte Carlo) - จำลอง 3,000 พอร์ตโฟลิโอ
        num_portfolios = 3000
        all_weights = np.zeros((num_portfolios, len(st.session_state.portfolio)))
        ret_arr = np.zeros(num_portfolios)
        vol_arr = np.zeros(num_portfolios)
        sharpe_arr = np.zeros(num_portfolios)

        for i in range(num_portfolios):
            # สุ่มน้ำหนัก
            weights = np.array(np.random.random(len(st.session_state.portfolio)))
            weights = weights / np.sum(weights) # Normalize ให้รวมเป็น 1
            all_weights[i,:] = weights
            
            # คำนวณ Return, Volatility, Sharpe Ratio
            ret_arr[i] = np.sum(daily_returns.mean() * weights) * 252
            vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_arr[i] = ret_arr[i] / vol_arr[i] # สมมติ Risk Free Rate = 0 ง่ายๆ

        # สร้าง Plotly Scatter Plot
        fig = px.scatter(
            x=vol_arr, y=ret_arr, color=sharpe_arr,
            labels={'x': 'Risk (Volatility)', 'y': 'Expected Return', 'color': 'Sharpe Ratio'},
            title='Efficient Frontier (Monte Carlo Simulation)',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(coloraxis_showscale=False)
        fig.update_traces(marker_size=2.5)
        # เพิ่มจุด "พอร์ตของคุณ" (ดาวสีแดง)
        fig.add_trace(go.Scatter(
            x=[user_volatility], y=[user_return],
            mode='markers+text',
            marker=dict(color='red', size=15, symbol='star'),
            name='My Portfolio',
            text=['YOU'], textposition="top center"
        ))

        # หาจุด Max Sharpe (จุดที่ดีที่สุดในทางทฤษฎี)
        max_sharpe_idx = sharpe_arr.argmax()
        fig.add_trace(go.Scatter(
            x=[vol_arr[max_sharpe_idx]], y=[ret_arr[max_sharpe_idx]],
            mode='markers',
            marker=dict(color='orange', size=12, symbol='diamond'),
            name='Max Sharpe (Optimal)'
        ))

        st.plotly_chart(fig, use_container_width=True)
        
        # สรุปผล Performance
        st.success(f"🎯 **ผลลัพธ์พอร์ตของคุณ:** Return: {user_return*100:.2f}% | Risk: {user_volatility*100:.2f}%")
        st.info(f"💎 **พอร์ตที่แนะนำ (Max Sharpe):** Return: {ret_arr[max_sharpe_idx]*100:.2f}% | Risk: {vol_arr[max_sharpe_idx]*100:.2f}%")

else:
    st.warning("⚠️ กรุณาเพิ่มหุ้นอย่างน้อย 2 ตัวเพื่อคำนวณ Efficient Frontier")