import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


st.set_page_config(page_title="Crypto Correlation Analyzer", layout="wide")


st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
        }
        .stSelectbox {
            margin-bottom: 20px;
        }
        .css-1d391kg {
            padding: 2rem 1rem;
        }
        .stProgress .st-bo {
            background-color: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)


st.title("🚀 Cryptocurrency Correlation Analyzer")
st.markdown("---")


col1, col2 = st.columns([1, 3])

with col1:
    st.sidebar.header("📊 Analysis Parameters")
    
   
    st.sidebar.subheader("📅 Date Range")
    start_date = st.sidebar.date_input(
        "Start Date",
        datetime.now() - timedelta(days=365)
    )
    end_date = st.sidebar.date_input(
        "End Date",
        datetime.now()
    )

 
    crypto_list = [
        'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD',
        'SOL-USD', 'DOT-USD', 'DOGE-USD', 'AVAX-USD', 'MATIC-USD'
    ]
    selected_cryptos = st.sidebar.multiselect(
        "🪙 Select Cryptocurrencies",
        crypto_list,
        default=['BTC-USD', 'ETH-USD', 'BNB-USD']
    )

  
    analysis_type = st.sidebar.selectbox(
        "📈 Select Analysis Type",
        ["Price Movement", "Correlation Analysis", "Returns Distribution", "Rolling Correlation"]
    )

@st.cache_data
def fetch_crypto_data(symbols, start_date, end_date):
    data = pd.DataFrame()
    with st.spinner('Fetching cryptocurrency data...'):
        for symbol in symbols:
            try:
                crypto_data = yf.download(symbol, start=start_date, end=end_date)['Adj Close']
                data[symbol] = crypto_data
            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {str(e)}")
    return data


if len(selected_cryptos) > 0:
    data = fetch_crypto_data(selected_cryptos, start_date, end_date)

    if not data.empty:
        if analysis_type == "Price Movement":
            st.subheader("📈 Price Movement Analysis")
            
            normalized_data = data / data.iloc[0] * 100
            fig = go.Figure()
            for crypto in selected_cryptos:
                fig.add_trace(go.Scatter(
                    x=normalized_data.index,
                    y=normalized_data[crypto],
                    name=crypto,
                    mode='lines'
                ))
            
            fig.update_layout(
                title="Normalized Price Movement (%)",
                xaxis_title="Date",
                yaxis_title="Normalized Price",
                hovermode='x unified',
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("📊 Basic Statistics")
            stats_df = data.describe()
            st.dataframe(stats_df.style.highlight_max(axis=1))
            
        elif analysis_type == "Correlation Analysis":
            st.subheader("🔄 Correlation Analysis")
            correlation_matrix = data.corr()
            
            fig = px.imshow(
                correlation_matrix,
                labels=dict(color="Correlation"),
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            fig.update_layout(
                title="Correlation Matrix",
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_type == "Returns Distribution":
            st.subheader("📊 Returns Distribution Analysis")
            
            returns = data.pct_change().dropna()
            fig = make_subplots(rows=len(selected_cryptos), cols=1,
                              subplot_titles=[f"{crypto} Daily Returns Distribution" 
                                            for crypto in selected_cryptos])
            
            for i, crypto in enumerate(selected_cryptos, 1):
                fig.add_trace(
                    go.Histogram(x=returns[crypto], name=crypto,
                               nbinsx=50, histnorm='probability'),
                    row=i, col=1
                )
            
            fig.update_layout(
                height=300*len(selected_cryptos),
                showlegend=False,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("📈 Returns Statistics")
            returns_stats = returns.describe()
            st.dataframe(returns_stats.style.highlight_max(axis=1))
            
        elif analysis_type == "Rolling Correlation":
            st.subheader("🔄 Rolling Correlation Analysis")
            
            window_size = st.slider(
                "Select Rolling Window Size (days)", 
                min_value=7, 
                max_value=180, 
                value=30,
                help="Adjust the window size to see how correlations change over time"
            )
            
            if len(selected_cryptos) >= 2:
                fig = go.Figure()
                
                for i in range(len(selected_cryptos)):
                    for j in range(i+1, len(selected_cryptos)):
                        rolling_corr = data[selected_cryptos[i]].rolling(
                            window=window_size
                        ).corr(data[selected_cryptos[j]])
                        
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=rolling_corr,
                            name=f'{selected_cryptos[i]} vs {selected_cryptos[j]}',
                            mode='lines'
                        ))
                
                fig.update_layout(
                    title=f"{window_size}-Day Rolling Correlation",
                    xaxis_title="Date",
                    yaxis_title="Correlation Coefficient",
                    hovermode='x unified',
                    template='plotly_white',
                    yaxis_range=[-1, 1]
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least two cryptocurrencies for correlation analysis.")
        
        st.markdown("---")
        st.subheader("💾 Download Data")
        csv = data.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="crypto_data.csv",
            mime="text/csv",
        )
else:
    st.warning("Please select at least one cryptocurrency to analyze .")

st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Created with ❤️ in Python and Streamlit</p>
    </div>
""", unsafe_allow_html=True)