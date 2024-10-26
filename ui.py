import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from pydantic import BaseModel, model_validator

import seaborn as sns
import cohere
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# Cohere integration for chatbot functionality
def generate_text_with_cohere(prompt, cohere_client):
    response = cohere_client.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=300,
        temperature=0.75,
        k=0,
        stop_sequences=[]
    )
    return response.generations[0].text

cohere_api_key = 'IDlfxdy11paDxht8zQKuLQZkR61dhaPRTwdNzkpF'
co = cohere.Client(cohere_api_key)

template = """
You are a helpful assistant. Please answer the question based on the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

st.set_page_config(page_title="Agri-Commodity Forecast & Chatbot", layout="wide")


st.markdown(
    """
    <style>
    body {
        background-color: #f0f4f8;
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background-color: #2e7d32;
        color: #ffffff;
    }
    h1, h2, h3, h4 {
        color: #1b5e20;
    }
    .stButton>button {
        background-color: #1b5e20;
        color: white;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #388e3c;
    }
    .stTextInput>div>input {
        border: 1px solid #1b5e20;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
class MyModel(BaseModel):
    value: int

@model_validator(mode='before')
    def check_value(cls, values):
        if values.get('value') < 0:
            raise ValueError('Value must be non-negative')
        return values
@st.cache_data
def load_data():
    df = pd.read_csv('commodity.csv')
    return df

data = load_data()

st.sidebar.header("📊 Data Overview")
st.sidebar.dataframe(data, height=300)

st.sidebar.markdown("### Select a Commodity and Column for Forecasting")
specific_commodities = data['Commodity'].unique().tolist()
commodity = st.sidebar.selectbox("Select Commodity", specific_commodities)

st.sidebar.image('/home/murali/Downloads/Untitled.jpeg', use_column_width=False, width=250)

left_col, right_col = st.columns([3, 1])

with left_col:
    st.title("🌾 Agricultural Commodity Forecasting & Chatbot Integration")
    st.markdown("Gain insights into future prices of various agricultural commodities and interact with an AI-powered chatbot for more detailed information.")

    st.image('/home/murali/Downloads/main-image.png', use_column_width=False, width=500)

    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    if not numeric_columns:
        st.error("No numeric columns available for forecasting.")
    else:
        target_column = st.sidebar.selectbox("Select Column to Forecast", numeric_columns)

        st.header(f"📈 {commodity} {target_column} Forecasting")

        commodity_data = data[data['Commodity'] == commodity]
        if commodity_data.empty:
            st.error(f"No valid data available for {commodity}. Please select a different commodity.")
        else:
            commodity_data = commodity_data.dropna(subset=[target_column])

            if commodity_data.empty:
                st.error(f"No valid data available for {commodity} after cleaning. Please select a different commodity.")
            else:
                st.markdown("### ARIMA Model Configuration")
                order = st.selectbox("Select ARIMA Order (p, d, q)", [(1, 1, 1), (2, 1, 2), (1, 0, 1)])

                scaler = MinMaxScaler(feature_range=(0, 1))
                data_scaled = scaler.fit_transform(commodity_data[target_column].values.reshape(-1, 1))
                model = ARIMA(data_scaled, order=order)
                model_fit = model.fit()

                steps = st.slider("Steps to Forecast into the Future", 1, 100, 10)
                forecast = model_fit.forecast(steps=steps)
                forecast = scaler.inverse_transform(forecast.reshape(-1, 1))

                original_values = commodity_data[target_column].values
                forecast_index = range(len(commodity_data), len(commodity_data) + steps)
                forecast_series = pd.Series(forecast.flatten(), index=forecast_index)

                st.subheader(f"🔵 Original {target_column}")

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(commodity_data.index, original_values, label=f'Original {target_column}', color='blue')
                ax.set_xlabel("Index")
                ax.set_ylabel(f"{target_column}")
                ax.set_title(f'{commodity} - Original {target_column}')
                ax.legend()
                st.pyplot(fig)

                st.subheader(f"🔴 Forecasted {target_column}")

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(forecast_series.index, forecast_series, label=f'Forecasted {target_column}', color='red', linestyle='--')
                ax.set_xlabel("Index")
                ax.set_ylabel(f"{target_column}")
                ax.set_title(f'{commodity} - Forecasted {target_column}')
                ax.legend()
                st.pyplot(fig)

                st.markdown(f"### Forecasted {target_column} for Next {steps} Steps")
                st.write(forecast)

                st.markdown("### Distribution of Values with Percentages")
                pie_data = commodity_data[target_column].value_counts()

                fig, ax = plt.subplots(figsize=(8, 8))
                wedges, texts, autotexts = ax.pie(
                    pie_data, labels=pie_data.index, autopct='%1.1f%%',
                    startangle=140, colors=sns.color_palette('Set3'),
                    wedgeprops=dict(width=0.3, edgecolor='w')
                )

                for text in autotexts:
                    text.set_color('white')
                    text.set_weight('bold')
                    text.set_fontsize(12)

                ax.set_title(f'{commodity} - {target_column} Distribution')
                st.pyplot(fig)

                st.markdown("### Enhanced Bar Graph of Values")
                bar_data = commodity_data[target_column].head(10)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=bar_data.index, y=bar_data, palette='Set2', ax=ax)
                ax.set_xlabel("Index")
                ax.set_ylabel(f"{target_column}")
                ax.set_title(f'{commodity} - Top 10 {target_column} Values')
                st.pyplot(fig)

# 1. Supply Chain Management Visualization
with left_col:
    st.header("📦 Supply Chain Management Visualization")
    st.markdown("This section visualizes the supply chain process from production to the consumer.")

    supply_chain = {
        "Farm": "Production",
        "Processor": "Processing",
        "Distributor": "Distribution",
        "Retailer": "Retail",
        "Consumer": "Consumption"
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(list(supply_chain.keys()), list(supply_chain.values()), marker='o', color='green')
    ax.set_xlabel("Supply Chain Stage")
    ax.set_ylabel("Process")
    ax.set_title("Supply Chain Management Visualization")
    st.pyplot(fig)

# 2. Inventory Management
with left_col:
    st.header("📦 Inventory Management")
    st.markdown("Track the current inventory levels for selected commodities.")

    inventory_data = commodity_data.groupby('State')[target_column].sum().reset_index()
    inventory_data.columns = ['State', 'Total Inventory']

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=inventory_data['State'], y=inventory_data['Total Inventory'], palette='coolwarm', ax=ax)
    ax.set_xlabel("State")
    ax.set_ylabel("Total Inventory")
    ax.set_title(f"{commodity} - Total Inventory by State")
    st.pyplot(fig)

# 3. Price Awareness for Consumers
with left_col:
    st.header("📈 Price Awareness for Consumers")
    st.markdown("This section helps consumers stay informed about price changes.")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(commodity_data.index, commodity_data[target_column], color='blue', label=f'{target_column} Prices')
    ax.axhline(commodity_data[target_column].mean(), color='red', linestyle='--', label='Average Price')
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{target_column}")
    ax.set_title(f'{commodity} - {target_column} Price Awareness')
    ax.legend()
    st.pyplot(fig)

    
with right_col:
    st.header("💬 Chat with the Assistant")

    # Extract relevant details from the dataset
    unique_states = data['State'].unique().tolist()
    commodities = data['Commodity'].unique().tolist()
    columns = data.columns.tolist()

    # Provide a summary of the data
    summary = data.describe().to_dict()

    context = f"""
    - Commodities in the dataset: {commodities}
    - States in the dataset: {unique_states}
    - Available columns: {columns}
    - Summary of numeric data: {summary}

    Current data context:
    {data.head().to_dict()}

    Forecasting details:
    For the selected commodity and column, the ARIMA model forecasts future values. The parameters of the model and the forecasted results are displayed in the app.

    Please ask about anything related to this data.
    """

    question = st.text_input("Ask the chatbot a question", placeholder="What would you like to know?")

    if st.button("Get Answer"):
        if question:
            prompt_text = prompt_template.format(context=context, question=question)
            response = generate_text_with_cohere(prompt=prompt_text, cohere_client=co)
            st.write(f"**Chatbot's Answer:** {response}")
        else:
            st.error("Please ask a question.")

