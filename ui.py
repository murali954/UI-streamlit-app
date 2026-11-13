
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cohere

from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Agri-Commodity Forecast & Chatbot",
    page_icon="🌾",
    layout="wide"
)


# --------------------------------------------------
# 2. CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #f0f4f8;
}

h1, h2, h3, h4 {
    color: #1b5e20;
}

.stButton > button {
    background-color: #1b5e20;
    color: white;
    border-radius: 8px;
}

.stButton > button:hover {
    background-color: #388e3c;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #e8f5e9;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. COHERE CONFIGURATION
# --------------------------------------------------

# Store your API key in .streamlit/secrets.toml
# or set the COHERE_API_KEY environment variable.

try:
    COHERE_API_KEY = st.secrets.get("COHERE_API_KEY", "")
except Exception:
    COHERE_API_KEY = ""

if not COHERE_API_KEY:
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

co = None
COHERE_INIT_ERROR = None

if COHERE_API_KEY:
    try:
        co = cohere.ClientV2(api_key=COHERE_API_KEY)
    except Exception as e:
        COHERE_INIT_ERROR = str(e)


def generate_text_with_cohere(prompt, cohere_client):
    """
    Send a prompt to Cohere using the V2 Chat API and always return
    a plain-text answer. Raises a useful error if Cohere returns no text.
    """
    if cohere_client is None:
        raise ValueError(
            "Cohere client is not initialized. Check your API key and configuration."
        )

    response = cohere_client.chat(
        model="command-a-03-2025",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=500,
        temperature=0.7
    )

    # Cohere V2 returns content as a list of typed content blocks.
    message = getattr(response, "message", None)
    content = getattr(message, "content", None) if message else None

    text_parts = []

    if isinstance(content, str):
        text_parts.append(content)
    elif content:
        for item in content:
            item_text = getattr(item, "text", None)

            # Also support dictionary-shaped response blocks.
            if item_text is None and isinstance(item, dict):
                item_text = item.get("text")

            if isinstance(item_text, str) and item_text.strip():
                text_parts.append(item_text.strip())

    answer = "\\n".join(text_parts).strip()

    if not answer:
        raise RuntimeError(
            "Cohere returned an empty response. Check that the model is available "
            "for your API key and try again."
        )

    return answer


# --------------------------------------------------
# 4. LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("commodity.csv")

    df.columns = df.columns.str.strip()

    return df


try:
    data = load_data()

except FileNotFoundError:
    st.error(
        "commodity.csv not found. Place it in the same folder as ui.py."
    )
    st.stop()

except Exception as e:
    st.error("Unable to load dataset: {}".format(e))
    st.stop()


required_columns = ["Commodity", "State"]

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:
    st.error(
        "Missing required columns: {}".format(missing_columns)
    )
    st.stop()

data["Commodity"] = data["Commodity"].astype(str)
data["State"] = data["State"].astype(str)


# --------------------------------------------------
# 5. SIDEBAR
# --------------------------------------------------

st.sidebar.header("📊 Data Overview")

st.sidebar.dataframe(
    data,
    height=300
)

st.sidebar.markdown("### Select Commodity")

specific_commodities = sorted(
    data["Commodity"].dropna().unique().tolist()
)

commodity = st.sidebar.selectbox(
    "Select Commodity",
    specific_commodities
)


# --------------------------------------------------
# 6. SEASONAL CROP RECOMMENDATIONS
# --------------------------------------------------

season_crops = {

    "Rainy": [
        "Paddy",
        "Soybean",
        "Groundnut",
        "Mango",
        "Turmeric",
        "Black Gram (Urd Beans)",
        "Bottle Gourd",
        "Cauliflower",
        "Cucumber (Kheera)",
        "Green Chilli",
        "Mousambi (Sweet Lime)",
        "Onion",
        "Peas Wet",
        "Tomato",
        "Dry Chillies",
        "Mahua Seed (Hippe Seed)",
        "Mahua",
        "Banana - Green",
        "Bhindi (Ladies Finger)",
        "Bitter Gourd",
        "Brinjal",
        "Cabbage",
        "Capsicum",
        "Chikoos (Sapota)",
        "Coriander (Leaves)",
        "Garlic",
        "Guar",
        "Little Gourd (Kundru)",
        "Tobacco",
        "Arecanut (Betelnut/Supari)"
    ],

    "Winter": [
        "Wheat",
        "Barley",
        "Mustard",
        "Chickpeas (Bengal Gram)",
        "Soyabean",
        "Raddish",
        "Papaya (Raw)",
        "Pumpkin",
        "Plum",
        "Pomegranate",
        "Potato",
        "Kodo Millet (Varagu)"
    ],

    "Summer": [
        "Cotton",
        "Maize",
        "Sorghum",
        "Jowar (Sorghum)",
        "Sesamum (Sesame, Gingelly)",
        "Ground Nut Seed",
        "Sweet Potato",
        "Tinda"
    ]

}

selected_season = st.sidebar.selectbox(
    "Select Season",
    list(season_crops.keys())
)

st.sidebar.markdown(
    "### Recommended Crops for {} Season".format(
        selected_season
    )
)

st.sidebar.write(
    ", ".join(season_crops[selected_season])
)


# --------------------------------------------------
# 7. HEADER AND LAYOUT
# --------------------------------------------------

left_col, right_col = st.columns([3, 1])

with left_col:

    st.title(
        "🌾 Agricultural Commodity Forecasting & Chatbot Integration"
    )

    st.markdown(
        "Gain insights into future prices of agricultural "
        "commodities and interact with an AI-powered chatbot."
    )

    if os.path.exists("main-image.png"):
        st.image("main-image.png", width=500)

    if os.path.exists("Untitled.jpeg"):
        st.sidebar.image("Untitled.jpeg", width=250)


# --------------------------------------------------
# 8. COMMODITY FORECASTING
# --------------------------------------------------

numeric_columns = data.select_dtypes(
    include=[np.number]
).columns.tolist()

commodity_data = data[
    data["Commodity"] == commodity
].copy()

forecast_values = None
target_column = None
steps = None

with left_col:

    if not numeric_columns:

        st.error("No numeric columns available for forecasting.")

    elif commodity_data.empty:

        st.error("No data available for selected commodity.")

    else:

        target_column = st.sidebar.selectbox(
            "Select Column to Forecast",
            numeric_columns
        )

        commodity_data[target_column] = pd.to_numeric(
            commodity_data[target_column],
            errors="coerce"
        )

        commodity_data = commodity_data.dropna(
            subset=[target_column]
        )

        st.header(
            "📈 {} {} Forecasting".format(
                commodity,
                target_column
            )
        )

        if len(commodity_data) < 10:

            st.warning(
                "At least 10 valid observations are recommended "
                "for forecasting."
            )

        else:

            st.markdown("### ARIMA Model Configuration")

            order = st.selectbox(
                "Select ARIMA Order (p, d, q)",
                [
                    (1, 1, 1),
                    (2, 1, 2),
                    (1, 0, 1)
                ]
            )

            steps = st.slider(
                "Steps to Forecast into the Future",
                1,
                100,
                10
            )

            if st.button("Generate Forecast"):

                try:

                    scaler = MinMaxScaler(
                        feature_range=(0, 1)
                    )

                    original_values = commodity_data[
                        target_column
                    ].values

                    data_scaled = scaler.fit_transform(
                        original_values.reshape(-1, 1)
                    ).flatten()

                    model = ARIMA(
                        data_scaled,
                        order=order
                    )

                    model_fit = model.fit()

                    forecast = model_fit.forecast(
                        steps=steps
                    )

                    forecast = np.asarray(
                        forecast
                    ).reshape(-1, 1)

                    forecast = scaler.inverse_transform(
                        forecast
                    )

                    forecast_values = forecast.flatten()

                    st.session_state["forecast_values"] = (
                        forecast_values
                    )

                    st.session_state["forecast_column"] = (
                        target_column
                    )

                    st.session_state["forecast_commodity"] = (
                        commodity
                    )

                    st.success(
                        "Forecast generated successfully!"
                    )

                except Exception as e:

                    st.error(
                        "Forecasting error: {}".format(e)
                    )

            # Display stored forecast

            if (
                "forecast_values" in st.session_state
                and st.session_state.get("forecast_commodity") == commodity
                and st.session_state.get("forecast_column") == target_column
            ):

                forecast_values = st.session_state[
                    "forecast_values"
                ]

                original_values = commodity_data[
                    target_column
                ].values

                forecast_index = range(
                    len(commodity_data),
                    len(commodity_data) + len(forecast_values)
                )

                forecast_series = pd.Series(
                    forecast_values,
                    index=forecast_index
                )

                # Original graph

                st.subheader(
                    "🔵 Original {}".format(target_column)
                )

                fig, ax = plt.subplots(figsize=(10, 5))

                ax.plot(
                    commodity_data.index,
                    original_values,
                    label="Original {}".format(target_column),
                    color="blue"
                )

                ax.set_xlabel("Index")
                ax.set_ylabel(target_column)
                ax.set_title(
                    "{} - Original {}".format(
                        commodity,
                        target_column
                    )
                )

                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

                # Forecast graph

                st.subheader(
                    "🔴 Forecasted {}".format(target_column)
                )

                fig, ax = plt.subplots(figsize=(10, 5))

                ax.plot(
                    forecast_series.index,
                    forecast_series.values,
                    label="Forecasted {}".format(target_column),
                    color="red",
                    linestyle="--"
                )

                ax.set_xlabel("Index")
                ax.set_ylabel(target_column)

                ax.set_title(
                    "{} - Forecasted {}".format(
                        commodity,
                        target_column
                    )
                )

                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

                st.markdown(
                    "### Forecasted {} for Next {} Steps".format(
                        target_column,
                        len(forecast_values)
                    )
                )

                st.dataframe(
                    pd.DataFrame({
                        "Forecast Step": range(
                            1,
                            len(forecast_values) + 1
                        ),
                        "Predicted Value": forecast_values
                    })
                )

                # Distribution graph

                st.markdown(
                    "### Distribution of Values with Percentages"
                )

                pie_data = commodity_data[
                    target_column
                ].value_counts().head(8)

                fig, ax = plt.subplots(figsize=(8, 8))

                ax.pie(
                    pie_data.values,
                    labels=pie_data.index,
                    autopct="%1.1f%%",
                    startangle=140,
                    colors=sns.color_palette("Set3"),
                    wedgeprops={
                        "width": 0.3,
                        "edgecolor": "w"
                    }
                )

                ax.set_title(
                    "{} - {} Distribution".format(
                        commodity,
                        target_column
                    )
                )

                st.pyplot(fig)
                plt.close(fig)

                # Bar graph

                st.markdown(
                    "### Enhanced Bar Graph of Values"
                )

                bar_data = commodity_data[
                    target_column
                ].head(10)

                fig, ax = plt.subplots(figsize=(10, 6))

                sns.barplot(
                    x=bar_data.index.astype(str),
                    y=bar_data.values,
                    palette="Set2",
                    ax=ax
                )

                ax.set_xlabel("Index")
                ax.set_ylabel(target_column)

                ax.set_title(
                    "{} - Top 10 {} Values".format(
                        commodity,
                        target_column
                    )
                )

                st.pyplot(fig)
                plt.close(fig)


# --------------------------------------------------
# 9. SUPPLY CHAIN MANAGEMENT
# --------------------------------------------------

with left_col:

    st.header(
        "📦 Supply Chain Management Visualization"
    )

    st.markdown(
        "This section visualizes the supply chain process "
        "from production to the consumer."
    )

    supply_chain = {
        "Farm": "Production",
        "Processor": "Processing",
        "Distributor": "Distribution",
        "Retailer": "Retail",
        "Consumer": "Consumption"
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        list(supply_chain.keys()),
        list(supply_chain.values()),
        marker="o",
        color="green"
    )

    ax.set_xlabel("Supply Chain Stage")
    ax.set_ylabel("Process")
    ax.set_title("Supply Chain Management Visualization")

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# 10. INVENTORY MANAGEMENT
# --------------------------------------------------

with left_col:

    st.header("📦 Inventory Management")

    st.markdown(
        "Track current inventory levels for selected commodities."
    )

    if not commodity_data.empty and target_column:

        inventory_data = (
            commodity_data.groupby("State")[target_column]
            .sum()
            .reset_index()
        )

        inventory_data.columns = [
            "State",
            "Total Inventory"
        ]

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.barplot(
            data=inventory_data,
            x="State",
            y="Total Inventory",
            palette="coolwarm",
            ax=ax
        )

        ax.set_xlabel("State")
        ax.set_ylabel("Total Inventory")
        ax.set_title(
            "{} - Total Inventory by State".format(
                commodity
            )
        )

        plt.xticks(rotation=45)

        st.pyplot(fig)
        plt.close(fig)

    else:

        st.info("Inventory data is not available.")


# --------------------------------------------------
# 11. PRICE AWARENESS
# --------------------------------------------------

with left_col:

    st.header(
        "📈 Price Awareness for Consumers"
    )

    st.markdown(
        "This section helps consumers stay informed "
        "about commodity price changes."
    )

    if not commodity_data.empty and target_column:

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            commodity_data.index,
            commodity_data[target_column],
            color="blue",
            label="{} Prices".format(target_column)
        )

        average_price = commodity_data[
            target_column
        ].mean()

        ax.axhline(
            average_price,
            color="red",
            linestyle="--",
            label="Average Price"
        )

        ax.set_xlabel("Time / Index")
        ax.set_ylabel(target_column)

        ax.set_title(
            "{} - {} Price Awareness".format(
                commodity,
                target_column
            )
        )

        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

        st.metric(
            "Average {}".format(target_column),
            "{:.2f}".format(average_price)
        )


# --------------------------------------------------
# 12. COHERE AI CHATBOT
# --------------------------------------------------

with right_col:

    st.header("💬 Chat with the Assistant")

    st.markdown(
        "Ask questions about agricultural commodities, "
        "prices, states, and forecasting."
    )

    if co is None:
        if COHERE_INIT_ERROR:
            st.error("Unable to initialize Cohere: {}".format(COHERE_INIT_ERROR))
        else:
            st.warning(
                "Cohere API key is missing. Configure your API key "
                "to enable the chatbot."
            )

    # Prepare dataset context

    unique_states = data["State"].dropna().unique().tolist()

    commodities = data[
        "Commodity"
    ].dropna().unique().tolist()

    columns = data.columns.tolist()

    summary = data.describe(
        include="all"
    ).to_string()

    sample_data = data.head(10).to_string(
        index=False
    )

    context = """
Agricultural Commodity Dataset Information:

Commodities:
{commodities}

States:
{states}

Available Columns:
{columns}

Dataset Summary:
{summary}

Sample Dataset:
{sample_data}

Selected Commodity:
{commodity}

Forecasting Information:
The application uses an ARIMA model to forecast
future values of the selected commodity.

Answer questions based on the supplied dataset.
If information is unavailable, clearly mention that.
Do not invent prices, statistics, or predictions.
""".format(
        commodities=commodities,
        states=unique_states,
        columns=columns,
        summary=summary,
        sample_data=sample_data,
        commodity=commodity
    )

    question = st.text_input(
        "Ask the chatbot a question",
        placeholder="What would you like to know?"
    )

    if st.button("Get Answer"):

        if not question.strip():

            st.error("Please enter a question.")

        elif co is None:

            st.error(
                "Please configure your Cohere API key first."
            )

        else:

            prompt_text = """
You are a helpful agricultural assistant.

Answer the user's question using the provided
agricultural dataset context.

Context:
{context}

Question:
{question}

Provide a clear, accurate and useful answer.
""".format(
                context=context,
                question=question
            )

            try:

                with st.spinner(
                    "Generating answer using Cohere AI..."
                ):

                    response = generate_text_with_cohere(
                        prompt=prompt_text,
                        cohere_client=co
                    )

                if not isinstance(response, str) or not response.strip():
                    raise RuntimeError("The chatbot returned an empty answer.")

                st.markdown("### Chatbot's Answer")
                st.markdown(response)

            except Exception as e:

                st.error(
                    "Cohere API Error: {}".format(e)
                )


# --------------------------------------------------
# 13. FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">
        <p>🌾 Agricultural Commodity Forecasting System</p>
        <p>ARIMA Forecasting | Data Visualization | Cohere AI Chatbot</p>
    </div>
    """,
    unsafe_allow_html=True
)
