import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import streamlit as st

# Setting up the page

st.set_page_config(page_title="Ai Stock Predictor",layout="centered")
st.title("AI Stock Price Predictor")
st.write("Enter any stock of your choice and our model will predict whether the price of the stock increases or decreases tomorrow")
st.write("Let us help you make more informed investments")

# User input

ticker = st.text_input("Enter a Stock ticker of your choice (eg: AMZN,AAPL etc.) : ")

# When button is clicked

if st.button("Predict"):
    try:
        data=yf.ticker(ticker).history(period="max")
        if data.empty:
            st.error("Invalid ticker or no available data. Please try again")
        else:
            # Cleaning the data
            del data["Dividends"]
            del data["Stock Splits"]
            data["Tomorrow"] = data["Close"].shift(-1)
            data["Target"] = (data["Tomorrow"]>data["Close"]).astype("int")
            data = data.loc["1990-01-01"].copy()

            # Splitting the data
            train = data.iloc[ :-100]
            test = data.iloc[-100: ]
            predictors = ["Open","Close","High","Low","Volume"]

            # Training the model
            model = RandomForestClassifier(n_estimators=100,min_samples_split=100,random_state=1)
            model.fit(train[predictors],train["Target"])

            # Testing the model
            preds = model.predict(test[predictors])
            precision = precision_score(test["Target"],preds)

            # Making the final prediction
            latest_data = data.iloc[-1][predictors].values.reshape(1,-1)
            tomorrow_pred = model.predict(latest_data)[0]
            direction = "⬆️ UP (1)" if tomorrow_pred == 1 else "⬇️ DOWN (0)"

            # Displaying the results
            st.subheader(f"Results for {ticker}")
            st.metric(label="AI prediction for tomorrow's price",value=direction)
            st.write(f"**Model Precision:** {precision:.2f}")

            # Chart
            st.subheader("Stock Closing Price Chart")
            st.line_chart(data["Close"])




    except Exception as e:
        st.error(f"An error occurred : {e}")