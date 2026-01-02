import pickle
import pandas as pd
import numpy as np
import streamlit as st
import datetime


class Commodity:
    def __init__(self):
       pass

    def load_models(self):
        try:
            with open("models/le_commodity_name.pkl", "rb") as file:
                self.le_commodity_name = pickle.load(file)
            with open("models/le_district.pkl", "rb") as file:
                self.le_district = pickle.load(file)
            with open("models/le_market.pkl", "rb") as file:
                self.le_market = pickle.load(file)
            with open("models/le_state.pkl", "rb") as file:
                self.le_state = pickle.load(file)
            with open("models/model.pkl", "rb") as file:
                self.model = pickle.load(file)
            with open("models/scaler.pkl", "rb") as file:
                self.scaler = pickle.load(file)
            print("Models loaded successfully")
        except Exception as e:
            print(e)

    def predict(self):
        self.commodity_name = self.le_commodity_name.transform([self.commodity_name])
        self.district = self.le_district.transform([self.district])
        self.state = self.le_state.transform([self.state])
        self.market = self.le_market.transform([self.market])

        X = pd.DataFrame(
            {
                "commodity_name": [self.commodity_name[0]],
                "state": [self.state[0]],
                "district": [self.district[0]],
                "market": [self.market[0]],
                "year": [self.year],
                "month": [self.month],
                "date": [self.date],
            }
        )
        y = self.model.predict(X)

        return y

    def load_form(self):
        st.title("Agricultural Commodity Price Prediction")
        self.load_models()
        with st.form("commodity_form"):
            self.commodity_name = st.selectbox("Commodity", self.le_commodity_name.classes_)
            self.state = st.selectbox("State", self.le_state.classes_)
            self.district = st.selectbox("District", self.le_district.classes_)
            self.market = st.selectbox("Market", self.le_market.classes_)

            date = st.date_input(
                "Date",
                min_value=datetime.date(2017, 1, 1),
                max_value=datetime.date(2040, 12, 31),
                format="DD/MM/YYYY",
            )
            
            self.year = date.year
            self.month = date.month
            self.date = date.day
            
            submit = st.form_submit_button('predict')
        
        if submit:
            output = self.predict()
            
            st.write(output)
            
if __name__ == '__main__':
    commodity = Commodity()
    commodity.load_form()
