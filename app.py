import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")
st.title("🚗 Car Price Prediction System")
st.write("Enter the car's specifications to predict its price (MSRP, US dollars).")
st.caption("Model: Random Forest, selected by 10-fold cross validation on 11,199 cars (typical error about $15,400).")

MARKET_CATEGORIES = ['Unknown', 'Crossover', 'Flex Fuel', 'Luxury', 'Luxury,Performance', 'Hatchback', 'Performance', 'Crossover,Luxury', 'Luxury,High-Performance', 'Exotic,High-Performance', 'Hatchback,Performance', 'Factory Tuner,Luxury,High-Performance', 'High-Performance', 'Hybrid', 'Crossover,Luxury,Performance', 'Factory Tuner,High-Performance', 'Factory Tuner,Performance', 'Flex Fuel,Performance', 'Diesel', 'Exotic,Luxury,High-Performance', 'Crossover,Hatchback', 'Hatchback,Hybrid', 'Crossover,Performance', 'Crossover,Flex Fuel', 'Exotic,Factory Tuner,Luxury,High-Performance', 'Luxury,Hybrid', 'Diesel,Luxury', 'Hatchback,Luxury', 'Crossover,Hybrid', 'Flex Fuel,Luxury', 'Hatchback,Luxury,Performance', 'Exotic,Luxury,Performance', 'Crossover,Luxury,Diesel', 'Flex Fuel,Luxury,High-Performance', 'Factory Tuner,Luxury,Performance', 'Flex Fuel,Luxury,Performance', 'Crossover,Factory Tuner,Luxury,High-Performance', 'Crossover,Luxury,Hybrid', 'Hatchback,Factory Tuner,Performance', 'Exotic,Factory Tuner,High-Performance', 'Flex Fuel,Diesel', 'Hatchback,Diesel', 'Exotic,Flex Fuel,Factory Tuner,Luxury,High-Performance', 'Hatchback,Factory Tuner,High-Performance', 'Luxury,High-Performance,Hybrid', 'Exotic,Luxury', 'Luxury,Performance,Hybrid', 'Exotic,Flex Fuel,Luxury,High-Performance', 'Crossover,Flex Fuel,Luxury', 'Exotic,Performance', 'Hatchback,Factory Tuner,Luxury,Performance', 'Crossover,Luxury,High-Performance', 'Hatchback,Flex Fuel', 'Crossover,Diesel', 'Crossover,Hatchback,Luxury', 'Crossover,Flex Fuel,Luxury,Performance', 'Crossover,Flex Fuel,Performance', 'Crossover,Hatchback,Factory Tuner,Performance', 'Crossover,Hatchback,Performance', 'Crossover,Factory Tuner,Luxury,Performance', 'Crossover,Factory Tuner,Performance', 'Exotic,Factory Tuner,Luxury,Performance', 'Hatchback,Luxury,Hybrid', 'Crossover,Luxury,Performance,Hybrid', 'Factory Tuner,Luxury', 'Flex Fuel,Hybrid', 'Flex Fuel,Performance,Hybrid', 'Crossover,Exotic,Luxury,High-Performance', 'Crossover,Exotic,Luxury,Performance', 'Exotic,Luxury,High-Performance,Hybrid', 'Flex Fuel,Factory Tuner,Luxury,High-Performance', 'Performance,Hybrid']
VEHICLE_STYLES = ['Sedan', 'Coupe', 'Convertible', 'Wagon', '4dr Hatchback', '2dr Hatchback', '4dr SUV', '2dr SUV', 'Convertible SUV', 'Passenger Minivan', 'Cargo Minivan', 'Crew Cab Pickup', 'Regular Cab Pickup', 'Extended Cab Pickup', 'Cargo Van', 'Passenger Van']

@st.cache_resource
def load_model():
    return joblib.load("car_price_model.pkl")

model = load_model()

col1, col2 = st.columns(2)
with col1:
    engine_hp = st.number_input("Engine HP", 50, 1100, 300, 5)
    year = st.number_input("Year", 1990, 2017, 2015, 1)
    cylinders = st.number_input("Engine Cylinders", 0, 16, 6, 1)
    popularity = st.number_input("Popularity", 1, 6000, 1385, 1)
with col2:
    city_mpg = st.number_input("City MPG", 5, 140, 19, 1)
    highway_mpg = st.number_input("Highway MPG", 10, 360, 27, 1)
    market = st.selectbox("Market Category", MARKET_CATEGORIES, index=MARKET_CATEGORIES.index("Luxury"))
    style = st.selectbox("Vehicle Style", VEHICLE_STYLES, index=0)

if st.button("Predict Price"):
    row = pd.DataFrame([{
        "Engine HP": engine_hp, "Year": year, "Engine Cylinders": cylinders,
        "city mpg": city_mpg, "highway MPG": highway_mpg, "Popularity": popularity,
        "Market Category": market, "Vehicle Style": style,
    }])
    price = model.predict(row)[0]
    st.success(f"Predicted price (MSRP): ${price:,.2f}")
