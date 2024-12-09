import streamlit as st
import pandas as pd
import joblib  # For saving and loading the model
from sklearn.pipeline import Pipeline

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv('standardized_locations_dataset.csv')

# Add weighted features based on proximity categories
def add_weighted_features(data):
    data['Weighted_Beachfront'] = (data['beach_proximity'] == 'Beachfront').astype(int) * 2.5
    data['Weighted_Seaview'] = (data['beach_proximity'] == 'Sea view').astype(int) * 2.0
    data['Weighted_Lakefront'] = (data['lake_proximity'] == 'Lakefront').astype(int) * 1.8
    data['Weighted_Lakeview'] = (data['lake_proximity'] == 'Lake view').astype(int) * 1.5
    return data

# Calculate mean price per cent per location (to avoid data leakage)
def add_location_mean_price(data, target_col='Price', area_col='Area', location_col='Location'):
    data['Price_per_cent'] = data[target_col] / data[area_col]
    mean_price_per_location = (
        data.groupby(location_col)['Price_per_cent']
        .mean()
        .rename("Mean_Price_per_Cent")
        .reset_index()
    )
    data = pd.merge(data, mean_price_per_location, on=location_col, how='left')
    return data.drop(columns=['Price_per_cent'])

# Predict function
def predict_price(pipeline, training_data, area, location, beach_proximity, lake_proximity, density):
    # Map proximity inputs to weights
    beach_weights = {'Inland': 0, 'Sea view': 2.0, 'Beachfront': 2.5}
    lake_weights = {'Inland': 0, 'Lake view': 1.5, 'Lakefront': 1.8}
    price_per_cent_mean = training_data.loc[training_data['Location'] == location, 'Price'].sum() / \
                          training_data.loc[training_data['Location'] == location, 'Area'].sum()

    if pd.isna(price_per_cent_mean):
        raise ValueError(f"No training data found for location '{location}' to calculate Mean_Price_per_Cent.")

    # Calculate weights
    weighted_beachfront = beach_weights[beach_proximity]
    weighted_seaview = beach_weights[beach_proximity]
    weighted_lakefront = lake_weights[lake_proximity]
    weighted_lakeview = lake_weights[lake_proximity] * 0.75
    area_density = area * (1 if density == 'High' else 0)

    # Create input DataFrame
    input_data = pd.DataFrame([{
        'Location': location,
        'Area': area,
        'Mean_Price_per_Cent': price_per_cent_mean,
        'Weighted_Beachfront': weighted_beachfront,
        'Weighted_Seaview': weighted_seaview,
        'Weighted_Lakefront': weighted_lakefront,
        'Weighted_Lakeview': weighted_lakeview,
        'Area_Density': area_density,
        'density': density
    }])

    # Predict
    return pipeline.predict(input_data)[0]

# Load data and model
data = load_data()
pipeline = joblib.load('price_prediction_pipeline.pkl')  # Save your pipeline earlier using joblib.dump()

# Streamlit UI
st.title("Real Estate Price Predictor")
st.write("Predict the price of plots based on features like location, proximity to amenities, and area.")

area = st.number_input("Enter the area in cents:", min_value=1.0, step=0.1)
location = st.selectbox("Select the location:", options=data['Location'].unique())
beach_proximity = st.selectbox("Select beach proximity:", options=['Inland', 'Sea view', 'Beachfront'])
lake_proximity = st.selectbox("Select lake proximity:", options=['Inland', 'Lake view', 'Lakefront'])
density = st.selectbox("Select density:", options=['Low', 'High'])

if st.button("Predict Price"):
    try:
        predicted_price = predict_price(pipeline, data, area, location, beach_proximity, lake_proximity, density)
        st.success(f"Predicted Price for the plot: ₹{predicted_price:,.2f}")
    except ValueError as e:
        st.error(str(e))
