import streamlit as st
import pandas as pd
import joblib
import requests
import bz2
import io

# # Function to download file from GitHub Release
# def download_file_from_github(url):
#     response = requests.get(url)
#     if response.status_code == 200:
#         return io.BytesIO(response.content)  # Return as BytesIO object
#     else:
#         raise Exception(f"Failed to download the file, status code: {response.status_code}")

# GitHub Release URL for your model (Make sure it's the .pkl.bz2 file)
model_url = 'https://github.com/xkimberlx/FYP/releases/download/v1.0.0/best_rf_model.pkl.bz2'

# # Download the model file
# model_file = download_file_from_github(model_url)

# Decompress and load the model using Python's built-in bz2 module
with bz2.BZ2File(model_file, 'rb') as f:
    model = joblib.load(f)

# Load the saved scaler (Make sure scaler.pkl is accessible)
scaler = joblib.load('scaler.pkl')

# Custom CSS to change background image
st.markdown(
    """
    <style>
    body {
        background-image: url('https://media.istockphoto.com/id/1329026011/photo/real-estate-growth-chart-3d-illustration.jpg?s=612x612&w=0&k=20&c=pSdRUY0_eRIc5646LB8CUh5YhI-7JFtzgNMAmtr5ufM=');
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        height: 100vh;
        margin: 0;
    }

    .stApp {
        background-color: rgba(255, 255, 255, 0.7);  /* Optional: makes background slightly transparent */
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit Web App title
st.title("Rental Price Prediction")

# Input widgets for the user
property_type = st.selectbox("Select property type", ['Apartment', 'Condominium', 'Duplex', 'Flat', 'Landed House', 'Service Residence', 'Studio', 'Townhouse Condo'])
furnished = st.selectbox("How furnished is the property?", ['Fully Furnished', 'Not Furnished', 'Partially Furnished'])
size_sqft = st.number_input("Enter size (in square feet)", min_value=1, max_value=10000, value=1000)
region = st.radio("Select region", ('Kuala Lumpur', 'Selangor'))
rooms = st.slider("Select number of rooms", 1, 5, value=1)
near_ktm_lrt = st.radio("Is the property near KTM/LRT?", ('No', 'Yes'))
parking = st.radio("Is parking available?", ('No', 'Yes'))
additional_facilities_required = st.radio("Require additional facilities?\n\n(eg. Playground, Barbeque area, Multipurpose hall, Gymnasium, Lift, Sauna, Minimart, Swimming Pool, Security, Tennis Court, Squash Court, Jogging Track, Club house)" , ('No', 'Yes'))

# Convert categorical inputs to numeric
parking = 1 if parking == 'Yes' else 0
near_ktm_lrt = 1 if near_ktm_lrt == 'Yes' else 0
region = 1 if region == 'Kuala Lumpur' else 0
additional_facilities_required = 1 if additional_facilities_required == 'Yes' else 0

# One-hot encoding for furnished status
furnished_mapping = {
    'Fully Furnished': [1, 0, 0],
    'Not Furnished': [0, 1, 0],
    'Partially Furnished': [0, 0, 1]
}
furnished_encoded = furnished_mapping[furnished]

# One-hot encoding for property type
property_type_mapping = {
    'Apartment': [1, 0, 0, 0, 0, 0, 0, 0],
    'Condominium': [0, 1, 0, 0, 0, 0, 0, 0],
    'Duplex': [0, 0, 1, 0, 0, 0, 0, 0],
    'Flat': [0, 0, 0, 1, 0, 0, 0, 0],
    'Landed House': [0, 0, 0, 0, 1, 0, 0, 0],
    'Service Residence': [0, 0, 0, 0, 0, 1, 0, 0],
    'Studio': [0, 0, 0, 0, 0, 0, 1, 0],
    'Townhouse Condo': [0, 0, 0, 0, 0, 0, 0, 1]
}
property_type_encoded = property_type_mapping[property_type]

# Prepare the input data as a DataFrame
input_data = {
    'size_sqft': size_sqft,
    'rooms': rooms,
    'near_ktm_lrt': near_ktm_lrt,
    'parking': parking,
    'region': region,
    'require_facilities': additional_facilities_required,
    'furnished_Fully Furnished': furnished_encoded[0],
    'furnished_Not Furnished': furnished_encoded[1],
    'furnished_Partially Furnished': furnished_encoded[2],
    'property_type_Apartment': property_type_encoded[0],
    'property_type_Condominium': property_type_encoded[1],
    'property_type_Duplex': property_type_encoded[2],
    'property_type_Flat': property_type_encoded[3],
    'property_type_Landed House': property_type_encoded[4],
    'property_type_Service Residence': property_type_encoded[5],
    'property_type_Studio': property_type_encoded[6],
    'property_type_Townhouse Condo': property_type_encoded[7]
}

input_df = pd.DataFrame([input_data])

# Scale numerical features
input_df[['size_sqft', 'rooms']] = scaler.transform(input_df[['size_sqft', 'rooms']])

# Prediction button
if st.button('Predict Rent Price'):
    # Make the prediction
    prediction = model.predict(input_df)
    
    # Display the result
    st.write(f"Predicted Monthly Rent: RM {prediction[0]:,.2f}")
