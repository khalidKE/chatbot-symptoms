import streamlit as st
import pandas as pd
from math import radians, sin, cos,  , atan2

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("recommend.csv")

df = load_data()

st.title("Doctor Recommendation System ")

# Dictionary of cities + coordinates
locations = {
    "Cairo": (30.0444, 31.2357),
    "Giza": (30.0131, 31.2089),
    "Alexandria": (31.2001, 29.9187),
    "Mansoura": (31.0409, 31.3785),
    "Tanta": (30.7865, 31.0004),
    "Assiut": (27.1800, 31.1837),
    "Zagazig": (30.5877, 31.5020),
    "Sohag": (26.5560, 31.6948),
    "Ismailia": (30.5965, 32.2715),
    "Fayoum": (29.3084, 30.8429)
}

# Specialty selection
specialties = df["specialty"].unique()
selected_specialty = st.selectbox("Select Specialty:", specialties)

# Patient location selection
st.subheader("Set Your Location:")
selected_city = st.selectbox("Select City:", list(locations.keys()))
user_location = locations[selected_city]

# Function to calculate distance (Haversine formula)
def haversine(coord1, coord2):
    R = 6371  # Earth's radius in km
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# Filter doctors by specialty
filtered = df[df["specialty"] == selected_specialty].copy()

# Calculate distance for each doctor
filtered["distance_km"] = filtered.apply(
    lambda row: haversine(user_location, (row["latitude"], row["longitude"])),
    axis=1
)

# Sort by highest rating, then closest distance
filtered = filtered.sort_values(by=["rating", "distance_km"], ascending=[False, True])

# Display top 10 doctors
st.subheader("Top Recommended Doctors:")
st.dataframe(filtered[["doctor_name", "specialty", "location", "rating", "experience_years", "num_reviews", "distance_km"]].head(10))
