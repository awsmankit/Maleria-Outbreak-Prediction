import requests
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model and scaler from pickle files
model_filename = 'trained_model.pkl'
scaler_filename = 'scaler.pkl'
label_encoder_filename = 'label_encoder.pkl'

with open(model_filename, 'rb') as file:
    model = pickle.load(file)

with open(scaler_filename, 'rb') as file:
    scaler = pickle.load(file)

# Function to fetch weather data from Open Weather API
def fetch_weather_data(city):
    api_key = 'e71c1740fa171b936f30f5a6659faf10'  # Replace with your Open Weather API key
    base_url = 'http://api.openweathermap.org/data/2.5/weather'

    # Send GET request to Open Weather API
    response = requests.get(base_url, params={'q': city, 'appid': api_key})

    if response.status_code == 200:
        data = response.json()

        # Extract relevant weather features
        weather_data = {
            'maxTemp': data['main']['temp_max'],
            'minTemp': data['main']['temp_min'],
            'avgHumidity': data['main']['humidity'],
            'Rainfall': data['rain']['3h'] if 'rain' in data else 0.0
        }

        return pd.DataFrame(weather_data, index=[0])
    else:
        print('Error:', response.status_code)
        return None

# Get the city name from the user
city_name = input('Enter the city name: ')

# Prompt the user to enter values for 'Positive' and 'pf'
positive = float(input('Enter the value for Positive: '))
pf = float(input('Enter the value for pf: '))

# Fetch weather data for the given city
weather_df = fetch_weather_data(city_name)
if weather_df is not None:
    # Set the 'Positive' and 'pf' values
    weather_df['Positive'] = positive
    weather_df['pf'] = pf

    # Scale the weather data using the saved scaler
    weather_scaled = scaler.transform(weather_df)

    # Make the prediction using the trained model
    outbreak_prediction = model.predict(weather_scaled)

    # Load the label encoder
    with open(label_encoder_filename, 'rb') as file:
        label_encoder = pickle.load(file)

    # Convert the prediction label back to the original class
    outbreak_label = label_encoder.inverse_transform(outbreak_prediction)

    print('Malaria Outbreak Prediction for', city_name, ':', outbreak_label[0])
