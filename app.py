from flask import Flask, render_template, request
import requests
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load the trained model and scaler from pickle files
model_filename = 'trained_model.pkl'
scaler_filename = 'scaler.pkl'
label_encoder_filename = 'label_encoder.pkl'

with open(model_filename, 'rb') as file:
    model = pickle.load(file)

with open(scaler_filename, 'rb') as file:
    scaler = pickle.load(file)

with open(label_encoder_filename, 'rb') as file:
    label_encoder = pickle.load(file)

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


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        city_name = request.form['city_name']
        positive = float(request.form['positive'])
        pf = float(request.form['pf'])

        weather_df = fetch_weather_data(city_name)
        if weather_df is not None:
            weather_df['Positive'] = positive
            weather_df['pf'] = pf

            weather_scaled = scaler.transform(weather_df)

            outbreak_prediction = model.predict(weather_scaled)
            outbreak_label = label_encoder.inverse_transform(outbreak_prediction)

        return render_template('index.html', prediction=True, city=city_name, outbreak=outbreak_label[0])
    return render_template('index.html', prediction=False)


if __name__ == '__main__':
    app.run(debug=True)
