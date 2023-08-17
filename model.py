import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
Dataset = 'outbreak_detect.xlsx'
df = pd.read_excel(Dataset)

# Impute missing values using mean (for numerical variables)
imputer = SimpleImputer(strategy='mean')
df['Rainfall'] = imputer.fit_transform(df[['Rainfall']])

# Label Encoding
label_encoder = LabelEncoder()
df['Outbreak'] = label_encoder.fit_transform(df['Outbreak'])

# Split the data into features (X) and target variable (y)
X = df.drop('Outbreak', axis=1)
y = df['Outbreak']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling and Normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a random forest classifier
model = RandomForestClassifier()

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy:", accuracy)

# Save the trained model and scaler to files using pickle
model_filename = 'trained_model.pkl'
scaler_filename = 'scaler.pkl'
label_encoder_filename = 'label_encoder.pkl'

with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler, file)

with open(label_encoder_filename, 'wb') as file:
    pickle.dump(label_encoder, file)

print("Model and scaler saved as", model_filename, "and", scaler_filename, "and", label_encoder_filename)
