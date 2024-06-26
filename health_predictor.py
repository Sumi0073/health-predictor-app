import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify

# Simulated user data
data = {
    'height': [170, 160, 180, 175, 165],
    'weight': [70, 60, 80, 75, 65],
    'age': [30, 25, 40, 35, 28],
    'gender': ['M', 'F', 'M', 'M', 'F'],
    'family_history': ['None', 'Diabetes', 'Heart Disease', 'Cancer', 'Hypertension'],
    'current_symptoms': ['Cough', 'Fever', 'Chest Pain', 'Headache', 'Fatigue'],
    'disease': [0, 1, 1, 0, 1]  # 0 = No disease, 1 = Disease
}

df = pd.DataFrame(data)

# Encode categorical variables
le_gender = LabelEncoder()
le_family_history = LabelEncoder()
le_symptoms = LabelEncoder()

# Fit LabelEncoders with all possible values
df['gender'] = le_gender.fit_transform(df['gender'])
df['family_history'] = le_family_history.fit_transform(df['family_history'])
df['current_symptoms'] = le_symptoms.fit_transform(df['current_symptoms'])

# Basic Exploratory Data Analysis (EDA) using Matplotlib
# Visualizing distributions and correlations
plt.figure(figsize=(12, 6))

# Distribution of age
plt.subplot(1, 2, 1)
plt.hist(df['age'], bins=10, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribution of Age')

# Correlation heatmap
plt.subplot(1, 2, 2)
corr_matrix = df.corr()
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title('Correlation Heatmap')

plt.tight_layout()
plt.show()

# Features and target variable
X = df.drop('disease', axis=1)
y = df['disease']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Create Flask API
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Preprocess input data
    input_data = pd.DataFrame([data])
    input_data['gender'] = le_gender.transform(input_data['gender'])
    input_data['family_history'] = le_family_history.transform(input_data['family_history'])
    input_data['current_symptoms'] = le_symptoms.transform(input_data['current_symptoms'])

    # Predict
    prediction = clf.predict(input_data)
    disease_prediction = 'Disease' if prediction[0] == 1 else 'No Disease'

    return jsonify({'prediction': disease_prediction})

if __name__ == '__main__':
    app.run(debug=True)
