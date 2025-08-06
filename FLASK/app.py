from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the saved model
with open(r'C:\Users\seglu\OneDrive\Desktop\Churn_prediction_model\FLASK\model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # ❗Corrected: 'index.htm' ➜ 'index.html'

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        # Handle JSON API request
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({'prediction': int(prediction[0])})
    else:
        # Handle HTML form submission
        form_values = [
            request.form['location'],
            request.form['account_type'],
            int(request.form['age']),
            int(request.form['login_frequency']),
            int(request.form['total_spent']),
            int(request.form['customer_service_interaction_count']),
            int(request.form['registration_date_year']),
            int(request.form['registration_date_month']),
            int(request.form['registration_date_day']),
            1 if request.form['gender'] == 'male' else 0  # example encoding
        ]
        features = np.array(form_values).reshape(1, -1)
        prediction = model.predict(features)
        return render_template('index.html', prediction=int(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
