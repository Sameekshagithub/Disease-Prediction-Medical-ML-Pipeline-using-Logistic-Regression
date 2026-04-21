"""
Disease Prediction Flask App - Port 8080
Medical ML with Logistic Regression
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import json
import os
import traceback

app = Flask(__name__)

# ── Load Model ──────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, 'disease_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    with open(os.path.join(MODEL_DIR, 'model_info.json')) as f:
        info = json.load(f)
    return model, scaler, info

try:
    model, scaler, model_info = load_artifacts()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️  Model not found. Run the Jupyter notebook first. Error: {e}")
    model, scaler, model_info = None, None, {}

FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

FEATURE_RANGES = {
    'Pregnancies':              {'min': 0,    'max': 20,   'step': 1,    'unit': 'count',  'label': 'Pregnancies'},
    'Glucose':                  {'min': 50,   'max': 250,  'step': 1,    'unit': 'mg/dL',  'label': 'Glucose'},
    'BloodPressure':            {'min': 20,   'max': 130,  'step': 1,    'unit': 'mmHg',   'label': 'Blood Pressure'},
    'SkinThickness':            {'min': 0,    'max': 100,  'step': 1,    'unit': 'mm',     'label': 'Skin Thickness'},
    'Insulin':                  {'min': 0,    'max': 850,  'step': 1,    'unit': 'μU/mL',  'label': 'Insulin'},
    'BMI':                      {'min': 10,   'max': 70,   'step': 0.1,  'unit': 'kg/m²',  'label': 'BMI'},
    'DiabetesPedigreeFunction': {'min': 0.05, 'max': 2.5,  'step': 0.01, 'unit': 'score',  'label': 'Diabetes Pedigree'},
    'Age':                      {'min': 18,   'max': 90,   'step': 1,    'unit': 'years',  'label': 'Age'},
}

# ── Routes ───────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html',
                           features=FEATURE_RANGES,
                           model_info=model_info,
                           model_ready=(model is not None))

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please run the Jupyter notebook first.'}), 503

    try:
        data = request.get_json()
        values = [float(data.get(f, 0)) for f in FEATURES]
        arr = np.array(values).reshape(1, -1)
        arr_sc = scaler.transform(arr)

        prob = float(model.predict_proba(arr_sc)[0][1])
        pred = int(model.predict(arr_sc)[0])

        # Feature contributions (coefficients × scaled values)
        coeffs = model_info.get('coefficients', {})
        contributions = {}
        for i, feat in enumerate(FEATURES):
            contributions[feat] = round(float(arr_sc[0][i] * model.coef_[0][i]), 4)

        # Risk level
        if prob < 0.3:
            risk = 'Low'
            risk_color = '#27ae60'
        elif prob < 0.6:
            risk = 'Moderate'
            risk_color = '#f39c12'
        else:
            risk = 'High'
            risk_color = '#e74c3c'

        return jsonify({
            'prediction': pred,
            'probability': round(prob, 4),
            'probability_pct': round(prob * 100, 1),
            'risk_level': risk,
            'risk_color': risk_color,
            'label': 'Diabetic' if pred == 1 else 'Non-Diabetic',
            'contributions': contributions,
            'top_factors': sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        })

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 400

@app.route('/metrics')
def metrics():
    return jsonify(model_info)

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

# ── Run ──────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
