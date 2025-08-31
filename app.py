from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Configure for production
app.config['ENV'] = 'production'
app.config['DEBUG'] = False

# Load the trained model
model = None
try:
    with open('kidney.pkl', 'rb') as file:
        model = pickle.load(file)
    print("✅ Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Model has predict method: {hasattr(model, 'predict')}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please ensure the kidney.pkl file exists and is compatible with your scikit-learn version.")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if model is available
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model is not available. Please restart the application.'
        }), 500
    
    try:
        # Get data from form
        data = request.get_json()
        
        # Extract all 24 features from form
        all_features = [
            float(data['age']),
            float(data['bp']),
            float(data['sg']),
            int(data['al']),
            int(data['su']),
            float(data['bgr']),
            float(data['bu']),
            float(data['sc']),
            float(data['sod']),
            float(data['pot']),
            float(data['hemo']),
            int(data['pcv']),
            float(data['rc']),
            int(data['rbc']),
            int(data['pc']),
            int(data['pcc']),
            int(data['ba']),
            int(data['wc']),
            int(data['htn']),
            int(data['dm']),
            int(data['cad']),
            int(data['appet']),
            int(data['pe']),
            int(data['ane'])
        ]
        
        # Extract only the 18 features the model expects
        # Features: age, bp, al, su, rbc, pc, pcc, ba, bgr, bu, sc, pot, wc, htn, dm, cad, pe, ane
        features = [
            all_features[0],   # age
            all_features[1],   # bp
            all_features[3],   # al
            all_features[4],   # su
            all_features[13],  # rbc
            all_features[14],  # pc
            all_features[15],  # pcc
            all_features[16],  # ba
            all_features[5],   # bgr
            all_features[6],   # bu
            all_features[7],   # sc
            all_features[9],   # pot
            all_features[17],  # wc
            all_features[18],  # htn
            all_features[19],  # dm
            all_features[20],  # cad
            all_features[22],  # pe
            all_features[23]   # ane
        ]
        
        # Reshape features for prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
        
        # Get result and confidence
        result = "CHRONIC KIDNEY DISEASE (CKD)" if prediction == 1 else "NO CHRONIC KIDNEY DISEASE"
        confidence = max(probability) * 100
        
        # Prepare response
        response = {
            'success': True,
            'result': result,
            'confidence': round(confidence, 2),
            'prediction': int(prediction),
            'recommendation': "Please consult a nephrologist immediately for proper diagnosis and treatment." if prediction == 1 else "Continue with regular health checkups and maintain a healthy lifestyle."
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
