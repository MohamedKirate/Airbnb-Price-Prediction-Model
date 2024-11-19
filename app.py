from flask import Flask, render_template, request, jsonify
import pandas as pd
from src.pipeline.prediction import prediction as predict_function 

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('prediction.html')
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            neighbourhood_group_cleansed = data.get('neighbourhood')
            room_type = data.get('room_type')
            accommodates = data.get('accommodates')
            bathrooms = data.get('bathrooms')
            availability_365 = data.get('availability_365')
            bedrooms = data.get('bedrooms')
            minimum_nights= data.get('minimum_nights')
            beds = data.get('beds')

            input_data = pd.DataFrame([{
                'neighbourhood_group_cleansed': neighbourhood_group_cleansed,
                'room_type': room_type,
                'accommodates': accommodates,
                'bathrooms': bathrooms,
                'availability_365': availability_365,
                'bedrooms': bedrooms,
                'minimum_nights':minimum_nights,
                'beds': beds
            }])

            input_data['text'] = input_data.select_dtypes('O').apply(
                lambda row: ' '.join(row.values), axis=1
            )
            input_data['text'] = input_data['text'].str.replace('/', ' ')
            input_data = input_data.drop(columns=['neighbourhood_group_cleansed', 'room_type'], axis=1)

            y_predict = predict_function(input_data)

            return jsonify({'success': True, 'prediction': round(float(y_predict[0]),2)})
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')



