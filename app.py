from flask import Flask, jsonify, request

from src.pipelines.prediction_pipeline import PredictionPipeline, CustomClass
from src.exception import CustomException
import sys
import logging


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json()
        logging.info("Received prediction request")
        custom_data = CustomClass(
            age=int(json_data.get('age')),
            workclass=int(json_data.get('workclass')),
            education_num=int(json_data.get('education_num')),
            marital_status=int(json_data.get('marital_status')),
            occupation=int(json_data.get('occupation')),
            relationship=int(json_data.get('relationship')), 
            race=int(json_data.get('race')),
            sex=int(json_data.get('sex')),
            capital_gain=int(json_data.get('capital_gain')),
            capital_loss=int(json_data.get('capital_loss')),
            hours_per_week=int(json_data.get('hours_per_week')),
            native_country=int(json_data.get('native_country'))
        )

        data_df = custom_data.get_data_as_dataframe()
        prediction_pipeline = PredictionPipeline()
        preds = prediction_pipeline.predict(data_df)
        output = preds[0]
        logging.info("Prediction successful")
        return jsonify({
            'status': 'success',
            'prediction': output,
            'income': '>50K' if output == 1 else '<=50K'})
    

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5010, debug=True)
