from flask import Flask , request, jsonify
from joblib import dump, load
import traceback, sys, json
import pandas as pd
from flask_cors import CORS



app = Flask(__name__)


# Enable CORS for all routes
CORS(app)


@app.route('/predict', methods=['POST']) # Your API endpoint URL would consist /predict
def predict():
    if lr:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(lr.predict(query))
            prediction = [int(value) for value in prediction]

            return jsonify({'prediction': prediction})

        except Exception as e:
          app.logger.error("An error occurred: %s", str(e))  # Log the error
          return jsonify({'error': 'An internal server error occurred'}), 500

    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line argument
    except:
        port = 12345 # If you don't provide any port then the port will be set to 12345
    lr = load('rl_model.pkl') # Load "lr_model.pkl"
    print ('Model loaded')
    model_columns = load('lr_model_columns.pkl') # Load "r_lmodel_columns.pkl"
    print ('Model columns loaded')
    app.run(port=port, debug=False)

