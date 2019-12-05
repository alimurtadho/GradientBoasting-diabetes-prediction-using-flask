import pickle
import pandas as pd
from flask import Flask, jsonify, request, abort

app = Flask(__name__)

classifier = pickle.load(open('model.pkl','rb'))

@app.route('/predict', methods=['POST'])
def ApiCall():
    if not request.json:
	    abort(400)

    y_train_json = request.json
    y_train = pd.DataFrame(y_train_json,index=[0])

    y_pred = classifier.predict(y_train)    
    return jsonify({'Species':y_pred.item()})

if __name__ == '__main__':
    app.run(debug=True)