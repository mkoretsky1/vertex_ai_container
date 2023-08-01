import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
import sklearn
import pickle as pkl
import json

app = Flask(__name__)

pkl_path = 'model/GP2_merge_APRIL_2023_ready_genotools_callrate_sex_ancestry_umap_linearsvc_ancestry_model.pkl'
pkl_in = open(pkl_path, 'rb')
pipe_clf = pkl.load(pkl_in)
pkl_in.close()

@app.route('/predict', methods=['POST','GET'])
def predict():
    input = request.json.get('instances')

    prediction = pipe_clf.predict(input)

    int_pred = []

    for pred in prediction:
        int_pred.append(int(pred))

    output = {
                'predictions': 
                    int_pred
             }

    return jsonify(output)

@app.route('/healthz')
def healtz():
    return 'OK'

if __name__=='__main__':
    app.run(host='0.0.0.0')

