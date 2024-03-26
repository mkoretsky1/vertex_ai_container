import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
import sklearn
import pickle as pkl
import json

app = Flask(__name__)

# model path
pkl_path = 'model/neuroChip_AFR_NG_hg38_updatedIDs_callrate_ancestry_umap_linearsvc_ancestry_model.pkl'

# load model
pkl_in = open(pkl_path, 'rb')
pipe_clf = pkl.load(pkl_in)
pkl_in.close()

# prediction function
@app.route('/predict', methods=['POST','GET'])
def predict():
    # get instances from json
    input = request.json.get('instances')

    # predict
    prediction = pipe_clf.predict(input)

    int_pred = []

    # get list of predictions as int
    for pred in prediction:
        int_pred.append(int(pred))

    # create output json
    output = {
                'predictions': 
                    int_pred
             }

    return jsonify(output)

# health function
@app.route('/healthz')
def healtz():
    return 'OK'

if __name__=='__main__':
    app.run(host='0.0.0.0')

