import os
import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics

from google.cloud import aiplatform

reference_pca_path = 'sample_data/reference_pcs.csv'
projected_pca_path = 'sample_data/GP2_QC_round3_S4_projected_new_pca.txt'

reference_pca = pd.read_csv(reference_pca_path)
labels = np.array(reference_pca[['label']]).flatten()
le = preprocessing.LabelEncoder()
encoded_labels = le.fit_transform(labels)

projected_pca = pd.read_csv(projected_pca_path, sep='\s+')
# samp_labels = np.array(projected_pca[['label']]).flatten()
# encoded_samp_labels = le.transform(samp_labels)
projected_pca = projected_pca.drop(columns=['FID','IID','label'], axis=1)

num_splits = round((projected_pca.shape[0] / 2000), 0)

projected_pca_arr = np.array(projected_pca)

project = 'genotools'
location = 'us-central1'
endpoint = '9131351709702946816'

aiplatform.init(project=project, location=location)
endpoint = aiplatform.Endpoint(endpoint)

full_pred = []

if num_splits > 0:

    for arr in np.array_split(projected_pca_arr, num_splits):
        arr = arr.tolist()
        prediction = endpoint.predict(instances=arr)
        pred = prediction.predictions
        pred = [int(i) for i in pred]

        full_pred += pred

else:
    arr = projected_pca_arr.tolist()
    prediction = endpoint.predict(instances=arr)
    pred = prediction.predictions
    pred = [int(i) for i in pred]
    full_pred += pred

predictions = pd.Series(le.inverse_transform(full_pred))
print(predictions.value_counts())

# score = metrics.balanced_accuracy_score(encoded_samp_labels, full_pred)
# print(score)
