import sys

import numpy as np
import pandas as pd
import firebase_admin
import firebase_admin.firestore
from google.cloud import storage

import constants

# Use the application default credentials to initialize app
cred = firebase_admin.credentials.ApplicationDefault()
try:
    firebase_admin.initialize_app(cred, {'projectId': constants.PROJECT_ID_PROD})
except:
    pass

db = firebase_admin.firestore.client()
storage_client = storage.Client()

if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1], header=None)

    bucket = None
    for i in range(len(df)):
        if i % 100 == 0:
            print(f"Checked {i} rows")
        bucket_name, storage_path = df.iloc[i, 0].split('/', 1)
        if not bucket:
            bucket = storage_client.get_bucket(bucket_name)
        silhouette_blob = bucket.get_blob(storage_path)
        if not silhouette_blob or not silhouette_blob.exists():
            print(f"{storage_path} does not exist")
            scanId = storage_path.split('/')[2]
            userId = storage_path.split('/')[1]
            fixedUserId = db.collection('scans').document(scanId).get().to_dict()['userId']
            # replace old path with new path
            storage_path = storage_path.replace(userId, fixedUserId)
            silhouette_blob = bucket.get_blob(storage_path)
            if not silhouette_blob or not silhouette_blob.exists():
                print(f"{storage_path} still does not exist")
                df.drop(i, inplace=True)
                continue
            else:
                print(f"Replaced {userId} with {fixedUserId} for {scanId}")
            df.iloc[i, 0] = '/'.join([bucket_name, storage_path])

    import os
    in_filename = os.path.basename(sys.argv[1])
    out_filename = f"{in_filename.split('.')[0]}_fixed.csv"
    df.to_csv(os.path.join(os.path.dirname(sys.argv[1]), out_filename), index=False, header=None)
    print('Done')


