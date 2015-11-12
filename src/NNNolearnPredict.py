import pickle
import csv
import numpy as np
import data_manager


X_valid = data_manager.load_valid_data()
X_valid -= X_valid.mean()
X_valid /= X_valid.std()
X_valid  = X_valid.reshape(-1,1,38,38)
X_valid = X_valid.astype(np.float32)
with open('./CNNMODELS/convnet.pickle', 'rb') as input_file:
	model = pickle.load(input_file)

preds = model.predict(X_valid)

with open('predictions.csv','w',newline='') as predictions:
        writer = csv.writer(predictions)
        writer.writerow(['Id','Prediction'])
        counter = 1
        for prediction in preds:
            writer.writerow([counter,prediction])
            counter += 1
