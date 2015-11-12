import numpy as np
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

X_DATA = np.load('../data/rotatedTrimedX.npz')['arr_0']
y_DATA = np.load('../data/rotatedTrimedY.npz')['arr_0']

y_DATA = y_DATA.astype(np.int32)

X_DATA -= X_DATA.mean()
X_DATA /= X_DATA.std()
X_DATA = X_DATA.reshape(-1,1,38,38)
X_DATA = X_DATA.astype(np.float32)

with open('convnet1D.pickle', 'rb') as input_file:
	model1 = pickle.load(input_file)

with open('convnet2D.pickle', 'rb') as input_file:
	model2 = pickle.load(input_file)

with open('convnet3D.pickle', 'rb') as input_file:
	model3 = pickle.load(input_file)

X_train,X_valid,y_train,y_valid = train_test_split(X_DATA,y_DATA,test_size=.2,random_state=42)

preds1 = model1.predict(X_valid)
preds2 = model2.predict(X_valid)
preds3 = model3.predict(X_valid)

score1 = accuracy_score(y_valid,preds1)
score2 = accuracy_score(y_valid,preds2)
score3 = accuracy_score(y_valid,preds3)

print (score1)
print (score2)
print (score3)

with open('preds1.pickle', 'wb') as f:
    pickle.dump(preds1, f, -1)
with open('preds2.pickle', 'wb') as f:
    pickle.dump(preds2, f, -1)
with open('preds3.pickle', 'wb') as f:
    pickle.dump(preds3, f, -1)

with open('y_valid.pickle', 'wb') as f:
    pickle.dump(y_valid, f, -1)
