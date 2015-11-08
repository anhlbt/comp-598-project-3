import data_manager
import numpy as np
import lasagne
import theano
import NNLasagne
import theano.tensor as T
import csv
_,_,testX = data_manager.load_test_data()

testX = testX.reshape(-1,1,48,48)
testX = testX.astype(np.float32)
# Prepare Theano variables for inputs and targets
input_var = T.tensor4('inputs')



network = NNLasagne.build_cnn(input_var)
with np.load('model.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)


    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))

    preds = []
    for i in range(testX.shape[0]):
        temp = np.zeros((1,1,48,48),dtype=np.float32)
        temp[0,0,:,:] = testX[i,0,:,:]
        preds.append(int(predict_fn(temp)))



    with open('predictions.csv','w',newline='') as predictions:
            writer = csv.writer(predictions)
            writer.writerow(['Id','Prediction'])
            counter = 1
            for prediction in preds:
                writer.writerow([counter,prediction])
                counter += 1
    
