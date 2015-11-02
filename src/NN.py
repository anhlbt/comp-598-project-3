__author__ = 'Josh'

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

from sklearn.cross_validation    import train_test_split
from sklearn.feature_selection   import f_classif
from sklearn.feature_selection   import SelectKBest

from numpy.random import multivariate_normal
import data_manager

X,y = data_manager.load_test_data()

#X = SelectKBest(f_classif,k=2304).fit_transform(X,y)


#PYBRAIN STUFF..
alldata = ClassificationDataSet(X.shape[1],1,nb_classes = 10)

#From the documentation, you have to add each example separately...
for example,target in zip(X,y):
    alldata.addSample(example,[target])

#Basic Train test split
tstdata_temp, trndata_temp = alldata.splitWithProportion( 0.25 )

###########################################################
#This is a hack fix for a known pybrain error with polymorphism...
tstdata = ClassificationDataSet(X.shape[1],1,nb_classes = 10)
for n in range(0, tstdata_temp.getLength()):
    tstdata.addSample( tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1] )

trndata = ClassificationDataSet(X.shape[1],1,nb_classes = 10)
for n in range(0, trndata_temp.getLength()):
    trndata.addSample( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )
############################################################

#Convert to have one output Neuron Per class     
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )


print ("Training Size: ", len(trndata))
print ("Features and Classes: ", trndata.indim, trndata.outdim)

#builds feed forward network with 100 hidden neurons, can change the outclass..
fnn = buildNetwork( trndata.indim, 10, trndata.outdim, outclass=SoftmaxLayer )

#Can change the trainer, many are available
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

#number of training iterations
for i in range(100):
    trainer.trainEpochs( 1 )
    trnresult = percentError( trainer.testOnClassData(),trndata['class'] )
    tstresult = percentError( trainer.testOnClassData(dataset=tstdata ), tstdata['class'] )

    print ("epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult)
