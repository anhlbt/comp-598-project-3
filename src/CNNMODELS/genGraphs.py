import numpy as np
import matplotlib.pyplot as plt

model1csv = 'model1stats.csv'
model2csv = 'model2stats.csv'
model3csv = 'model3stats.csv'

model1np = np.loadtxt(open(model1csv,"rb"),delimiter=",",skiprows=1)
model2np = np.loadtxt(open(model2csv,"rb"),delimiter=",",skiprows=1)
model3np = np.loadtxt(open(model3csv,"rb"),delimiter=",",skiprows=1)


def plot_train_test(train_sizes,train_scores,test_scores):

	x = np.arange(train_sizes)
	plot = plt.figure()
	
	plt.plot(x, train_scores, 'o-', color="r",
	             label="Training loss")
	plt.plot(x, test_scores, 'o-', color="g",
	             label="Validation loss")

	plt.legend(loc="best")

	plt.xlabel('#epochs')
	plt.ylabel('mean cross-entropy loss')

	plt.show()

	return plot

n = model1np.shape[0]
#col1: epoch, col2:trainloss, col3:valloss, col4: valacc
plt1 = plot_train_test(n,model1np[:,1],model1np[:,2])
plt2 = plot_train_test(n,model2np[:,1],model2np[:,2])
plt3 = plot_train_test(n,model3np[:,1],model3np[:,2])

plt1.savefig("traintestcurve1.png", format="png")
plt2.savefig("traintestcurve2.png", format="png")
plt3.savefig("traintestcurve3.png", format="png")

#plot all validation errors
x = np.arange(n)
plot = plt.figure()

plt.plot(x, model1np[:,3], 'o-', color="r",
             label="Model 1 validation accuracy")

plt.plot(x, model2np[:,3], 'o-', color="g",
             label="Model 2 validation accuracy")

plt.plot(x, model3np[:,3], 'o-', color="b",
             label="Model 3 validation accuracy")

plt.legend(loc="best")

plt.xlabel('#epochs')
plt.ylabel('Validation accuracy')

plt.show()

plot.savefig('accuracycurve.png',format="png")