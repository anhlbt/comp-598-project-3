import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

with open('y_valid.pickle', 'rb') as input_file:
	y_valid = pickle.load(input_file)

with open('preds1.pickle', 'rb') as input_file:
	preds1 = pickle.load(input_file)

with open('preds2.pickle', 'rb') as input_file:
	preds2 = pickle.load(input_file)

with open('preds3.pickle', 'rb') as input_file:
	preds3 = pickle.load(input_file)



def plot_confusion_matrix(cm, title='CNN Confusion matrix', cmap=plt.cm.jet):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(10)
    target_names = ['0','1','2','3','4','5','6','7','8','9']
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



np.set_printoptions(precision=4)

cm1 = confusion_matrix(y_valid,preds1)
cm2 = confusion_matrix(y_valid,preds2)
cm3 = confusion_matrix(y_valid,preds3)

print('Confusion matrix')
#print(cm1)
fig1 = plt.figure()
plot_confusion_matrix(cm1, title='Confusion matrix 1')

plt.show()
fig1.savefig("confmat1.png", format="png")

# Normalize the confusion matrix by row (i.e by the number of samples

cm_normalized1 = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
#print(cm_normalized1)
fig1n = plt.figure()
plot_confusion_matrix(cm_normalized1, title='Normalized confusion matrix 1')

plt.show()
fig1n.savefig("confmat1n.png", format="png")

print('Confusion matrix')
#print(cm2)
fig2 = plt.figure()
plot_confusion_matrix(cm2, title='Confusion matrix 2')

plt.show()
fig2.savefig("confmat2.png", format="png")

# Normalize the confusion matrix by row (i.e by the number of samples

cm_normalized2 = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
#print(cm_normalized2)
fig2n = plt.figure()
plot_confusion_matrix(cm_normalized2, title='Normalized confusion matrix 2')

plt.show()
fig2n.savefig("confmat2n.png", format="png")

print('Confusion matrix')
#print(cm3)
fig3 = plt.figure()
plot_confusion_matrix(cm3, title='Confusion matrix 3')

plt.show()
fig3.savefig("confmat3.png", format="png")

# Normalize the confusion matrix by row (i.e by the number of samples

cm_normalized3 = cm3.astype('float') / cm3.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
#print(cm_normalized3)
fig3n = plt.figure()
plot_confusion_matrix(cm_normalized3, title='Normalized confusion matrix 3')

plt.show()
fig3n.savefig("confmat3n.png", format="png")