


import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp























import numpy
import scipy
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
import pandas as pd
import numpy as np


#pybrain module imports.
import sys
sys.path.append('/home/golam/pybrain/')
import pybrain
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer






def load_training_dataSet(fileName):
    data = pd.read_csv(fileName, sep=',', header=None)
    #data.columns = ["state", "outcome"]
    return data

myclones_data = load_training_dataSet('Datasets/new_dataset_with_new_features.csv')
myclones_data = myclones_data.values


inputDim = 8;


means = [(-1,0),(2,4),(3,1)]
cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
alldata = ClassificationDataSet(inputDim, 1, nb_classes=2)


#input = np.array([ myclones_data[n][16], myclones_data[n][17], myclones_data[n][18], myclones_data[n][15],myclones_data[n][11],myclones_data[n][12],   myclones_data[n][26], myclones_data[n][27]] )

for n in xrange(len(myclones_data)):
    #for klass in range(3):
    input = np.array(
        [myclones_data[n][16], myclones_data[n][17], myclones_data[n][18], myclones_data[n][15], myclones_data[n][11],
         myclones_data[n][12], myclones_data[n][26], myclones_data[n][27]])
    #print (n, "-->", input)
    alldata.addSample(input, int(myclones_data[n][35]))


tstdata, trndata = alldata.splitWithProportion( 0.85 )

print("Class Label --> ", int(tstdata.getSample(1)[1]))

tmp_tst_for_validation = tstdata



tstdata_new = ClassificationDataSet(inputDim, 1, nb_classes=2)
for n in xrange(0, tstdata.getLength()):
    tstdata_new.addSample( tstdata.getSample(n)[0], tstdata.getSample(n)[1] )

trndata_new = ClassificationDataSet(inputDim, 1, nb_classes=2)
for n in xrange(0, trndata.getLength()):
    trndata_new.addSample( trndata.getSample(n)[0], trndata.getSample(n)[1])

trndata = trndata_new
tstdata = tstdata_new

#print("Before --> ", trndata)

trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

#print("After --> ", trndata)

#print "Number of training patterns: ", len(trndata)
#print "Input and output dimensions: ", trndata.indim, trndata.outdim
#print "First sample (input, target, class):"
#print trndata['input'][0], trndata['target'][0], trndata['class'][0]


fnn = buildNetwork( trndata.indim, 107, trndata.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1,learningrate=0.05 , verbose=True, weightdecay=0.001)





trainer.trainEpochs(1)
trainer.testOnData(tstdata, verbose=True)
alg_score =  np.array([fnn.activate(x) for x, _ in tstdata])


print ("Printing Test Data: ")
print (alg_score[:, 0])

#print (tmp_tst_for_validation)

alg_y_test = []
for test_index in range(len(tmp_tst_for_validation)):
    alg_y_test.append(int(tmp_tst_for_validation.getSample(test_index)[1]) )



alg_y = label_binarize(alg_y_test, classes=[0, 1, 2])




print ("Dim --> ", len(tstdata))
print ("Dim --> ", len(trndata))























































import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

print ("Binarize : ",y)

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]




# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

print ("y score --> ", y_score)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(alg_y[:, i], alg_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print ("run i ", i)


# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(alg_y[:, 0:2].ravel(), alg_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])







plt.figure()
lw = 5
plt.plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='ROC curve (area = 0.87)')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('ROC - Auto Clone Validator', fontsize=20)
plt.legend(loc="lower right")
plt.show()
