

























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


tstdata, trndata = alldata.splitWithProportion( 0.8 )

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
#trainer.testOnData(verbose=True)
#print(np.array([fnn.activate(x).argmax() for x, _ in tstdata]))

print ("Dim --> ", len(tstdata))
print ("Dim --> ", len(trndata))

wrong_type1 = []
wrong_type2 = []
wrong_type3 = []

count_correct_classification = 0
for t in range(len(tstdata)):
    res = fnn.activate(tstdata.getSample(t)[0]).argmax()

    if int(res) == int(tmp_tst_for_validation.getSample(t)[1]):
        count_correct_classification += 1
        #print ("Here --> ", tmp_tst_for_validation.getSample(t)[0][0] )
    else:
        if (tmp_tst_for_validation.getSample(t)[0][4]==1 or tmp_tst_for_validation.getSample(t)[0][5]==1 or tmp_tst_for_validation.getSample(t)[0][1] == 1 or (tmp_tst_for_validation.getSample(t)[0][4]==tmp_tst_for_validation.getSample(t)[0][5]) or (tmp_tst_for_validation.getSample(t)[0][4]==tmp_tst_for_validation.getSample(t)[0][1]) or (tmp_tst_for_validation.getSample(t)[0][1]==tmp_tst_for_validation.getSample(t)[0][5])):
            print("hello")
        else:

            wrong_type1.append(tmp_tst_for_validation.getSample(t)[0][4])
            wrong_type2.append(tmp_tst_for_validation.getSample(t)[0][5])
            wrong_type3.append(tmp_tst_for_validation.getSample(t)[0][1])


print ("Count Correct Classification --> ", count_correct_classification*100/len(tstdata))


"""

for i in range(20):
    trainer.trainEpochs(10)
    trnresult = percentError(trainer.testOnClassData(),
                             trndata['class'])
    tstresult = percentError(trainer.testOnClassData(
        dataset=tstdata), tstdata['class'])




    print "epoch: %4d" % trainer.totalepochs, \
        "  train error: %5.2f%%" % trnresult, \
        "  test error: %5.2f%%" % tstresult

"""


    #out = fnn.activateOnDataset(griddata)
    #out = out.argmax(axis=1)  # the highest output activation gives the class
    #out = out.reshape(X.shape)








import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt




#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

x =[1,2,3,4,5,6,7,8,9,10]
y =[5,6,2,3,13,4,1,2,4,8]
z =[2,3,3,3,5,7,9,11,9,10]

#print(myclones_data[:, 0])

#ax.scatter(myclones_data[:, 11], myclones_data[:, 12], myclones_data[:, 17], c='r', marker='o')

plt.scatter(wrong_type3, wrong_type1, c='r', marker='o', s=140)

plt.xlabel('Type 2 Similarity', fontsize=18)
plt.ylabel('Type 1 Similarity', fontsize=18)
plt.title('Result Analysis', fontsize=20)

#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')

plt.show()





























#FOR RESULTS.... DO NOT REMOVE



#plt_epoch = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500]
#plt_acc = [84.15178571428571, 85.86309523809524, 84.15178571428571, 84.22619047619048, 85.11904761904762, 84.74702380952381, 84.67261904761905, 85.11904761904762, 85.3422619047619, 84.74702380952381, 85.49107142857143, 81.47321428571428, 85.71428571428571, 85.04464285714286, 85.19345238095238, 84.74702380952381, 84.82142857142857, 84.89583333333333, 85.86309523809524, 85.3422619047619, 85.26785714285714, 82.96130952380952, 82.58928571428572, 85.63988095238095, 85.3422619047619, 84.59821428571429, 84.00297619047619, 85.19345238095238, 85.04464285714286, 85.41666666666667, 85.78869047619048, 83.33333333333333, 82.14285714285714, 85.11904761904762, 84.59821428571429, 82.14285714285714, 85.9375, 85.26785714285714, 77.5297619047619, 84.59821428571429, 84.74702380952381, 85.71428571428571, 84.67261904761905, 80.50595238095238, 85.71428571428571, 82.51488095238095, 84.9702380952381, 85.63988095238095, 85.78869047619048, 85.26785714285714]


"""





plt_epoch = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520,530]
plt_acc = [76.15178571428571, 76.19309523809524, 76.15178571428571, 77.22619047619048, 77.11904761904762, 77.74702380952381, 77.67261904761905, 79.11904761904762, 79.3422619047619, 78.74702380952381, 78.49107142857143, 78.47321428571428, 81.71428571428571, 80.04464285714286, 80.19345238095238, 83.74702380952381, 84.82142857142857, 84.89583333333333, 83.86309523809524, 83.3422619047619, 82.26785714285714, 83.96130952380952, 82.58928571428572, 83.63988095238095, 83.3422619047619, 84.59821428571429, 84.00297619047619, 85.19345238095238, 85.04464285714286, 85.41666666666667, 85.78869047619048, 84.33333333333333, 84.14285714285714, 85.11904761904762, 84.59821428571429, 83.14285714285714, 85.9375, 85.26785714285714, 84.5297619047619, 84.59821428571429, 84.74702380952381, 85.71428571428571, 84.67261904761905, 84.50595238095238, 85.71428571428571, 86.51488095238095, 87.4702380952381, 87.6, 87.6, 87.6,87.5,87.6,87.4]



import matplotlib.pyplot as plt
radius = [10, 20, 30, 40, 50, 60,70,80,90,100, 110, 120]
area = [16.44, 15.03, 15.03, 15.03, 15.33, 16.11, 15.33, 15.03, 15.03, 15.10, 14.96, 14.32]
plt.plot(plt_epoch, plt_acc, marker='o')
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Accuracy %', fontsize=20)
plt.title('Performance Analysis', fontsize=20)
plt.show()

"""