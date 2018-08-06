# System wise validation of code clones
# Created on: 9 Oct, 2017
# Copyright: Golam Mostaeen






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


inputDim = 3;


means = [(-1,0),(2,4),(3,1)]
cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
alldata = ClassificationDataSet(inputDim, 1, nb_classes=2)


#input = np.array([ myclones_data[n][16], myclones_data[n][17], myclones_data[n][18], myclones_data[n][15],myclones_data[n][11],myclones_data[n][12],   myclones_data[n][26], myclones_data[n][27]] )


#15 ==>type1simtok
#16 ==>type3simtok
#18 ==>type2simtok


for n in xrange(len(myclones_data)):
    #for klass in range(3):
    input = np.array(
        #[myclones_data[n][16], myclones_data[n][18], myclones_data[n][15], myclones_data[n][26], myclones_data[n][27]])
        [myclones_data[n][16], myclones_data[n][18], myclones_data[n][15]])
    #print (n, "-->", input)
    alldata.addSample(input, int(myclones_data[n][35]))


tstdata, trndata = alldata.splitWithProportion( 0.001 )

print "Printing Test Data:"
print(tstdata)

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

print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]


fnn = buildNetwork( trndata.indim, 107, trndata.outdim, bias=True,  outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1,learningrate=0.05 , verbose=True, weightdecay=0.001)



"""
ticks = arange(-3.,6.,0.2)
X, Y = meshgrid(ticks, ticks)
# need column vectors in dataset, not arrays
griddata = ClassificationDataSet(7,1, nb_classes=2)
for i in xrange(X.size):
    griddata.addSample([X.ravel()[i],Y.ravel()[i]], [0])
griddata._convertToOneOfMany()  # this is still needed to make the fnn feel comfy

"""



#trainer.trainEpochs(1)
#trainer.testOnData(verbose=True)
#print(np.array([fnn.activate(x) for x, _ in tstdata]))





for i in range(1):
    trainer.trainEpochs(10)
    trnresult = percentError(trainer.testOnClassData(),
                             trndata['class'])
    tstresult = percentError(trainer.testOnClassData(
        dataset=tstdata), tstdata['class'])




    print "epoch: %4d" % trainer.totalepochs, \
        "  train error: %5.2f%%" % trnresult, \
        "  test error: %5.2f%%" % tstresult





def getManualValidationResult(fileName, cloneID):
    userValidatedFile = load_training_dataSet(fileName)
    userValidatedFile = userValidatedFile.values
    validation_res = 'NOT_FOUND'

    for i in xrange(len(userValidatedFile)):
        if int(userValidatedFile[i][1]) == int(cloneID):
            validation_res = userValidatedFile[i][2]
            break

    return validation_res







def getManualValidationResult2(dir, fileCount, cloneID):

    for i in range(0, fileCount):
        userValidatedFile = load_training_dataSet(dir + 'Set_' +str(i)+'.clones.res')
        userValidatedFile = userValidatedFile.values
        validation_res = 'NOT_FOUND'

        for i in xrange(len(userValidatedFile)):
            if int(userValidatedFile[i][1]) == int(cloneID):
                validation_res = userValidatedFile[i][2]
                return validation_res

    return validation_res








#RESULT VALIDATION FOR SMALLER SYSTEM AND INDV USERS
my_validation_data = load_training_dataSet('Datasets/System_wise_clone_features/System_99.csv')
my_validation_data = my_validation_data.values
my_validation_data = my_validation_data[1:len(my_validation_data)-1,:]



correct_prediction_count = 0
total_sample_count =0
false_pos_count = 0

user_TP = 0

ML_TP = 0
ML_TN = 0
ML_FP = 0
ML_FN = 0

undecided_count = 0


#4 ==>type1simtok
#5 ==>type2simtok
#6 ==>type3simtok


for n in xrange(len(my_validation_data)):
   res = fnn.activate([my_validation_data[n][6],  my_validation_data[n][5], my_validation_data[n][4] ])

   #print res
   #print "clone id--> ",my_validation_data[n][0]


   predict_val = 'false'
   if res[1] >= res [0]:
            predict_val='true'

   #manual_validation_res = getManualValidationResult('Datasets/User_Validation_Results/System53/Set_0.clones.res', my_validation_data[n][0])
   manual_validation_res = getManualValidationResult2('Datasets/User_Validation_Results/System99/', 2, my_validation_data[n][0])

   #print predict_val
   #print manual_validation_res

   manual_validation_res = str(manual_validation_res).lower()



   if manual_validation_res != 'undecided' and manual_validation_res != 'not_found':
       total_sample_count = total_sample_count + 1
       if predict_val == manual_validation_res:
            correct_prediction_count = correct_prediction_count +1

       #User count on true positive clones
       if  manual_validation_res == 'true':
            user_TP = user_TP + 1


       #ML
       if manual_validation_res == 'true'  and predict_val == 'true':
            ML_TP = ML_TP + 1

       elif manual_validation_res == 'false' and predict_val == 'false':
           ML_TN = ML_TN + 1

       elif manual_validation_res == 'false' and predict_val == 'true':
           ML_FP = ML_FP + 1

       elif manual_validation_res == 'true' and predict_val == 'false':
           ML_FN = ML_FN + 1

       else:
           print manual_validation_res, " -------------------> " , predict_val



print "Total Samples ", total_sample_count
print "Correct count ", correct_prediction_count
print "Accuracy: ", correct_prediction_count*100/total_sample_count, "%"

#print "User TP : ", user_TP
print "ML TP : ", ML_TP
print "ML FP : ", ML_FP
print "ML FN : ", ML_FN
print "ML TN : ", ML_TN


print 'SUM: ', ML_TP + ML_FP + ML_FN + ML_TN


print "TPR: ", ML_TP / (ML_TP + ML_FN)


"""



correct_prediction_count_f = 0
false_count = 0
for f in xrange(len(myclones_data)/2):

   if (int(myclones_data[f][35]) == 0):
       false_count = false_count + 1
       res_f =fnn.activate([  myclones_data[f][16], myclones_data[f][18], myclones_data[f][15] ])
       predict_val_f = 0
       if res_f[1] < res_f[0]:
                predict_val_f=1
       if predict_val_f == 1:
           correct_prediction_count_f = correct_prediction_count_f +1

print "Total Samples ", false_count
print "Correct coount ", correct_prediction_count_f
print "Accuracy: ", correct_prediction_count_f * 100 / false_count

"""

#print fnn.activate([.7,.2,.3,.4,.5])

    #out = fnn.activateOnDataset(griddata)
    #out = out.argmax(axis=1)  # the highest output activation gives the class
    #out = out.reshape(X.shape)

"""

    figure(1)
    ioff()  # interactive graphics off
    clf()  # clear the plot
    hold(True)  # overplot on
    for c in [0, 1, 2]:
        here, _ = where(tstdata['class'] == c)
        plot(tstdata['input'][here, 0], tstdata['input'][here, 1], 'o')
    if out.max() != out.min():  # safety check against flat field
        contourf(X, Y, out)  # plot the contour
    ion()  # interactive graphics on
    draw()  # update the plot

ioff()
show()
"""








####################################################
"""
import matplotlib.pyplot as plt
import numpy as np

# fake up some data
spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
data = np.concatenate((spread, center, flier_high, flier_low), 0)

# basic plot
plt.boxplot(data)

# notched plot
plt.figure()
plt.boxplot(data, 1)

# change outlier point symbols
plt.figure()
plt.boxplot(data, 0, 'gD')

# don't show outlier points
plt.figure()
plt.boxplot(data, 0, '')

# horizontal boxes
plt.figure()
plt.boxplot(data, 0, 'rs', 0)

# change whisker length
plt.figure()
plt.boxplot(data, 0, 'rs', 0, 0.75)

# fake up some more data
spread = np.random.rand(50) * 100
center = np.ones(25) * 40
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
d2 = np.concatenate((spread, center, flier_high, flier_low), 0)
data.shape = (-1, 1)
d2.shape = (-1, 1)
# data = concatenate( (data, d2), 1 )
# Making a 2-D array only works if all the columns are the
# same length.  If they are not, then use a list instead.
# This is actually more efficient because boxplot converts
# a 2-D array into a list of vectors internally anyway.
data = [data, d2, d2[::2, 0]]
# multiple box plots on one figure
plt.figure()
plt.boxplot(data)

plt.show()

"""
















