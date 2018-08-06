# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# load data
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv('Datasets/col_deleted.csv')
array = dataframe.values

print (array.dtype)


#[7,8,11,12,13,14,22,23]
X = array[:, 0:28].astype(float)



#X = array[:,0:8]
Y = array[:,28].astype(float)

print (Y.dtype)


print (X.dtype)
# feature extraction
test = SelectKBest(score_func=chi2, k=8)
fit = test.fit(X, Y)
# summarize scores
numpy.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:10,:])