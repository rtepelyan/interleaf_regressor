import interleaf_regressor
import numpy as np
import sklearn.metrics as skmet
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

clfParams = {
  'early_stopping':True,
  'tol':1e-4,
  'max_iter':200,
  'activation':'tanh',
  'hidden_layer_sizes':(200,20,10),
}

# this example is somewhat forced, see the readme for a description of the use case for interleaf regression
numRows = 10000
Xdata = np.random.random((numRows, 2))
Ydata = np.zeros((numRows,))
for xcounter, xrow in enumerate(Xdata):
  Ydata[xcounter] = xrow[0]*3 + xrow[1]*2 - xrow[0]*xrow[1] + np.sin(xrow[0]*2*np.pi) + np.cos(xrow[1]*2*np.pi) - 0.5
Ydata -= min(Ydata)
Ydata /= max(Ydata)

X_train, X_test, y_train, y_test = train_test_split(Xdata, Ydata, test_size=0.25, random_state=0)

print('creating interleaf_regressor')
reg = interleaf_regressor.InterleafRegressor(MLPClassifier, clfParams, lowerBound=0, upperBound=1, numberOfClassifiers=64, classifierResolution=2, n_jobs=1)

print('fitting')
reg.fit(X_train, y_train)

print('evaluating')
y_pred_mode = reg.predict(X_test, predictionMode='mode')
y_pred_mean = reg.predict(X_test, predictionMode='mean')

print('using mode predictions:')
print('exp var score is', skmet.explained_variance_score(y_test, y_pred_mode))
print('mean abs err is', skmet.mean_absolute_error(y_test, y_pred_mode))
print('rmse is', np.sqrt(skmet.mean_squared_error(y_test, y_pred_mode)))
print('median abs err is', skmet.median_absolute_error(y_test, y_pred_mode))
print('r2 is', skmet.r2_score(y_test, y_pred_mode))

print('\nusing mean predictions:')
print('exp var score is', skmet.explained_variance_score(y_test, y_pred_mean))
print('mean abs err is', skmet.mean_absolute_error(y_test, y_pred_mean))
print('rmse is', np.sqrt(skmet.mean_squared_error(y_test, y_pred_mean)))
print('median abs err is', skmet.median_absolute_error(y_test, y_pred_mean))
print('r2 is', skmet.r2_score(y_test, y_pred_mean))
