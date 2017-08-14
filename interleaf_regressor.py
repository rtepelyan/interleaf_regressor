from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import sklearn.multiclass as skmc
import numpy as np

class InterleafRegressor(BaseEstimator, RegressorMixin):
  def __init__(self, baseClassifierClass, baseClassifierParams, lowerBound=0, upperBound=1, numberOfClassifiers=4, classifierResolution=4, n_jobs=1):
    # get the initial arguments into the new class instance
    self.baseClassifierClass = baseClassifierClass
    self.baseClassifierParams = baseClassifierParams
    self.lowerBound = lowerBound
    self.upperBound = upperBound
    self.n_jobs = n_jobs
    if(self.upperBound <= self.lowerBound):
      raise ValueError('Upper bound is not greater than lower bound')
    self.numberOfClassifiers = numberOfClassifiers
    if(self.numberOfClassifiers <= 0):
      raise ValueError('The number of classifiers must be greater than 0')
    self.classifierResolution = classifierResolution
    if(self.classifierResolution <= 0):
      raise ValueError('The classifier resolution must be greater than 0')
    # compute internal components
    # first is the output support
    self.pdfSupportSize = self.numberOfClassifiers*self.classifierResolution
    self.pdfSupportInterval = (self.upperBound-self.lowerBound)/self.pdfSupportSize
    self.boundsInterval = (self.upperBound-self.lowerBound)/self.classifierResolution
    self.pdfSupportVals = np.array([i*self.pdfSupportInterval for i in range(self.pdfSupportSize)])+self.lowerBound+self.pdfSupportInterval*0.5
    # next is the set of classifiers, their corresponding class boundaries, and the overlap of each pdf support segment with each set of bounds as a fraction of the bounds size
    self.classifiers = []
    self.classifierBounds = []
    self.matchingBoundsIndicesAndOverlapFractions = []
    for i in range(self.numberOfClassifiers):
      # ith classifier
      baseClf = self.baseClassifierClass()
      baseClf.set_params(**self.baseClassifierParams)
      self.classifiers.append(skmc.OneVsRestClassifier(baseClf, n_jobs=self.n_jobs))

      # ith set of bounds
      ithBounds = []
      curLowerBound = self.lowerBound - i*self.pdfSupportInterval
      curUpperBound = curLowerBound + self.boundsInterval
      ithBounds.append((max(self.lowerBound, curLowerBound), min(self.upperBound, curUpperBound)))
      while(curUpperBound < self.upperBound):
        curLowerBound += self.boundsInterval
        curUpperBound += self.boundsInterval
        ithBounds.append((max(self.lowerBound, curLowerBound), min(self.upperBound, curUpperBound)))
      self.classifierBounds.append(ithBounds)

      # ith set of overlaps between pdf support segments and bounds
      self.matchingBoundsIndicesAndOverlapFractions.append([])
      for j in range(self.pdfSupportSize):
        self.matchingBoundsIndicesAndOverlapFractions[i].append([])
        curLowerBound = self.lowerBound + j*self.pdfSupportInterval
        curUpperBound = curLowerBound + self.pdfSupportInterval
        for k,b in enumerate(self.classifierBounds[i]):
          overlap = min(curUpperBound, b[1]) - max(curLowerBound, b[0])
          if(overlap < 0):
            continue
          else:
            self.matchingBoundsIndicesAndOverlapFractions[i][j].append((k, overlap/(b[1]-b[0])))

    # all done with the constructor
    return None

  def fit(self, X, y):
    X, y = check_X_y(X, y)
    for yval in y:
      if(yval < self.lowerBound):
        raise ValueError('got a value', yval, ', which is less than the lower bound')
      if(yval > self.upperBound):
        raise ValueError('got a value', yval, ', which is greater than the upper bound')
    for i in range(self.numberOfClassifiers):
      # print('fitting classifier', i+1, 'of', self.numberOfClassifiers)
      clf = self.classifiers[i]
      bounds = self.classifierBounds[i]
      classMatrix = np.zeros((len(y), len(bounds)))
      for j in range(len(y)):
        for k in range(len(bounds)):
          if(y[j] >= bounds[k][0] and ((y[j] < bounds[k][1]) or (k == len(bounds)-1))):
            classMatrix[j][k] = 1
      try:
        clf.fit(X, classMatrix)
      except e:
        print('caught exception while fitting:', e)
        raise e
      del classMatrix
    return self

  def predict_pdf(self, X):
    pdfs = np.zeros((X.shape[0], self.pdfSupportSize))
    for i in range(self.numberOfClassifiers):
      clf = self.classifiers[i]
      pbounds = clf.predict_proba(X)
      pbounds /= pbounds.sum(axis=1).reshape(-1,1)
      for xcounter, xrow in enumerate(pbounds):
        for j in range(self.pdfSupportSize):
          for tup in self.matchingBoundsIndicesAndOverlapFractions[i][j]:
            pdfs[xcounter][j] += tup[1]*xrow[tup[0]]
    for prow in pdfs:
      prow /= np.sum(prow)
    return pdfs

  def predict(self, X, predictionMode='mode'):
    pdfs = self.predict_pdf(X)
    if(predictionMode == 'mode'):
      modeVals = np.zeros((X.shape[0],))
      for xcounter, xrow in enumerate(pdfs):
        maxp = -1
        modeVal = self.lowerBound
        for i,p in enumerate(xrow):
          if(p > maxp):
            maxp = p
            modeVal = self.pdfSupportVals[i]
        modeVals[xcounter] = modeVal
      del pdfs
      return modeVals
    elif(predictionMode == 'mean'):
      meanVals = pdfs.dot(self.pdfSupportVals)
      del pdfs
      return meanVals
    else:
      raise ValueError('prediction mode must be "mode" or "mean"')
    return None
