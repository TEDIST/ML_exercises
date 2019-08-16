from sklearn import svm
import numpy as np
np.info(svm.SVC)
X2=0
y2=0
from sklearn import svm
clf2 = svm.SVC(kernel='rbf',gamma=50,C=1.0)
clf2.fit(X2,y2.ravel())
y2_pred = clf2.predict(X2)