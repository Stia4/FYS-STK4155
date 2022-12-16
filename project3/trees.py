import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix
import csv
import matplotlib.image as mpimg
import pandas as pd

from DataProcessing import load_data


params = [{'suit':suit, 'num':rank} for suit in range(1,5) for rank in range(1,14)] # All 52 suit+rank combinations

X_train, X_test, t_train, t_test = load_data(params)#, size=170//5)

reshape = lambda x : x.reshape((x.shape[0], x.shape[1]*x.shape[2]))


basetree = DecisionTreeClassifier(criterion="log_loss")
mytree = BaggingClassifier(basetree, n_estimators=50) # max_samples

# fit data
mytree = mytree.fit(reshape(X_train), t_train)

# find score
score = mytree.score(reshape(X_test), t_test)
print(score)


