import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

d = pd.DataFrame({"GOLD" : gold, 
                  "RB" : pyConTextPredictions,
                  "ETDS" : etDSPredictions})

d['RB'] = np.where(d.GOLD == d.RB, 1, 0).astype('int64')
d['ETDS_AGREE'] = np.where(d.GOLD == d.ETDS, 1, 0).astype('int64')


contingency = confusion_matrix(d.PYCONTEXT_AGREE, d.ETDS_AGREE)
print(contingency)
