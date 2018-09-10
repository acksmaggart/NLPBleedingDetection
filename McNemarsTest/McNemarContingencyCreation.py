import pandas as pd
import numpy as np

d = pd.DataFrame({"GOLD" : gold, 
                  "RB" : pyConTextPredictions,
                  "ETDS" : etDSPredictions})

rb_positive = d.loc[d.GOLD == 1, "RB"]
etds_positive = d.loc[d.GOLD == 1, "ETDS"]

b = np.sum(np.where(np.logical_and(rb_positive==1, etds_positive==0), 1, 0), dtype=np.float)
c = np.sum(np.where(np.logical_and(rb_positive==0, etds_positive==1), 1, 0), dtype=np.float)

chi_square  = ((b - c) ** 2) / (b + c)
