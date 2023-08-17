import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed
from typing import List
from sklearn import metrics
from transformers import EvalPrediction


class Metric(object):
    """
    所有评价指标的父类
    """
    def __call__(self, p: EvalPrediction):
        """
            Evaluation output (always contains labels), to be used to compute metrics.has to return a dictionary string to float.

            Parameters:
                predictions (`np.ndarray`): Predictions of the model.
                label_ids (`np.ndarray`): Targets to be matched.
                inputs (`np.ndarray`, *optional*)
        """
        raise NotImplemented
