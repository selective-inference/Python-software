from rpy2.robjects.packages import importr
from rpy2 import robjects

utils = importr('utils')
utils.install_packages("glmnet")

base = importr('base')
print(base.R_home())

glmnet = importr('glmnet')
from selection.tests.instance import gaussian_instance
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import numpy as np
import regreg.api as rr
from selection.api import randomization
