import sys
sys.path.insert(0,'src/')
sys.path.insert(0,'bartpy/') #https://github.com/JakeColtman/bartpy
from data_simulation import *

from bartpy.sklearnmodel import SklearnModel as bart
from bartpy.features.featureselection import SelectNullDistributionThreshold, SelectSplitProportionThreshold
from bartpy.diagnostics.features import *
from bartpy.features.featureimportance import feature_importance

from sklearn.metrics import confusion_matrix,f1_score
from sklearn.metrics import roc_curve,roc_auc_score

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value
    https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])

gwas_data = gwas_simulated_data(1000, 100, 8, prop_tc = 0.05)
y, tc, X, col = gwas_data.generate_samples()
X = pd.DataFrame(X).sample(frac=1.0).values
y = y.astype('float')


model = bart(n_samples=1000, n_burn=50, n_trees=10, store_in_sample_predictions=False, n_jobs=1) # Use default parameters
model.fit(X, y)

feature_importance(model, X, y,0, n_permutations = 5, n_k_fold_splits = 2)
