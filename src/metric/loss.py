import numpy as np

def symmetric_mean_absolute_percentage_error(
    y_true, y_pred
):
    """Symmetric mean absolute percentage error (SMAPE or sMAPE) is an accuracy measure based on percentage (or relative) errors

    Args:
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    """
    assert len(y_true)==len(y_pred)
    n=len(y_pred)
    abs_diff=np.abs(np.array(y_true)-np.array(y_pred))
    pred_abs=np.abs(y_pred)
    true_abs=np.abs(y_true)
    
    ape=abs_diff/((pred_abs+true_abs)/2)
    smape=np.sum(ape)*1/n
    
    return smape
    
