"""
    src/models/utils/preproc.py
    ---------------------------

    Helper script for serialize / deserialize sklearn.preprocessors
    which enable store parameters in form of .pkl files and thus no
    need to store the entire preprocessor object. 


    Would you like me to show you a @singledispatch version where you just register
     new preprocessors (e.g., QuantileTransformer) instead of editing big 
     if/elif blocks? That’d make it super extensible.
"""
import numpy as np

from enum import Enum
from typing import Any, Dict, Final, Union
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler


class PreprocType:
    STANDARD    : Final[str] = "standard"
    ONE_HOT     : Final[str] = "one-hot"
    MIN_MAX     : Final[str] = "min-max"

Preproc = Union[StandardScaler, OneHotEncoder, MinMaxScaler]

def preproc_to_param(proc: Preproc, proc_type: PreprocType) -> Dict[str, Any]:
    """
        Serializes sklearn.preprocessing object into a JSON-Safe 
        dictionary.
    """
    if proc_type is PreprocType.STANDARD:
        return {"mean"              : proc.mean_,
                "scale"             : proc.scale_,
                "var"               : getattr(proc, "var_", proc.scale_**2),
                "n_samples_seen"    : getattr(proc, "n_samples_seen_", None),}
    
    if proc_type is PreprocType.ONE_HOT:
        params: Dict[str, Any] = {
            "categories"        : [np.asarray(c) for c in proc.categories_],
            "handle_unknown"    : proc.handle_unknown,
            "drop"              : proc.drop,
            "sparse_output"     : proc.sparse_output,
        }
        if hasattr(proc, "min_frequency"): 
            params["min_frequency"] = proc.min_frequency
        if hasattr(proc, "max_categories"):
            params["max_categories"] = proc.max_categories
        return params
    

    if proc_type is PreprocType.MIN_MAX:
        return {"data_min"      : proc.data_min_,
                "data_max"      : proc.data_max_,
                "data_range"    : proc.data_range_,
                "scale"         : proc.scale_,
                "min"           : proc.min_,
                "feature_range" : getattr(proc, "feature_range", (0, 1)),}
    
    raise ValueError(f"Unknown preprocessing type: {proc_type}")


#################################################################################

def param_to_preproc(param: Dict[str, Any], proc_type: PreprocType) -> Preproc:
    """
        Deserialize JSON-safe dict back into sklearn preprocessor. 
        Reconstructs the sklearn.preprocessor object back from the
        JSON-Safe representation
    """
    if proc_type is PreprocType.STANDARD:
        proc = StandardScaler()
        proc.mean_ = np.array(param["mean"])
        proc.scale_ = np.array(param["scale"])
        proc.var_ = np.array(param.get("var", proc.scale_**2))
        proc.n_samples_seen_ = param.get("n_samples_seen", None)
        proc.n_features_in_ = proc.mean_.shape[0]
        return proc

    if proc_type is PreprocType.ONE_HOT:
        proc = OneHotEncoder(
            categories=[np.asarray(c) for c in param["categories"]],
            handle_unknown=param.get("handle_unknown", "ignore"),
            drop=param.get("drop", None),
            sparse_output=param.get("sparse_output", False),
            min_frequency=param.get("min_frequency", None),
            max_categories=param.get("max_categories", None),
        )
        # Pretend fitted
        proc.categories_ = [np.asarray(c) for c in param["categories"]]
        proc.n_features_in_ = len(proc.categories_)
        return proc

    if proc_type is PreprocType.MIN_MAX:
        proc = MinMaxScaler(feature_range=param.get("feature_range", (0, 1)))
        proc.data_min_ = np.array(param["data_min"])
        proc.data_max_ = np.array(param["data_max"])
        proc.data_range_ = np.array(param["data_range"])
        proc.scale_ = np.array(param["scale"])
        proc.min_ = np.array(param["min"])
        proc.n_features_in_ = proc.data_min_.shape[0]
        return proc

    raise ValueError(f"Unknown preprocessing type: {proc_type}")
