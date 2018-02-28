"""
As part of forming the selective likelihood ratio, various reconstructions
of parts of the original randomization are necessary.

In this module, generally speaking: 

- `internal` refers to coordinates internal to a given query
as each query can represent its data in its own coordinates;

- `full` refers to the coordinate system of the original randomization
and is the sum of a `score` as well as an `opt` term

 
"""
import numpy as np

def reconstruct_internal(data_state, data_transform):
    """
    Reconstruct some internal state data
    based on an affine mapping from `data_state` to the
    internal coordinates of the query.
    """

    data_state = np.atleast_2d(data_state)
    data_linear, data_offset = data_transform
    if data_linear is not None:
        return np.squeeze(data_linear.dot(data_state.T) + data_offset[:,None]).T
    else:
        return np.squeeze(data_offset)

def reconstruct_full_from_data(opt_transform, score_transform, data_state, data_transform, opt_state):
    """
    Reconstruct original randomization state from state data
    and optimization state.
    """

    internal_state = reconstruct_internal(data_state, data_transform)
    return np.squeeze(reconstruct_full_from_internal(opt_transform, score_transform, internal_state, opt_state))

def reconstruct_opt(opt_transform, opt_state):
    """
    Reconstruct part of the original randomization state 
    in terms of optimization state.
    """
    opt_linear, opt_offset = opt_transform
    if opt_linear is not None:
        opt_state = np.atleast_2d(opt_state)
        return np.squeeze(opt_linear.dot(opt_state.T) + opt_offset[:, None]).T
    else:
        return opt_offset

def reconstruct_score(score_transform, internal_state):
    """
    Reconstruct part of the original randomization state 
    determined by the score of the loss from 
    a query's internal coordinates.
    """
    score_linear, score_offset = score_transform
    return score_linear.dot(internal_state.T).T + score_offset

def reconstruct_full_from_internal(opt_transform, score_transform, internal_state, opt_state):
    """
    Reconstruct original randomization state from internal state data
    and optimization state.
    """
    randomization_score = reconstruct_score(score_transform, internal_state)
    randomization_opt = reconstruct_opt(opt_transform, opt_state)
    return randomization_score + randomization_opt

