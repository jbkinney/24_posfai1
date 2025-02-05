import numpy as np
import pandas as pd

def get_nanmask_lclc(L,C):
    # Create mask for nans
    ls = np.arange(L).astype(int)
    ls1 = np.tile(ls.reshape([L, 1, 1, 1]),
                  [1, C, L, C])
    ls2 = np.tile(ls.reshape([1, 1, L, 1]),
                  [L, C, 1, C])
    nanmask_lclc = (ls2 - ls1 < 1)
    return nanmask_lclc

def get_one_hot_encoding(x, alphabet):
    """
    Convert a sequence array to a one-hot encoded matrix.

    Parameters
    ----------
    x: (np.ndarray)
        (N,) array of input sequences, each of length L

    alphabet: (np.ndarray)
        (C,) array describing the alphabet sequences are drawn from.

    Returns
    -------
    x_lc: (np.ndarray)
        Array of one-hot encoded sequences, shaped as (N, L, C), stored floats.
    """

    # Get dimensions
    L = len(x[0])
    N = len(x)
    C = len(alphabet)

    # Shape sequences as array of int8s
    x_arr = np.frombuffer(bytes(''.join(x), 'utf-8'),
                          np.int8, N * L).reshape([N, L])

    # Create alphabet as array of int8s
    alphabet_arr = np.frombuffer(bytes(''.join(alphabet), 'utf-8'),
                                 np.int8, C)

    # Compute (N,L,C) grid of one-hot encoded values
    x_nlc = (x_arr[:, :, np.newaxis] ==
             alphabet_arr[np.newaxis, np.newaxis, :]).astype(float)

    return x_nlc

def evaluate_pairwise_model(theta_dict, x_nlc):
    """Evaluates a pairwise model given theta_dict and sequences x"""
    
    # Extract unfixed parameters
    theta_0 = theta_dict['theta_0'].copy()
    theta_lc = theta_dict['theta_lc'].copy()
    theta_lclc = theta_dict['theta_lclc'].copy()
    
    # Get dimensions
    L, C = theta_lc.shape

    # Get nanmask
    nanmask_lclc = get_nanmask_lclc(L,C)
    
    # Remove nans from pairwise parameters
    theta_lclc[nanmask_lclc] = 0.0
    
    # Compute phi
    phi = theta_0
    phi = phi + np.reshape(np.sum(theta_lc 
                                  * np.reshape(x_nlc, (-1, L, C)),
                                  axis=(1, 2)),
                           (-1, 1))
    phi = phi + np.reshape(np.sum(theta_lclc 
                                  * np.reshape(x_nlc, (-1, L, C, 1, 1)) 
                                  * np.reshape(x_nlc, (-1, 1, 1, L, C)),
                                  axis=(1, 2, 3, 4)),
                           (-1, 1))

    # Return phi
    return phi

def evaluate_additive_model(theta_dict, x_nlc):
    """Evaluates an additive model given theta_dict and sequences x"""
    
    # Extract unfixed parameters
    theta_0 = theta_dict['theta_0'].copy()
    theta_lc = theta_dict['theta_lc'].copy()
    
    # Get dimensions
    L, C = theta_lc.shape
    
    # Compute phi
    phi = theta_0
    phi = phi + np.reshape(np.sum(theta_lc 
                                  * np.reshape(x_nlc, (-1, L, C)),
                                  axis=(1, 2)),
                           (-1, 1))

    # Return phi
    return phi

def fix_pairwise_model_gauge(theta_dict, p_lc):
    """Fixes the hierarchical gauge for a pairwise model"""
    
    # Extract unfixed parameters
    theta_0 = theta_dict['theta_0'].copy()
    theta_lc = theta_dict['theta_lc'].copy()
    theta_lclc = theta_dict['theta_lclc'].copy()
    
    # Get dimensions
    L, C = theta_lc.shape

    # Get nanmask
    nanmask_lclc = get_nanmask_lclc(L,C)
    
    # Remove nans from pairwise parameters
    theta_lclc[nanmask_lclc] = 0.0

    # Need this
    _ = np.newaxis
    
    # Fix 0th order parameter
    fixed_theta_0 = theta_0 \
        + np.sum(p_lc * theta_lc) \
        + np.sum(theta_lclc * p_lc[:, :, _, _] * p_lc[_, _, :, :])


    ## WOW! We were missing 2 terms! Need to fix this within MAVE-NN 
    # Fix 1st order parameters
    fixed_theta_lc = theta_lc \
        - np.sum(theta_lc * p_lc, axis=1)[:, _] \
        + np.sum(theta_lclc * p_lc[_, _, :, :],
                 axis=(2, 3)) \
        + np.sum(theta_lclc * p_lc[:, :, _, _], \
                 axis=(0, 1)) \
        - np.sum(theta_lclc * p_lc[:, :, _, _] * p_lc[_, _, :, :],
                 axis=(1, 2, 3))[:, _] \
        - np.sum(theta_lclc * p_lc[:, :, _, _] * p_lc[_, _, :, :],
                 axis=(0, 1, 3))[:, _]

    # Fix 2nd order parameters
    fixed_theta_lclc = theta_lclc \
        - np.sum(theta_lclc * p_lc[:, :, _, _],
                 axis=1)[:, _, :, :] \
        - np.sum(theta_lclc * p_lc[_, _, :, :],
                 axis=3)[:, :, :, _] \
        + np.sum(theta_lclc * p_lc[:, :, _, _] * p_lc[_, _, :, :],
                 axis=(1, 3))[:, _, :, _]
    
    # Restore Nans
    fixed_theta_lclc[nanmask_lclc] = np.NaN
        
    # Return dictionary of parameters
    fixed_theta_dict =  {
        'theta_0': fixed_theta_0,
        'theta_lc': fixed_theta_lc,
        'theta_lclc': fixed_theta_lclc,
    }
    return fixed_theta_dict