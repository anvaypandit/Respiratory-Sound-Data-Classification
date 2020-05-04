# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:53:14 2020

@author: erice
"""

import numpy as np
from numba import jit
from librosa import util
#from .. import util
from librosa.util.exceptions import ParameterError

__all__ = ['lpc']

def lpc(y, order):
    """Linear Prediction Coefficients via Burg's method

    This function applies Burg's method to estimate coefficients of a linear
    filter on `y` of order `order`.  Burg's method is an extension to the
    Yule-Walker approach, which are both sometimes referred to as LPC parameter
    estimation by autocorrelation.

    It follows the description and implementation approach described in the
    introduction in [1]_.  N.B. This paper describes a different method, which
    is not implemented here, but has been chosen for its clear explanation of
    Burg's technique in its introduction.

    .. [1] Larry Marple
           A New Autoregressive Spectrum Analysis Algorithm
           IEEE Transactions on Accoustics, Speech, and Signal Processing
           vol 28, no. 4, 1980

    Parameters
    ----------
    y : np.ndarray
        Time series to fit

    order : int > 0
        Order of the linear filter

    Returns
    -------
    a : np.ndarray of length order + 1
        LP prediction error coefficients, i.e. filter denominator polynomial

    Raises
    ------
    ParameterError
        - If y is not valid audio as per `util.valid_audio`
        - If order < 1 or not integer
    FloatingPointError
        - If y is ill-conditioned

    See also
    --------
    scipy.signal.lfilter

    Examples
    --------
    Compute LP coefficients of y at order 16 on entire series

    >>> y, sr = librosa.load(librosa.util.example_audio_file(), offset=30,
    ...                      duration=10)
    >>> librosa.lpc(y, 16)

    Compute LP coefficients, and plot LP estimate of original series

    >>> import matplotlib.pyplot as plt
    >>> import scipy
    >>> y, sr = librosa.load(librosa.util.example_audio_file(), offset=30,
    ...                      duration=0.020)
    >>> a = librosa.lpc(y, 2)
    >>> y_hat = scipy.signal.lfilter([0] + -1*a[1:], [1], y)
    >>> plt.figure()
    >>> plt.plot(y)
    >>> plt.plot(y_hat, linestyle='--')
    >>> plt.legend(['y', 'y_hat'])
    >>> plt.title('LP Model Forward Prediction')
    >>> plt.show()

    """
    if not isinstance(order, int) or order < 1:
        raise ParameterError("order must be an integer > 0")

    util.valid_audio(y, mono=True)

    return __lpc(y, order)



@jit(nopython=True)
def __lpc(y, order):
    # This implementation follows the description of Burg's algorithm given in
    # section III of Marple's paper referenced in the docstring.
    #
    # We use the Levinson-Durbin recursion to compute AR coefficients for each
    # increasing model order by using those from the last. We maintain two
    # arrays and then flip them each time we increase the model order so that
    # we may use all the coefficients from the previous order while we compute
    # those for the new one. These two arrays hold ar_coeffs for order M and
    # order M-1.  (Corresponding to a_{M,k} and a_{M-1,k} in eqn 5)

    dtype = y.dtype.type
    ar_coeffs = np.zeros(order+1, dtype=dtype)
    ar_coeffs[0] = dtype(1)
    ar_coeffs_prev = np.zeros(order+1, dtype=dtype)
    ar_coeffs_prev[0] = dtype(1)

    # These two arrays hold the forward and backward prediction error. They
    # correspond to f_{M-1,k} and b_{M-1,k} in eqns 10, 11, 13 and 14 of
    # Marple. First they are used to compute the reflection coefficient at
    # order M from M-1 then are re-used as f_{M,k} and b_{M,k} for each
    # iteration of the below loop
    fwd_pred_error = y[1:]
    bwd_pred_error = y[:-1]

    # DEN_{M} from eqn 16 of Marple.
    den = np.dot(fwd_pred_error, fwd_pred_error) \
          + np.dot(bwd_pred_error, bwd_pred_error)

    for i in range(order):
        if den <= 0:
            raise FloatingPointError('numerical error, input ill-conditioned?')

        # Eqn 15 of Marple, with fwd_pred_error and bwd_pred_error
        # corresponding to f_{M-1,k+1} and b{M-1,k} and the result as a_{M,M}
        #reflect_coeff = dtype(-2) * np.dot(bwd_pred_error, fwd_pred_error) / dtype(den)
        reflect_coeff = dtype(-2) * np.dot(bwd_pred_error, fwd_pred_error) / dtype(den)

        # Now we use the reflection coefficient and the AR coefficients from
        # the last model order to compute all of the AR coefficients for the
        # current one.  This is the Levinson-Durbin recursion described in
        # eqn 5.
        # Note 1: We don't have to care about complex conjugates as our signals
        # are all real-valued
        # Note 2: j counts 1..order+1, i-j+1 counts order..0
        # Note 3: The first element of ar_coeffs* is always 1, which copies in
        # the reflection coefficient at the end of the new AR coefficient array
        # after the preceding coefficients
        ar_coeffs_prev, ar_coeffs = ar_coeffs, ar_coeffs_prev
        for j in range(1, i + 2):
            ar_coeffs[j] = ar_coeffs_prev[j] + reflect_coeff * ar_coeffs_prev[i - j + 1]

        # Update the forward and backward prediction errors corresponding to
        # eqns 13 and 14.  We start with f_{M-1,k+1} and b_{M-1,k} and use them
        # to compute f_{M,k} and b_{M,k}
        fwd_pred_error_tmp = fwd_pred_error
        fwd_pred_error = fwd_pred_error + reflect_coeff * bwd_pred_error
        bwd_pred_error = bwd_pred_error + reflect_coeff * fwd_pred_error_tmp

        # SNIP - we are now done with order M and advance. M-1 <- M

        # Compute DEN_{M} using the recursion from eqn 17.
        #
        # reflect_coeff = a_{M-1,M-1}      (we have advanced M)
        # den =  DEN_{M-1}                 (rhs)
        # bwd_pred_error = b_{M-1,N-M+1}   (we have advanced M)
        # fwd_pred_error = f_{M-1,k}       (we have advanced M)
        # den <- DEN_{M}                   (lhs)
        #

        q = dtype(1) - reflect_coeff**2
        den = q*den - bwd_pred_error[-1]**2 - fwd_pred_error[0]**2

        # Shift up forward error.
        #
        # fwd_pred_error <- f_{M-1,k+1}
        # bwd_pred_error <- b_{M-1,k}
        #
        # N.B. We do this after computing the denominator using eqn 17 but
        # before using it in the numerator in eqn 15.
        fwd_pred_error = fwd_pred_error[1:]
        bwd_pred_error = bwd_pred_error[:-1]

    return ar_coeffs