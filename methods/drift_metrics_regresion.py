#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def normalize(distances_list):
    """
    Method to normalize a vector, in this case used to normalize the
    weighted euclidean distance vector.

    Parameters:
    -----------

    distances_list : list of floats
        List of floats that we want to normalize

    Return:
    -------

    NONE

    """

    xmin = min(distances_list)
    xmax = max(distances_list)
    for i, x in enumerate(distances_list):
        distances_list[i] = (x - xmin) / (xmax - xmin)


def weighted_euclidean_dist(a, b, index):
    """
    Function for performing the weighted euclidean between two batches.
    The formula is:
        𝑑(𝐴, 𝐵) = √(∑𝑤(𝐴_𝑖 − 𝐵_𝑖)^2)

    Where A, B are the batches what we want to compare and w is the weight
    that we want to give to feature i.
    In our case, this weight is given by a linear scalar to know how old its
    the batch that we are comparing with.

    Parameters:
    -----------

    a : list of floats
        1st batch to compare

    b : list of floats
        2nd batch to compare

    index : float
        weight of the euclidean distance

    Return:
    -------

    distance : float

    """

    return np.sqrt(np.sum(index * (a - b) ** 2))


def drift_single_batch(batch, matrix, tranformation = "log"):
    """
    IMPORTANT!! : ALL THE DATA NEEDS TO BE TIME ORDERED

    Function to perform the metric of drift comparing a batch with all the
    batches that the matrix has.

    This metric is logarithm of a weighted euclidean distance that is weighted
    with a linear scale of how old is the batch of the matrix that we are
    comparing with. This distance shows the changes in the distribution that
    the batches are creating over the time, and with this weight we prioritize
    that the distribution of the new batches is similar to the newer ones.

    The value of the metric doesn't show anything useful, only works as a
    long term tendency, we need to compare it with other batches to know if it
    have drifted (increasing or decreasing trend = drift)

    Parameters:
    -----------

    batch : list of floats
        List of floats with the data of the batch

    matrix : matrix
        Matrix of batches
    
    tranformation : string
        If tranformation = "log" it performs the logarithm of the distance, if not it do only the weighted euclidean distance 

    Return:
    -------

    drift value : float

    """
    aux_list = []
    for i in range(len(matrix)):
        idx = (len(matrix) - abs(i) - 1) / len(matrix)
        if tranformation == "log":
            dst = np.log(weighted_euclidean_dist(matrix.iloc[i], batch, idx))
        else:
            dst = weighted_euclidean_dist(matrix.iloc[i], batch, idx)
        aux_list.append(dst)
    return np.ma.masked_invalid(aux_list).min()
