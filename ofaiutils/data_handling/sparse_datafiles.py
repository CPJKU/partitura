#!/usr/bin/env python

import sys,os
import numpy as np
from scipy.io import loadmat, savemat, mmwrite, mmread
from scipy.sparse import csr_matrix,coo_matrix,lil_matrix
from scipy.sparse import vstack as sp_vstack

def stack_csr_npz(fns, outfile = None, return_indices = False):
    """
    load multiple sparse matrices (in CSR format) from the filenames
    in `fns`, and stack them vertically and return tht result; If
    `outfile` is specified, save the result to that destination

    :param fns: a list of filenames
    :param outfile: the name of the file to save the stacked CSR matrices to
    :param return_indices: if True, return an array with the index of the file to which each row belongs

    :returns: a CSR matrix

    """
    
    data = []

    idx = []
    for i, fn in enumerate(fns):
        data.append(csr_from_file(fn))
        idx.append(np.ones(data[-1].shape[0])*i)
    data = sp_vstack(data).tocsr()

    if outfile:
        csr_to_file(outfile, data)

    if return_indices:
        return data.astype(np.float), np.vstack(idx)
    else:
        return data.astype(np.float)

def csr_from_file(fn):
    """
    load a sparse matrix from a file; the file can be in matrix market
    format (.mtx), in a custom ".npz" format, or in matlab format
    (.mat). The extension is used to determine the filetype

    :param fn: the file to read the data from

    :returns: a sparse matrix

    """
    
    if fn.endswith('.mtx'):
        d = mmread(fn)
    else:
        if fn.endswith('.npz'):
            v = np.load(fn)
            d = csr_matrix((v['data'],v['indices'],v['indptr']),shape=v['shape'])
        elif fn.endswith('.mat'):
            v = loadmat(fn)
            d = csr_matrix((v['data'].reshape((-1,)),
                            v['indices'].reshape((-1,)),
                            v['indptr'].reshape((-1,))),
                           shape=v['shape'])

        else:
            print('Warning: unkown file format')
            return None
        try:
            v.close()
        except:
            pass
    return d

def csr_to_file(fn, A):
    """
    save a sparse matrix to a file; the data can be saved in matrix
    market format (.mtx), in a custom ".npz" format, or in matlab
    format (.mat). The extension is used to determine the filetype;
    The data will be converted to CSR format before saving

    :param fn: the file to read the data from
    :param A: a sparse matrix

    """

    if not isinstance(A, csr_matrix):
        A = A.tocsr()
    d = {'data': A.data, 
         'indices': A.indices, 
         'indptr': A.indptr, 
         'shape': A.shape}
    if fn.endswith('.npz'):
        np.savez(fn,**d)
    elif fn.endswith('.mat'):
        savemat(fn,d)
    elif fn.endswith('.mtx'):
        mmwrite(fn,A,field='real')

class ResizeError: pass

def resize_csr(a, shape):
    
    # check for shape incompatibilities
    if len(a.shape) != 2 or len(shape) != 2:
        raise NotImplementedError

    target_shape = list(shape)

    # infer dimension 0 if it is specified as -1
    if target_shape[0] == -1 and target_shape[1] > 0 :
        
        if (a.shape[0] * a.shape[1]) % target_shape[1] > 0:
            raise ResizeError
        else:
            target_shape[0] = ( a.shape[0] * a.shape[1] ) / target_shape[1]

    # infer dimension 1 if it is specified as -1
    if target_shape[1] == -1 and target_shape[0] > 0 :
        if (a.shape[0] * a.shape[1]) % target_shape[0] > 0:
            raise ResizeError
        else:
            target_shape[1] = ( a.shape[0] * a.shape[1] ) / target_shape[0]

    # check for shape incompatibilities        
    if a.shape[0] * a.shape[1] != target_shape[0] * target_shape[1]:
        raise ResizeError

    if len(a.indices) == 0:
        a._shape = tuple(target_shape)
        return None
        
    for i in range(len(a.indptr)-1):
        a.indices[a.indptr[i] : a.indptr[i+1]] += i*a.shape[1]

    # finished checking for shape incompatibilities, now do the reshaping

    row_idx = a.indices / target_shape[1] 

    a.indices %= target_shape[1] 

    indptr = np.zeros(target_shape[0], dtype = np.int)

    indptr[np.arange(np.max(row_idx) + 1)] = np.bincount(row_idx)
    a.indptr = np.r_[0, np.cumsum(indptr)]

    a._shape = tuple(target_shape)

    return None

