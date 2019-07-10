# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8

from libcpp.map cimport map
from libcpp.vector cimport vector
cimport numpy as np

cdef extern from "voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)

ctypedef np.float32_t REAL_t

# BLAS routine signatures
ctypedef void (*scopy_ptr) (const int *N, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*dsdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*snrm2_ptr) (const int *N, const float *X, const int *incX) nogil
ctypedef void (*sscal_ptr) (const int *N, const float *alpha, const float *X, const int *incX) nogil

cdef scopy_ptr scopy
cdef saxpy_ptr saxpy
cdef sdot_ptr sdot
cdef dsdot_ptr dsdot
cdef snrm2_ptr snrm2
cdef sscal_ptr sscal

# precalculated sigmoid table
DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6
cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE

DEF MAX_SENTENCE_LEN = 10000

# function implementations swapped based on BLAS detected in word2vec_inner.pyx init()
ctypedef REAL_t (*our_dot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef void (*our_saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil

cdef our_dot_ptr our_dot
cdef our_saxpy_ptr our_saxpy

# for when fblas.sdot returns a double
cdef REAL_t our_dot_double(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil

# for when fblas.sdot returns a float
cdef REAL_t our_dot_float(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil

# for when no blas available
cdef REAL_t our_dot_noblas(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
cdef void our_saxpy_noblas(const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil

# to support random draws from negative-sampling cum_table
cdef unsigned long long bisect_left(np.uint32_t *a, unsigned long long x, unsigned long long lo, unsigned long long hi) nogil

cdef unsigned long long random_int32(unsigned long long *next_random) nogil

###################################################################################################################################

cdef struct Word2VecConfig:
    REAL_t running_training_loss, alpha
    int hs, negative, sample, compute_loss, size, window, cbow_mean, workers, sg
    int use_sub, use_head, use_hyper, use_merger
    
    # to segment sentences 
    int sentence_idx[MAX_SENTENCE_LEN + 1]
    
    # for token (head only)
    np.uint32_t indexes[MAX_SENTENCE_LEN]      
    np.uint32_t reduced_windows[MAX_SENTENCE_LEN]      
    
    # embeddings to train: use_sub --> use_head --> use_hyper
    map[int, REAL_t * ] syn0_map   
    # embeddings of negative tokens
    REAL_t *syn1neg
    
    # for sub fields only                  
    map[int, np.uint32_t *] LookUp_map
    map[int, np.uint32_t *] EndIdx_map
    map[int, REAL_t *] LengInv_map

    # for hyper fields only
    # map[int, np.uint32_t *] hyper_indexes # to improve it in the future.
    np.uint32_t pos_indexes[MAX_SENTENCE_LEN]      
    
    REAL_t *grad_mem
    REAL_t *work
    REAL_t *neu1
    REAL_t *work_m
    REAL_t *neu_m
    REAL_t *word_locks
    np.uint32_t *cum_table

    unsigned long long cum_table_len
    unsigned long long next_random

###################################################################################################################################

cdef init_w2v_config(
    Word2VecConfig *c, 
    model, 
    alpha, 
    compute_loss, 
    _work, 
    _neu1, 
    _work_m, 
    _neu_m, 
    _grad_mem)


cdef unsigned long long fieldembed_negsamp( 
    const REAL_t alpha, 
    const int size,
    const int negative, 
    np.uint32_t *cum_table, 
    unsigned long long cum_table_len, 

    const np.uint32_t indexes[MAX_SENTENCE_LEN],
    # map[int, np.uint32_t *] hyper_indexes, 
    const np.uint32_t pos_indexes[MAX_SENTENCE_LEN],
    int i, # right word loc_idx
    int j, # left  word loc_idx start
    int k, # left  word loc_idx end

    int use_head,               
    int use_sub,               
    int use_hyper,
    int use_merger,

    map[int, REAL_t * ] syn0_map,
    map[int, np.uint32_t *] LookUp_map,
    map[int, np.uint32_t *] EndIdx_map,
    map[int, REAL_t *] LengInv_map,

    REAL_t *syn1neg, 
    REAL_t *word_locks,
    
    REAL_t *grad_mem,
    REAL_t *neu1,  
    REAL_t *work,

    REAL_t *neu_m,  
    REAL_t *work_m,

    int cbow_mean, 
    unsigned long long next_random, 
    const int _compute_loss, 
    REAL_t *_running_training_loss_param) nogil