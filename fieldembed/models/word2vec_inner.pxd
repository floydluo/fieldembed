# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8
#
# shared type definitions for word2vec_inner
# used by both word2vec_inner.pyx (automatically) and doc2vec_inner.pyx (by explicit cimport)


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


cdef struct Word2VecConfig:
    int hs, negative, sample, compute_loss, size, window, cbow_mean, workers
    int sg # added by jjluo
    int use_head #--> NEW for 0X1
    int use_sub  #--> NEW for 0X1

    REAL_t running_training_loss, alpha

    REAL_t *syn0
    
    #########################
    REAL_t *syn0_1
    np.uint32_t *syn0_1_LookUp
    np.uint32_t *syn0_1_EndIdx
    REAL_t *syn0_1_LengInv
    int syn0_1_leng_max
    #########################

    REAL_t *word_locks
    REAL_t *work
    REAL_t *neu1

    REAL_t *work2
    REAL_t *neu2

    int codelens[MAX_SENTENCE_LEN]
    
    np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    np.uint32_t indexes[MAX_SENTENCE_LEN]
    int sentence_idx[MAX_SENTENCE_LEN + 1]
    np.uint32_t *cum_table

    # For negative sampling
    REAL_t *syn1neg
    unsigned long long cum_table_len
    
    # for sampling (negative and frequent-word downsampling)
    unsigned long long next_random


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

cdef unsigned long long w2v_fast_sentence_sg_neg(
    const int negative, 
    np.uint32_t *cum_table, 
    unsigned long long cum_table_len,
    REAL_t *syn0, 
    REAL_t *syn1neg, 
    const int size, 
    const np.uint32_t word_index,
    const np.uint32_t word2_index, 
    const REAL_t alpha, 
    REAL_t *work,
    unsigned long long next_random, 
    REAL_t *word_locks,
    const int _compute_loss, 
    REAL_t *_running_training_loss_param) nogil

cdef unsigned long long w2v_fast_sentence_cbow_neg(
    const int negative, 
    np.uint32_t *cum_table, 
    unsigned long long cum_table_len, 
    int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1,  
    REAL_t *syn0, 
    REAL_t *syn1neg, 
    const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], 
    const REAL_t alpha, 
    REAL_t *work,
    int i, int j, int k, 
    int cbow_mean, 
    unsigned long long next_random, 
    REAL_t *word_locks,
    const int _compute_loss, 
    REAL_t *_running_training_loss_param) nogil


cdef init_w2v_config(
    Word2VecConfig *c, 
    model, 
    alpha, 
    compute_loss, 
    _work, 
    _neu1=*) # could learn this method.

#--> NEW for 0X1
cdef init_w2v_config_0X1(
    Word2VecConfig *c, 
    model, 
    alpha, 
    compute_loss, 
    _work, 
    _neu1, 
    _work2, 
    _neu2)

cdef unsigned long long fieldembed_token_neg( 
    const REAL_t alpha, 
    const int size,
    const int negative, 
    np.uint32_t *cum_table, 
    unsigned long long cum_table_len, 

    const np.uint32_t indexes[MAX_SENTENCE_LEN], 
    int i, # right word loc_idx
    int j, # left  word loc_idx start
    int k, # left  word loc_idx end

    REAL_t *syn0, 
    REAL_t *syn1neg, 
    REAL_t *word_locks,
    REAL_t *neu1,  
    REAL_t *work,
    
    int cbow_mean, 
    unsigned long long next_random, 
    const int _compute_loss, 
    REAL_t *_running_training_loss_param) nogil
#=================================================#

#--> NEW for 0X1
cdef unsigned long long fieldembed_token_neg_0X1( 
    const REAL_t alpha, 
    const int size,
    const int negative, 
    np.uint32_t *cum_table, 
    unsigned long long cum_table_len, 

    const np.uint32_t indexes[MAX_SENTENCE_LEN], 
    int i, # right word loc_idx
    int j, # left  word loc_idx start
    int k, # left  word loc_idx end

    int use_head,                # 
    int use_sub,                 # 
    REAL_t *syn0, 
    
    REAL_t *syn0_1,
    np.uint32_t *syn0_1_LookUp,  # 
    np.uint32_t *syn0_1_EndIdx,  # 
    REAL_t *syn0_1_LengInv,     # currently, it is not in use.
    int syn0_1_leng_max,         # currently, it is not in use.

    REAL_t *syn1neg, 
    REAL_t *word_locks,

    REAL_t *neu1,  
    REAL_t *work,

    REAL_t *neu2,                # 
    REAL_t *work2,               # 

    int cbow_mean, 
    unsigned long long next_random, 
    const int _compute_loss, 
    REAL_t *_running_training_loss_param) nogil
#=================================================#


#####################################################################################
cdef init_w2v_config_0X1_neat(
    Word2VecConfig *c, 
    model, 
    alpha, 
    compute_loss, 
    _work, 
    _neu1)

cdef unsigned long long fieldembed_token_neg_0X1_neat( 
    const REAL_t alpha, 
    const int size,
    const int negative, 
    np.uint32_t *cum_table, 
    unsigned long long cum_table_len, 

    const np.uint32_t indexes[MAX_SENTENCE_LEN], 
    int i, # right word loc_idx
    int j, # left  word loc_idx start
    int k, # left  word loc_idx end

    int use_head,                # 
    int use_sub,                 # 
    REAL_t *syn0, 
    
    REAL_t *syn0_1,
    np.uint32_t *syn0_1_LookUp,  # 
    np.uint32_t *syn0_1_EndIdx,  # 
    REAL_t *syn0_1_LengInv,      # 
    int syn0_1_leng_max,         # currently, it is not in use.

    REAL_t *syn1neg, 
    REAL_t *word_locks,

    REAL_t *neu1,  
    REAL_t *work,
    
    # int sg,
    int cbow_mean, 
    unsigned long long next_random, 
    const int _compute_loss, 
    REAL_t *_running_training_loss_param) nogil