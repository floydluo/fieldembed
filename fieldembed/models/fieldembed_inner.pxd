# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8
#
# shared type definitions for word2vec_inner
# used by both word2vec_inner.pyx (automatically) and doc2vec_inner.pyx (by explicit cimport)
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

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


cdef struct Word2VecConfig:
    int hs, negative, sample, compute_loss, size, window, cbow_mean, workers
    int sg # added by jjluo
    int use_head    #--> NEW for 0X1
    int use_sub     #--> NEW for 0X1
    int use_merger  #--> NEW for M0X1
    int use_pos     # ########################### -------------> use pos
    int use_token

    REAL_t running_training_loss, alpha

    REAL_t *syn0
    REAL_t *syn_p  # ########################### -------------> use pos
    
    #########################
    REAL_t *syn0_1
    np.uint32_t *syn0_1_LookUp
    np.uint32_t *syn0_1_EndIdx
    REAL_t *syn0_1_LengInv
    int syn0_1_leng_max
    #########################
    #########################
    REAL_t *syn0_2
    np.uint32_t *syn0_2_LookUp
    np.uint32_t *syn0_2_EndIdx
    REAL_t *syn0_2_LengInv
    int syn0_2_leng_max
    #########################
    #########################
    REAL_t *syn0_3
    np.uint32_t *syn0_3_LookUp
    np.uint32_t *syn0_3_EndIdx
    REAL_t *syn0_3_LengInv
    int syn0_3_leng_max
    #########################
    #########################
    REAL_t *syn0_4
    np.uint32_t *syn0_4_LookUp
    np.uint32_t *syn0_4_EndIdx
    REAL_t *syn0_4_LengInv
    int syn0_4_leng_max
    #########################
    #########################
    REAL_t *syn0_5
    np.uint32_t *syn0_5_LookUp
    np.uint32_t *syn0_5_EndIdx
    REAL_t *syn0_5_LengInv
    int syn0_5_leng_max
    #########################
    #########################
    REAL_t *syn0_6
    np.uint32_t *syn0_6_LookUp
    np.uint32_t *syn0_6_EndIdx
    REAL_t *syn0_6_LengInv
    int syn0_6_leng_max
    #########################

    REAL_t *word_locks
    
    REAL_t *work
    REAL_t *neu1

    REAL_t *work2
    REAL_t *neu2

    REAL_t *work3
    REAL_t *neu3

    REAL_t *work4
    REAL_t *neu4

    REAL_t *work5
    REAL_t *neu5

    REAL_t *work6
    REAL_t *neu6

    REAL_t *work7
    REAL_t *neu7

    REAL_t *work_m
    REAL_t *neu_m

    REAL_t *work_p  # ########################### -------------> use pos
    REAL_t *neu_p   # ########################### -------------> use pos

    int codelens[MAX_SENTENCE_LEN]
    
    np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    np.uint32_t indexes[MAX_SENTENCE_LEN]
    np.uint32_t indexes_pos[MAX_SENTENCE_LEN]
    int sentence_idx[MAX_SENTENCE_LEN + 1]
    np.uint32_t *cum_table

    # For negative sampling
    REAL_t *syn1neg
    unsigned long long cum_table_len
    
    # for sampling (negative and frequent-word downsampling)
    unsigned long long next_random


#--> NEW for M0X1
cdef init_w2v_config_M0X1(
    Word2VecConfig *c, 
    model, 
    alpha, 
    compute_loss, 
    _work, 
    _neu1, 
    _work2, 
    _neu2,
    _work_m,
    _neu_m)


#--> NEW for M0X2
cdef init_w2v_config_M0X2(
    Word2VecConfig *c, 
    model, 
    alpha, 
    compute_loss, 
    _work, 
    _neu1, 
    _work2, 
    _neu2,
    _work3, 
    _neu3,
    _work_m,
    _neu_m)


#--> NEW for M0XY
cdef init_w2v_config_M0XY(
    Word2VecConfig *c, 
    model, 
    alpha, 
    compute_loss, 
    _work, 
    _neu1, 
    _work2, 
    _neu2,
    _work3, 
    _neu3,
    _work4, 
    _neu4,
    _work5, 
    _neu5,
    _work6, 
    _neu6,
    _work7, 
    _neu7,
    _work_m,
    _neu_m)

#--> NEW for M0XY_P
cdef init_w2v_config_M0XY_P(
    Word2VecConfig *c, 
    model, 
    alpha, 
    compute_loss, 
    _work, 
    _neu1, 
    _work2, 
    _neu2,
    _work3, 
    _neu3,
    _work4, 
    _neu4,
    _work5, 
    _neu5,
    _work6, 
    _neu6,
    _work7, 
    _neu7,
    _work_m,
    _neu_m,
    _work_p,  # ########################### -------------> use pos
    _neu_p)   # ########################### -------------> use pos


#--> NEW for M0X1
cdef unsigned long long fieldembed_token_neg_M0X1( 
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
    int use_merger,
    
    REAL_t *syn0, 
    
    REAL_t *syn0_1,
    
    np.uint32_t *syn0_1_LookUp,  # 
    np.uint32_t *syn0_1_EndIdx,  # 
    REAL_t *syn0_1_LengInv,    # currently, it is not in use.
    int syn0_1_leng_max,         # currently, it is not in use.

    REAL_t *syn1neg, 
    REAL_t *word_locks,

    REAL_t *neu1,  
    REAL_t *work,

    REAL_t *neu2,                # 
    REAL_t *work2,               # 

    REAL_t *neu_m,               # 
    REAL_t *work_m,              # 

    int cbow_mean, 
    unsigned long long next_random, 
    const int _compute_loss, 
    REAL_t *_running_training_loss_param) nogil
#===========================================================================================#

#--> NEW for M0X1
cdef unsigned long long fieldembed_token_neg_M0X2( 
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
    int use_sub,                 # #--------------> this value will be 2 instead of 1
    int use_merger,
    REAL_t *syn0, 
    
    REAL_t *syn0_1,
    np.uint32_t *syn0_1_LookUp,  # 
    np.uint32_t *syn0_1_EndIdx,  # 
    REAL_t *syn0_1_LengInv,      # 
    int syn0_1_leng_max,         # currently, it is not in use.

    REAL_t *syn0_2,              # add the second #--------------> add this line
    np.uint32_t *syn0_2_LookUp,  # add the second #--------------> add this line
    np.uint32_t *syn0_2_EndIdx,  # add the second #--------------> add this line
    REAL_t *syn0_2_LengInv,      # add the second #--------------> add this line
    int syn0_2_leng_max,         # add the second #--------------> add this line


    REAL_t *syn1neg, 
    REAL_t *word_locks,

    REAL_t *neu1,  
    REAL_t *work,

    REAL_t *neu2,                # 
    REAL_t *work2,               # 

    REAL_t *neu3,                # add the second #--------------> add this line
    REAL_t *work3,               # add the second #--------------> add this line


    REAL_t *neu_m,               # 
    REAL_t *work_m,              # 


    int cbow_mean, 
    unsigned long long next_random, 
    const int _compute_loss, 
    REAL_t *_running_training_loss_param) nogil

#--> NEW for M0XY
cdef unsigned long long fieldembed_token_neg_M0XY( 
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
    int use_sub,                 # #--------------> this value will be 2 instead of 1
    int use_merger,
    REAL_t *syn0, 
    
    REAL_t *syn0_1,
    np.uint32_t *syn0_1_LookUp,  # 
    np.uint32_t *syn0_1_EndIdx,  # 
    REAL_t *syn0_1_LengInv,      # 
    int syn0_1_leng_max,         # currently, it is not in use.

    REAL_t *syn0_2,              # add the second #--------------> add this line
    np.uint32_t *syn0_2_LookUp,  # add the second #--------------> add this line
    np.uint32_t *syn0_2_EndIdx,  # add the second #--------------> add this line
    REAL_t *syn0_2_LengInv,      # add the second #--------------> add this line
    int syn0_2_leng_max,         # add the second #--------------> add this line

    REAL_t *syn0_3,              # add the second #--------------> add this line
    np.uint32_t *syn0_3_LookUp,  # add the second #--------------> add this line
    np.uint32_t *syn0_3_EndIdx,  # add the second #--------------> add this line
    REAL_t *syn0_3_LengInv,      # add the second #--------------> add this line
    int syn0_3_leng_max,         # add the second #--------------> add this line

    REAL_t *syn0_4,              # add the second #--------------> add this line
    np.uint32_t *syn0_4_LookUp,  # add the second #--------------> add this line
    np.uint32_t *syn0_4_EndIdx,  # add the second #--------------> add this line
    REAL_t *syn0_4_LengInv,      # add the second #--------------> add this line
    int syn0_4_leng_max,         # add the second #--------------> add this line

    REAL_t *syn0_5,              # add the second #--------------> add this line
    np.uint32_t *syn0_5_LookUp,  # add the second #--------------> add this line
    np.uint32_t *syn0_5_EndIdx,  # add the second #--------------> add this line
    REAL_t *syn0_5_LengInv,      # add the second #--------------> add this line
    int syn0_5_leng_max,         # add the second #--------------> add this line

    REAL_t *syn0_6,              # add the second #--------------> add this line
    np.uint32_t *syn0_6_LookUp,  # add the second #--------------> add this line
    np.uint32_t *syn0_6_EndIdx,  # add the second #--------------> add this line
    REAL_t *syn0_6_LengInv,      # add the second #--------------> add this line
    int syn0_6_leng_max,         # add the second #--------------> add this line


    REAL_t *syn1neg, 
    REAL_t *word_locks,

    REAL_t *neu1,  
    REAL_t *work,

    REAL_t *neu2,                # 
    REAL_t *work2,               # 

    REAL_t *neu3,                # add the second #--------------> add this line
    REAL_t *work3,               # add the second #--------------> add this line

    REAL_t *neu4,                # add the second #--------------> add this line
    REAL_t *work4,               # add the second #--------------> add this line

    REAL_t *neu5,                # add the second #--------------> add this line
    REAL_t *work5,               # add the second #--------------> add this line

    REAL_t *neu6,                # add the second #--------------> add this line
    REAL_t *work6,               # add the second #--------------> add this line

    REAL_t *neu7,                # add the second #--------------> add this line
    REAL_t *work7,               # add the second #--------------> add this line

    REAL_t *neu_m,               # 
    REAL_t *work_m,              # 

    int cbow_mean, 
    unsigned long long next_random, 
    const int _compute_loss, 
    REAL_t *_running_training_loss_param) nogil
    #===========================================================================================#

#--> NEW for M0XY_P
cdef unsigned long long fieldembed_token_neg_M0XY_P( 
    const REAL_t alpha, 
    const int size,
    const int negative, 
    np.uint32_t *cum_table, 
    unsigned long long cum_table_len, 

    const np.uint32_t indexes[MAX_SENTENCE_LEN], 
    const np.uint32_t indexes_pos[MAX_SENTENCE_LEN], 
    int i, # right word loc_idx
    int j, # left  word loc_idx start
    int k, # left  word loc_idx end

    int use_head,                # 
    int use_sub,                 # #--------------> this value will be 2 instead of 1
    int use_merger,
    int use_token,  # #--------------> pos
    int use_pos,    # #--------------> pos
    REAL_t *syn0, 
    REAL_t *syn_p,  # #--------------> pos
    
    REAL_t *syn0_1,
    np.uint32_t *syn0_1_LookUp,  # 
    np.uint32_t *syn0_1_EndIdx,  # 
    REAL_t *syn0_1_LengInv,      # 
    int syn0_1_leng_max,         # currently, it is not in use.

    REAL_t *syn0_2,              # add the second #--------------> add this line
    np.uint32_t *syn0_2_LookUp,  # add the second #--------------> add this line
    np.uint32_t *syn0_2_EndIdx,  # add the second #--------------> add this line
    REAL_t *syn0_2_LengInv,      # add the second #--------------> add this line
    int syn0_2_leng_max,         # add the second #--------------> add this line

    REAL_t *syn0_3,              # add the second #--------------> add this line
    np.uint32_t *syn0_3_LookUp,  # add the second #--------------> add this line
    np.uint32_t *syn0_3_EndIdx,  # add the second #--------------> add this line
    REAL_t *syn0_3_LengInv,      # add the second #--------------> add this line
    int syn0_3_leng_max,         # add the second #--------------> add this line

    REAL_t *syn0_4,              # add the second #--------------> add this line
    np.uint32_t *syn0_4_LookUp,  # add the second #--------------> add this line
    np.uint32_t *syn0_4_EndIdx,  # add the second #--------------> add this line
    REAL_t *syn0_4_LengInv,      # add the second #--------------> add this line
    int syn0_4_leng_max,         # add the second #--------------> add this line

    REAL_t *syn0_5,              # add the second #--------------> add this line
    np.uint32_t *syn0_5_LookUp,  # add the second #--------------> add this line
    np.uint32_t *syn0_5_EndIdx,  # add the second #--------------> add this line
    REAL_t *syn0_5_LengInv,      # add the second #--------------> add this line
    int syn0_5_leng_max,         # add the second #--------------> add this line

    REAL_t *syn0_6,              # add the second #--------------> add this line
    np.uint32_t *syn0_6_LookUp,  # add the second #--------------> add this line
    np.uint32_t *syn0_6_EndIdx,  # add the second #--------------> add this line
    REAL_t *syn0_6_LengInv,      # add the second #--------------> add this line
    int syn0_6_leng_max,         # add the second #--------------> add this line


    REAL_t *syn1neg, 
    REAL_t *word_locks,

    REAL_t *neu1,  
    REAL_t *work,

    REAL_t *neu2,                # 
    REAL_t *work2,               # 

    REAL_t *neu3,                # add the second #--------------> add this line
    REAL_t *work3,               # add the second #--------------> add this line

    REAL_t *neu4,                # add the second #--------------> add this line
    REAL_t *work4,               # add the second #--------------> add this line

    REAL_t *neu5,                # add the second #--------------> add this line
    REAL_t *work5,               # add the second #--------------> add this line

    REAL_t *neu6,                # add the second #--------------> add this line
    REAL_t *work6,               # add the second #--------------> add this line

    REAL_t *neu7,                # add the second #--------------> add this line
    REAL_t *work7,               # add the second #--------------> add this line

    REAL_t *neu_m,               # 
    REAL_t *work_m,              # 

    REAL_t *neu_p,               # #--------------> pos
    REAL_t *work_p,              # #--------------> pos

    int cbow_mean, 
    unsigned long long next_random, 
    const int _compute_loss, 
    REAL_t *_running_training_loss_param)  nogil
    #===========================================================================================#
