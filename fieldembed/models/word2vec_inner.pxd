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


cdef struct Word2VecConfig:
    int hs, negative, sample, compute_loss, size, window, cbow_mean, workers

    REAL_t running_training_loss, alpha

    REAL_t *syn0
    REAL_t *word_locks
    REAL_t *work
    REAL_t *neu1

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


# cdef struct Word2VecConfig_NLPText:
#     int hs, negative, sample, compute_loss, size, window, cbow_mean, workers
    
#     REAL_t running_training_loss, alpha

#     REAL_t *syn0
#     REAL_t *word_locks
    

#     # interesting
#     REAL_t *work  # change in each pair
#     REAL_t *neu1  # change in each pair

#     int codelens[MAX_SENTENCE_LEN]
    
#     # change in each batch
#     # interesting
#     np.uint32_t reduced_windows[MAX_SENTENCE_LEN]   # This is produced inside each batch, make sure that batch size < MAX_SENTENCE_LEN
#     np.uint32_t indexes[MAX_SENTENCE_LEN]         # <Token    Info>
#     int sentence_idx[MAX_SENTENCE_LEN + 1]    # <Sentence Info>
#     np.uint32_t *cum_table # this is constant. 

#     # For negative sampling
#     REAL_t *syn1neg
    
#     unsigned long long cum_table_len
#     # for sampling (negative and frequent-word downsampling)
#     unsigned long long next_random




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
    _neu1=*)


# cdef unsigned long long w2v_nlptext_sg_neg(
#     const int negative, 
#     np.uint32_t *cum_table, 
#     unsigned long long cum_table_len,
#     REAL_t *syn0, 
#     REAL_t *syn1neg, 
#     const int size, 
#     const np.uint32_t word_index,
#     const np.uint32_t word2_index, 
#     const REAL_t alpha, 
#     REAL_t *work,
#     unsigned long long next_random, 
#     REAL_t *word_locks,
#     const int _compute_loss, 
#     REAL_t *_running_training_loss_param) nogil


# cdef unsigned long long w2v_nlptext_cbow_neg( 
#     const int negative, 
#     np.uint32_t *cum_table, 
#     unsigned long long cum_table_len, 
#     int codelens[MAX_SENTENCE_LEN],
#     REAL_t *neu1,  
#     REAL_t *syn0, 
#     REAL_t *syn1neg, 
#     const int size,
#     np.uint32_t *indexes, 
#     const REAL_t alpha, 
#     REAL_t *work,
#     int i, int j, int k, 
#     int cbow_mean, 
#     unsigned long long next_random, 
#     REAL_t *word_locks,
#     const int _compute_loss, 
#     REAL_t *_running_training_loss_param) nogil


# cdef init_w2v_config_nlptext(
#     Word2VecConfig_NLPText *c, 
#     model, 
#     alpha, 
#     compute_loss,
#     _work,
#     _neu1=*)