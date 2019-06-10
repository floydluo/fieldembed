#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Optimized cython functions for training :class:`~gensim.models.word2vec.Word2Vec` model."""

import cython
import numpy as np

cimport numpy as np

from libc.math cimport exp
from libc.math cimport log
from libc.string cimport memset

# scipy <= 0.15
try:
    from scipy.linalg.blas import fblas
except ImportError:
    # in scipy > 0.15, fblas function has been removed
    import scipy.linalg.blas as fblas

REAL = np.float32

DEF MAX_SENTENCE_LEN = 10000

cdef scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)  # y = x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x
cdef sdot_ptr  sdot =<sdot_ptr> PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)  # sqrt(x^2)
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer) # x = alpha * x

DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE
cdef REAL_t[EXP_TABLE_SIZE] LOG_TABLE

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0


############################################## UTILS TOOL
# for when fblas.sdot returns a double
cdef REAL_t our_dot_double(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    return <REAL_t>dsdot(N, X, incX, Y, incY)

# for when fblas.sdot returns a float
cdef REAL_t our_dot_float(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    return <REAL_t>sdot(N, X, incX, Y, incY)

# for when no blas available
cdef REAL_t our_dot_noblas(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    # not a true full dot()-implementation: just enough for our cases
    cdef int i
    cdef REAL_t a
    a = <REAL_t>0.0
    for i from 0 <= i < N[0] by 1:
        a += X[i] * Y[i]
    return a

# for when no blas available
cdef void our_saxpy_noblas(const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil:
    cdef int i
    for i from 0 <= i < N[0] by 1:
        Y[i * (incY[0])] = (alpha[0]) * X[i * (incX[0])] + Y[i * (incY[0])]

# to support random draws from negative-sampling cum_table
cdef inline unsigned long long bisect_left(np.uint32_t *a, unsigned long long x, unsigned long long lo, unsigned long long hi) nogil:
    cdef unsigned long long mid
    while hi > lo:
        mid = (lo + hi) >> 1
        if a[mid] >= x:
            hi = mid
        else:
            lo = mid + 1
    return lo

# this quick & dirty RNG apparently matches Java's (non-Secure)Random
# note this function side-effects next_random to set up the next number
cdef inline unsigned long long random_int32(unsigned long long *next_random) nogil:
    cdef unsigned long long this_random = next_random[0] >> 16
    next_random[0] = (next_random[0] * <unsigned long long>25214903917ULL + 11) & 281474976710655ULL
    return this_random
############################################## UTILS TOOL


cdef init_w2v_config(Word2VecConfig *c, model, alpha, compute_loss, _work, _neu1=None):
    # c[0].hs = model.hs
    c[0].sg = model.sg
    c[0].negative = model.negative
    c[0].sample = (model.vocabulary.sample != 0)
    c[0].cbow_mean = model.standard_grad
    c[0].window = model.window
    c[0].workers = model.workers

    c[0].compute_loss = (1 if compute_loss else 0)
    c[0].running_training_loss = model.running_training_loss

    #######################################################################
    # print(model.wv.vectors.shape)
    c[0].syn0 = <REAL_t *>(np.PyArray_DATA(model.wv.vectors))
    #######################################################################

    c[0].word_locks = <REAL_t *>(np.PyArray_DATA(model.trainables.vectors_lockf))
    c[0].alpha = alpha
    c[0].size = model.wv.vector_size

    if c[0].negative:
        try:
            c[0].syn1neg = <REAL_t *>(np.PyArray_DATA(model.wv_neg.vectors))
        except:
            c[0].syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        c[0].cum_table = <np.uint32_t *>(np.PyArray_DATA(model.vocabulary.cum_table))
        c[0].cum_table_len = len(model.vocabulary.cum_table)
    if c[0].negative or c[0].sample:
        c[0].next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    c[0].work = <REAL_t *>np.PyArray_DATA(_work)

    if _neu1 is not None:
        c[0].neu1 = <REAL_t *>np.PyArray_DATA(_neu1)
################################################################# OLD WAY

################################################################# OLD WAY
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
    REAL_t *_running_training_loss_param) nogil:
    #=================================================#
        
    # cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_dot, log_e_f_dot
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f_dot = our_dot(&size, &syn0[row1], &ONE, &syn1neg[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha

        if _compute_loss == 1:
            f_dot = (f_dot if d == 0  else -f_dot)
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot

        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, &syn0[row1], &ONE, &syn1neg[row2], &ONE)

    our_saxpy(&size, &word_locks[word2_index], work, &ONE, &syn0[row1], &ONE)

    return next_random

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
    REAL_t *_running_training_loss_param) nogil:
    #=================================================#


    

    # cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, count, inv_count = 1.0, label, log_e_f_dot, f_dot
    cdef np.uint32_t target_index, word_index
    cdef int d, m

    word_index = indexes[i]

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i:
            continue
        else:
            count += ONEF
            our_saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF/count
    # if cbow_mean:
    sscal(&size, &inv_count, neu1, &ONE)  # (does this need BLAS-variants like saxpy?)

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f_dot = our_dot(&size, neu1, &ONE, &syn1neg[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha

        if _compute_loss == 1:
            f_dot = (f_dot if d == 0  else -f_dot)
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot

        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)

    if cbow_mean:  # divide error over summed window vectors // if standard_grad
        sscal(&size, &inv_count, work, &ONE)  # (does this need BLAS-variants like saxpy?)

    for m in range(j,k):
        if m == i:
            continue
        else:
            our_saxpy(&size, &word_locks[indexes[m]], work, &ONE, &syn0[indexes[m]*size], &ONE)

    return next_random

################################################################# Field Embedding WITH NLPText
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
    REAL_t *_running_training_loss_param) nogil:
    #=======================================================================#




    # cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL

    cdef REAL_t count, inv_count = 1.0
    cdef REAL_t label, f, g, f_dot, log_e_f_dot
    
    cdef np.uint32_t target_index, word_index
    cdef int d, m


    word_index = indexes[i]  ########### S: get index for right token voc_idx


    #################################### S: calculate hProj from syn0
    # neu1 ===> hProj
    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k): # sg case: j = k; loop left tokens here
        if m == i:
            continue
        else:
            count += ONEF
            our_saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF/count
    #if cbow_mean:
    sscal(&size, &inv_count, neu1, &ONE)  # (does this need BLAS-variants like saxpy? # no, you don't)
    #################################### E: calculate hProj from syn0

    #################################### S: calculate hProj_grad and update syn1neg
    # work ===> hProj_grad
    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index # word_index is vocab_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f_dot = our_dot(&size, neu1, &ONE, &syn1neg[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha

        if _compute_loss == 1:
            f_dot = (f_dot if d == 0  else -f_dot)
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]

        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE) # accumulate work
        our_saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)
    #################################### E: calculate hProj_grad and update syn1neg


    #################################### S: update syn0 gradient
    if cbow_mean:  # divide error over summed window vectors # if standard grad
        # big questions here!!!
        sscal(&size, &inv_count, work, &ONE)  # (does this need BLAS-variants like saxpy?)

    for m in range(j,k): 
        if m == i:
            continue
        else:
            # Here,actually, it looks up the indexes again.
            # Why not store these items some where?
            # Is it a case to trade off between time and space?
            our_saxpy(&size, &word_locks[indexes[m]], work, &ONE, &syn0[indexes[m]*size], &ONE)
    #################################### E: update syn0 gradient

    return next_random
################################################################# Field Embedding WITH NLPText

################################################################# Field Embedding 0X1
#--> NEW for 0X1
cdef init_w2v_config_0X1(
    Word2VecConfig *c, 
    model, 
    alpha, 
    compute_loss, 
    _work, 
    _neu1, 
    _work2, 
    _neu2):

    c[0].sg = model.sg
    c[0].negative = model.negative
    c[0].sample = (model.vocabulary.sample != 0)
    c[0].cbow_mean = model.standard_grad
    c[0].window = model.window
    c[0].workers = model.workers
    c[0].compute_loss = (1 if compute_loss else 0)
    c[0].running_training_loss = model.running_training_loss

    #######################################################################
    # c[0].total_fields = len(sg_model.field_headfields_meta) # new; total_fields = 1
    # assume that total_fields is 1    
    # 0 X 1 
    c[0].use_head = model.use_head # loop head
    # print('\n')
    if c[0].use_head:
        # print(model.field_head[0][1].vectors.shape)
        c[0].syn0  = <REAL_t *>(np.PyArray_DATA(model.field_head[0][1].vectors)) # currently, use this
    
    c[0].use_sub  = model.use_sub
    if c[0].use_sub:
        c[0].syn0_1 = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][0][0].vectors)) # currently, use this
        c[0].syn0_1_LookUp   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][0][1]))  # lookup
        c[0].syn0_1_EndIdx   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][0][2]))  # endIdx
        c[0].syn0_1_LengInv  = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][0][3]))  # leng_inv
        c[0].syn0_1_leng_max = model.field_sub[0][0][4]                                    # leng_max
    #######################################################################

    c[0].word_locks = <REAL_t *>(np.PyArray_DATA(model.trainables.vectors_lockf))
    c[0].alpha = alpha
    c[0].size = model.wv.vector_size

    if c[0].negative:
        c[0].syn1neg = <REAL_t *>(np.PyArray_DATA(model.wv_neg.vectors))
        c[0].cum_table = <np.uint32_t *>(np.PyArray_DATA(model.vocabulary.cum_table))
        c[0].cum_table_len = len(model.vocabulary.cum_table)
    if c[0].negative or c[0].sample:
        c[0].next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    c[0].work = <REAL_t *>np.PyArray_DATA(_work)
    c[0].neu1 = <REAL_t *>np.PyArray_DATA(_neu1)

    #######################################################################
    c[0].work2 = <REAL_t *>np.PyArray_DATA(_work2)
    c[0].neu2  = <REAL_t *>np.PyArray_DATA(_neu2)
    #######################################################################

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
    REAL_t *syn0_1_LengInv,      # 
    int syn0_1_leng_max,         # currently, it is not in use.

    REAL_t *syn1neg, 
    REAL_t *word_locks,

    REAL_t *neu1,  
    REAL_t *work,

    REAL_t *neu2,                # 
    REAL_t *work2,               # 
    # int sg,
    int cbow_mean, 
    unsigned long long next_random, 
    const int _compute_loss, 
    REAL_t *_running_training_loss_param) nogil:
    #===========================================================================================#

    # cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    
    cdef REAL_t label
    cdef REAL_t f_dot,  f,  g,  log_e_f_dot
    cdef REAL_t g2
    
    cdef int d, m  # d is for looping negative, m is for looping left words, 
    cdef int n # n is for looping left word's grain, shoud n be an int?
    cdef int left_word
    cdef int gs, ge
    cdef np.uint32_t target_index, word_index,  grain_index # should left_word be an int?

    cdef REAL_t count,  inv_count = 1.0
    # cdef REAL_t count2, inv_count2 = 1.0
    cdef REAL_t word_lenginv = 1.0

    # Here word_index is np.uint32_t. very interesting
    word_index = indexes[i]  ########### S: get index for right token voc_idx
    # because indexes is np.int32_t

    #################################### S: Count the left tokens number
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i: # j, m, i, k are int
            continue
        else:
            count += ONEF
    if count > (<REAL_t>0.5):  # when using sg, count is 1. count is cw in word2vec.c
        inv_count = ONEF/count
    # else: inv_count = 1.0
    #################################### E: Count the left tokens number


    #################################### E: calculate hProj from syn0
    if use_head: # this is correct
        memset(neu1, 0, size * cython.sizeof(REAL_t))
        for m in range(j, k): # sg case: k = j+1; loop left tokens here
            if m == i: # j, m, i, k are int
                continue
            else:
                our_saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
        # if not sg:
        sscal(&size, &inv_count, neu1, &ONE)  # (does this need BLAS-variants like saxpy? # no, you don't)
    #################################### E: calculate hProj from syn0


    #################################### S: calculate hProj from syn0
    if use_sub: # this is correct
        memset(neu2, 0, size * cython.sizeof(REAL_t))
        # count2 = <REAL_t>0.0 // different weight: using their code directly, don't need to reproduce it.
        for m in range(j, k): # sg case: j = k; loop left tokens here
            if m == i:
                continue
            else:
                left_word  = indexes[m]                  # left_word: uint32 to int
                ###################################################################
                word_lenginv = syn0_1_LengInv[left_word] # word_lenginv: REAL_t
                gs = syn0_1_EndIdx[left_word-1]
                ge = syn0_1_EndIdx[left_word]
                for n in range(gs, ge):
                    # n is also np.uint_32
                    # should n be an int? just like m?
                    grain_index = syn0_1_LookUp[n] # syn0_1_LookUp is a np.uint_32
                    # grain_index is also np.uint_32
                    our_saxpy(&size, &word_lenginv, &syn0_1[grain_index * size],  &ONE, neu2, &ONE)
                ###################################################################
        # if not sg:
        sscal(&size, &inv_count, neu2, &ONE)  # (does this need BLAS-variants like saxpy? # no, you don't)
    #################################### E: calculate hProj from syn0


    #################################### S: calculate hProj_grad and update syn1neg
    if use_head:
        memset(work,  0, size * cython.sizeof(REAL_t))
    if use_sub:
        memset(work2, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        # d is int
        if d == 0:
            target_index = word_index # word_index is vocab_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue 
            label = <REAL_t>0.0

        row2 = target_index * size # target_index: np.uint32, size: int; row2: long long 
        ##########################################################################
        if use_head:
            f_dot = our_dot(&size, neu1, &ONE, &syn1neg[row2], &ONE)
            if _compute_loss == 1: # TODO
                f_dot = (f_dot if d == 0  else -f_dot)
                if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                    continue # this is still an issue
                log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]

            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue # quit: this is unreasonable.
            f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

            g = (label - f) * alpha
            our_saxpy(&size, &g,  &syn1neg[row2], &ONE, work, &ONE) # accumulate work

        if use_sub:
            ################################################################
            f_dot = our_dot(&size, neu2, &ONE, &syn1neg[row2], &ONE)
            if _compute_loss == 1: # TODO
                f_dot = (f_dot if d == 0  else -f_dot)
                if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                    continue # this is still an issue
                log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]
            
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue # quit: this is unreasonable.
            f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            
            g2 = (label - f) * alpha
            our_saxpy(&size, &g2, &syn1neg[row2], &ONE, work2, &ONE) # accumulate work
            ################################################################
        #########################################################################

        ##########################################################################
        if use_head:
            our_saxpy(&size, &g,  neu1, &ONE, &syn1neg[row2], &ONE)
        if use_sub:
            our_saxpy(&size, &g2, neu2, &ONE, &syn1neg[row2], &ONE)
        ##########################################################################
    #################################### E: calculate hProj_grad and update syn1neg


    #################################### S: update syn0 gradient
    if cbow_mean:  # use standard grad
        # set cbow_mean = 1 is the standard gradient
        # other wise, it is using a larger graident step size
        if use_head:
            sscal(&size, &inv_count, work,  &ONE)  # (does this need BLAS-variants like saxpy?)
        if use_sub:
            sscal(&size, &inv_count, work2, &ONE)  
  
    if use_head:
        for m in range(j,k): 
            if m == i:
                continue
            else:
                our_saxpy(&size, &word_locks[indexes[m]], work, &ONE, &syn0[indexes[m]*size], &ONE)

    if use_sub:
        for m in range(j, k): # sg case: j = k; loop left tokens here
            if m == i:
                continue
            else:
                ############### This four lines are important ###############
                left_word = indexes[m] # left_word  #  from uint32 to int 
                
                word_lenginv = syn0_1_LengInv[left_word] # word_lenginv: REAL_t
                gs = syn0_1_EndIdx[left_word-1]     #  from uint32 to int 
                ge = syn0_1_EndIdx[left_word]       #  from uint32 to int 
                for n in range(gs, ge):             #  n is int
                    grain_index = syn0_1_LookUp[n]  # grain_index is uint
                    our_saxpy(&size, &word_lenginv, work2, &ONE, &syn0_1[grain_index * size], &ONE)       
    ################################### E: update syn0 gradient
    return next_random
################################################################# Field Embedding 0X1


##############################################
def train_batch_sg(model, sentences, alpha, _work, compute_loss):

    cdef Word2VecConfig c
    cdef int i, j, k, g
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end

    init_w2v_config(&c, model, alpha, compute_loss, _work)


    # prepare C structures so we can go "full C" and release the Python GIL
    vlookup = model.wv.vocab
    c.sentence_idx[0] = 0  # indices of the first sentence always start at 0
    for sent in sentences:
        if not sent:
            continue  # ignore empty sentences; leave effective_sentences unchanged
        for token in sent:
            word = vlookup[token] if token in vlookup else None
            if word is None:
                continue  # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            if c.sample and word.sample_int < random_int32(&c.next_random):
                continue
            c.indexes[effective_words] = word.index

            effective_words += 1
            if effective_words == MAX_SENTENCE_LEN:
                break  # TODO: log warning, tally overflow?

        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        # indices of sentence number X are between <sentence_idx[X], sentence_idx[X])
        effective_sentences += 1
        c.sentence_idx[effective_sentences] = effective_words

        if effective_words == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?

    # precompute "reduced window" offsets in a single randint() call
    for i, item in enumerate(model.random.randint(0, c.window, effective_words)):
        c.reduced_windows[i] = item

    # release GIL & train on all sentences
    with nogil:
        for sent_idx in range(effective_sentences):
            idx_start = c.sentence_idx[sent_idx]
            idx_end = c.sentence_idx[sent_idx + 1]
            for i in range(idx_start, idx_end):
                j = i - c.window + c.reduced_windows[i]
                if j < idx_start:
                    j = idx_start
                k = i + c.window + 1 - c.reduced_windows[i]
                if k > idx_end:
                    k = idx_end
                for j in range(j, k):
                    if j == i:
                        continue
                    # if c.hs:
                    #     w2v_fast_sentence_sg_hs(c.points[i], c.codes[i], c.codelens[i], c.syn0, c.syn1, c.size, c.indexes[j], c.alpha, c.work, c.word_locks, c.compute_loss, &c.running_training_loss)
                    if c.negative:
                        c.next_random = w2v_fast_sentence_sg_neg(c.negative, c.cum_table, c.cum_table_len, c.syn0, c.syn1neg, c.size, c.indexes[i], c.indexes[j], c.alpha, c.work, c.next_random, c.word_locks, c.compute_loss, &c.running_training_loss)

    model.running_training_loss = c.running_training_loss
    return effective_words

############################################
def train_batch_cbow(model, sentences, alpha, _work, _neu1, compute_loss):
    cdef Word2VecConfig c
    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end

    init_w2v_config(&c, model, alpha, compute_loss, _work, _neu1)

    # prepare C structures so we can go "full C" and release the Python GIL
    vlookup = model.wv.vocab
    c.sentence_idx[0] = 0  # indices of the first sentence always start at 0
    for sent in sentences:
        if not sent:
            continue  # ignore empty sentences; leave effective_sentences unchanged
        for token in sent:
            word = vlookup[token] if token in vlookup else None
            if word is None:
                continue  # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            if c.sample and word.sample_int < random_int32(&c.next_random):
                continue
            c.indexes[effective_words] = word.index
            effective_words += 1
            if effective_words == MAX_SENTENCE_LEN:
                break  # TODO: log warning, tally overflow?

        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        # indices of sentence number X are between <sentence_idx[X], sentence_idx[X])
        effective_sentences += 1
        c.sentence_idx[effective_sentences] = effective_words

        if effective_words == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?

    # precompute "reduced window" offsets in a single randint() call
    for i, item in enumerate(model.random.randint(0, c.window, effective_words)):
        c.reduced_windows[i] = item

    # release GIL & train on all sentences
    with nogil:
        for sent_idx in range(effective_sentences):
            idx_start = c.sentence_idx[sent_idx]
            idx_end = c.sentence_idx[sent_idx + 1]
            for i in range(idx_start, idx_end):
                j = i - c.window + c.reduced_windows[i]
                if j < idx_start:
                    j = idx_start
                k = i + c.window + 1 - c.reduced_windows[i]
                if k > idx_end:
                    k = idx_end
                
                if c.negative:
                    c.next_random = w2v_fast_sentence_cbow_neg(c.negative, c.cum_table, c.cum_table_len, 
                                                               c.codelens, c.neu1, c.syn0, c.syn1neg, c.size, 
                                                               c.indexes, c.alpha, c.work, i, j, k, c.cbow_mean, 
                                                               c.next_random, c.word_locks, 
                                                               c.compute_loss, &c.running_training_loss)

    model.running_training_loss = c.running_training_loss
    return effective_words



cdef int SUBSAMPLING = 1
##############################################
def train_batch_sg_nlptext(model, indexes, sentence_idx, alpha, _work, compute_loss, subsampling = SUBSAMPLING):

    cdef Word2VecConfig c
    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end
    cdef int word_vocidx

    # prepare c with store the information for this whole job 
    init_w2v_config(&c, model, alpha, compute_loss, _work)

    if subsampling:
        vlookup = model.wv.vocab_values
        for sent_idx in range(len(sentence_idx)):
            # step1: get every sentence's idx_start and idx_end
            if sent_idx == 0:
                idx_start = 0
            else:
                idx_start = sentence_idx[sent_idx-1]
            idx_end = sentence_idx[sent_idx]

            # step2: loop every tokens in this sentence, drop special tokens and use subsampling
            for word_vocidx in indexes[idx_start: idx_end]:
                if word_vocidx <= 3:
                    continue
                if c.sample and vlookup[word_vocidx].sample_int < random_int32(&c.next_random): # 
                    continue
                # NOTICE: c.sentence_idx[0] = 0  # indices of the first sentence always start at 0
                # my sentence_idx is not started from 0
                c.indexes[effective_words] = word_vocidx
                effective_words +=1
                if effective_words == MAX_SENTENCE_LEN:
                    break  # TODO: log warning, tally overflow?

            # step3: add the new idx_end for this sentence, that is, the value of effective_words
            c.sentence_idx[effective_sentences] = effective_words
            effective_sentences += 1
            if effective_words == MAX_SENTENCE_LEN:
                break  # TODO: log warning, tally overflow?

    else:
        # In this case, we don't drop special tokens or use subsampling 
        effective_words = len(indexes)
        effective_sentences = len(sentence_idx) # different from the original sentence_idx and effective_sentences
        for i, item in enumerate(indexes):
            c.indexes[i] = item
        for i, item in enumerate(sentence_idx):
            c.sentence_idx[i] = item

    # use dynamic windows
    for i, item in enumerate(model.random.randint(0, c.window, effective_words)):
        c.reduced_windows[i] = item

    with nogil: # LESSION: you should notice this nogil, otherwise the threads are rubbish
        for sent_idx in range(effective_sentences):

            # idx_start and idx_end
            if sent_idx == 0:
                idx_start = 0
            else:
                idx_start = c.sentence_idx[sent_idx-1]
            idx_end = c.sentence_idx[sent_idx]
            # then indexes[idx_start: idx_end] is the current sentence.
            # print(idx_start, idx_end)

            for i in range(idx_start, idx_end):
                j = i - c.window + c.reduced_windows[i]
                if j < idx_start:
                    j = idx_start
                k = i + c.window + 1 - c.reduced_windows[i]
                if k > idx_end:
                    k = idx_end
                # print(j, i, k)
                for j in range(j, k): # change the first j to another name: such as t.
                    if j == i:
                        continue
                    if c.negative:
                        c.next_random = w2v_fast_sentence_sg_neg(c.negative, c.cum_table, c.cum_table_len, c.syn0, c.syn1neg, 
                                                                 c.size, 
                                                                 c.indexes[i], c.indexes[j], c.alpha, c.work, c.next_random, c.word_locks, c.compute_loss, &c.running_training_loss)

    model.running_training_loss = c.running_training_loss
    return effective_words


##############################################
def train_batch_cbow_nlptext(model, indexes, sentence_idx, alpha, _work, _neu1, compute_loss, subsampling = SUBSAMPLING):

    cdef Word2VecConfig c
    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end
    cdef int word_vocidx


    init_w2v_config(&c, model, alpha, compute_loss, _work, _neu1) # this is the difference between sg and cbow
    
    if subsampling:
        vlookup = model.wv.vocab_values
        for sent_idx in range(len(sentence_idx)):
            # step1: get every sentence's idx_start and idx_end
            if sent_idx == 0:
                idx_start = 0
            else:
                idx_start = sentence_idx[sent_idx-1]
            idx_end = sentence_idx[sent_idx]

            # step2: loop every tokens in this sentence, drop special tokens and use subsampling
            for word_vocidx in indexes[idx_start: idx_end]:
                if word_vocidx <= 3:
                    continue
                if c.sample and vlookup[word_vocidx].sample_int < random_int32(&c.next_random):
                    continue
                # NOTICE: c.sentence_idx[0] = 0  # indices of the first sentence always start at 0
                # my sentence_idx is not started from 0
                c.indexes[effective_words] = word_vocidx
                effective_words +=1
                if effective_words == MAX_SENTENCE_LEN:
                    break  # TODO: log warning, tally overflow?

            # step3: add the new idx_end for this sentence, that is, the value of effective_words
            c.sentence_idx[effective_sentences] = effective_words
            effective_sentences += 1
            if effective_words == MAX_SENTENCE_LEN:
                break  # TODO: log warning, tally overflow?

    else:
        # In this case, we don't drop special tokens or use subsampling 
        effective_words = len(indexes)
        effective_sentences = len(sentence_idx) # different from the original sentence_idx and effective_sentences
        for i, item in enumerate(indexes):
            c.indexes[i] = item
        for i, item in enumerate(sentence_idx):
            c.sentence_idx[i] = item

    # precompute "reduced window" offsets in a single randint() call
    for i, item in enumerate(model.random.randint(0, c.window, effective_words)):
        c.reduced_windows[i] = item

    with nogil: # LESSION: you should notice this nogil, otherwise the threads are rubbish
        for sent_idx in range(effective_sentences):

            # idx_start and idx_end
            if sent_idx == 0:
                idx_start = 0
            else:
                idx_start = c.sentence_idx[sent_idx-1]      # this is the difference between nlptext or original version
            idx_end = c.sentence_idx[sent_idx]              # this is the difference between nlptext or original version
            # then indexes[idx_start: idx_end] is the current sentence.

            for i in range(idx_start, idx_end):
                j = i - c.window + c.reduced_windows[i]
                if j < idx_start:
                    j = idx_start
                k = i + c.window + 1 - c.reduced_windows[i]
                if k > idx_end:
                    k = idx_end
                # print(j, i, k)
                if c.negative:
                    c.next_random = w2v_fast_sentence_cbow_neg(c.negative, c.cum_table, c.cum_table_len, 
                                                               c.codelens, c.neu1, c.syn0, c.syn1neg, c.size, 
                                                               c.indexes, c.alpha, c.work, i, j, k, c.cbow_mean, 
                                                               c.next_random, c.word_locks, 
                                                               c.compute_loss, &c.running_training_loss)

    model.running_training_loss = c.running_training_loss
    return effective_words


##############################################
def train_batch_fieldembed_token(model, indexes, sentence_idx, alpha, _work, _neu1, compute_loss, subsampling = SUBSAMPLING):

    cdef Word2VecConfig c
    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end

    cdef int word_vocidx
    # cdef int sg
    # print('before init')
    init_w2v_config(&c, model, alpha, compute_loss, _work, _neu1) # this is the difference between sg and cbow
    
    if subsampling:
        vlookup = model.wv.vocab_values
        for sent_idx in range(len(sentence_idx)):
            # step1: get every sentence's idx_start and idx_end
            if sent_idx == 0:
                idx_start = 0
            else:
                idx_start = sentence_idx[sent_idx-1]
            idx_end = sentence_idx[sent_idx]

            # step2: loop every tokens in this sentence, drop special tokens and use subsampling
            for word_vocidx in indexes[idx_start: idx_end]:
                if word_vocidx <= 3:
                    continue
                if c.sample and vlookup[word_vocidx].sample_int < random_int32(&c.next_random):
                    continue
                # NOTICE: c.sentence_idx[0] = 0  # indices of the first sentence always start at 0
                # my sentence_idx is not started from 0
                c.indexes[effective_words] = word_vocidx
                effective_words +=1
                if effective_words == MAX_SENTENCE_LEN:
                    break  # TODO: log warning, tally overflow?

            # step3: add the new idx_end for this sentence, that is, the value of effective_words
            c.sentence_idx[effective_sentences] = effective_words
            effective_sentences += 1
            if effective_words == MAX_SENTENCE_LEN:
                break  # TODO: log warning, tally overflow?

    else:
        # In this case, we don't drop special tokens or use downsampling 
        effective_words = len(indexes)
        effective_sentences = len(sentence_idx) # different from the original sentence_idx and effective_sentences
        for i, item in enumerate(indexes):
            c.indexes[i] = item
        for i, item in enumerate(sentence_idx):
            c.sentence_idx[i] = item

    # precompute "reduced window" offsets in a single randint() call
    for i, item in enumerate(model.random.randint(0, c.window, effective_words)):
        c.reduced_windows[i] = item


    with nogil: # LESSION: you should notice this nogil, otherwise the threads are rubbish
        for sent_idx in range(effective_sentences):
            # idx_start and idx_end
            idx_end = c.sentence_idx[sent_idx]
            if sent_idx == 0:
                idx_start = 0
            else:
                idx_start = c.sentence_idx[sent_idx-1]
            
            # then indexes[idx_start: idx_end] is the current sentence.
            # print(idx_start, idx_end)
            for i in range(idx_start, idx_end):
                j = i - c.window + c.reduced_windows[i]
                if j < idx_start:
                    j = idx_start
                k = i + c.window + 1 - c.reduced_windows[i]
                if k > idx_end:
                    k = idx_end
                # print(j, i, k)
                if c.sg == 1:
                    for j in range(j, k): # change the first j to another name: such as t.
                        if j == i:
                            continue
                        # build the batch here
                        c.next_random = fieldembed_token_neg(c.alpha, c.size, c.negative, c.cum_table, c.cum_table_len, 
                            c.indexes, i, j, j + 1, c.syn0, c.syn1neg, c.word_locks, c.neu1, c.work, c.cbow_mean, 
                            c.next_random, c.compute_loss, &c.running_training_loss)
                else:
                    # build the batch here
                    c.next_random = fieldembed_token_neg(c.alpha, c.size, c.negative, c.cum_table, c.cum_table_len, 
                            c.indexes, i, j, k, c.syn0, c.syn1neg, c.word_locks, c.neu1, c.work, c.cbow_mean, 
                            c.next_random, c.compute_loss, &c.running_training_loss)

    model.running_training_loss = c.running_training_loss
    return effective_words


##############################################
#--> NEW for 0X1
def train_batch_fieldembed_0X1(model, indexes, sentence_idx, alpha, _work, _neu1, _work2, _neu2, compute_loss, subsampling = 1):

    cdef Word2VecConfig c
    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end
    cdef int word_vocidx
    # cdef int sg
    # print('before init')
    init_w2v_config_0X1(&c, model, alpha, compute_loss,  _work, _neu1, _work2, _neu2) # this is the difference between sg and cbow
    
    if subsampling:
        vlookup = model.wv.vocab_values
        for sent_idx in range(len(sentence_idx)):
            # step1: get every sentence's idx_start and idx_end
            if sent_idx == 0:
                idx_start = 0
            else:
                idx_start = sentence_idx[sent_idx-1]
            idx_end = sentence_idx[sent_idx]

            # step2: loop every tokens in this sentence, drop special tokens and use downsampling
            for word_vocidx in indexes[idx_start: idx_end]:
                if word_vocidx <= 3:
                    continue
                if c.sample and vlookup[word_vocidx].sample_int < random_int32(&c.next_random):
                    continue
                # NOTICE: c.sentence_idx[0] = 0  # indices of the first sentence always start at 0
                # my sentence_idx is not started from 0
                c.indexes[effective_words] = word_vocidx
                effective_words +=1
                if effective_words == MAX_SENTENCE_LEN:
                    break  # TODO: log warning, tally overflow?

            # step3: add the new idx_end for this sentence, that is, the value of effective_words
            c.sentence_idx[effective_sentences] = effective_words
            effective_sentences += 1
            if effective_words == MAX_SENTENCE_LEN:
                break  # TODO: log warning, tally overflow?

    else:
        # In this case, we don't drop special tokens or use downsampling 
        effective_words = len(indexes)
        effective_sentences = len(sentence_idx) # different from the original sentence_idx and effective_sentences
        for i, item in enumerate(indexes):
            c.indexes[i] = item
        for i, item in enumerate(sentence_idx):
            c.sentence_idx[i] = item

    # precompute "reduced window" offsets in a single randint() call
    for i, item in enumerate(model.random.randint(0, c.window, effective_words)):
        c.reduced_windows[i] = item

    with nogil: # LESSION: you should notice this nogil, otherwise the threads are rubbish
        for sent_idx in range(effective_sentences):
            # idx_start and idx_end
            idx_end = c.sentence_idx[sent_idx]
            if sent_idx == 0:
                idx_start = 0
            else:
                idx_start = c.sentence_idx[sent_idx-1]

            for i in range(idx_start, idx_end):
                j = i - c.window + c.reduced_windows[i]
                if j < idx_start:
                    j = idx_start
                k = i + c.window + 1 - c.reduced_windows[i]
                if k > idx_end:
                    k = idx_end
                # print(j, i, k)

                if c.sg == 1:
                    for j in range(j, k): # change the first j to another name: such as t.
                        if j == i:
                            continue
                        c.next_random = fieldembed_token_neg_0X1(c.alpha, c.size, c.negative, c.cum_table, c.cum_table_len, 
                            c.indexes, i, j, j + 1, 
                            c.use_head, c.use_sub,  # new
                            c.syn0, 
                            c.syn0_1, c.syn0_1_LookUp, c.syn0_1_EndIdx, c.syn0_1_LengInv, c.syn0_1_leng_max, # new
                            c.syn1neg, c.word_locks, 
                            c.neu1, c.work, 
                            c.neu2, c.work2,        # new
                            c.cbow_mean, c.next_random, c.compute_loss, &c.running_training_loss)
                else:
                    # build the batch here
                    c.next_random = fieldembed_token_neg_0X1(c.alpha, c.size, c.negative, c.cum_table, c.cum_table_len, 
                            c.indexes, i, j, k, 
                            c.use_head, c.use_sub,  # new
                            c.syn0, 
                            c.syn0_1, c.syn0_1_LookUp, c.syn0_1_EndIdx, c.syn0_1_LengInv, c.syn0_1_leng_max, # new
                            c.syn1neg, c.word_locks, 
                            c.neu1, c.work, 
                            c.neu2, c.work2,        # new
                            c.cbow_mean, c.next_random, c.compute_loss, &c.running_training_loss)

    model.running_training_loss = c.running_training_loss
    return effective_words


def init():
    """Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized into table EXP_TABLE.
     Also calculate log(sigmoid(x)) into LOG_TABLE.

    Returns
    -------
    {0, 1, 2}
        Enumeration to signify underlying data type returned by the BLAS dot product calculation.
        0 signifies double, 1 signifies double, and 2 signifies that custom cython loops were used
        instead of BLAS.

    """
    global our_dot
    global our_saxpy

    cdef int i
    cdef float *x = [<float>10.0]
    cdef float *y = [<float>0.01]
    cdef float expected = <float>0.1
    cdef int size = 1
    cdef double d_res
    cdef float *p_res

    # build the sigmoid table
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = <REAL_t>exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))
        LOG_TABLE[i] = <REAL_t>log( EXP_TABLE[i] )

    # check whether sdot returns double or float
    d_res = dsdot(&size, x, &ONE, y, &ONE)
    p_res = <float *>&d_res
    if abs(d_res - expected) < 0.0001:
        our_dot = our_dot_double
        our_saxpy = saxpy
        # our_saxpy = our_saxpy_noblas
        return 0  # double
    elif abs(p_res[0] - expected) < 0.0001:
        our_dot = our_dot_float
        our_saxpy = saxpy
        # our_saxpy = our_saxpy_noblas
        return 1  # float
    else:
        # neither => use cython loops, no BLAS
        # actually, the BLAS is so messed up we'll probably have segfaulted above and never even reach here
        our_dot = our_dot_noblas
        our_saxpy = our_saxpy_noblas
        return 2

FAST_VERSION = init()  # initialize the module
MAX_WORDS_IN_BATCH = MAX_SENTENCE_LEN
