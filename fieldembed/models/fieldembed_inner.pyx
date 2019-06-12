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
    _neu_m):

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

    c[0].use_merger = model.use_merger


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

    c[0].work_m = <REAL_t *>np.PyArray_DATA(_work_m)
    c[0].neu_m  = <REAL_t *>np.PyArray_DATA(_neu_m)
    
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
    REAL_t *syn0_1_LengInv,      # 
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
    REAL_t *_running_training_loss_param) nogil:
    #===========================================================================================#

    # cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    
    cdef REAL_t label
    cdef REAL_t f_dot,   f,   g,   log_e_f_dot
    cdef REAL_t g2
    cdef REAL_t g_m
    
    cdef int d, m  # d is for looping negative, m is for looping left words, 
    cdef int n # n is for looping left word's grain, shoud n be an int?
    cdef int left_word
    cdef int gs, ge
    cdef np.uint32_t target_index, word_index, grain_index # should left_word be an int?

    cdef REAL_t count,  inv_count = 1.0 # for left word number
    # cdef REAL_t count2, inv_count2 = 1.0
    cdef REAL_t word_lenginv = 1.0
    # cedf REAL_t channel_no = use_head + use_sub
    cdef REAL_t channel_no_inv = ONEF / (use_head + use_sub) # for merger, the channel number


    # Here word_index is np.uint32_t. very interesting
    word_index = indexes[i]  ########### S: get index for right token voc_idx
    # because indexes is np.int32_t

    #################################### 
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i: # j, m, i, k are int
            continue
        else:
            count += ONEF
    if count > (<REAL_t>0.5):  # when using sg, count is 1. count is cw in word2vec.c
        inv_count = ONEF/count
    # else: inv_count = 1.0
    #################################### 


    #################################### S: calculate hProj from syn0
    # neu1 ===> hProj1
    if use_head:
        memset(neu1, 0, size * cython.sizeof(REAL_t))
        count = <REAL_t>0.0
        for m in range(j, k): # sg case: j = k; loop left tokens here
            if m == i: # j, m, i, k are int
                continue
            else: # size is an int
                our_saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
        sscal(&size, &inv_count, neu1, &ONE)  # (does this need BLAS-variants like saxpy? # no, you don't)
    #################################### E: calculate hProj from syn0


    #################################### S: calculate hProj from syn0
    # neu2 ===> hProj2
    if use_sub:
        memset(neu2, 0, size * cython.sizeof(REAL_t))
        # count2 = <REAL_t>0.0
        for m in range(j, k): # sg case: j = k; loop left tokens here
            # m is tk_loc_id
            if m == i:
                continue
            else:
                ############### This four lines are important ###############
                left_word   = indexes[m] # left_word  #
                word_lenginv = syn0_1_LengInv[left_word]
                
                gs = syn0_1_EndIdx[left_word-1]     # 
                ge = syn0_1_EndIdx[left_word]       # 
                for n in range(gs, ge):             # 
                    # n is also np.uint_32
                    # should n be an int? just like m?
                    grain_index = syn0_1_LookUp[n] # syn0_1_LookUp is a np.uint_32
                    # grain_index is also np.uint_32
                    our_saxpy(&size, &word_lenginv, &syn0_1[grain_index * size],  &ONE, neu2, &ONE)

        # if count2 > (<REAL_t>0.5):
        #     inv_count2 = ONEF/count2
        sscal(&size, &inv_count, neu2, &ONE)  # (does this need BLAS-variants like saxpy? # no, you don't)
    #################################### E: calculate hProj from syn0

    if use_merger:
        memset(neu_m, 0, size * cython.sizeof(REAL_t))
        if use_head:
            our_saxpy(&size, &channel_no_inv, neu1,  &ONE, neu_m, &ONE)
        if use_sub:
            our_saxpy(&size, &channel_no_inv, neu2,  &ONE, neu_m, &ONE)


    #################################### S: calculate hProj_grad and update syn1neg
    # work ===> hProj_grad
    if use_head:
        memset(work,  0,  size * cython.sizeof(REAL_t))
    if use_sub:
        memset(work2, 0,  size * cython.sizeof(REAL_t))
    if use_merger:
        memset(work_m, 0, size * cython.sizeof(REAL_t))

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
                # loss1
                f_dot = (f_dot if d == 0  else -f_dot)
                if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                    continue # this is still an issue
                log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]

            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

            g = (label - f) * alpha
            our_saxpy(&size, &g,  &syn1neg[row2], &ONE, work, &ONE) # accumulate work

        if use_sub:
            f_dot = our_dot(&size, neu2, &ONE, &syn1neg[row2], &ONE)
            if _compute_loss == 1: # TODO
                # loss1
                f_dot = (f_dot if d == 0  else -f_dot)
                if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                    continue # this is still an issue
                log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]

            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            
            g2 = (label - f) * alpha
            our_saxpy(&size, &g2, &syn1neg[row2], &ONE, work2, &ONE) # accumulate work

        if use_merger:
            f_dot = our_dot(&size, neu_m, &ONE, &syn1neg[row2], &ONE)
            if _compute_loss == 1: # TODO
                # loss1
                f_dot = (f_dot if d == 0  else -f_dot)
                if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                    continue # this is still an issue
                log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            
            g_m = (label - f) * alpha
            our_saxpy(&size, &g_m, &syn1neg[row2], &ONE, work_m, &ONE) # accumulate work
        #########################################################################

        ##########################################################################
        if use_head:
            our_saxpy(&size, &g,   neu1, &ONE, &syn1neg[row2], &ONE)
        if use_sub:
            our_saxpy(&size, &g2,  neu2, &ONE, &syn1neg[row2], &ONE)
        if use_merger:
            our_saxpy(&size, &g_m, neu_m, &ONE, &syn1neg[row2], &ONE)
        ##########################################################################
    #################################### E: calculate hProj_grad and update syn1neg

    #################################### S: update syn0 gradient
    if use_merger:
        if use_head:
            our_saxpy(&size, &channel_no_inv, work_m, &ONE, work,  &ONE)
        if use_sub:
            our_saxpy(&size, &channel_no_inv, work_m, &ONE, work2, &ONE)

    if cbow_mean:  # if use standard gradient here...
        if use_head:
            sscal(&size, &inv_count,  work,  &ONE)  # (does this need BLAS-variants like saxpy?)
        if use_sub:
            sscal(&size, &inv_count,  work2, &ONE)

    if use_head:
        for m in range(j,k): 
            if m == i:
                continue
            else:
                # Here,actually, it looks up the indexes again.
                # Why not store these items some where?
                # Is it a case to trade off between time and space?
                our_saxpy(&size, &word_locks[indexes[m]], work, &ONE, &syn0[indexes[m]*size], &ONE)

    if use_sub:
        for m in range(j, k): # sg case: j = k; loop left tokens here
            # m is tk_loc_id
            if m == i:
                continue
            else:
                ############### This four lines are important ###############
                left_word = indexes[m] # left_word  #
                word_lenginv = syn0_1_LengInv[left_word]
                
                gs = syn0_1_EndIdx[left_word-1]     # 
                ge = syn0_1_EndIdx[left_word]       # 
                for n in range(gs, ge):             # 
                    # try:
                    grain_index = syn0_1_LookUp[n]
                    # print('From  ', gr_loc_id, 'To', gr_voc_id)
                    our_saxpy(&size, &word_lenginv, work2, &ONE, &syn0_1[grain_index * size], &ONE)
                    
    ################################### E: update syn0 gradient

    return next_random
################################################################# Field Embedding WITH NLPText

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
    _neu_m):

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

        c[0].syn0_2 = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][1][0].vectors)) # currently, use this <---------------------
        c[0].syn0_2_LookUp   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][1][1]))  # lookup       <---------------------
        c[0].syn0_2_EndIdx   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][1][2]))  # endIdx       <---------------------
        c[0].syn0_2_LengInv  = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][1][3]))  # leng_inv          <---------------------
        c[0].syn0_2_leng_max = model.field_sub[0][1][4]                                    # leng_max     <---------------------

    #######################################################################

    c[0].use_merger = model.use_merger

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

    #######################################################################
    c[0].work3 = <REAL_t *>np.PyArray_DATA(_work3)  # <---------------------
    c[0].neu3 = <REAL_t *>np.PyArray_DATA(_neu3)    # <---------------------
    #######################################################################

    c[0].work_m = <REAL_t *>np.PyArray_DATA(_work_m)
    c[0].neu_m  = <REAL_t *>np.PyArray_DATA(_neu_m)
    
#--> NEW for M0X2
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
    REAL_t *_running_training_loss_param) nogil:
    #===========================================================================================#

    # cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    
    cdef REAL_t label
    cdef REAL_t f_dot,   f,   g,   log_e_f_dot
    cdef REAL_t g2
    cdef REAL_t g3  # add the second #--------------> this time add a single g3 is ok.
    cdef REAL_t g_m
    
    cdef int d, m  # d is for looping negative, m is for looping left words, 
    cdef int n # n is for looping left word's grain, shoud n be an int?
    cdef int left_word
    cdef int gs, ge
    cdef np.uint32_t target_index, word_index, grain_index # should left_word be an int?

    cdef REAL_t count,  inv_count = 1.0 # for left word number
    # cdef REAL_t count2, inv_count2 = 1.0
    cdef REAL_t word_lenginv = 1.0
    # cedf REAL_t channel_no = use_head + use_sub
    cdef REAL_t channel_no_inv = ONEF / (use_head + use_sub) # for merger, the channel number


    # Here word_index is np.uint32_t. very interesting
    word_index = indexes[i]  ########### S: get index for right token voc_idx
    # because indexes is np.int32_t

    #################################### 
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i: # j, m, i, k are int
            continue
        else:
            count += ONEF
    if count > (<REAL_t>0.5):  # when using sg, count is 1. count is cw in word2vec.c
        inv_count = ONEF/count
    # else: inv_count = 1.0
    #################################### 


    #################################### S: calculate hProj from syn0
    # neu1 ===> hProj1
    if use_head:
        memset(neu1, 0, size * cython.sizeof(REAL_t))
        count = <REAL_t>0.0
        for m in range(j, k): # sg case: j = k; loop left tokens here
            if m == i: # j, m, i, k are int
                continue
            else: # size is an int
                our_saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
    #################################### E: calculate hProj from syn0

    # print('inter use_sub hProj')
    #################################### S: calculate hProj from syn0
    # neu2 ===> hProj2
    if use_sub:
        # the first subfield
        # print('allocate neu2')
        memset(neu2, 0, size * cython.sizeof(REAL_t))
        # print('allocate neu3')
        memset(neu3, 0, size * cython.sizeof(REAL_t)) # -------------> change syn0_1_LengInv to syn0_2_LengInv
        # count2 = <REAL_t>0.0
        # print('before the loop')
        for m in range(j, k): # sg case: j = k; loop left tokens here
            # m is tk_loc_id
            # print('in loop')
            if m == i:
                continue
            else:
                ############### This four lines are important ###############
                # print('in the first one')
                left_word   = indexes[m] # left_word  #
                
                word_lenginv = syn0_1_LengInv[left_word]
                gs = syn0_1_EndIdx[left_word-1]     # 
                ge = syn0_1_EndIdx[left_word]       # 
                for n in range(gs, ge):             # 
                    # n is also np.uint_32
                    # should n be an int? just like m?
                    grain_index = syn0_1_LookUp[n] # syn0_1_LookUp is a np.uint_32
                    # grain_index is also np.uint_32
                    our_saxpy(&size, &word_lenginv, &syn0_1[grain_index * size],  &ONE, neu2, &ONE)

                ################################################ copy this block
                # print('in the second one')
                word_lenginv = syn0_2_LengInv[left_word]  # -------------> change syn0_1_LengInv to syn0_2_LengInv
                gs = syn0_2_EndIdx[left_word-1]     #     #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                ge = syn0_2_EndIdx[left_word]       #     #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                for n in range(gs, ge):             # 
                    grain_index = syn0_2_LookUp[n] # syn0_1_LookUp is a np.uint_32 #--------------> change syn0_1_LookUp to syn0_2_LookUp
                    our_saxpy(&size, &word_lenginv, &syn0_2[grain_index * size],  &ONE, neu3, &ONE) #--------------> change syn0_1 to syn0_2
                ################################################ copy this block
    #################################### E: calculate hProj from syn0
    # print('go out of use_sub hProj')

    if use_head:
        sscal(&size, &inv_count, neu1, &ONE)  # (does this need BLAS-variants like saxpy? # no, you don't)
    if use_sub:
        sscal(&size, &inv_count, neu2, &ONE)  # (does this need BLAS-variants like saxpy? # no, you don't)
        sscal(&size, &inv_count, neu3, &ONE)  # (does this need BLAS-variants like saxpy? # no, you don't) #--------------> change neu2 to neu3


    if use_merger:
        memset(neu_m, 0, size * cython.sizeof(REAL_t))
        if use_head:
            our_saxpy(&size, &channel_no_inv, neu1,  &ONE, neu_m, &ONE)
        if use_sub:
            our_saxpy(&size, &channel_no_inv, neu2,  &ONE, neu_m, &ONE)
            our_saxpy(&size, &channel_no_inv, neu3,  &ONE, neu_m, &ONE)  #--------------> add this line; notice neu3


    #################################### S: calculate hProj_grad and update syn1neg
    # work ===> hProj_grad
    if use_head:
        memset(work,  0,  size * cython.sizeof(REAL_t))
    if use_sub:
        memset(work2, 0,  size * cython.sizeof(REAL_t))
        memset(work3, 0,  size * cython.sizeof(REAL_t)) #--------------> add this line; notice work3
    if use_merger:
        memset(work_m, 0, size * cython.sizeof(REAL_t))


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
            if _compute_loss == 1: 
                f_dot = (f_dot if d == 0  else -f_dot)
                if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                    continue # this is still an issue
                log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]
            
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            
            g = (label - f) * alpha
            our_saxpy(&size, &g,  &syn1neg[row2], &ONE, work, &ONE) # accumulate work

        if use_sub:
            f_dot = our_dot(&size, neu2, &ONE, &syn1neg[row2], &ONE)
            if _compute_loss == 1: # TODO
                f_dot = (f_dot if d == 0  else -f_dot)
                if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                    continue # this is still an issue
                log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]
            
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

            g2 = (label - f) * alpha
            our_saxpy(&size, &g2, &syn1neg[row2], &ONE, work2, &ONE) # accumulate work
            
            ################################################ copy this block
            f_dot = our_dot(&size, neu3, &ONE, &syn1neg[row2], &ONE)
            if _compute_loss == 1: # TODO
                # loss1
                f_dot = (f_dot if d == 0  else -f_dot)
                if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                    continue # this is still an issue
                log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]
            
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            #---------------------------------------------------------------------------------------- check rules: g & nue & work change at the same time
            g3 = (label - f) * alpha                                                                  #--------------> just change the g2 to g3
            our_saxpy(&size, &g3, &syn1neg[row2], &ONE, work3, &ONE) # accumulate work                #--------------> just change the work2 to work3
            ################################################ copy this block

        if use_merger:
            f_dot = our_dot(&size, neu_m, &ONE, &syn1neg[row2], &ONE)
            if _compute_loss == 1: # TODO
                f_dot = (f_dot if d == 0  else -f_dot)
                if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                    continue # this is still an issue
                log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]
            
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            
            g_m = (label - f) * alpha
            our_saxpy(&size, &g_m, &syn1neg[row2], &ONE, work_m, &ONE) # accumulate work
        #########################################################################

        ##########################################################################
        if use_head:
            our_saxpy(&size, &g,   neu1,  &ONE, &syn1neg[row2], &ONE)
        if use_sub:
            our_saxpy(&size, &g2,  neu2,  &ONE, &syn1neg[row2], &ONE)
            our_saxpy(&size, &g3,  neu3,  &ONE, &syn1neg[row2], &ONE) #--------------> add this line; change g2 to g3, and neu2 to neu3
        if use_merger:
            our_saxpy(&size, &g_m, neu_m, &ONE, &syn1neg[row2], &ONE)
        ##########################################################################
    #################################### E: calculate hProj_grad and update syn1neg

    #################################### S: update syn0 gradient
    if use_merger:
        if use_head:
            our_saxpy(&size, &channel_no_inv, work_m, &ONE, work,  &ONE)
        if use_sub:
            our_saxpy(&size, &channel_no_inv, work_m, &ONE, work2, &ONE)
            our_saxpy(&size, &channel_no_inv, work_m, &ONE, work3, &ONE) #--------------> add this line; change work2 to work3

    if cbow_mean:  # if use standard gradient; usually the cbow_mean (standard gradient is 1)
        if use_head:
            sscal(&size, &inv_count,  work,  &ONE)  # (does this need BLAS-variants like saxpy?)
        if use_sub:
            sscal(&size, &inv_count,  work2, &ONE)
            sscal(&size, &inv_count,  work3, &ONE)  #--------------> add this line; change work2 to work3

    if use_head:
        for m in range(j,k): 
            if m == i:
                continue
            else:
                our_saxpy(&size, &word_locks[indexes[m]], work, &ONE, &syn0[indexes[m]*size], &ONE)

    if use_sub:
        for m in range(j, k): # sg case: j = k; loop left tokens here
            # m is tk_loc_id
            if m == i:
                continue
            else:
                ############### This four lines are important ###############
                left_word = indexes[m] # left_word  #
                
                word_lenginv = syn0_1_LengInv[left_word]
                gs = syn0_1_EndIdx[left_word-1]     # 
                ge = syn0_1_EndIdx[left_word]       # 
                for n in range(gs, ge):             # 
                    # try:
                    grain_index = syn0_1_LookUp[n]
                    # print('From  ', gr_loc_id, 'To', gr_voc_id)
                    our_saxpy(&size, &word_lenginv, work2, &ONE, &syn0_1[grain_index * size], &ONE)

                # left_word = indexes[m] # left_word  #
                ###############################################################
                word_lenginv = syn0_2_LengInv[left_word] #--------------> change syn0_1_LengInv to syn0_2_LengInv
                gs = syn0_2_EndIdx[left_word-1]     #    #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                ge = syn0_2_EndIdx[left_word]       #    #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                for n in range(gs, ge):             #
                    grain_index = syn0_2_LookUp[n]  #    #--------------> change syn0_1_LookUp to syn0_2_LookUp
                    our_saxpy(&size, &word_lenginv, work3, &ONE, &syn0_2[grain_index * size], &ONE) #--------------> change syn0_1 to syn0_2 and work2 to work3
                ###############################################################

    return next_random
################################################################# Field Embedding WITH NLPText

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
    _neu_m):


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
        # print('use head')
        # print(model.field_head[0][1].vectors.shape)
        c[0].syn0  = <REAL_t *>(np.PyArray_DATA(model.field_head[0][1].vectors)) # currently, use this
        c[0].work = <REAL_t *>np.PyArray_DATA(_work)
        c[0].neu1 = <REAL_t *>np.PyArray_DATA(_neu1)
    
    
    c[0].use_sub  = model.use_sub
    if c[0].use_sub >= 1 :
        # print('In sub 1')
        c[0].syn0_1 = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][0][0].vectors)) # currently, use this
        c[0].syn0_1_LookUp   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][0][1]))  # lookup
        c[0].syn0_1_EndIdx   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][0][2]))  # endIdx
        c[0].syn0_1_LengInv  = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][0][3]))  # leng_inv
        c[0].syn0_1_leng_max = model.field_sub[0][0][4]                                    # leng_max
        #######################################################################
        c[0].work2 = <REAL_t *>np.PyArray_DATA(_work2)
        c[0].neu2  = <REAL_t *>np.PyArray_DATA(_neu2)
        #######################################################################
        
        if c[0].use_sub >= 2 :
            # print('In sub 2')
            c[0].syn0_2 = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][1][0].vectors)) # currently, use this <---------------------
            c[0].syn0_2_LookUp   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][1][1]))  # lookup       <---------------------
            c[0].syn0_2_EndIdx   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][1][2]))  # endIdx       <---------------------
            c[0].syn0_2_LengInv  = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][1][3]))  # leng_inv          <---------------------
            c[0].syn0_2_leng_max = model.field_sub[0][1][4]                                    # leng_max     <---------------------
            #######################################################################
            c[0].work3 = <REAL_t *>np.PyArray_DATA(_work3)  # <---------------------
            c[0].neu3 = <REAL_t *>np.PyArray_DATA(_neu3)    # <---------------------
            #######################################################################
            
            if c[0].use_sub >= 3 :
                # print('In sub 3')
                c[0].syn0_3 = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][2][0].vectors)) # currently, use this <---------------------
                c[0].syn0_3_LookUp   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][2][1]))  # lookup       <---------------------
                c[0].syn0_3_EndIdx   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][2][2]))  # endIdx       <---------------------
                c[0].syn0_3_LengInv  = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][2][3]))  # leng_inv          <---------------------
                c[0].syn0_3_leng_max = model.field_sub[0][2][4]                                    # leng_max     <---------------------
                #######################################################################
                c[0].work4 = <REAL_t *>np.PyArray_DATA(_work4)  # <---------------------
                c[0].neu4 = <REAL_t *>np.PyArray_DATA(_neu4)    # <---------------------
                #######################################################################
            
                if c[0].use_sub >= 4 :
                    # print('In sub 4')
                    c[0].syn0_4 = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][3][0].vectors)) # currently, use this <---------------------
                    c[0].syn0_4_LookUp   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][3][1]))  # lookup       <---------------------
                    c[0].syn0_4_EndIdx   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][3][2]))  # endIdx       <---------------------
                    c[0].syn0_4_LengInv  = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][3][3]))  # leng_inv          <---------------------
                    c[0].syn0_4_leng_max = model.field_sub[0][3][4]                                    # leng_max     <---------------------
                    #######################################################################
                    c[0].work5 = <REAL_t *>np.PyArray_DATA(_work5)  # <---------------------
                    c[0].neu5 = <REAL_t *>np.PyArray_DATA(_neu5)    # <---------------------
                    #######################################################################
            
                    if c[0].use_sub >= 5 :
                        # print('In sub 5')
                        c[0].syn0_5 = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][4][0].vectors)) # currently, use this <---------------------
                        c[0].syn0_5_LookUp   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][4][1]))  # lookup       <---------------------
                        c[0].syn0_5_EndIdx   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][4][2]))  # endIdx       <---------------------
                        c[0].syn0_5_LengInv  = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][4][3]))  # leng_inv          <---------------------
                        c[0].syn0_5_leng_max = model.field_sub[0][4][4]                                    # leng_max     <---------------------
                        #######################################################################
                        c[0].work6 = <REAL_t *>np.PyArray_DATA(_work6)  # <---------------------
                        c[0].neu6 = <REAL_t *>np.PyArray_DATA(_neu6)    # <---------------------
                        #######################################################################
                        
                        if c[0].use_sub >= 6 :
                            # print('In sub 6')
                            c[0].syn0_6 = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][6][0].vectors)) # currently, use this <---------------------
                            c[0].syn0_6_LookUp   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][6][1]))  # lookup       <---------------------
                            c[0].syn0_6_EndIdx   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][6][2]))  # endIdx       <---------------------
                            c[0].syn0_6_LengInv  = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][6][3]))  # leng_inv          <---------------------
                            c[0].syn0_6_leng_max = model.field_sub[0][6][4]                                    # leng_max     <---------------------
                            #######################################################################
                            c[0].work7 = <REAL_t *>np.PyArray_DATA(_work7)  # <---------------------
                            c[0].neu7 = <REAL_t *>np.PyArray_DATA(_neu7)    # <---------------------
                            #######################################################################

    #######################################################################
    c[0].use_merger = model.use_merger
    if c[0].use_merger:
        # print('use merger')
        c[0].work_m = <REAL_t *>np.PyArray_DATA(_work_m)
        c[0].neu_m  = <REAL_t *>np.PyArray_DATA(_neu_m)

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
    REAL_t *_running_training_loss_param) nogil:
    #===========================================================================================#

    # cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    
    cdef REAL_t label
    cdef REAL_t f_dot,   f,   g,   log_e_f_dot
    cdef REAL_t g2
    cdef REAL_t g3  # add the second #--------------> this time add a single g3 is ok.
    cdef REAL_t g4  # add the second #--------------> this time add a single g3 is ok.
    cdef REAL_t g5  # add the second #--------------> this time add a single g3 is ok.
    cdef REAL_t g6  # add the second #--------------> this time add a single g3 is ok.
    cdef REAL_t g7  # add the second #--------------> this time add a single g3 is ok.
    cdef REAL_t g_m
    
    cdef int d, m  # d is for looping negative, m is for looping left words, 
    cdef int n # n is for looping left word's grain, shoud n be an int?
    cdef int left_word
    cdef int gs, ge
    cdef np.uint32_t target_index, word_index, grain_index # should left_word be an int?

    cdef REAL_t count,  inv_count = 1.0 # for left word number
    # cdef REAL_t count2, inv_count2 = 1.0
    cdef REAL_t word_lenginv = 1.0
    # cedf REAL_t channel_no = use_head + use_sub
    cdef REAL_t channel_no_inv = ONEF / (use_head + use_sub) # for merger, the channel number

    cdef int restrict = 1
    # Here word_index is np.uint32_t. very interesting
    word_index = indexes[i]  ########### S: get index for right token voc_idx
    # because indexes is np.int32_t

    #################################### 
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i: # j, m, i, k are int
            continue
        else:
            count += ONEF
    if count > (<REAL_t>0.5):  # when using sg, count is 1. count is cw in word2vec.c
        inv_count = ONEF/count
    # else: inv_count = 1.0
    #################################### 

    # print('Inverse channel num:', channel_no_inv)
    # print('Inverse count   num:', inv_count)

    ############################################################################################################# |----> for neu: all channel's projection
    # neu1 ===> hProj1
    if use_head:
        memset(neu1, 0, size * cython.sizeof(REAL_t))
        count = <REAL_t>0.0
        for m in range(j, k): # sg case: j = k; loop left tokens here
            if m == i: # j, m, i, k are int
                continue
            else: # size is an int
                our_saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
    #################################### E: calculate hProj from syn0

    # print('inter use_sub hProj')
    #################################### S: calculate hProj from syn0
    # neu2 ===> hProj2
    if use_sub >= 1:
        memset(neu2, 0, size * cython.sizeof(REAL_t))
        if use_sub >= 2:
            memset(neu3, 0, size * cython.sizeof(REAL_t)) # -------------> change syn0_1_LengInv to syn0_2_LengInv
            if use_sub >= 3:
                memset(neu4, 0, size * cython.sizeof(REAL_t)) # -------------> change syn0_1_LengInv to syn0_2_LengInv
                if use_sub >= 4:
                    memset(neu5, 0, size * cython.sizeof(REAL_t)) # -------------> change syn0_1_LengInv to syn0_2_LengInv
                    if use_sub >= 5:
                        memset(neu6, 0, size * cython.sizeof(REAL_t)) # -------------> change syn0_1_LengInv to syn0_2_LengInv
                        if use_sub >= 6:
                            memset(neu7, 0, size * cython.sizeof(REAL_t)) # -------------> change syn0_1_LengInv to syn0_2_LengInv
          
        for m in range(j, k): # sg case: j = k; loop left tokens here
            if m == i:
                continue
            else:
                ############### This four lines are important ###############
                left_word   = indexes[m] # left_word  #
                
                word_lenginv = syn0_1_LengInv[left_word]
                gs = syn0_1_EndIdx[left_word-1]     # 
                ge = syn0_1_EndIdx[left_word]       # \
                # print('in 1', gs, ge, (ge-gs) *word_lenginv, word_lenginv)
                for n in range(gs, ge):             # 
                    grain_index = syn0_1_LookUp[n] # syn0_1_LookUp is a np.uint_32
                    our_saxpy(&size, &word_lenginv, &syn0_1[grain_index * size],  &ONE, neu2, &ONE)

                ################################################ copy this block
                if use_sub >= 2:
                    # print('in the second one')
                    word_lenginv = syn0_2_LengInv[left_word]  # -------------> change syn0_1_LengInv to syn0_2_LengInv
                    gs = syn0_2_EndIdx[left_word-1]     #     #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                    ge = syn0_2_EndIdx[left_word]       #     #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                    # print('in 2', gs, ge, (ge-gs) *word_lenginv, word_lenginv)
                    for n in range(gs, ge):             # 
                        grain_index = syn0_2_LookUp[n] # syn0_1_LookUp is a np.uint_32 #--------------> change syn0_1_LookUp to syn0_2_LookUp
                        our_saxpy(&size, &word_lenginv, &syn0_2[grain_index * size],  &ONE, neu3, &ONE) #--------------> change syn0_1 to syn0_2, and neu3 to neu4

                    if use_sub >=3:
                        # print('in the second one')
                        word_lenginv = syn0_3_LengInv[left_word]  # -------------> change syn0_1_LengInv to syn0_2_LengInv
                        gs = syn0_3_EndIdx[left_word-1]     #     #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                        ge = syn0_3_EndIdx[left_word]       #     #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                        # print('in 3', gs, ge, (ge-gs) *word_lenginv,  word_lenginv)
                        for n in range(gs, ge):             # 
                            grain_index = syn0_3_LookUp[n] # syn0_1_LookUp is a np.uint_32 #--------------> change syn0_1_LookUp to syn0_2_LookUp
                            our_saxpy(&size, &word_lenginv, &syn0_3[grain_index * size],  &ONE, neu4, &ONE) #--------------> change syn0_1 to syn0_2, and 

                        if use_sub >=4:
                            # print('in the second one')
                            word_lenginv = syn0_4_LengInv[left_word]  # -------------> change syn0_1_LengInv to syn0_2_LengInv
                            gs = syn0_4_EndIdx[left_word-1]     #     #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                            ge = syn0_4_EndIdx[left_word]       #     #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                            # print('in 4', gs, ge, (ge-gs) *word_lenginv, word_lenginv)
                            for n in range(gs, ge):             # 
                                grain_index = syn0_4_LookUp[n] # syn0_1_LookUp is a np.uint_32 #--------------> change syn0_1_LookUp to syn0_2_LookUp
                                our_saxpy(&size, &word_lenginv, &syn0_4[grain_index * size],  &ONE, neu5, &ONE) #--------------> change syn0_1 to syn0_2, and 

                            if use_sub >=5:
                                # print('in the second one')
                                word_lenginv = syn0_5_LengInv[left_word]  # -------------> change syn0_1_LengInv to syn0_2_LengInv
                                gs = syn0_5_EndIdx[left_word-1]     #     #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                                ge = syn0_5_EndIdx[left_word]       #     #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                                # print('in 5', gs, ge, (ge-gs) *word_lenginv, word_lenginv)
                                for n in range(gs, ge):             # 
                                    grain_index = syn0_5_LookUp[n] # syn0_1_LookUp is a np.uint_32 #--------------> change syn0_1_LookUp to syn0_2_LookUp
                                    our_saxpy(&size, &word_lenginv, &syn0_5[grain_index * size],  &ONE, neu6, &ONE) #--------------> change syn0_1 to syn0_2, and 

                                if use_sub >=6:
                                    # print('in the second one')
                                    word_lenginv = syn0_6_LengInv[left_word]  # -------------> change syn0_1_LengInv to syn0_2_LengInv
                                    gs = syn0_6_EndIdx[left_word-1]     #     #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                                    ge = syn0_6_EndIdx[left_word]       #     #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                                    # print('in 6', gs, ge, (ge-gs) *word_lenginv,  word_lenginv)
                                    for n in range(gs, ge):             # 
                                        grain_index = syn0_6_LookUp[n] # syn0_1_LookUp is a np.uint_32 #--------------> change syn0_1_LookUp to syn0_2_LookUp
                                        our_saxpy(&size, &word_lenginv, &syn0_6[grain_index * size],  &ONE, neu7, &ONE) #--------------> change syn0_1 to syn0_2, and 
    ############################################################################################################# |----> for neu: all channel's projection


    ############################################################################################################# |----> forposteprocessing neu: all channel's projection
    if use_head:
        sscal(&size, &inv_count, neu1, &ONE)  # (does this need BLAS-variants like saxpy? # no, you don't)
    if use_sub >= 1:
        sscal(&size, &inv_count, neu2, &ONE)
        if use_sub >= 2:
            sscal(&size, &inv_count, neu3, &ONE)
            if use_sub >= 3:
                sscal(&size, &inv_count, neu4, &ONE)
                if use_sub >= 4:
                    sscal(&size, &inv_count, neu5, &ONE)
                    if use_sub >= 5:
                        sscal(&size, &inv_count, neu6, &ONE)
                        if use_sub >= 6:
                            sscal(&size, &inv_count, neu7, &ONE)

    if use_merger:
        memset(neu_m, 0, size * cython.sizeof(REAL_t))
        if use_head:
            our_saxpy(&size, &channel_no_inv, neu1,  &ONE, neu_m, &ONE)
        if use_sub >= 1:
            our_saxpy(&size, &channel_no_inv, neu2,  &ONE, neu_m, &ONE)
            if use_sub >= 2:
                our_saxpy(&size, &channel_no_inv, neu3,  &ONE, neu_m, &ONE)
                if use_sub >= 3:
                    our_saxpy(&size, &channel_no_inv, neu4,  &ONE, neu_m, &ONE)
                    if use_sub >= 4:
                        our_saxpy(&size, &channel_no_inv, neu5,  &ONE, neu_m, &ONE)
                        if use_sub >= 5:
                            our_saxpy(&size, &channel_no_inv, neu6,  &ONE, neu_m, &ONE)
                            if use_sub >= 6:
                                our_saxpy(&size, &channel_no_inv, neu7,  &ONE, neu_m, &ONE)


    #################################### S: calculate hProj_grad and update syn1neg
    # work ===> hProj_grad
    if use_head:
        memset(work,  0,  size * cython.sizeof(REAL_t))
    
    if use_sub >= 1:
        memset(work2, 0,  size * cython.sizeof(REAL_t))
        if use_sub >= 2:
            memset(work3, 0,  size * cython.sizeof(REAL_t))
            if use_sub >= 3:
                memset(work4, 0,  size * cython.sizeof(REAL_t))
                if use_sub >= 4:
                    memset(work5, 0,  size * cython.sizeof(REAL_t))
                    if use_sub >= 5:
                        memset(work6, 0,  size * cython.sizeof(REAL_t))
                        if use_sub >= 6:
                            memset(work7, 0,  size * cython.sizeof(REAL_t))
    
    if use_merger:
        memset(work_m, 0, size * cython.sizeof(REAL_t))


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
            if _compute_loss == 1: 
                f_dot = (f_dot if d == 0  else -f_dot)
                if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                    continue # this is still an issue
                log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]

            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            
            g = (label - f) * alpha
            our_saxpy(&size, &g,  &syn1neg[row2], &ONE, work, &ONE) # accumulate work

        if use_sub >= 1:
            f_dot = our_dot(&size, neu2, &ONE, &syn1neg[row2], &ONE)
            
            if _compute_loss == 1: # TODO
                f_dot = (f_dot if d == 0  else -f_dot)
                if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                    continue # this is still an issue
                log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]
            
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            
            g2 = (label - f) * alpha
            our_saxpy(&size, &g2, &syn1neg[row2], &ONE, work2, &ONE) # accumulate work
            

            if use_sub >= 2:
                ################################################ copy this block
                f_dot = our_dot(&size, neu3, &ONE, &syn1neg[row2], &ONE)
                
                if _compute_loss == 1: # TODO
                    # loss1
                    f_dot = (f_dot if d == 0  else -f_dot)
                    if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                        continue # this is still an issue
                    log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                    _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]
                
                if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                    continue
                f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                
                g3 = (label - f) * alpha                                                                  #--------------> just change the g2 to g3
                our_saxpy(&size, &g3, &syn1neg[row2], &ONE, work3, &ONE) # accumulate work                #--------------> just change the work2 to work3
                ################################################ copy this block

                if use_sub >= 3:
                    ################################################ copy this block
                    f_dot = our_dot(&size, neu4, &ONE, &syn1neg[row2], &ONE)
                    if _compute_loss == 1: # TODO
                        # loss1
                        f_dot = (f_dot if d == 0  else -f_dot)
                        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                            continue # this is still an issue
                        log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                        _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]
                    
                    if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                        continue
                    f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                    
                    g4 = (label - f) * alpha                                                                  #--------------> just change the g2 to g3
                    our_saxpy(&size, &g4, &syn1neg[row2], &ONE, work4, &ONE) # accumulate work                #--------------> just change the work2 to work3
                    ################################################ copy this block

                    if use_sub >= 4:
                        ################################################ copy this block
                        f_dot = our_dot(&size, neu5, &ONE, &syn1neg[row2], &ONE)
                        if _compute_loss == 1: # TODO
                            f_dot = (f_dot if d == 0  else -f_dot)
                            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                                continue # this is still an issue
                            log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                            _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]
                        
                        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                            continue
                        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                        
                        g5 = (label - f) * alpha                                                                  #--------------> just change the g2 to g3
                        our_saxpy(&size, &g5, &syn1neg[row2], &ONE, work5, &ONE) # accumulate work                #--------------> just change the work2 to work3
                        ################################################ copy this block

                        if use_sub >= 5:
                            ################################################ copy this block
                            f_dot = our_dot(&size, neu6, &ONE, &syn1neg[row2], &ONE)
                            if _compute_loss == 1: # TODO
                                f_dot = (f_dot if d == 0  else -f_dot)
                                if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                                    continue # this is still an issue
                                log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                                _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]
                            
                            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                                continue
                            f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                            
                            g6 = (label - f) * alpha                                                                  #--------------> just change the g2 to g3
                            our_saxpy(&size, &g6, &syn1neg[row2], &ONE, work6, &ONE) # accumulate work                #--------------> just change the work2 to work3
                            ################################################ copy this block

                            if use_sub >= 6:
                                ################################################ copy this block
                                f_dot = our_dot(&size, neu7, &ONE, &syn1neg[row2], &ONE)
                                if _compute_loss == 1: # TODO
                                    f_dot = (f_dot if d == 0  else -f_dot)
                                    if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                                        continue # this is still an issue
                                    log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                                    _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]
                                
                                if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                                    continue
                                f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                                
                                g7 = (label - f) * alpha                                                                  #--------------> just change the g2 to g3
                                our_saxpy(&size, &g7, &syn1neg[row2], &ONE, work7, &ONE) # accumulate work                #--------------> just change the work2 to work3
                                ################################################ copy this block

        if use_merger:
            f_dot = our_dot(&size, neu_m, &ONE, &syn1neg[row2], &ONE)
            if _compute_loss == 1: # TODO
                f_dot = (f_dot if d == 0  else -f_dot)
                if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                    continue # this is still an issue
                log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]
            
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                            
            g_m = (label - f) * alpha
            our_saxpy(&size, &g_m, &syn1neg[row2], &ONE, work_m, &ONE) # accumulate work
        #########################################################################

        ##########################################################################
        if use_head:
            our_saxpy(&size, &g,   neu1,  &ONE, &syn1neg[row2], &ONE)

        if use_sub >= 1:
            our_saxpy(&size, &g2,  neu2,  &ONE, &syn1neg[row2], &ONE)
            if use_sub >= 2:
                our_saxpy(&size, &g3,  neu3,  &ONE, &syn1neg[row2], &ONE)
                if use_sub >= 3:
                    our_saxpy(&size, &g4,  neu4,  &ONE, &syn1neg[row2], &ONE)
                    if use_sub >= 4:
                        our_saxpy(&size, &g5,  neu5,  &ONE, &syn1neg[row2], &ONE)
                        if use_sub >= 5:
                            our_saxpy(&size, &g6,  neu6,  &ONE, &syn1neg[row2], &ONE)
                            if use_sub >= 6:
                                our_saxpy(&size, &g7,  neu7,  &ONE, &syn1neg[row2], &ONE)

        if use_merger:
            our_saxpy(&size, &g_m, neu_m, &ONE, &syn1neg[row2], &ONE)
        ##########################################################################
    #################################### E: calculate hProj_grad and update syn1neg

    #################################### S: update syn0 gradient
    if use_merger:
        if use_head:
            our_saxpy(&size, &channel_no_inv, work_m, &ONE, work,  &ONE)

        if use_sub >= 1:
            our_saxpy(&size, &channel_no_inv, work_m, &ONE, work2, &ONE)
            if use_sub >= 2:
                our_saxpy(&size, &channel_no_inv, work_m, &ONE, work3, &ONE)
                if use_sub >= 3:
                    our_saxpy(&size, &channel_no_inv, work_m, &ONE, work4, &ONE)
                    if use_sub >= 4:
                        our_saxpy(&size, &channel_no_inv, work_m, &ONE, work5, &ONE)
                        if use_sub >= 5:
                            our_saxpy(&size, &channel_no_inv, work_m, &ONE, work6, &ONE)
                            if use_sub >= 6:
                                our_saxpy(&size, &channel_no_inv, work_m, &ONE, work7, &ONE)

    if cbow_mean:  # divide error over summed window vectors # confusing here? to see the result tomorrow
        if use_head:
            sscal(&size, &inv_count,  work,  &ONE)  # (does this need BLAS-variants like saxpy?)

        if use_sub >= 1:
            sscal(&size, &inv_count,  work2, &ONE)
            if use_sub >= 2:
                sscal(&size, &inv_count,  work3, &ONE)
                if use_sub >= 3:
                    sscal(&size, &inv_count,  work4, &ONE)
                    if use_sub >= 4:
                        sscal(&size, &inv_count,  work5, &ONE)
                        if use_sub >= 5:
                            sscal(&size, &inv_count,  work6, &ONE)
                            if use_sub >= 6:
                                sscal(&size, &inv_count,  work7, &ONE)

    if use_head:
        for m in range(j,k): 
            if m == i:
                continue
            else:
                our_saxpy(&size, &word_locks[indexes[m]], work, &ONE, &syn0[indexes[m]*size], &ONE)

    if use_sub >= 1:
        for m in range(j, k): # sg case: j = k; loop left tokens here
            if m == i:
                continue
            else:
                ############### This four lines are important ###############
                left_word = indexes[m] # left_word  #
                
                word_lenginv = syn0_1_LengInv[left_word]
                gs = syn0_1_EndIdx[left_word-1]     # 
                ge = syn0_1_EndIdx[left_word]       # 
                for n in range(gs, ge):             # 
                    grain_index = syn0_1_LookUp[n]
                    our_saxpy(&size, &word_lenginv, work2, &ONE, &syn0_1[grain_index * size], &ONE)

                ###############################################################
                if use_sub >= 2:
                    word_lenginv = syn0_2_LengInv[left_word] #--------------> change syn0_1_LengInv to syn0_2_LengInv
                    gs = syn0_2_EndIdx[left_word-1]     #    #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                    ge = syn0_2_EndIdx[left_word]       #    #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                    for n in range(gs, ge):             #
                        grain_index = syn0_2_LookUp[n]  #    #--------------> change syn0_1_LookUp to syn0_2_LookUp
                        our_saxpy(&size, &word_lenginv, work3, &ONE, &syn0_2[grain_index * size], &ONE) #--------------> change syn0_1 to syn0_2 and work2 to work3
                    ###############################################################
                    if use_sub >= 3:
                        word_lenginv = syn0_3_LengInv[left_word] #--------------> change syn0_1_LengInv to syn0_2_LengInv
                        gs = syn0_3_EndIdx[left_word-1]     #    #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                        ge = syn0_3_EndIdx[left_word]       #    #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                        for n in range(gs, ge):             #
                            grain_index = syn0_3_LookUp[n]  #    #--------------> change syn0_1_LookUp to syn0_2_LookUp
                            our_saxpy(&size, &word_lenginv, work4, &ONE, &syn0_3[grain_index * size], &ONE) #--------------> change syn0_1 to syn0_2 and work2 to work3
                        ###############################################################
                        if use_sub >= 4:
                            word_lenginv = syn0_4_LengInv[left_word] #--------------> change syn0_1_LengInv to syn0_2_LengInv
                            gs = syn0_4_EndIdx[left_word-1]     #    #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                            ge = syn0_4_EndIdx[left_word]       #    #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                            for n in range(gs, ge):             #
                                grain_index = syn0_4_LookUp[n]  #    #--------------> change syn0_1_LookUp to syn0_2_LookUp
                                our_saxpy(&size, &word_lenginv, work5, &ONE, &syn0_4[grain_index * size], &ONE) #--------------> change syn0_1 to syn0_2 and work2 to work3
                            ###############################################################
                            if use_sub >= 5:
                                word_lenginv = syn0_5_LengInv[left_word] #--------------> change syn0_1_LengInv to syn0_2_LengInv
                                gs = syn0_5_EndIdx[left_word-1]     #    #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                                ge = syn0_5_EndIdx[left_word]       #    #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                                for n in range(gs, ge):             #
                                    grain_index = syn0_5_LookUp[n]  #    #--------------> change syn0_1_LookUp to syn0_2_LookUp
                                    our_saxpy(&size, &word_lenginv, work6, &ONE, &syn0_5[grain_index * size], &ONE) #--------------> change syn0_1 to syn0_2 and work2 to work3
                                ###############################################################
                                if use_sub >= 6:
                                    word_lenginv = syn0_6_LengInv[left_word] #--------------> change syn0_1_LengInv to syn0_2_LengInv
                                    gs = syn0_6_EndIdx[left_word-1]     #    #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                                    ge = syn0_6_EndIdx[left_word]       #    #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                                    for n in range(gs, ge):             #
                                        grain_index = syn0_6_LookUp[n]  #    #--------------> change syn0_1_LookUp to syn0_2_LookUp
                                        our_saxpy(&size, &word_lenginv, work7, &ONE, &syn0_6[grain_index * size], &ONE) #--------------> change syn0_1 to syn0_2 and work2 to work3
                                    ###############################################################

    return next_random
################################################################# Field Embedding WITH NLPText


#--> NEW for M0XY
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
    _work_p,   # ########################### -------------> use pos
    _neu_p):   # ########################### -------------> use pos


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
    # if c[0].use_head:

    # print('I am good here, before use token')
    c[0].use_token = model.use_token # loop head
    if c[0].use_token:
        c[0].syn0  = <REAL_t *>(np.PyArray_DATA(model.field_head[model.field_idx['token']][1].vectors)) #|---------------> changed 0 to 'token'
        c[0].work = <REAL_t *>np.PyArray_DATA(_work)
        c[0].neu1 = <REAL_t *>np.PyArray_DATA(_neu1)

    # ---------------------------------->
    # print('I am good here, before use pos')
    c[0].use_pos = model.use_pos
    if c[0].use_pos:
        c[0].syn_p  = <REAL_t *>(np.PyArray_DATA(model.field_head[model.field_idx['pos']][1].vectors)) #|---------------> changed 0 to 'pos'
        c[0].work_p = <REAL_t *>np.PyArray_DATA(_work_p)
        c[0].neu_p  = <REAL_t *>np.PyArray_DATA(_neu_p)
    # ---------------------------------->
    

    # print('I am good here, before use sub')
    c[0].use_sub  = model.use_sub
    if c[0].use_sub >= 1 :
        # print('In sub 1')
        c[0].syn0_1 = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][0][0].vectors)) # currently, use this
        c[0].syn0_1_LookUp   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][0][1]))  # lookup
        c[0].syn0_1_EndIdx   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][0][2]))  # endIdx
        c[0].syn0_1_LengInv  = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][0][3]))  # leng_inv
        c[0].syn0_1_leng_max = model.field_sub[0][0][4]                                    # leng_max
        #######################################################################
        c[0].work2 = <REAL_t *>np.PyArray_DATA(_work2)
        c[0].neu2  = <REAL_t *>np.PyArray_DATA(_neu2)
        #######################################################################
        
        if c[0].use_sub >= 2 :
            # print('In sub 2')
            c[0].syn0_2 = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][1][0].vectors)) # currently, use this <---------------------
            c[0].syn0_2_LookUp   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][1][1]))  # lookup       <---------------------
            c[0].syn0_2_EndIdx   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][1][2]))  # endIdx       <---------------------
            c[0].syn0_2_LengInv  = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][1][3]))  # leng_inv          <---------------------
            c[0].syn0_2_leng_max = model.field_sub[0][1][4]                                    # leng_max     <---------------------
            #######################################################################
            c[0].work3 = <REAL_t *>np.PyArray_DATA(_work3)  # <---------------------
            c[0].neu3 = <REAL_t *>np.PyArray_DATA(_neu3)    # <---------------------
            #######################################################################
            
            if c[0].use_sub >= 3 :
                # print('In sub 3')
                c[0].syn0_3 = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][2][0].vectors)) # currently, use this <---------------------
                c[0].syn0_3_LookUp   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][2][1]))  # lookup       <---------------------
                c[0].syn0_3_EndIdx   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][2][2]))  # endIdx       <---------------------
                c[0].syn0_3_LengInv  = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][2][3]))  # leng_inv          <---------------------
                c[0].syn0_3_leng_max = model.field_sub[0][2][4]                                    # leng_max     <---------------------
                #######################################################################
                c[0].work4 = <REAL_t *>np.PyArray_DATA(_work4)  # <---------------------
                c[0].neu4 = <REAL_t *>np.PyArray_DATA(_neu4)    # <---------------------
                #######################################################################
            
                if c[0].use_sub >= 4 :
                    # print('In sub 4')
                    c[0].syn0_4 = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][3][0].vectors)) # currently, use this <---------------------
                    c[0].syn0_4_LookUp   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][3][1]))  # lookup       <---------------------
                    c[0].syn0_4_EndIdx   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][3][2]))  # endIdx       <---------------------
                    c[0].syn0_4_LengInv  = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][3][3]))  # leng_inv          <---------------------
                    c[0].syn0_4_leng_max = model.field_sub[0][3][4]                                    # leng_max     <---------------------
                    #######################################################################
                    c[0].work5 = <REAL_t *>np.PyArray_DATA(_work5)  # <---------------------
                    c[0].neu5 = <REAL_t *>np.PyArray_DATA(_neu5)    # <---------------------
                    #######################################################################
            
                    if c[0].use_sub >= 5 :
                        # print('In sub 5')
                        c[0].syn0_5 = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][4][0].vectors)) # currently, use this <---------------------
                        c[0].syn0_5_LookUp   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][4][1]))  # lookup       <---------------------
                        c[0].syn0_5_EndIdx   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][4][2]))  # endIdx       <---------------------
                        c[0].syn0_5_LengInv  = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][4][3]))  # leng_inv          <---------------------
                        c[0].syn0_5_leng_max = model.field_sub[0][4][4]                                    # leng_max     <---------------------
                        #######################################################################
                        c[0].work6 = <REAL_t *>np.PyArray_DATA(_work6)  # <---------------------
                        c[0].neu6 = <REAL_t *>np.PyArray_DATA(_neu6)    # <---------------------
                        #######################################################################
                        
                        if c[0].use_sub >= 6 :
                            # print('In sub 6')
                            c[0].syn0_6 = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][6][0].vectors)) # currently, use this <---------------------
                            c[0].syn0_6_LookUp   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][6][1]))  # lookup       <---------------------
                            c[0].syn0_6_EndIdx   = <np.uint32_t *>(np.PyArray_DATA(model.field_sub[0][6][2]))  # endIdx       <---------------------
                            c[0].syn0_6_LengInv  = <REAL_t *>(np.PyArray_DATA(model.field_sub[0][6][3]))  # leng_inv          <---------------------
                            c[0].syn0_6_leng_max = model.field_sub[0][6][4]                                    # leng_max     <---------------------
                            #######################################################################
                            c[0].work7 = <REAL_t *>np.PyArray_DATA(_work7)  # <---------------------
                            c[0].neu7 = <REAL_t *>np.PyArray_DATA(_neu7)    # <---------------------
                            #######################################################################

    #######################################################################
    c[0].use_merger = model.use_merger
    if c[0].use_merger:
        # print('use merger')
        c[0].work_m = <REAL_t *>np.PyArray_DATA(_work_m)
        c[0].neu_m  = <REAL_t *>np.PyArray_DATA(_neu_m)

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


#--> NEW for M0XY
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
    REAL_t *_running_training_loss_param) nogil:
    #===========================================================================================#

    # cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    
    cdef REAL_t label
    cdef REAL_t f_dot,   f,   g,   log_e_f_dot
    cdef REAL_t g2
    cdef REAL_t g3  # add the second #--------------> this time add a single g3 is ok.
    cdef REAL_t g4  # add the second #--------------> this time add a single g3 is ok.
    cdef REAL_t g5  # add the second #--------------> this time add a single g3 is ok.
    cdef REAL_t g6  # add the second #--------------> this time add a single g3 is ok.
    cdef REAL_t g7  # add the second #--------------> this time add a single g3 is ok.
    cdef REAL_t g_m
    cdef REAL_t g_p  # #--------------> pos
    
    cdef int d, m  # d is for looping negative, m is for looping left words, 
    cdef int n # n is for looping left word's grain, shoud n be an int?
    cdef int left_word
    cdef int gs, ge
    cdef np.uint32_t target_index, word_index, grain_index # should left_word be an int?

    cdef REAL_t count,  inv_count = 1.0 # for left word number
    # cdef REAL_t count2, inv_count2 = 1.0
    cdef REAL_t word_lenginv = 1.0
    # cedf REAL_t channel_no = use_head + use_sub
    cdef REAL_t channel_no_inv = ONEF / (use_head + use_sub) # for merger, the channel number

    # cdef int restrict = USE_STRICT
    # Here word_index is np.uint32_t. very interesting
    word_index = indexes[i]  ########### S: get index for right token voc_idx
    # because indexes is np.int32_t

    #################################### 
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i: # j, m, i, k are int
            continue
        else:
            count += ONEF
    if count > (<REAL_t>0.5):  # when using sg, count is 1. count is cw in word2vec.c
        inv_count = ONEF/count
    # else: inv_count = 1.0
    #################################### 

    # print('Inverse channel num:', channel_no_inv)
    # print('Inverse count   num:', inv_count)

    # print('0')
    ############################################################################################################# |----> for neu: all channel's projection
    # neu1 ===> hProj1
    if use_token:
        memset(neu1, 0, size * cython.sizeof(REAL_t))
        count = <REAL_t>0.0
        for m in range(j, k): # sg case: j = k; loop left tokens here
            if m == i: # j, m, i, k are int
                continue
            else: # size is an int
                our_saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)

    # print('1')
    #################################### E: calculate hProj from syn0
    if use_pos:
        memset(neu_p, 0, size * cython.sizeof(REAL_t))
        # print('in pos 1')
        count = <REAL_t>0.0
        for m in range(j, k): # sg case: j = k; loop left tokens here
            if m == i: # j, m, i, k are int
                continue
            else: # size is an int
                our_saxpy(&size, &ONEF, &syn_p[indexes_pos[m] * size], &ONE, neu_p, &ONE)
    #################################### E: calculate hProj from syn0
    # print('out pos 2')

    # print('inter use_sub hProj')
    #################################### S: calculate hProj from syn0
    # neu2 ===> hProj2
    if use_sub >= 1:
        memset(neu2, 0, size * cython.sizeof(REAL_t))
        if use_sub >= 2:
            memset(neu3, 0, size * cython.sizeof(REAL_t)) # -------------> change syn0_1_LengInv to syn0_2_LengInv
            if use_sub >= 3:
                memset(neu4, 0, size * cython.sizeof(REAL_t)) # -------------> change syn0_1_LengInv to syn0_2_LengInv
                if use_sub >= 4:
                    memset(neu5, 0, size * cython.sizeof(REAL_t)) # -------------> change syn0_1_LengInv to syn0_2_LengInv
                    if use_sub >= 5:
                        memset(neu6, 0, size * cython.sizeof(REAL_t)) # -------------> change syn0_1_LengInv to syn0_2_LengInv
                        if use_sub >= 6:
                            memset(neu7, 0, size * cython.sizeof(REAL_t)) # -------------> change syn0_1_LengInv to syn0_2_LengInv
          
        for m in range(j, k): # sg case: j = k; loop left tokens here
            if m == i:
                continue
            else:
                ############### This four lines are important ###############
                left_word   = indexes[m] # left_word  #
                
                word_lenginv = syn0_1_LengInv[left_word]
                gs = syn0_1_EndIdx[left_word-1]     # 
                ge = syn0_1_EndIdx[left_word]       # \
                # print('in 1', gs, ge, (ge-gs) *word_lenginv, word_lenginv)
                for n in range(gs, ge):             # 
                    grain_index = syn0_1_LookUp[n] # syn0_1_LookUp is a np.uint_32
                    our_saxpy(&size, &word_lenginv, &syn0_1[grain_index * size],  &ONE, neu2, &ONE)

                ################################################ copy this block
                if use_sub >= 2:
                    # print('in the second one')
                    word_lenginv = syn0_2_LengInv[left_word]  # -------------> change syn0_1_LengInv to syn0_2_LengInv
                    gs = syn0_2_EndIdx[left_word-1]     #     #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                    ge = syn0_2_EndIdx[left_word]       #     #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                    # print('in 2', gs, ge, (ge-gs) *word_lenginv, word_lenginv)
                    for n in range(gs, ge):             # 
                        grain_index = syn0_2_LookUp[n] # syn0_1_LookUp is a np.uint_32 #--------------> change syn0_1_LookUp to syn0_2_LookUp
                        our_saxpy(&size, &word_lenginv, &syn0_2[grain_index * size],  &ONE, neu3, &ONE) #--------------> change syn0_1 to syn0_2, and neu3 to neu4

                    if use_sub >=3:
                        # print('in the second one')
                        word_lenginv = syn0_3_LengInv[left_word]  # -------------> change syn0_1_LengInv to syn0_2_LengInv
                        gs = syn0_3_EndIdx[left_word-1]     #     #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                        ge = syn0_3_EndIdx[left_word]       #     #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                        # print('in 3', gs, ge, (ge-gs) *word_lenginv,  word_lenginv)
                        for n in range(gs, ge):             # 
                            grain_index = syn0_3_LookUp[n] # syn0_1_LookUp is a np.uint_32 #--------------> change syn0_1_LookUp to syn0_2_LookUp
                            our_saxpy(&size, &word_lenginv, &syn0_3[grain_index * size],  &ONE, neu4, &ONE) #--------------> change syn0_1 to syn0_2, and 

                        if use_sub >=4:
                            # print('in the second one')
                            word_lenginv = syn0_4_LengInv[left_word]  # -------------> change syn0_1_LengInv to syn0_2_LengInv
                            gs = syn0_4_EndIdx[left_word-1]     #     #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                            ge = syn0_4_EndIdx[left_word]       #     #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                            # print('in 4', gs, ge, (ge-gs) *word_lenginv, word_lenginv)
                            for n in range(gs, ge):             # 
                                grain_index = syn0_4_LookUp[n] # syn0_1_LookUp is a np.uint_32 #--------------> change syn0_1_LookUp to syn0_2_LookUp
                                our_saxpy(&size, &word_lenginv, &syn0_4[grain_index * size],  &ONE, neu5, &ONE) #--------------> change syn0_1 to syn0_2, and 

                            if use_sub >=5:
                                # print('in the second one')
                                word_lenginv = syn0_5_LengInv[left_word]  # -------------> change syn0_1_LengInv to syn0_2_LengInv
                                gs = syn0_5_EndIdx[left_word-1]     #     #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                                ge = syn0_5_EndIdx[left_word]       #     #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                                # print('in 5', gs, ge, (ge-gs) *word_lenginv, word_lenginv)
                                for n in range(gs, ge):             # 
                                    grain_index = syn0_5_LookUp[n] # syn0_1_LookUp is a np.uint_32 #--------------> change syn0_1_LookUp to syn0_2_LookUp
                                    our_saxpy(&size, &word_lenginv, &syn0_5[grain_index * size],  &ONE, neu6, &ONE) #--------------> change syn0_1 to syn0_2, and 

                                if use_sub >=6:
                                    # print('in the second one')
                                    word_lenginv = syn0_6_LengInv[left_word]  # -------------> change syn0_1_LengInv to syn0_2_LengInv
                                    gs = syn0_6_EndIdx[left_word-1]     #     #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                                    ge = syn0_6_EndIdx[left_word]       #     #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                                    # print('in 6', gs, ge, (ge-gs) *word_lenginv,  word_lenginv)
                                    for n in range(gs, ge):             # 
                                        grain_index = syn0_6_LookUp[n] # syn0_1_LookUp is a np.uint_32 #--------------> change syn0_1_LookUp to syn0_2_LookUp
                                        our_saxpy(&size, &word_lenginv, &syn0_6[grain_index * size],  &ONE, neu7, &ONE) #--------------> change syn0_1 to syn0_2, and 
    ############################################################################################################# |----> for neu: all channel's projection
    # print('prepare gradient')

    ############################################################################################################# |----> forposteprocessing neu: all channel's projection
    if use_token:
        sscal(&size, &inv_count, neu1, &ONE)  # (does this need BLAS-variants like saxpy? # no, you don't)
    if use_pos:
        sscal(&size, &inv_count, neu_p, &ONE)  # (does this need BLAS-variants like saxpy? # no, you don't)
    if use_sub >= 1:
        sscal(&size, &inv_count, neu2, &ONE)
        if use_sub >= 2:
            sscal(&size, &inv_count, neu3, &ONE)
            if use_sub >= 3:
                sscal(&size, &inv_count, neu4, &ONE)
                if use_sub >= 4:
                    sscal(&size, &inv_count, neu5, &ONE)
                    if use_sub >= 5:
                        sscal(&size, &inv_count, neu6, &ONE)
                        if use_sub >= 6:
                            sscal(&size, &inv_count, neu7, &ONE)

    # print('use mearger')
    if use_merger:
        memset(neu_m, 0, size * cython.sizeof(REAL_t))
        if use_token:
            our_saxpy(&size, &channel_no_inv, neu1,  &ONE, neu_m, &ONE)
        if use_pos:
            our_saxpy(&size, &channel_no_inv, neu_p,  &ONE, neu_m, &ONE)
        if use_sub >= 1:
            our_saxpy(&size, &channel_no_inv, neu2,  &ONE, neu_m, &ONE)
            if use_sub >= 2:
                our_saxpy(&size, &channel_no_inv, neu3,  &ONE, neu_m, &ONE)
                if use_sub >= 3:
                    our_saxpy(&size, &channel_no_inv, neu4,  &ONE, neu_m, &ONE)
                    if use_sub >= 4:
                        our_saxpy(&size, &channel_no_inv, neu5,  &ONE, neu_m, &ONE)
                        if use_sub >= 5:
                            our_saxpy(&size, &channel_no_inv, neu6,  &ONE, neu_m, &ONE)
                            if use_sub >= 6:
                                our_saxpy(&size, &channel_no_inv, neu7,  &ONE, neu_m, &ONE)


    #################################### S: calculate hProj_grad and update syn1neg
    # work ===> hProj_grad
    if use_token:
        memset(work,  0,  size * cython.sizeof(REAL_t))

    # print('use work_p')
    if use_pos:
        memset(work_p,  0,  size * cython.sizeof(REAL_t))
    
    if use_sub >= 1:
        memset(work2, 0,  size * cython.sizeof(REAL_t))
        if use_sub >= 2:
            memset(work3, 0,  size * cython.sizeof(REAL_t))
            if use_sub >= 3:
                memset(work4, 0,  size * cython.sizeof(REAL_t))
                if use_sub >= 4:
                    memset(work5, 0,  size * cython.sizeof(REAL_t))
                    if use_sub >= 5:
                        memset(work6, 0,  size * cython.sizeof(REAL_t))
                        if use_sub >= 6:
                            memset(work7, 0,  size * cython.sizeof(REAL_t))
    
    if use_merger:
        memset(work_m, 0, size * cython.sizeof(REAL_t))


    # print('go to negative')
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
        if use_token:
            f_dot = our_dot(&size, neu1, &ONE, &syn1neg[row2], &ONE)
            if _compute_loss == 1: 
                f_dot = (f_dot if d == 0  else -f_dot)
                if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                    continue # this is still an issue
                log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]

            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            # print('how could f in token', f_dot)
            f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            
            g = (label - f) * alpha
            our_saxpy(&size, &g,  &syn1neg[row2], &ONE, work, &ONE) # accumulate work

        # print('-------- for use pos')
        if use_pos:
            # print('are wrong')
            f_dot = our_dot(&size, neu_p, &ONE, &syn1neg[row2], &ONE)
            # print('without this line')
            if _compute_loss == 1: 
                f_dot = (f_dot if d == 0  else -f_dot)
                if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                    continue # this is still an issue
                log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]


            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            
            # print('how could f', f_dot)
            # print(<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2)))
            f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            # print('without this line')
            
            g_p = (label - f) * alpha
            # print('you must be wrong')
            our_saxpy(&size, &g_p,  &syn1neg[row2], &ONE, work_p, &ONE) # accumulate work
            # print('without this line')

        # print('-------- out for use pos')

        if use_sub >= 1:
            f_dot = our_dot(&size, neu2, &ONE, &syn1neg[row2], &ONE)
            
            if _compute_loss == 1: # TODO
                f_dot = (f_dot if d == 0  else -f_dot)
                if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                    continue # this is still an issue
                log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]
            
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

            g2 = (label - f) * alpha
            our_saxpy(&size, &g2, &syn1neg[row2], &ONE, work2, &ONE) # accumulate work
            

            if use_sub >= 2:
                ################################################ copy this block
                f_dot = our_dot(&size, neu3, &ONE, &syn1neg[row2], &ONE)
                
                if _compute_loss == 1: # TODO
                    # loss1
                    f_dot = (f_dot if d == 0  else -f_dot)
                    if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                        continue # this is still an issue
                    log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                    _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]
                
                if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                    continue
                f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

                g3 = (label - f) * alpha                                                                  #--------------> just change the g2 to g3
                our_saxpy(&size, &g3, &syn1neg[row2], &ONE, work3, &ONE) # accumulate work                #--------------> just change the work2 to work3
                ################################################ copy this block

                if use_sub >= 3:
                    ################################################ copy this block
                    f_dot = our_dot(&size, neu4, &ONE, &syn1neg[row2], &ONE)
                    if _compute_loss == 1: # TODO
                        # loss1
                        f_dot = (f_dot if d == 0  else -f_dot)
                        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                            continue # this is still an issue
                        log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                        _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]
                    
                    if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                        continue
                    f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                    
                    g4 = (label - f) * alpha                                                                  #--------------> just change the g2 to g3
                    our_saxpy(&size, &g4, &syn1neg[row2], &ONE, work4, &ONE) # accumulate work                #--------------> just change the work2 to work3
                    ################################################ copy this block

                    if use_sub >= 4:
                        ################################################ copy this block
                        f_dot = our_dot(&size, neu5, &ONE, &syn1neg[row2], &ONE)
                        if _compute_loss == 1: # TODO
                            f_dot = (f_dot if d == 0  else -f_dot)
                            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                                continue # this is still an issue
                            log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                            _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]
                        
                        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                            continue
                        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                        
                        g5 = (label - f) * alpha                                                                  #--------------> just change the g2 to g3
                        our_saxpy(&size, &g5, &syn1neg[row2], &ONE, work5, &ONE) # accumulate work                #--------------> just change the work2 to work3
                        ################################################ copy this block

                        if use_sub >= 5:
                            ################################################ copy this block
                            f_dot = our_dot(&size, neu6, &ONE, &syn1neg[row2], &ONE)
                            if _compute_loss == 1: # TODO
                                f_dot = (f_dot if d == 0  else -f_dot)
                                if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                                    continue # this is still an issue
                                log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                                _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]
                            
                            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                                continue
                            f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                            
                            g6 = (label - f) * alpha                                                                  #--------------> just change the g2 to g3
                            our_saxpy(&size, &g6, &syn1neg[row2], &ONE, work6, &ONE) # accumulate work                #--------------> just change the work2 to work3
                            ################################################ copy this block

                            if use_sub >= 6:
                                ################################################ copy this block
                                f_dot = our_dot(&size, neu7, &ONE, &syn1neg[row2], &ONE)
                                if _compute_loss == 1: # TODO
                                    f_dot = (f_dot if d == 0  else -f_dot)
                                    if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                                        continue # this is still an issue
                                    log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                                    _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]
                                
                                if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                                    continue
                                f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                                
                                g7 = (label - f) * alpha                                                                  #--------------> just change the g2 to g3
                                our_saxpy(&size, &g7, &syn1neg[row2], &ONE, work7, &ONE) # accumulate work                #--------------> just change the work2 to work3
                                ################################################ copy this block


        if use_merger:
            f_dot = our_dot(&size, neu_m, &ONE, &syn1neg[row2], &ONE)
            if _compute_loss == 1: # TODO
                f_dot = (f_dot if d == 0  else -f_dot)
                if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                    continue # this is still an issue
                log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
                _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot # it seems when using *i, to query it, use *[0]
            
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

            g_m = (label - f) * alpha
            our_saxpy(&size, &g_m, &syn1neg[row2], &ONE, work_m, &ONE) # accumulate work
        #########################################################################


        # print('update syn1neg')
        ##########################################################################
        if use_token:
            our_saxpy(&size, &g,   neu1,  &ONE, &syn1neg[row2], &ONE)

        if use_pos:
            our_saxpy(&size, &g_p, neu_p, &ONE, &syn1neg[row2], &ONE)

        if use_sub >= 1:
            our_saxpy(&size, &g2,  neu2,  &ONE, &syn1neg[row2], &ONE)
            if use_sub >= 2:
                our_saxpy(&size, &g3,  neu3,  &ONE, &syn1neg[row2], &ONE)
                if use_sub >= 3:
                    our_saxpy(&size, &g4,  neu4,  &ONE, &syn1neg[row2], &ONE)
                    if use_sub >= 4:
                        our_saxpy(&size, &g5,  neu5,  &ONE, &syn1neg[row2], &ONE)
                        if use_sub >= 5:
                            our_saxpy(&size, &g6,  neu6,  &ONE, &syn1neg[row2], &ONE)
                            if use_sub >= 6:
                                our_saxpy(&size, &g7,  neu7,  &ONE, &syn1neg[row2], &ONE)

        if use_merger:
            our_saxpy(&size, &g_m, neu_m, &ONE, &syn1neg[row2], &ONE)
        ##########################################################################
    #################################### E: calculate hProj_grad and update syn1neg

    # print('assgn merger')
    #################################### S: update syn0 gradient
    if use_merger:
        if use_token:
            our_saxpy(&size, &channel_no_inv, work_m, &ONE, work,  &ONE)

        if use_pos:
            our_saxpy(&size, &channel_no_inv, work_m, &ONE, work_p,  &ONE)

        if use_sub >= 1:
            our_saxpy(&size, &channel_no_inv, work_m, &ONE, work2, &ONE)
            if use_sub >= 2:
                our_saxpy(&size, &channel_no_inv, work_m, &ONE, work3, &ONE)
                if use_sub >= 3:
                    our_saxpy(&size, &channel_no_inv, work_m, &ONE, work4, &ONE)
                    if use_sub >= 4:
                        our_saxpy(&size, &channel_no_inv, work_m, &ONE, work5, &ONE)
                        if use_sub >= 5:
                            our_saxpy(&size, &channel_no_inv, work_m, &ONE, work6, &ONE)
                            if use_sub >= 6:
                                our_saxpy(&size, &channel_no_inv, work_m, &ONE, work7, &ONE)

    if cbow_mean:  # if true, use standard grad
        # divide error over summed window vectors # confusing here? to see the result tomorrow
        # if cbow_mean means do the right gradient method
        # otherwise it is not.
        if use_token:
            sscal(&size, &inv_count,  work,  &ONE)  # (does this need BLAS-variants like saxpy?)

        if use_token:
            sscal(&size, &inv_count,  work_p,  &ONE)  # (does this need BLAS-variants like saxpy?)

        if use_sub >= 1:
            sscal(&size, &inv_count,  work2, &ONE)
            if use_sub >= 2:
                sscal(&size, &inv_count,  work3, &ONE)
                if use_sub >= 3:
                    sscal(&size, &inv_count,  work4, &ONE)
                    if use_sub >= 4:
                        sscal(&size, &inv_count,  work5, &ONE)
                        if use_sub >= 5:
                            sscal(&size, &inv_count,  work6, &ONE)
                            if use_sub >= 6:
                                sscal(&size, &inv_count,  work7, &ONE)

    if use_token:
        for m in range(j,k): 
            if m == i:
                continue
            else:
                our_saxpy(&size, &word_locks[indexes[m]], work, &ONE, &syn0[indexes[m]*size],      &ONE)

    # print('update pos')
    if use_pos:
        for m in range(j,k): 
            if m == i:
                continue
            else:
                our_saxpy(&size, &ONEF,                   work, &ONE, &syn_p[indexes_pos[m]*size], &ONE)
    # print('update pos end')

    if use_sub >= 1:
        for m in range(j, k): # sg case: j = k; loop left tokens here
            if m == i:
                continue
            else:
                ############### This four lines are important ###############
                left_word = indexes[m] # left_word  #
                
                word_lenginv = syn0_1_LengInv[left_word]
                gs = syn0_1_EndIdx[left_word-1]     # 
                ge = syn0_1_EndIdx[left_word]       # 
                for n in range(gs, ge):             # 
                    grain_index = syn0_1_LookUp[n]
                    our_saxpy(&size, &word_lenginv, work2, &ONE, &syn0_1[grain_index * size], &ONE)

                ###############################################################
                if use_sub >= 2:
                    word_lenginv = syn0_2_LengInv[left_word] #--------------> change syn0_1_LengInv to syn0_2_LengInv
                    gs = syn0_2_EndIdx[left_word-1]     #    #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                    ge = syn0_2_EndIdx[left_word]       #    #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                    for n in range(gs, ge):             #
                        grain_index = syn0_2_LookUp[n]  #    #--------------> change syn0_1_LookUp to syn0_2_LookUp
                        our_saxpy(&size, &word_lenginv, work3, &ONE, &syn0_2[grain_index * size], &ONE) #--------------> change syn0_1 to syn0_2 and work2 to work3
                    ###############################################################
                    if use_sub >= 3:
                        word_lenginv = syn0_3_LengInv[left_word] #--------------> change syn0_1_LengInv to syn0_2_LengInv
                        gs = syn0_3_EndIdx[left_word-1]     #    #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                        ge = syn0_3_EndIdx[left_word]       #    #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                        for n in range(gs, ge):             #
                            grain_index = syn0_3_LookUp[n]  #    #--------------> change syn0_1_LookUp to syn0_2_LookUp
                            our_saxpy(&size, &word_lenginv, work4, &ONE, &syn0_3[grain_index * size], &ONE) #--------------> change syn0_1 to syn0_2 and work2 to work3
                        ###############################################################
                        if use_sub >= 4:
                            word_lenginv = syn0_4_LengInv[left_word] #--------------> change syn0_1_LengInv to syn0_2_LengInv
                            gs = syn0_4_EndIdx[left_word-1]     #    #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                            ge = syn0_4_EndIdx[left_word]       #    #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                            for n in range(gs, ge):             #
                                grain_index = syn0_4_LookUp[n]  #    #--------------> change syn0_1_LookUp to syn0_2_LookUp
                                our_saxpy(&size, &word_lenginv, work5, &ONE, &syn0_4[grain_index * size], &ONE) #--------------> change syn0_1 to syn0_2 and work2 to work3
                            ###############################################################
                            if use_sub >= 5:
                                word_lenginv = syn0_5_LengInv[left_word] #--------------> change syn0_1_LengInv to syn0_2_LengInv
                                gs = syn0_5_EndIdx[left_word-1]     #    #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                                ge = syn0_5_EndIdx[left_word]       #    #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                                for n in range(gs, ge):             #
                                    grain_index = syn0_5_LookUp[n]  #    #--------------> change syn0_1_LookUp to syn0_2_LookUp
                                    our_saxpy(&size, &word_lenginv, work6, &ONE, &syn0_5[grain_index * size], &ONE) #--------------> change syn0_1 to syn0_2 and work2 to work3
                                ###############################################################
                                if use_sub >= 6:
                                    word_lenginv = syn0_6_LengInv[left_word] #--------------> change syn0_1_LengInv to syn0_2_LengInv
                                    gs = syn0_6_EndIdx[left_word-1]     #    #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                                    ge = syn0_6_EndIdx[left_word]       #    #--------------> change syn0_1_EndIdx to syn0_2_EndIdx
                                    for n in range(gs, ge):             #
                                        grain_index = syn0_6_LookUp[n]  #    #--------------> change syn0_1_LookUp to syn0_2_LookUp
                                        our_saxpy(&size, &word_lenginv, work7, &ONE, &syn0_6[grain_index * size], &ONE) #--------------> change syn0_1 to syn0_2 and work2 to work3
                                    ###############################################################

    return next_random
################################################################# Field Embedding WITH NLPText


cdef int SUBSAMPLING = 1
##############################################
#--> NEW for M0X1
def train_batch_fieldembed_M0X1(model, 
    indexes, sentence_idx, alpha, _work, _neu1, _work2, _neu2, _work_m, _neu_m, compute_loss = 0, 
    subsampling = SUBSAMPLING):

    cdef Word2VecConfig c
    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end
    # cdef int sg
    # print('before init')
    init_w2v_config_M0X1(&c, model, alpha, compute_loss,  _work, _neu1, _work2, _neu2, _work_m, _neu_m) # this is the difference between sg and cbow
    
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
            for loc_idx in range(idx_start, idx_end):
                # loc_idx = i + idx_start
                word_vocidx = indexes[loc_idx]

                if word_vocidx <= 3:
                    continue
                if c.sample and vlookup[word_vocidx].sample_int < random_int32(&c.next_random):
                    continue

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
        effective_words = len(indexes)
        effective_sentences = len(sentence_idx) # a little different from the original sentence_idx and effective_sentences
        
        # print(indexes[:10])
        for i, item in enumerate(indexes):
            c.indexes[i] = item

        # for i, item in enumerate(indexes_pos):
        #     c.indexes_pos[i] = item

        for i, item in enumerate(sentence_idx):
            c.sentence_idx[i] = item


    # precompute "reduced window" offsets in a single randint() call
    for i, item in enumerate(model.random.randint(0, c.window, effective_words)):
        c.reduced_windows[i] = item

    # skip_ngram model
    # sg = c.sg
    # print(sg)
    with nogil: # LESSION: you should notice this nogil, otherwise the threads are rubbish
    # if True:
        # print(indexes[:10])

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
                # print(indexes[j:k])
                if c.sg == 1:
                    for j in range(j, k): # change the first j to another name: such as t.
                        if j == i:
                            continue
                        # build the batch here
                        c.next_random = fieldembed_token_neg_M0X1(c.alpha, c.size, c.negative, c.cum_table, c.cum_table_len, 
                            c.indexes, i, j, j + 1, 
                            c.use_head, c.use_sub, c.use_merger,# new
                            c.syn0, 
                            c.syn0_1, c.syn0_1_LookUp, c.syn0_1_EndIdx, c.syn0_1_LengInv, c.syn0_1_leng_max, # new
                            c.syn1neg, c.word_locks, 
                            c.neu1, c.work, 
                            c.neu2, c.work2,        # new
                            c.neu_m, c.work_m,      # new for M
                            c.cbow_mean, c.next_random, c.compute_loss, &c.running_training_loss)
                else:
                    # build the batch here
                    c.next_random = fieldembed_token_neg_M0X1(c.alpha, c.size, c.negative, c.cum_table, c.cum_table_len, 
                            c.indexes, i, j, k, 
                            c.use_head, c.use_sub, c.use_merger,# new
                            c.syn0, 
                            c.syn0_1, c.syn0_1_LookUp, c.syn0_1_EndIdx, c.syn0_1_LengInv, c.syn0_1_leng_max, # new
                            c.syn1neg, c.word_locks, 
                            c.neu1, c.work, 
                            c.neu2, c.work2,        # new
                            c.neu_m, c.work_m,      # new for M
                            c.cbow_mean, c.next_random, c.compute_loss, &c.running_training_loss)

    model.running_training_loss = c.running_training_loss
    return effective_words

##############################################
#--> NEW for M0X2
def train_batch_fieldembed_M0X2(model, indexes, sentence_idx, alpha, _work, _neu1, _work2, _neu2, _work3, _neu3, 
    _work_m, _neu_m, compute_loss = 0, subsampling = SUBSAMPLING):

    cdef Word2VecConfig c
    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end
    # cdef int sg
    # print('before init')
    init_w2v_config_M0X2(&c, model, alpha, compute_loss,  _work, _neu1, _work2, _neu2, _work3, _neu3, _work_m, _neu_m) # this is the difference between sg and cbow
    

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
            for loc_idx in range(idx_start, idx_end):
                # loc_idx = i + idx_start
                word_vocidx = indexes[loc_idx]

                if word_vocidx <= 3:
                    continue
                if c.sample and vlookup[word_vocidx].sample_int < random_int32(&c.next_random):
                    continue

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
        effective_words = len(indexes)
        effective_sentences = len(sentence_idx) # a little different from the original sentence_idx and effective_sentences
        
        # print(indexes[:10])
        for i, item in enumerate(indexes):
            c.indexes[i] = item

        for i, item in enumerate(sentence_idx):
            c.sentence_idx[i] = item


    # precompute "reduced window" offsets in a single randint() call
    for i, item in enumerate(model.random.randint(0, c.window, effective_words)):
        c.reduced_windows[i] = item

    # skip_ngram model
    # sg = c.sg
    # print(sg)
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
                # print(indexes[j:k])
                if c.sg == 1:
                    for j in range(j, k): # change the first j to another name: such as t.
                        if j == i:
                            continue
                        # build the batch here
                        c.next_random = fieldembed_token_neg_M0X2(c.alpha, c.size, c.negative, c.cum_table, c.cum_table_len, 
                            c.indexes, i, j, j + 1, 
                            c.use_head, c.use_sub, c.use_merger,# new
                            c.syn0, 
                            c.syn0_1, c.syn0_1_LookUp, c.syn0_1_EndIdx, c.syn0_1_LengInv, c.syn0_1_leng_max, # new
                            c.syn0_2, c.syn0_2_LookUp, c.syn0_2_EndIdx, c.syn0_2_LengInv, c.syn0_2_leng_max, # new
                            c.syn1neg, c.word_locks, 
                            c.neu1, c.work, 
                            c.neu2, c.work2,        # new
                            c.neu3, c.work3,        # new
                            c.neu_m, c.work_m,      # new for M
                            c.cbow_mean, c.next_random, c.compute_loss, &c.running_training_loss)
                else:
                    # build the batch here
                    c.next_random = fieldembed_token_neg_M0X2(c.alpha, c.size, c.negative, c.cum_table, c.cum_table_len, 
                            c.indexes, i, j, k, 
                            c.use_head, c.use_sub, c.use_merger,# new
                            c.syn0, 
                            c.syn0_1, c.syn0_1_LookUp, c.syn0_1_EndIdx, c.syn0_1_LengInv, c.syn0_1_leng_max, # new
                            c.syn0_2, c.syn0_2_LookUp, c.syn0_2_EndIdx, c.syn0_2_LengInv, c.syn0_2_leng_max, # new
                            c.syn1neg, c.word_locks, 
                            c.neu1, c.work, 
                            c.neu2, c.work2,        # new
                            c.neu3, c.work3,        # new
                            c.neu_m, c.work_m,      # new for M
                            c.cbow_mean, c.next_random, c.compute_loss, &c.running_training_loss)

    model.running_training_loss = c.running_training_loss
    return effective_words

##############################################
#--> NEW for M0XY
def train_batch_fieldembed_M0XY(model, indexes, sentence_idx, alpha, _work, 
    _neu1, _work2, _neu2, _work3, _neu3,_work4, _neu4,_work5, _neu5,_work6, _neu6,_work7, _neu7, 
    _work_m, _neu_m, compute_loss = 0, subsampling = SUBSAMPLING):

    cdef Word2VecConfig c
    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end


    # print(len(indexes))

    init_w2v_config_M0XY(&c, model, alpha, compute_loss,  _work, _neu1, _work2, _neu2, _work3, _neu3, _work4, _neu4,_work5, _neu5,_work6, _neu6,_work7, _neu7, _work_m, _neu_m) # this is the difference between sg and cbow
    
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
            for loc_idx in range(idx_start, idx_end):
                # loc_idx = i + idx_start
                word_vocidx = indexes[loc_idx]

                if word_vocidx <= 3:
                    continue
                if c.sample and vlookup[word_vocidx].sample_int < random_int32(&c.next_random):
                    continue

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
        effective_words = len(indexes)
        effective_sentences = len(sentence_idx) # a little different from the original sentence_idx and effective_sentences
        
        # print(indexes[:10])
        for i, item in enumerate(indexes):
            c.indexes[i] = item

        for i, item in enumerate(sentence_idx):
            c.sentence_idx[i] = item


    # precompute "reduced window" offsets in a single randint() call
    for i, item in enumerate(model.random.randint(0, c.window, effective_words)):
        c.reduced_windows[i] = item

    # skip_ngram model
    # sg = c.sg
    # print(sg)
    with nogil: # LESSION: you should notice this nogil, otherwise the threads are rubbish
    # if True:
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
                # print(indexes[j:k])
                if c.sg == 1:
                    for j in range(j, k): # change the first j to another name: such as t.
                        if j == i:
                            continue
                        # build the batch here
                        c.next_random = fieldembed_token_neg_M0XY(c.alpha, c.size, c.negative, c.cum_table, c.cum_table_len, 
                            c.indexes, i, j, j + 1, 
                            c.use_head, c.use_sub, c.use_merger,# new
                            c.syn0, 
                            c.syn0_1, c.syn0_1_LookUp, c.syn0_1_EndIdx, c.syn0_1_LengInv, c.syn0_1_leng_max, # new
                            c.syn0_2, c.syn0_2_LookUp, c.syn0_2_EndIdx, c.syn0_2_LengInv, c.syn0_2_leng_max, # new
                            c.syn0_3, c.syn0_3_LookUp, c.syn0_3_EndIdx, c.syn0_3_LengInv, c.syn0_3_leng_max, # new
                            c.syn0_4, c.syn0_4_LookUp, c.syn0_4_EndIdx, c.syn0_4_LengInv, c.syn0_4_leng_max, # new
                            c.syn0_5, c.syn0_5_LookUp, c.syn0_5_EndIdx, c.syn0_5_LengInv, c.syn0_5_leng_max, # new
                            c.syn0_6, c.syn0_6_LookUp, c.syn0_6_EndIdx, c.syn0_6_LengInv, c.syn0_6_leng_max, # new
                            c.syn1neg, c.word_locks, 
                            c.neu1, c.work, 
                            c.neu2, c.work2,        # new
                            c.neu3, c.work3,        # new
                            c.neu4, c.work4,        # new
                            c.neu5, c.work5,        # new
                            c.neu6, c.work6,        # new
                            c.neu7, c.work7,        # new
                            c.neu_m, c.work_m,      # new for M
                            c.cbow_mean, c.next_random, c.compute_loss, &c.running_training_loss)
                else:
                    # build the batch here
                    c.next_random = fieldembed_token_neg_M0XY(c.alpha, c.size, c.negative, c.cum_table, c.cum_table_len, 
                            c.indexes, i, j, k, 
                            c.use_head, c.use_sub, c.use_merger,# new
                            c.syn0, 
                            c.syn0_1, c.syn0_1_LookUp, c.syn0_1_EndIdx, c.syn0_1_LengInv, c.syn0_1_leng_max, # new
                            c.syn0_2, c.syn0_2_LookUp, c.syn0_2_EndIdx, c.syn0_2_LengInv, c.syn0_2_leng_max, # new
                            c.syn0_3, c.syn0_3_LookUp, c.syn0_3_EndIdx, c.syn0_3_LengInv, c.syn0_3_leng_max, # new
                            c.syn0_4, c.syn0_4_LookUp, c.syn0_4_EndIdx, c.syn0_4_LengInv, c.syn0_4_leng_max, # new
                            c.syn0_5, c.syn0_5_LookUp, c.syn0_5_EndIdx, c.syn0_5_LengInv, c.syn0_5_leng_max, # new
                            c.syn0_6, c.syn0_6_LookUp, c.syn0_6_EndIdx, c.syn0_6_LengInv, c.syn0_6_leng_max, # new
                            c.syn1neg, c.word_locks, 
                            c.neu1, c.work, 
                            c.neu2, c.work2,        # new
                            c.neu3, c.work3,        # new
                            c.neu4, c.work4,        # new
                            c.neu5, c.work5,        # new
                            c.neu6, c.work6,        # new
                            c.neu7, c.work7,        # new
                            c.neu_m, c.work_m,      # new for M
                            c.cbow_mean, c.next_random, c.compute_loss, &c.running_training_loss)

    model.running_training_loss = c.running_training_loss
    return effective_words

#--> NEW for M0XY_P
def train_batch_fieldembed_M0XY_P(model, indexes, sentence_idx, alpha, _work, 
    _neu1, _work2, _neu2, _work3, _neu3,_work4, _neu4,_work5, _neu5,_work6, _neu6,_work7, _neu7, 
    _work_m, _neu_m, _work_p, _neu_p, compute_loss = 0, subsampling = SUBSAMPLING):

    cdef Word2VecConfig c
    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end

    cdef int word_vocidx, pos_vocidx = 0
    cdef int loc_idx

    init_w2v_config_M0XY_P(&c, model, alpha, compute_loss, _work, _neu1, 
                           _work2, _neu2, _work3, _neu3, _work4, _neu4,_work5, _neu5,_work6, _neu6,_work7, _neu7,
                           _work_m, _neu_m, _work_p, _neu_p) # this is the difference between sg and cbow
    



    if c.use_pos:
        # print('use_pos')
        indexes_pos = indexes[1]
        indexes     = indexes[0] 
    else:
        indexes_pos = []
        indexes     = indexes

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
            for loc_idx in range(idx_start, idx_end):
                # loc_idx = i + idx_start
                word_vocidx = indexes[loc_idx]
                if c.use_pos:
                    # print('current loc_idx is', loc_idx)
                    pos_vocidx = indexes_pos[loc_idx] 
                    # print('get pos voc id is', pos_vocidx)

                ## filter high and low freq tokens
                if word_vocidx <= 3:
                    continue
                if c.sample and vlookup[word_vocidx].sample_int < random_int32(&c.next_random):
                    continue


                c.indexes[effective_words] = word_vocidx
                if c.use_pos:
                    c.indexes_pos[effective_words] = pos_vocidx

                effective_words +=1


                if effective_words == MAX_SENTENCE_LEN:
                    break  # TODO: log warning, tally overflow?

            # step3: add the new idx_end for this sentence, that is, the value of effective_words
            c.sentence_idx[effective_sentences] = effective_words
            effective_sentences += 1
            if effective_words == MAX_SENTENCE_LEN:
                break  # TODO: log warning, tally overflow?

    else:        
        effective_words = len(indexes)
        effective_sentences = len(sentence_idx) # a little different from the original sentence_idx and effective_sentences
        
        # print(indexes[:10])
        for i, item in enumerate(indexes):
            c.indexes[i] = item

        for i, item in enumerate(indexes_pos):
            c.indexes_pos[i] = item

        for i, item in enumerate(sentence_idx):
            c.sentence_idx[i] = item

    # print('in the gate of reduced_windows')
    # precompute "reduced window" offsets in a single randint() call
    for i, item in enumerate(model.random.randint(0, c.window, effective_words)):
        c.reduced_windows[i] = item

    # print('in the gate of nogil')
    # print('the leng of indexes:', len(indexes))
    # print('the leng of pos:', len(indexes_pos))
    with nogil: # LESSION: you should notice this nogil, otherwise the threads are rubbish

    # if True:
        for sent_idx in range(effective_sentences):
            # idx_start and idx_end
            idx_end = c.sentence_idx[sent_idx]
            if sent_idx == 0:
                idx_start = 0
            else:
                idx_start = c.sentence_idx[sent_idx-1]

            # print('in a sentence now')
            for i in range(idx_start, idx_end):
                j = i - c.window + c.reduced_windows[i]
                if j < idx_start:
                    j = idx_start
                k = i + c.window + 1 - c.reduced_windows[i]
                if k > idx_end:
                    k = idx_end
                # print(j, i, k)
                # print(indexes[j:k])
                if c.sg == 1:
                    for j in range(j, k): # change the first j to another name: such as t.
                        if j == i:
                            continue

                        # print('in sg')
                        # build the batch here
                        c.next_random = fieldembed_token_neg_M0XY_P(c.alpha, c.size, c.negative, c.cum_table, c.cum_table_len, 
                            c.indexes, c.indexes_pos, i, j, j + 1, 
                            c.use_head, c.use_sub, c.use_merger, c.use_token, c.use_pos, 
                            c.syn0, c.syn_p,
                            c.syn0_1, c.syn0_1_LookUp, c.syn0_1_EndIdx, c.syn0_1_LengInv, c.syn0_1_leng_max, # new
                            c.syn0_2, c.syn0_2_LookUp, c.syn0_2_EndIdx, c.syn0_2_LengInv, c.syn0_2_leng_max, # new
                            c.syn0_3, c.syn0_3_LookUp, c.syn0_3_EndIdx, c.syn0_3_LengInv, c.syn0_3_leng_max, # new
                            c.syn0_4, c.syn0_4_LookUp, c.syn0_4_EndIdx, c.syn0_4_LengInv, c.syn0_4_leng_max, # new
                            c.syn0_5, c.syn0_5_LookUp, c.syn0_5_EndIdx, c.syn0_5_LengInv, c.syn0_5_leng_max, # new
                            c.syn0_6, c.syn0_6_LookUp, c.syn0_6_EndIdx, c.syn0_6_LengInv, c.syn0_6_leng_max, # new
                            c.syn1neg, c.word_locks, 
                            c.neu1, c.work, 
                            c.neu2, c.work2,        # new
                            c.neu3, c.work3,        # new
                            c.neu4, c.work4,        # new
                            c.neu5, c.work5,        # new
                            c.neu6, c.work6,        # new
                            c.neu7, c.work7,        # new
                            c.neu_m, c.work_m,      # new for M
                            c.neu_p, c.work_p,      # new for P
                            c.cbow_mean, c.next_random, c.compute_loss, &c.running_training_loss)
                else:
                    # print('in cbow')
                    # build the batch here
                    c.next_random = fieldembed_token_neg_M0XY_P(c.alpha, c.size, c.negative, c.cum_table, c.cum_table_len, 
                            c.indexes, c.indexes_pos, i, j, k, 
                            c.use_head, c.use_sub, c.use_merger, c.use_token, c.use_pos, 
                            c.syn0, c.syn_p,
                            c.syn0_1, c.syn0_1_LookUp, c.syn0_1_EndIdx, c.syn0_1_LengInv, c.syn0_1_leng_max, # new
                            c.syn0_2, c.syn0_2_LookUp, c.syn0_2_EndIdx, c.syn0_2_LengInv, c.syn0_2_leng_max, # new
                            c.syn0_3, c.syn0_3_LookUp, c.syn0_3_EndIdx, c.syn0_3_LengInv, c.syn0_3_leng_max, # new
                            c.syn0_4, c.syn0_4_LookUp, c.syn0_4_EndIdx, c.syn0_4_LengInv, c.syn0_4_leng_max, # new
                            c.syn0_5, c.syn0_5_LookUp, c.syn0_5_EndIdx, c.syn0_5_LengInv, c.syn0_5_leng_max, # new
                            c.syn0_6, c.syn0_6_LookUp, c.syn0_6_EndIdx, c.syn0_6_LengInv, c.syn0_6_leng_max, # new
                            c.syn1neg, c.word_locks, 
                            c.neu1, c.work, 
                            c.neu2, c.work2,        # new
                            c.neu3, c.work3,        # new
                            c.neu4, c.work4,        # new
                            c.neu5, c.work5,        # new
                            c.neu6, c.work6,        # new
                            c.neu7, c.work7,        # new
                            c.neu_m, c.work_m,      # new for M
                            c.neu_p, c.work_p,      # new for P
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
