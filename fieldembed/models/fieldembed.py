from __future__ import division 

import logging
import sys
import os
import heapq
from timeit import default_timer
from datetime import datetime
from copy import deepcopy
from collections import defaultdict
import threading
import itertools
import warnings
from pprint import pprint
from queue import Queue, Empty

import numpy as np
from numpy import exp, dot, zeros, random, dtype, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, fromstring, sqrt,\
    empty, sum as np_sum, ones, logaddexp, log, outer

from scipy.special import expit
from six import iteritems, itervalues, string_types
from six.moves import range

from .. import utils, matutils  
from ..utils import deprecated
from ..utils import keep_vocab_item, call_on_class_only
from .keyedvectors import Vocab, Word2VecKeyedVectors
from .fieldembed_core import train_batch_fieldembed_0X1_neat

logger = logging.getLogger(__name__)

Field_Info = {
    #field: [head-subfield, subfield, subfield]
    'token':['token', 'char', 'subcomp', 'stroke', 'pinyin'],
    'pos'  :['pos'],
    'ner'  :['ner'],
}

Field_Idx = {
    #field: [head-subfield, subfield, subfield]
    'token':0,
    'pos'  :1,
    'ner'  :2,
}


class FieldEmbedding(utils.SaveLoad):

    def __init__(self, nlptext = None, Field_Settings = {}, mode = 'sg_nlptext', sg=0, use_merger = 0, 
        neg_init = 0, standard_grad = 1, size=100, alpha=0.025, window=5, sample=1e-3, seed=1, workers=3, 
        min_alpha=0.0001,negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=hash, iter=5,
        batch_words=MAX_WORDS_IN_BATCH, compute_loss=False, callbacks=()):

        self.standard_grad = standard_grad
        self.mode = mode
        self.callbacks = callbacks
        self.load = call_on_class_only
        self.Field_Settings = Field_Settings
        self.use_merger = use_merger
        self.neg_init = neg_init

        # here only do initializations for wv, vocabulary, and trainables
        # there is a self.wv, and its vectors may be empty.
        self.wv     = Word2VecKeyedVectors(size)
        self.wv_neg = Word2VecKeyedVectors(size)
       
        self.vocabulary = FieldEmbedVocab(sample=sample, ns_exponent=ns_exponent)
        self.trainables = FieldEmbedTrainables(seed=seed, vector_size=size, hashfxn=hashfxn)

        self.weights = {}
        self.field_info = {}
        self.field_idx  = {}

        self.field_sub  = []
        self.field_head = []
        self.field_hyper= []
        
        if vector_size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        
        self.window = int(window)
        self.random = random.RandomState(seed) # random = seed
        
        self.sg = int(sg)

        # self.hs = int(hs) # should be removed
        self.negative = int(negative)

        self.ns_exponent = ns_exponent
        self.cbow_mean = int(cbow_mean)
        self.compute_loss = bool(compute_loss)
        self.running_training_loss = 0
 
        self.alpha = float(alpha)
        self.min_alpha_yet_reached = float(alpha)
        self.min_alpha = float(min_alpha)

        self.corpus_count = 0
        self.corpus_total_words = 0


        self.vector_size = int(vector_size)
        self.workers = int(workers)
        self.epochs = epochs
        self.train_count = 0
        self.total_train_time = 0
        self.batch_words = batch_words
        self.model_trimmed_post_training = False
        self.callbacks = callbacks

        self.build_vocab(nlptext = nlptext)
        
        print('\n\n======== Training Start ....'); s = datetime.now()
        self.train(nlptext = nlptext, total_examples=self.corpus_count,
            total_words=self.corpus_total_words, epochs=self.epochs, 
            start_alpha=self.alpha, end_alpha=self.min_alpha, compute_loss=compute_loss)
        print('======== Training End ......'); e = datetime.now()
        print('======== Total Time: ', e - s)

    ################################################################################################################################################### build_vocab
    def build_vocab(self, nlptext = None, **kwargs):
        # scan_vocab and prepare_vocab
        # build .wv.vocab + .wv.index2word + .wv.cum_table
        update = False

        print('!!!======== Build_vocab based on NLPText....'); s = datetime.now()

        print('-------> Prepare Vocab....')
        total_words, corpus_count, report_values = self.vocabulary.scan_and_prepare_vocab(self, nlptext, 
                                                                    self.negative, update = update, **kwargs) 
        self.corpus_count = corpus_count
        self.corpus_total_words = total_words

        print('-------> Prepare Field Info....')

        self.prepare_field_info(nlptext)
        
        # self.create_field_embedding(size = self.size, )
        report_values['memory'] = self.estimate_memory(vocab_size=report_values['num_retained_words'])

        print('-------> Prepare Trainable Weight....')
        self.trainables.prepare_weights(self, self.negative, self.wv, update=update, vocabulary=self.vocabulary, neg_init = self.neg_init)

        print('======== The Voc and Parameters are Ready!'); e = datetime.now()
        print('======== Total Time: ', e - s)

    def prepare_field_info(self, nlptext):

        if len(self.Field_Settings) == 0:
            # CASE 1: no Field_Settings are provided!
            # Only use token itself.
            self.field_idx ['token']  = len(self.field_idx)
            self.field_info['token'] = ['token']
            self.field_head.append([])
            self.field_head[self.field_idx['token']] = [1, self.wv] # CB-010 / SG-010
            self.field_sub.append([])
            self.field_sub [self.field_idx['token']]  = []
            self.use_head = 1
            self.use_sub  = 0
            self.proj_num = self.use_head +  self.use_sub
            self.weights['token'] = self.wv 

        else:
            for channel, f_setting in self.Field_Settings.items():
                # Field_Settings is nlptext's Channel_Settings
                if channel in Field_Info:

                    hyper_field = channel
                    LGU, DGU  = nlptext.getGrainUnique(hyper_field, tagScheme = 'BIOES') # A LESSION 

                    wv = self.create_field_embedding(self.vector_size, hyper_field, LGU, DGU)
                    self.weights[hyper_field] = wv
                    
                    if hyper_field in self.field_info:
                        self.field_info[channel].append(channel)
                    else:
                        self.field_info[channel] = [channel]

                    if hyper_field not in self.field_idx:
                        self.field_idx[hyper_field]  = len(self.field_idx)

                    if self.field_idx[hyper_field] == len(self.field_head):
                        self.field_head.append([1, wv])
                    elif self.field_idx[hyper_field] < len(self.field_head):
                        self.field_head[self.field_idx[hyper_field]] = [1, wv]
                    else:
                        prnt('Error in field ')

                    if self.field_idx[hyper_field] == len(self.field_sub):
                        self.field_sub.append([])
                        self.field_sub[self.field_idx[channel]] = []

                else:
                    # when the channls are the sub fields
                    data = self.get_subfield_info(nlptext, channel, **f_setting) 
                    GU, LookUp, EndIdx, Leng_Inv, Leng_max, TU, LKP = data
                    LGU, DGU = GU
                    wv = self.create_field_embedding(self.vector_size, channel, LGU, DGU)
                    self.weights[channel] = wv
                    LTU, DTU = TU
                    # set different attributes
                    wv.LookUp  = LookUp
                    wv.EndIdx  = EndIdx
                    wv.Leng_Inv= Leng_Inv
                    wv.Leng_max= Leng_max
                    wv.LTU = LTU
                    wv.DTU = DTU
                    wv.LKP = LKP
                    
                    for field, subFields in  Field_Info.items():
                        if channel in subFields:
                            ######## deal with the field first
                            if field not in self.field_idx:
                                self.field_idx[field]  = len(self.field_idx)
                            
                            if self.field_idx[field] == len(self.field_head):
                                self.field_head.append([0, None])
                            ######## deal with the field first

                            if self.field_idx[field] == len(self.field_sub):
                                self.field_sub.append([])
                                self.field_sub[self.field_idx[field]] = [ [wv, LookUp, EndIdx, Leng_Inv, Leng_max] ]

                            elif self.field_idx[field] < len(self.field_sub):
                                self.field_sub[self.field_idx[field]].append([wv, LookUp, EndIdx, Leng_Inv, Leng_max])
                            else:
                                prnt('Error in subfield ')

                            if field in self.field_info:
                                self.field_info[field].append(channel)
                            else:
                                self.field_info[field] = [channel]
                            break
            
            self.use_head = sum([i[0] for i in self.field_head])
            self.use_sub  = sum([len(i) for i in self.field_sub])
            self.proj_num = self.use_head + self.use_sub
        
        pprint(self.field_idx)
        pprint(self.field_info)
        pprint(self.field_head)
        pprint(self.field_sub)
        for i, wv in self.weights.items():
            print(i, wv.vectors.shape)
        print('use_head:', self.use_head, 'use_sub:', self.use_sub)

    def get_subfield_info(self, nlptext, field = 'char', Max_Ngram = 1, end_grain = False):
        GU = nlptext.getGrainUnique(field, Max_Ngram=Max_Ngram, end_grain=end_grain)
        LKP, TU = nlptext.getLookUp(field, Max_Ngram=Max_Ngram, end_grain=end_grain)
        charLeng = np.array([len(i) for i in LKP], dtype = np.uint32)
        charLeng_max = np.max(charLeng)
        charEndIdx = np.cumsum(charLeng, dtype = np.uint32) # LESSION: ignoring the np.uint32 wastes me a lot of time
        charLookUp = np.array(list(itertools.chain.from_iterable(LKP)), dtype = np.uint32)
        charLeng_inv = 1 / charLeng
        charLeng_inv = charLeng_inv.astype(REAL)
        return GU, charLookUp, charEndIdx, charLeng_inv, charLeng_max, TU, LKP

    def create_field_embedding(self, size, channel, LGU, DGU):
        '''
            size: embedding size
            channel: field name
            LGU: index2str
            DGU: str2index
            --------------
            return and set a wv for fieldembed: wv_channel
        '''
        if channel == 'token':
            return self.wv
        else:
            gw = Word2VecKeyedVectors(size)
            gw.index2word = LGU 
            for gr in DGU:
                gw.vocab[gr] = Vocab(index=DGU[gr])
            self.__setattr__('wv_' + channel, gw)
            return self.__getattribute__('wv_' + channel)

    def estimate_memory(self, vocab_size=None, report=None):
        vocab_size = vocab_size or len(self.wv.vocab)
        report = report or {}
        report['vocab'] = vocab_size * (700 if self.hs else 500)
        report['vectors'] = vocab_size * self.vector_size * dtype(REAL).itemsize
        # if self.hs:
        #     report['syn1'] = vocab_size * self.trainables.layer1_size * dtype(REAL).itemsize
        if self.negative:
            report['syn1neg'] = vocab_size * self.trainables.layer1_size * dtype(REAL).itemsize
        report['total'] = sum(report.values())
        logger.info(
            "estimated required memory for %i words and %i dimensions: %i bytes",
            vocab_size, self.vector_size, report['total']
        )
        return report

    ################################################################################################################################################### train
    def train(self, nlptext = None, total_examples=None, 
        total_words=None, epochs=None, start_alpha=None, 
        end_alpha=None, word_count=0,queue_factor=2, 
        report_delay=1.0, compute_loss=False, callbacks=(), **kwargs):

        #---------------------------------------------------------------
        
        self.alpha = start_alpha or self.alpha
        self.min_alpha = end_alpha or self.min_alpha
        self.compute_loss = compute_loss
        self.running_training_loss = 0.0

        self._set_train_params(**kwargs)
        if callbacks:
            self.callbacks = callbacks
        self.epochs = epochs
        self._check_training_sanity(
            epochs=epochs,
            total_examples=total_examples,
            total_words=total_words, **kwargs)

        for callback in self.callbacks:
            callback.on_train_begin(self)

        trained_word_count = 0
        raw_word_count = 0
        start = default_timer() - 0.00001
        job_tally = 0

        for cur_epoch in range(self.epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(self)

            if nlptext is not None:
                trained_word_count_epoch, raw_word_count_epoch, job_tally_epoch = self._train_epoch_nlptext(
                    nlptext, cur_epoch=cur_epoch, total_examples=total_examples,
                    total_words=total_words, queue_factor=queue_factor, report_delay=report_delay
                )
            else:
                raise('No Training Data is Provided...')

            trained_word_count += trained_word_count_epoch
            raw_word_count += raw_word_count_epoch
            job_tally += job_tally_epoch

            for callback in self.callbacks:
                callback.on_epoch_end(self)

        total_elapsed = default_timer() - start
        self._log_train_end(raw_word_count, trained_word_count, total_elapsed, job_tally)

        self.train_count += 1  # number of times train() has been called
        self._clear_post_train()

        for callback in self.callbacks:
            callback.on_train_end(self)
        return trained_word_count, raw_word_count

    def _set_train_params(self, **kwargs):
        if 'compute_loss' in kwargs:
            self.compute_loss = kwargs['compute_loss']
        self.running_training_loss = 0

    def _check_training_sanity(self, epochs=None, total_examples=None, total_words=None, **kwargs):
        if self.alpha > self.min_alpha_yet_reached:
            logger.warning("Effective 'alpha' higher than previous training cycles")
        if self.model_trimmed_post_training:
            raise RuntimeError("Parameters for training were discarded using model_trimmed_post_training method")

        if not self.wv.vocab:  # should be set by `build_vocab`
            raise RuntimeError("you must first build vocabulary before training the model")

        if not hasattr(self, 'corpus_count'):
            raise ValueError(
                "The number of examples in the training corpus is missing. "
                "Please make sure this is set inside `build_vocab` function."
                "Call the `build_vocab` function before calling `train`."
            )

        if total_words is None and total_examples is None:
            raise ValueError(
                "You must specify either total_examples or total_words, for proper job parameters updation"
                "and progress calculations. "
                "The usual value is total_examples=model.corpus_count."
            )
        if epochs is None:
            raise ValueError("You must specify an explict epochs count. The usual value is epochs=model.epochs.")
        logger.info(
            "training model with %i workers on %i vocabulary and %i features, "
            "using sg=%s hs=%s sample=%s negative=%s window=%s",
            self.workers, len(self.wv.vocab), self.trainables.layer1_size, self.sg,
            self.hs, self.vocabulary.sample, self.negative, self.window
        )

    ####################################################################################### 
    def _train_epoch_nlptext(self, nlptext, cur_epoch=0, total_examples=None, total_words=None,queue_factor=2, report_delay=1.0):
        ########### preprocess
        sentences_endidx = nlptext.SENT['EndIDXTokens']
        tokens_vocidx    = nlptext.TOKEN['ORIGTokenIndex']
        print(len(tokens_vocidx))
        total_examples  =  len(sentences_endidx)
        total_words     =  len(tokens_vocidx)           


        print('Start getting batch infos')
        s = datetime.now(); print(s)
        batch_end_st_idx_list, job_no = nlptext.Calculate_Infos(self.batch_words)
        e = datetime.now(); print(e)
        print('The time of finding batch_end_st_idx_list:', e - s)
        print('Total job number is:', job_no)
        

        # sentences_endidx, tokens_vocidx, batch_end_st_idx_list, job_no, 
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        # in the future, make the selection here. or make selection here
        # make more worker_loop_nlptext1, 2, 3, 4, 5
        workers = [
            threading.Thread(
                target=self._worker_loop_nlptext,
                args=(job_queue, progress_queue, self.proj_num))
            for _ in range(self.workers)
        ]
        logger.info('\n the total_examples is:' + str(total_examples) + '   , the total words is:' + str(total_words) + '\n')
        workers.append(threading.Thread(
            target=self._job_producer_nlptext,
            args=(sentences_endidx, total_examples, tokens_vocidx, pos_vocidx, total_words, batch_end_st_idx_list, job_no, job_queue,), # data_iterable is sentences
            kwargs={'cur_epoch': cur_epoch,}))

        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        trained_word_count, raw_word_count, job_tally = self._log_epoch_progress(
            progress_queue, job_queue, cur_epoch=cur_epoch, total_examples=total_examples, total_words=total_words,
            report_delay=report_delay, is_corpus_file_mode=False)

        return trained_word_count, raw_word_count, job_tally

    def _job_producer_nlptext(self, sentences_endidx, total_examples, tokens_vocidx, pos_vocidx, 
        total_words, batch_end_st_idx_list, job_no, job_queue, cur_epoch=0):
        #---------------------------------------------------# 

        job_batch, batch_size = [], 0
        pushed_words, pushed_examples = 0, 0 # examples refers to sentences
        next_job_params = self._get_job_params(cur_epoch) # current learning rate: cur_alpha
    
        for idx in range(job_no):

            # start and end are batch's start sentence loc_id and end sentence loc_id
            # as python routines, batch is [start, end), left close right open
            start = batch_end_st_idx_list[idx-1] if idx > 0 else 0
            end   = batch_end_st_idx_list[idx]

            # print(start, end)
            # find the start sentence's start token loc_id, and
            # find the end sentence's start token loc_id. (as the end sentence is exluded)
            token_start = sentences_endidx[start-1] if start > 0 else 0
            token_end   = sentences_endidx[end  -1]

            token_indexes  = tokens_vocidx[token_start:token_end] # dtype = np.uint32
            if self.use_pos:
                pos_indexes = pos_vocidx[token_start:token_end]
                indexes = [token_indexes, pos_indexes]
            else:
                indexes = token_indexes
            # sentence_idx= np.array([i-token_start for i in sentences_endidx[start: end]], dtype = np.uint32)
            sentence_idx= [i-token_start for i in sentences_endidx[start: end]]
            # print('The start and end sent loc_id:', start, end)
            # print('The token start and end loc idx in each batch:', token_start, token_end)
            # print(sentence_idx[-1], len(indexes), '\n')
            # assaure that the input is correct
            # TODO
            # print_sentence()
            # print(len(indexes))
            job_queue.put((indexes,  sentence_idx, next_job_params))

            pushed_examples += len(sentence_idx)
            epoch_progress = 1.0 * pushed_examples / total_examples

            # prepare learning rate for next job
            next_job_params = self._update_job_params(next_job_params, epoch_progress, cur_epoch)

        if job_no == 0 and self.train_count == 0:
            logger.warning(
                "train() called with an empty iterator (if not intended, "
                "be sure to provide a corpus that offers restartable iteration = an iterable)."
            )
        # give the workers heads up that they can finish -- no more work!
        for _ in range(self.workers):
            job_queue.put(None) # at the end, give 4 None s if there are 4 calculation workers.
        logger.debug("----> Worker: Job Producer loop exiting, total %i jobs", job_no)

    def _get_job_params(self, cur_epoch):
        alpha = self.alpha - ((self.alpha - self.min_alpha) * float(cur_epoch) / self.epochs)
        return alpha

    def _update_job_params(self, job_params, epoch_progress, cur_epoch):
        start_alpha = self.alpha
        end_alpha = self.min_alpha
        progress = (cur_epoch + epoch_progress) / self.epochs
        next_alpha = start_alpha - (start_alpha - end_alpha) * progress
        next_alpha = max(end_alpha, next_alpha)
        self.min_alpha_yet_reached = next_alpha
        return next_alpha

    def _worker_loop_nlptext(self, job_queue, progress_queue, proj_num = 1):
        thread_private_mem = self._get_thread_working_mem(proj_num = proj_num) 
        merger_private_mem = self._get_thread_working_mem_for_meager()
        jobs_processed = 0
        while True:
            job = job_queue.get()
            if job is None:
                progress_queue.put(None)
                break  # no more jobs => quit this worker
            indexes, sentence_idx,  job_parameters = job 

            for callback in self.callbacks:
                callback.on_batch_begin(self)

            tally, raw_tally = self._do_train_job_nlptext(indexes, sentence_idx, job_parameters, 
                                                          thread_private_mem, 
                                                          merger_private_mem = merger_private_mem)
            for callback in self.callbacks:
                callback.on_batch_end(self)

            progress_queue.put((len(sentence_idx), tally, raw_tally))  # report back progress
            jobs_processed += 1
        logger.debug("o----> Worker exiting, processed %i jobs", jobs_processed)

    def _get_thread_working_mem(self, proj_num = 1):
        return [matutils.zeros_aligned(self.trainables.layer1_size * proj_num, dtype=REAL) for i in range(2)]

    def _get_thread_working_mem_for_meager(self):
        work_m = matutils.zeros_aligned(self.trainables.layer1_size, dtype=REAL) 
        neu_m  = matutils.zeros_aligned(self.trainables.layer1_size, dtype=REAL) 
        return work_m, neu_m

    def _do_train_job_nlptext(self, indexes, sentence_idx, alpha, inits, merger_private_mem = None):
        tally = 0
        if self.mode == 'fieldembed_0X1_neat':
            work, neu1 = inits
            tally += train_batch_fieldembed_0X1_neat(self, indexes, sentence_idx, alpha, work, neu1, self.compute_loss)

    ################################################################################################################################################### log
    def _log_epoch_progress(self, progress_queue=None, job_queue=None, cur_epoch=0, total_examples=None,total_words=None, report_delay=10.0, is_corpus_file_mode=None):

        example_count, trained_word_count, raw_word_count = 0, 0, 0
        start, next_report = default_timer() - 0.00001, 1.0
        job_tally = 0
        unfinished_worker_count = self.workers

        while unfinished_worker_count > 0:
            report = progress_queue.get()  # blocks if workers too slow
            if report is None:  # a thread reporting that it finished
                unfinished_worker_count -= 1
                logger.info("Worker thread finished; awaiting finish of %i more threads", unfinished_worker_count)
                continue
            examples, trained_words, raw_words = report
            job_tally += 1

            # update progress stats
            example_count += examples
            trained_word_count += trained_words  # only words in vocab & sampled
            raw_word_count += raw_words

            # log progress once every report_delay seconds
            elapsed = default_timer() - start
            if elapsed >= next_report:
                self._log_progress(
                    job_queue, progress_queue, cur_epoch, example_count, total_examples,
                    raw_word_count, total_words, trained_word_count, elapsed)
                next_report = elapsed + report_delay
        # all done; report the final stats
        elapsed = default_timer() - start
        self._log_epoch_end(
            cur_epoch, example_count, total_examples, raw_word_count, total_words,
            trained_word_count, elapsed, is_corpus_file_mode)
        self.total_train_time += elapsed
        return trained_word_count, raw_word_count, job_tally

    def _log_progress(self, job_queue, progress_queue, cur_epoch, example_count, total_examples,
        raw_word_count, total_words, trained_word_count, elapsed):
        if total_examples:
            # examples-based progress %
            logger.info(
                "EPOCH %i - PROGRESS: at %.2f%% examples, %.0f words/s, in_qsize %i, out_qsize %i",
                cur_epoch + 1, 
                100.0 * example_count / total_examples, 
                trained_word_count / elapsed,
                -1 if job_queue is None else utils.qsize(job_queue), 
                utils.qsize(progress_queue)
            )
        else:
            # words-based progress %
            logger.info(
                "EPOCH %i - PROGRESS: at %.2f%% words, %.0f words/s, in_qsize %i, out_qsize %i",
                cur_epoch + 1, 
                100.0 * raw_word_count / total_words, 
                trained_word_count / elapsed,
                -1 if job_queue is None else utils.qsize(job_queue), 
                utils.qsize(progress_queue)
            )

    def _log_epoch_end(self, cur_epoch, example_count, total_examples, raw_word_count, total_words,
        trained_word_count, elapsed, is_corpus_file_mode):
        logger.info(
            "EPOCH - %i : training on %i raw words (%i effective words) took %.1fs, %.0f effective words/s",
            cur_epoch + 1, raw_word_count, trained_word_count, elapsed, trained_word_count / elapsed
        )

        # don't warn if training in file-based mode, because it's expected behavior
        if is_corpus_file_mode:
            return

        # check that the input corpus hasn't changed during iteration
        if total_examples and total_examples != example_count:
            logger.warning(
                "EPOCH - %i : supplied example count (%i) did not equal expected count (%i)", cur_epoch + 1,
                example_count, total_examples
            )
        if total_words and total_words != raw_word_count:
            logger.warning(
                "EPOCH - %i : supplied raw word count (%i) did not equal expected count (%i)", cur_epoch + 1,
                raw_word_count, total_words
            )

    def _log_train_end(self, raw_word_count, trained_word_count, total_elapsed, job_tally):

        logger.info(
            "training on a %i raw words (%i effective words) took %.1fs, %.0f effective words/s",
            raw_word_count, trained_word_count, total_elapsed, trained_word_count / total_elapsed
        )
        if job_tally < 10 * self.workers:
            logger.warning(
                "under 10 jobs per worker: consider setting a smaller `batch_words' for smoother alpha decay"
            )

    ################################################################################################################################################### load and save
    def save(self, *args, **kwargs):
        kwargs['ignore'] = kwargs.get('ignore', ['vectors_norm', 'cum_table'])
        super(FieldEmbedding, self).save(*args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs):
        """Load a previously saved object (using :meth:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel.save`) from file.
        Also initializes extra instance attributes in case the loaded model does not include them.
        `*args` or `**kwargs` **MUST** include the fname argument (path to saved file).
        See :meth:`~gensim.utils.SaveLoad.load`.
        Parameters
        ----------
        *args : object
            Positional arguments passed to :meth:`~gensim.utils.SaveLoad.load`.
        **kwargs : object
            Key word arguments passed to :meth:`~gensim.utils.SaveLoad.load`.
        See Also
        --------
        :meth:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel.save`
            Method for save a model.
        Returns
        -------
        :class:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel`
            Model loaded from disk.
        Raises
        ------
        IOError
            When methods are called on instance (should be called from class).
        """
        model = super(FieldEmbedding, cls).load(*args, **kwargs)
        if not hasattr(model, 'ns_exponent'):
            model.ns_exponent = 0.75
        if not hasattr(model.vocabulary, 'ns_exponent'):
            model.vocabulary.ns_exponent = 0.75
        if model.negative and hasattr(model.wv, 'index2word'):
            model.vocabulary.make_cum_table(model.wv)  # rebuild cum_table from vocabulary
        if not hasattr(model, 'corpus_count'):
            model.corpus_count = None
        if not hasattr(model, 'corpus_total_words'):
            model.corpus_total_words = None
        if not hasattr(model.trainables, 'vectors_lockf') and hasattr(model.wv, 'vectors'):
            model.trainables.vectors_lockf = ones(len(model.wv.vectors), dtype=REAL)
        if not hasattr(model, 'random'):
            model.random = random.RandomState(model.trainables.seed)
        if not hasattr(model, 'train_count'):
            model.train_count = 0
            model.total_train_time = 0
        return model

    ################################################################################################################################################### others

    def _clear_post_train(self):
        self.wv.vectors_norm = None

    def clear_sims(self):
        self.wv.vectors_norm = None

    def reset_from(self, other_model):
        self.wv.vocab = other_model.wv.vocab
        self.wv.index2word = other_model.wv.index2word
        self.vocabulary.cum_table = other_model.vocabulary.cum_table
        self.corpus_count = other_model.corpus_count
        self.trainables.reset_weights(self.negative, self.wv)

    def get_latest_training_loss(self):
        return self.running_training_loss

    def delete_temporary_training_data(self, replace_word_vectors_with_normalized=False):
        if replace_word_vectors_with_normalized:
            self.init_sims(replace=True)
        self._minimize_model()

    @staticmethod
    def log_accuracy(section):
        return Word2VecKeyedVectors.log_accuracy(section)

    def __str__(self):
        return "%s(vocab=%s, size=%s, alpha=%s)" % (
            self.__class__.__name__, len(self.wv.index2word), self.wv.vector_size, self.alpha
        )


class FieldEmbedVocab(utils.SaveLoad):
    """Vocabulary used by :class:`~gensim.models.word2vec.Word2Vec`."""
    def __init__(self, sample=1e-3, sorted_vocab=True, null_word=0, max_final_vocab=None, ns_exponent=0.75):
        self.sample = sample
        self.sorted_vocab = sorted_vocab
        self.null_word = null_word
        self.cum_table = None  # for negative sampling
        self.raw_vocab = None
        self.max_final_vocab = max_final_vocab
        self.ns_exponent = ns_exponent
        self.LookUp  = None
        self.EndIdx  = None
        self.Leng_Inv= None
        self.Leng_max= None
        self.LTU     = None
        self.DTU     = None

    def scan_and_prepare_vocab(self, model, nlptext, negative,  update=False, sample=None, **kwargs):
        
        corpus_count       = nlptext.SENT['length']
        corpus_total_words = nlptext.TOKEN['length']
        LTU, DTU = nlptext.TokenUnique
        
        print('o-->', 'Get Vocab Frequency from NLPText')
        DTU_freq = nlptext.DTU_freq
        
        sample = sample or self.sample
        min_count = list(DTU_freq.values())[-1]

        specialTokens = nlptext.specialTokens

        self.effective_min_count = min_count # TODO: make it neater

        if not update:
            print('o-->', "Prepare WV's index2word, vocab...")
            s = datetime.now(); print('\tStart: ', s)
            
            logger.info("Loading a fresh vocabulary")
            # make stored settings match these applied settings
            self.min_count = min_count
            self.sample = sample
            model.wv.index2word = LTU # LTU
            model.wv.vocab = {}       # DTU
            for word, v in iteritems(DTU_freq):
                if word in specialTokens:
                    model.wv.vocab[word] = Vocab(count=0, index=DTU[word]) # the unk should not have the freq 
                else:
                    model.wv.vocab[word] = Vocab(count=v, index=DTU[word])

            ############################ log info ###########################
            drop_unique = 0 # TODO: not always the real
            drop_total = sum([DTU_freq[k] for k in specialTokens])
            retain_total = corpus_total_words - drop_total

            retain_words = LTU

            original_unique_total = len(retain_words) + drop_unique
            retain_unique_pct = len(retain_words) * 100 / max(original_unique_total, 1) # pct means percent
            logger.info(
                "effective_min_count=%d retains %i unique words (%i%% of original %i, drops %i)",
                self.effective_min_count, len(retain_words), retain_unique_pct, original_unique_total, drop_unique
            )
            original_total = retain_total + drop_total
            retain_pct = retain_total * 100 / max(original_total, 1)
            logger.info(
                "effective_min_count=%d leaves %i word corpus (%i%% of original %i, drops %i)",
                self.effective_min_count, retain_total, retain_pct, original_total, drop_total)
            ############################ log info ###########################

            e = datetime.now(); print('\tEnd  : ', e);print('\tTotal Time:', e - s )

        print('o-->', "Compute Token's sampel_int...")
        s = datetime.now(); print('\tStart: ', s)
        if not sample:
            # no words downsampled
            threshold_count = retain_total
        elif sample < 1.0:
            # traditional meaning: set parameter as proportion of total
            threshold_count = sample * retain_total
        else:
            # new shorthand: sample >= 1 means downsample all words with higher count than sample
            threshold_count = int(sample * (3 + sqrt(5)) / 2)


        downsample_total, downsample_unique = 0, 0
        for w in retain_words:
            v = DTU_freq[w]
            if v == 0:
                v = 1
           
            word_probability = (sqrt(v / threshold_count) + 1) * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            # if not dry_run:
            model.wv.vocab[w].sample_int = int(round(word_probability * 2**32))

        model.wv_neg.index2word = model.wv.index2word
        model.wv_neg.vocab      = model.wv.vocab
        
        e = datetime.now(); print('\tEnd  : ', e);print('\tTotal Time:', e - s )


        logger.info("sample=%g downsamples %i most-common words", sample, downsample_unique)
        logger.info(
            "downsampling leaves estimated %i word corpus (%.1f%% of prior %i)",
            downsample_total, downsample_total * 100.0 / max(retain_total, 1), retain_total
        )

        # return from each step: words-affected, resulting-corpus-size, extra memory estimates
        report_values = {
            'drop_unique': drop_unique, 'retain_total': retain_total, 'downsample_unique': downsample_unique,
            'downsample_total': int(downsample_total), 'num_retained_words': len(retain_words)
        }

        print('o-->', 'Compute Cum Table')
        s = datetime.now(); print('\tStart: ', s)
        self.make_cum_table(model.wv)
        e = datetime.now(); print('\tEnd  : ', e);print('\tTotal Time:', e - s )
        return corpus_total_words, corpus_count,  report_values

    def make_cum_table(self, wv, domain=2**31 - 1):
        vocab_size = len(wv.index2word)
        # here words' counts are not necessary for being ordered/
        self.cum_table = zeros(vocab_size, dtype=uint32)
        # compute sum of all power (Z in paper)
        train_words_pow = 0.0
        for word_index in range(vocab_size):
            train_words_pow += wv.vocab[wv.index2word[word_index]].count**self.ns_exponent
        cumulative = 0.0
        for word_index in range(vocab_size):
            cumulative      += wv.vocab[wv.index2word[word_index]].count**self.ns_exponent
            self.cum_table[word_index] = round(cumulative / train_words_pow * domain)
        if len(self.cum_table) > 0:
            assert self.cum_table[-1] == domain

class FieldEmbedTrainables(utils.SaveLoad):
    """Represents the inner shallow neural network used to train :class:`~gensim.models.word2vec.Word2Vec`."""
    def __init__(self, vector_size=100, seed=1, hashfxn=hash):
        self.hashfxn = hashfxn
        self.layer1_size = vector_size
        self.seed = seed

    def seeded_vector(self, seed_string, vector_size):
        """Get a random vector (but deterministic by seed_string)."""
        # Note: built-in hash() may vary by Python version or even (in Py3.x) per launch
        once = random.RandomState(self.hashfxn(seed_string) & 0xffffffff)
        return (once.rand(vector_size) - 0.5) / vector_size

    def prepare_weights(self, model, negative, wv, update=False, vocabulary=None, neg_init = 0):
        print('o-->', 'Prepare Trainable Parameters')
        s = datetime.now(); print('\tStart: ', s)
        self.reset_weights(model, negative, neg_init = neg_init)
        e = datetime.now(); print('\tEnd  : ', e);print('\tTotal Time:', e - s )

    ###############################  INIT WEIGHTS HERE ########################## 
    def reset_weights(self, model, negative, neg_init = 0):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting layer weights")

        # syn0
        for field, field_idx in model.field_idx.items():
            use, wv = model.field_head[field_idx]
            if use:
                wv.vectors = empty((len(wv.vocab), wv.vector_size), dtype=REAL)
                for i in range(len(wv.vocab)): 
                    wv.vectors[i] = self.seeded_vector(wv.index2word[i] + str(self.seed), self.layer1_size)

            for data in model.field_sub[field_idx]:
                wv = data[0]
                wv.vectors = empty((len(wv.vocab), wv.vector_size), dtype=REAL)
                for i in range(len(wv.vocab)): 
                    wv.vectors[i] = self.seeded_vector(wv.index2word[i] + str(self.seed), self.layer1_size)

        # syn1neg      
        if negative:
            if type(neg_init) == str:
                print('init neg with random:', neg_init)
                model.wv_neg.vectors = empty((len(model.wv.vocab), self.layer1_size), dtype=REAL)
                for i in range(len(wv.vocab)): 
                    # construct deterministic seed from word AND seed argument
                    model.wv_neg.vectors[i] = self.seeded_vector(model.wv_neg.index2word[i] + neg_int + str(self.seed), 
                                                                 self.layer1_size)
            else:    
                # we tend to init neg with zero vectors
                print('init neg with 0')
                model.wv_neg.vectors = zeros((len(model.wv.vocab), self.layer1_size), dtype=REAL)
            
            self.syn1neg = model.wv_neg.vectors

        model.wv.vocab_values = list(model.wv.vocab.values())
        model.wv_neg.vocab = model.wv.vocab 
        model.wv_neg.index2word = model.wv.index2word
        # this vectors_lockf is left for trainables
        self.vectors_lockf = ones(len(model.wv_neg.vocab), dtype=REAL)  # zeros suppress learning
        for i, wv in model.weights.items():
            print(i, wv.vectors.shape)
        print('use_head:', model.use_head, 'use_sub:', model.use_sub)
