from __future__ import division 
import re
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


from nlptext.utils.pyramid import read_file_chunk_string
from .utils import get_chunk_info


from . import utils, matutils  
from .utils import deprecated
from .utils import keep_vocab_item, call_on_class_only
from .keyedvectors import Vocab, Word2VecKeyedVectors
# from .fieldembed_core import train_batch_fieldembed_0X1_neat
from .fieldembed_core import train_batch_fieldembed_negsamp

logger = logging.getLogger(__name__)

MAX_WORDS_IN_BATCH = 20000

FIELD_INFO = {
    'sub':  ['char', 'subcomp', 'stroke', 'pinyin'],
    'head': ['token'],
    'hyper':['pos', 'ner']
}

class FieldEmbedding(utils.SaveLoad):

    def __init__(self, nlptext = None, Field_Settings = {},  train = True, mode = 'sg_nlptext', sg=0, use_merger = 0, 
        neg_init = 0, standard_grad = 1, size=100, alpha=0.025, min_alpha=0.0001, window=5, sample=1e-3, seed=1, workers=3, 
        negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=hash, iter=5,
        batch_words=MAX_WORDS_IN_BATCH, compute_loss=False, callbacks=()):
        '''

            nlptext: nlptext.base
            Field_Settings: field configurations.
            train: train embeddings or not.
            mode: 'what mode to use', TODO: remove it.
            sg: use skip-gram or cbow
            use_merger: use the merger projection. # TODO: adding the three modes 
            neg_init: method to initialize the syn1neg word embedding matrix.
            standard_grad: deal with cbow, cbow seems need a larger learning rate.

            ++Hyper Parameters++
            size:
            alpha:
            min_alpha:
            window:
            sample:
            seed:
            workers:

            compute_loss: we need to see the compute_loss

        '''
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
       
        # both vocabulary and trainables are methods to update wv and wv_neg
        # we can treat them as toolkits instead of data structures.
        # vocabulary will update wv and wv_neg 
        # the relationship between vocabulary and (wv, wv_neg) is very weird.
        self.vocabulary = FieldEmbedVocab(sample=sample, ns_exponent = ns_exponent)
        # trainables will create matrices as fields embeddings
        self.trainables = FieldEmbedTrainables(seed=seed, vector_size=size, hashfxn=hashfxn)

        self.weights = {}
        self.field_head = []
        self.field_sub  = []
        self.field_hyper= []
        self.use_head = 0
        self.use_sub  = 0
        self.use_hyper= 0

        vector_size = size
        
        if vector_size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        
        self.window = int(window)
        self.random = random.RandomState(seed) # random = seed
        
        self.sg = int(sg)
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
        print(self.workers)
        self.epochs = iter
        self.train_count = 0
        self.total_train_time = 0
        self.batch_words = batch_words
        self.model_trimmed_post_training = False
        self.callbacks = callbacks

        self.build_vocab(nlptext = nlptext)
        print("model's window size is:", window)
        print('finish build vocab')

        if train:
            print('\n\n======== Training Start ....'); s = datetime.now()
            self.train(nlptext = nlptext, total_examples=self.corpus_count,
                total_words=self.corpus_total_words, epochs=self.epochs, 
                start_alpha=self.alpha, end_alpha=self.min_alpha, compute_loss=compute_loss)
            print('======== Training End ......'); e = datetime.now()
            print('======== Total Time: ', e - s)

    ################################################################################################################################################### build_vocab
    def build_vocab(self, nlptext = None, **kwargs):
        print('-------> Prepare Field Info....')

        # produce self.use_head, self.use_sub, self.use_hyper
        # and self.weights, and self.field_sub information and so on.
        self.prepare_field_info(nlptext)
        print('======== Build_vocab based on NLPText....'); s = datetime.now()
        print('-------> Prepare Vocab....')

        total_words, corpus_count, report_values = self.vocabulary.scan_and_prepare_vocab(self, nlptext, 
                                                                    self.negative, update = False, **kwargs) 
        self.corpus_count = corpus_count
        self.corpus_total_words = total_words
        # self.create_field_embedding(size = self.size, )
        report_values['memory'] = self.estimate_memory(vocab_size=report_values['num_retained_words'])

        print('-------> Prepare Trainable Weight....')

        self.trainables.init_weights(model = self, neg_init = self.neg_init)

        print('======== The Voc and Parameters are Ready!'); e = datetime.now()
        print('======== Total Time: ', e - s)

    def prepare_field_info(self, nlptext):

        if len(self.Field_Settings) == 0:
            # case 1: no Field_Settings are provided, and only use token itself
            self.field_head.append([self.wv])
            self.weights['token'] = self.wv 

        else:
            for channel, f_setting in self.Field_Settings.items():
                if channel == 'token':
                    self.field_head.append([self.wv])
                    self.weights['token'] = self.wv 

                elif channel in FIELD_INFO['sub']:
                    # when the channel is a sub-field
                    data = self.get_subfield_info(nlptext, channel, **f_setting) 
                    (GU, TU, LKP), (LookUp, EndIdx, Leng_Inv, Leng_max) = data
                    # GU, LookUp, EndIdx, Leng_Inv, Leng_max, TU, LKP = data
                    LGU, DGU = GU
                    wv = self.create_field_embedding(self.vector_size, channel, LGU, DGU)
                    wv.TU = TU
                    wv.LKP = LKP
                    self.field_sub.append([wv, LookUp, EndIdx, Leng_Inv])
                    self.weights[channel] = wv

                elif channel in FIELD_INFO['hyper']:
                    # LGU, DGU  = nlptext.getGrainVocab(channel, tagScheme = 'BIOES') 
                    LGU, DGU  = nlptext.getGrainVocab(channel, **f_setting) 
                    wv = self.create_field_embedding(self.vector_size, channel, LGU, DGU)

                    self.field_hyper.append([wv])
                    self.weights[channel] = wv
                    print(self.weights[channel].LGU)
                    print(len(self.weights[channel].LGU))
            
        self.use_head = len(self.field_head)
        self.use_sub  = len(self.field_sub)
        self.use_hyper= len(self.field_hyper)
        self.proj_num = len(self.weights)
        print('use_head:', self.use_head, 'use_sub:', self.use_sub, 'use_hyper:', self.use_hyper)

    def get_subfield_info(self, nlptext, field = 'char', Min_Ngram = 1,  Max_Ngram = 1, end_grain = False, min_grain_freq = 1):
        GU = nlptext.getGrainVocab(field, Min_Ngram = Min_Ngram, Max_Ngram = Max_Ngram, end_grain = end_grain, min_grain_freq = min_grain_freq)
        LKP,TU = nlptext.getLookUp(field, Min_Ngram = Min_Ngram, Max_Ngram = Max_Ngram, end_grain = end_grain, min_grain_freq = min_grain_freq)
        Leng = np.array([len(i) for i in LKP], dtype = np.uint32)
        Leng_max = np.max(Leng)
        # LESSION: ignoring the np.uint32 wastes me a lot of time
        EndIdx = np.cumsum(Leng, dtype = np.uint32) 
        LookUp = np.array(list(itertools.chain.from_iterable(LKP)), dtype = np.uint32)
        Leng_Inv = 1 / Leng
        Leng_Inv = Leng_Inv.astype(REAL)
        return (GU, TU, LKP), (LookUp, EndIdx, Leng_Inv, Leng_max)

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
            # will update token wv in other place
            # self.wv.index2word = LGU 
            # self.wv.GU = (LGU, DGU)
            return self.wv
        else:
            gw = Word2VecKeyedVectors(size)
            gw.index2word = LGU 
            gw.GU = (LGU, DGU)
            for gr in DGU:
                gw.vocab[gr] = Vocab(index=DGU[gr])
            self.__setattr__('wv_' + channel, gw)
            return self.__getattribute__('wv_' + channel)

    def estimate_memory(self, vocab_size=None, report=None):
        vocab_size = vocab_size or len(self.wv.vocab)
        report = report or {}
        report['vocab'] = vocab_size * 500 
        report['vectors'] = vocab_size * self.vector_size * dtype(REAL).itemsize
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
                ## core things
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
            "using sg=%s sample=%s negative=%s window=%s",
            self.workers, len(self.wv.vocab), self.trainables.layer1_size, self.sg,
            self.vocabulary.sample, self.negative, self.window
        )

    ####################################################################################### 
    def _train_epoch_nlptext(self, nlptext, cur_epoch = 0, total_examples=None, total_words=None,queue_factor=2, report_delay=1.0):
        ########### preprocess
        # sentences_endidx = nlptext.SENT['EndIDXTokens']
        # tokens_vocidx    = nlptext.TOKEN['ORIGTokenIndex']
        # print(len(tokens_vocidx))
        total_examples  =  nlptext.SENT['length']
        total_words     =  nlptext.TOKEN['length']
        # hyper_vocidx = []      


        print('Start getting batch infos')
        
        chunkidx_2_endbyteidxs, chunkidx_2_cumlengoftexts = get_chunk_info(nlptext, 'sentence', BATCH_MAX_NUM = self.batch_words)
        job_no = len(chunkidx_2_cumlengoftexts)

        # print('The time of finding batch_end_st_idx_list:', e - s)
        print('Total job number is:', job_no)
        
        
        # sentences_endidx, tokens_vocidx, batch_end_st_idx_list, job_no, 
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        # in the future, make the selection here. or make selection here
        # make more worker_loop_nlptext1, 2, 3, 4, 5
        print('The workers number is:', self.workers)
        workers = [
            threading.Thread(
                target=self._worker_loop_nlptext,
                args=(job_queue, progress_queue, self.proj_num))
            for _ in range(self.workers)
        ]
        logger.info('\n the total_examples is:' + str(total_examples) + '   , the total words is:' + str(total_words) + '\n')
        workers.append(threading.Thread(
            target=self._job_producer_nlptext,
            args=(nlptext, chunkidx_2_endbyteidxs, chunkidx_2_cumlengoftexts, job_no, job_queue, total_examples,  total_words,), # data_iterable is sentences
            kwargs={'cur_epoch': cur_epoch,}))

        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        trained_word_count, raw_word_count, job_tally = self._log_epoch_progress(
            progress_queue, job_queue, cur_epoch=cur_epoch, total_examples=total_examples, total_words=total_words,
            report_delay=report_delay, is_corpus_file_mode=False)

        return trained_word_count, raw_word_count, job_tally

    def _job_producer_nlptext(self, nlptext, chunkidx_2_endbyteidxs, chunkidx_2_cumlengoftexts, job_no, job_queue, 
        total_examples,  total_words, cur_epoch=0):
        #---------------------------------------------------# 
        

        job_batch, batch_size = [], 0
        pushed_words, pushed_examples = 0, 0 # examples refers to sentences
        next_job_params = self._get_job_params(cur_epoch) # current learning rate: cur_alpha
    
        for chunk_idx in range(job_no):

            # get chunk_token_str
            channel = 'token'
            start_position = 0 if chunk_idx == 0 else chunkidx_2_endbyteidxs[channel][chunk_idx - 1] 
            end_postion = chunkidx_2_endbyteidxs[channel][chunk_idx] 
            path = nlptext.Channel_Hyper_Path[channel]
            chunk_token_str = re.split(' |\n', read_file_chunk_string(path, start_position, end_postion))
            token_num = len(chunk_token_str)

            # get chunk_hyper_idx
            # TODO: this needs update to keep the hyper channel order.
            chunk_hyper_idxs = []
            for channel in chunkidx_2_endbyteidxs:
                if channel != 'token' and channel in self.Field_Settings:
                    start_position = 0 if chunk_idx == 0 else chunkidx_2_endbyteidxs[channel][chunk_idx - 1] 
                    end_postion = chunkidx_2_endbyteidxs[channel][chunk_idx] 
                    path = nlptext.Channel_Hyper_Path[channel]
                    strings = read_file_chunk_string(path, start_position, end_postion)
                    grain_idx = re.split(' |\n', strings)
                    assert len(grain_idx) == token_num
                    f_settings = self.Field_Settings[channel]
                    bioes2tag = nlptext.getTrans(channel, f_settings['tagScheme'])
                    # shall we check its insanity?
                    bioes_idx = []
                    for vocidx in grain_idx:
                        if vocidx in bioes2tag:
                            tagidx = bioes2tag[vocidx]
                        else:
                            # print('Error for bioes', vocidx)
                            # print(strings)
                            tagidx = 0
                        bioes_idx.append(tagidx)
                    # bioes_idx =  [bioes2tag[vocidx] for vocidx in grain_idx]
                    chunk_hyper_idxs.append(bioes_idx)

            # sentence_idx= np.array([i-token_start for i in sentences_endidx[start: end]], dtype = np.uint32)
            sentence_idx = np.array(chunkidx_2_cumlengoftexts[chunk_idx], dtype = np.uint32)

            # this is important.
            # chunk_token_str, chunk_hyper_strs, sentence_idx, next_job_params
            job_queue.put((chunk_token_str, chunk_hyper_idxs, sentence_idx, next_job_params))

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
        # produce memory for projection vectors
        inits = self._get_thread_working_mem(proj_num = proj_num) 
        merger_mem = self._get_thread_working_mem_for_merger()
        grad_mem = self._get_thread_grad_mem(proj_num = proj_num)

        jobs_processed = 0

        idx = 0 # log

        while True:
            job = job_queue.get()
            if job is None:
                progress_queue.put(None)
                break  # no more jobs => quit this worker


            chunk_token_str, chunk_hyper_idxs, sentence_idx, alpha = job 

            # # log
            # idx = idx + 1 
            # if idx % 100 == 0:
            #     print(idx)
            #     # print(chunk_token_str)
            #     # print(chunk_hyper_idxs)
            #     print(alpha)

            #     for sent_idx in range(len(sentence_idx)):
            #         # step1: get every sentence's idx_start and idx_end
            #         if sent_idx == 0:
            #             idx_start = 0
            #         else:
            #             idx_start = sentence_idx[sent_idx-1]
            #         idx_end = sentence_idx[sent_idx]
            #         # print('orig-sent loc start and end idx')
            #         # print(idx_start, idx_end)
            #         print(' '.join(chunk_token_str[idx_start: idx_end]))
            #         print('\n')
            #     print('----')
            for callback in self.callbacks:
                callback.on_batch_begin(self)

            tally, raw_tally, loss = self._do_train_job_nlptext(chunk_token_str, chunk_hyper_idxs, sentence_idx, alpha, inits, merger_mem, grad_mem)
            for callback in self.callbacks:
                callback.on_batch_end(self)

            progress_queue.put((len(sentence_idx), tally, raw_tally, loss))  # report back progress
            jobs_processed += 1
        logger.debug("o----> Worker exiting, processed %i jobs", jobs_processed)

    def _get_thread_working_mem(self, proj_num = 1):
        return [matutils.zeros_aligned(self.vector_size * proj_num, dtype=REAL) for i in range(2)]

    def _get_thread_working_mem_for_merger(self):
        work_m = matutils.zeros_aligned(self.vector_size, dtype=REAL) 
        neu_m  = matutils.zeros_aligned(self.vector_size, dtype=REAL) 
        return work_m, neu_m

    def _get_thread_grad_mem(self, proj_num = 1):
        return matutils.zeros_aligned(self.proj_num, dtype=REAL) 

    def _do_train_job_nlptext(self, indexes, hyper_indexes, sentence_idx, alpha, inits, merger_mem, grad_mem):
        tally = 0
        work, neu1 = inits
        work_m, neu_m = merger_mem
        grad_mem = grad_mem
        tally_increase, loss = train_batch_fieldembed_negsamp(self, indexes, hyper_indexes, sentence_idx, 
                            alpha, work, neu1, work_m, neu_m, grad_mem, self.compute_loss)
        tally = tally_increase + tally
        return tally, sentence_idx[-1], loss

        # chunk_token_str = indexes
        # chunk_hyper_idxs = hyper_indexes

        # vlookup = self.wv.vocab
        # for sent_idx in range(len(sentence_idx)):
        #     try:
        #         if sent_idx == 0:
        #             idx_start = 0
        #         else:
        #             idx_start = sentence_idx[sent_idx-1]
        #         idx_end = sentence_idx[sent_idx]

        #         for loc_idx in range(idx_start, idx_end):
        #             token = chunk_token_str[loc_idx]
        #             word = vlookup[token] if token in vlookup else None
        #             if word is None:
        #                 continue
        #             word_vocidx = word.index 
        #     except:
        #         if sent_idx == 0:
        #             idx_start = 0
        #         else:
        #             idx_start = sentence_idx[sent_idx-1]
        #         idx_end = sentence_idx[sent_idx]
        #         total_leng = len(chunk_token_str)
        #         idx_end = total_leng if idx_end > total_leng else idx_end
        #         print(idx_start, idx_end)
        #         print(' '.join(chunk_token_str[idx_start: idx_end]))


        # return 0, sentence_idx[-1],  0

    ################################################################################################################################################### log
    def _log_epoch_progress(self, progress_queue=None, job_queue=None, cur_epoch=0, total_examples=None,total_words=None, report_delay=10.0, is_corpus_file_mode=None):

        example_count, trained_word_count, raw_word_count = 0, 0, 0
        start, next_report = default_timer() - 0.00001, 5.0
        job_tally = 0
        unfinished_worker_count = self.workers
        last_loss = 0
        while unfinished_worker_count > 0:
            report = progress_queue.get()  # blocks if workers too slow
            if report is None:  # a thread reporting that it finished
                unfinished_worker_count -= 1
                logger.info("Worker thread finished; awaiting finish of %i more threads", unfinished_worker_count)
                continue
            examples, trained_words, raw_words, loss = report
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
                    raw_word_count, total_words, trained_word_count, elapsed, loss - last_loss)
                next_report = elapsed + report_delay
            last_loss = loss
        # all done; report the final stats
        elapsed = default_timer() - start
        self._log_epoch_end(
            cur_epoch, example_count, total_examples, raw_word_count, total_words,
            trained_word_count, elapsed, is_corpus_file_mode)
        self.total_train_time += elapsed
        return trained_word_count, raw_word_count, job_tally

    def _log_progress(self, job_queue, progress_queue, cur_epoch, example_count, total_examples,
        raw_word_count, total_words, trained_word_count, elapsed, loss):
        if total_examples:
            # examples-based progress %
            logger.info(
                "EPOCH %i - PROGRESS: at %.2f%% examples, %.0f words/s, in_qsize %i, out_qsize %i, loss %.3f",
                cur_epoch + 1, 
                100.0 * example_count / total_examples, 
                trained_word_count / elapsed,
                -1 if job_queue is None else utils.qsize(job_queue), 
                utils.qsize(progress_queue),
                loss, 
            )
        else:
            # words-based progress %
            logger.info(
                "EPOCH %i - PROGRESS: at %.2f%% words, %.0f words/s, in_qsize %i, out_qsize %i, loss %.3f",
                cur_epoch + 1, 
                100.0 * raw_word_count / total_words, 
                trained_word_count / elapsed,
                -1 if job_queue is None else utils.qsize(job_queue), 
                utils.qsize(progress_queue),
                loss, 
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

    ###################################################################################################################################################
    def evals(self):
        # TODO: enrich this evaluation
        for channel, wv in self.weights.items():
            if channel in FIELD_INFO['hyper']:
                continue
            print(channel)
            print(wv.derivative_wv.lexical_evals())
        wv = self.wv_neg
        print('syn1neg')
        print(wv.lexical_evals())
        return

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
        self.LTU     = None
        self.DTU     = None

    def scan_and_prepare_vocab(self, model, nlptext, negative, update=False, sample=None, **kwargs):
        '''
            model: field embed model
            nlptext: nlptext base

        '''
        # this is sentence based. all right.
        corpus_count       = nlptext.SENT['length']
        corpus_total_words = nlptext.TOKEN['length']

        # up to know, the min_token_freq is considered already.
        LTU, DTU = nlptext.TokenVocab
        
        print('o-->', 'Get Token Vocab Frequency from NLPText')
        # ISSUE: contains the lower freq tokens or not.
        idx2freq = nlptext.idx2freq
        
        sample = sample or self.sample
        min_count = nlptext.min_token_freq

        # don't need to consider special tokens at this time
        # specialTokens = nlptext.specialTokens

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
            model.wv.GU = (LTU, DTU)
            for vocidx, freq in enumerate(idx2freq):
                # actually, wv.vocab is a combination of token2freq and token2idx
                model.wv.vocab[LTU[vocidx]] = Vocab(count=freq, index=vocidx)

            # original_token_num = nlptext.original_token_num
            original_unique_total = nlptext.original_vocab_token_num
            drop_unique = original_unique_total - len(LTU)
    
            retain_total = np.sum(idx2freq)
            drop_total   = corpus_total_words - retain_total

            retain_words = LTU

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
            # we use this one.
            # traditional meaning: set parameter as proportion of total
            threshold_count = sample * retain_total
        else:
            # new shorthand: sample >= 1 means downsample all words with higher count than sample
            threshold_count = int(sample * (3 + sqrt(5)) / 2)

        downsample_total, downsample_unique = 0, 0
        for w in retain_words:
            # v is freq
            v = model.wv.vocab[w].count
            word_probability = (sqrt(v / threshold_count) + 1) * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v

            # this is the same as create token2sample_int
            # or we can get idx2sample_int.
            model.wv.vocab[w].sample_int = int(round(word_probability * 2**32))

        model.wv_neg.index2word = model.wv.index2word
        model.wv_neg.vocab      = model.wv.vocab
        model.wv_neg.GU = (LTU, DTU)
        
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

        # update wv and wv_neg information.
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

    def init_weights(self, model, neg_init = 0):

        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting (initializing) layer weights")

        # syn0, many fields here.
        for field, wv in model.weights.items():
            wv.vectors = empty((len(wv.vocab), wv.vector_size), dtype=REAL)
            for i in range(len(wv.vocab)): 
                wv.vectors[i] = self.seeded_vector(wv.index2word[i] + str(self.seed), self.layer1_size)

        # syn1neg      
        # if negative:
        if type(neg_init) == str:
            print('Init syn1neg with random:', neg_init)
            model.wv_neg.vectors = empty((len(model.wv.vocab), self.layer1_size), dtype=REAL)
            for i in range(len(wv.vocab)): 
                # construct deterministic seed from word AND seed argument
                model.wv_neg.vectors[i] = self.seeded_vector(model.wv_neg.index2word[i] + neg_int + str(self.seed), self.layer1_size)
        else:    
            print('Init syn1neg with zeros')
            model.wv_neg.vectors = zeros((len(model.wv.vocab), self.layer1_size), dtype=REAL)
        
        # this is only for the convenient. trainable.syn1neg and model.wv_neg.vectors are the same thing.
        self.syn1neg = model.wv_neg.vectors

        # what is this?
        model.wv.vocab_values = list(model.wv.vocab.values())
        model.wv_neg.vocab = model.wv.vocab 
        model.wv_neg.index2word = model.wv.index2word
        # this vectors_lockf is left for trainables
        self.vectors_lockf = ones(len(model.wv_neg.vocab), dtype=REAL)  # zeros suppress learning
        for i, wv in model.weights.items():
            print(i, wv.vectors.shape)
        # print('use_head:', model.use_head, 'use_sub:', model.use_sub)
