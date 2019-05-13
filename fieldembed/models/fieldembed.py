from __future__ import division  # py3 "true division"

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

from .. import utils, matutils  # utility fnc for pickling, common scipy operations etc
from ..utils import deprecated
from ..utils import keep_vocab_item, call_on_class_only
from .keyedvectors import Vocab, Word2VecKeyedVectors
from .base_any2vec import BaseWordEmbeddingsModel


logger = logging.getLogger(__name__)

from .word2vec_inner import train_batch_sg_nlptext, train_batch_cbow_nlptext
from .word2vec_inner import train_batch_fieldembed_token
from .word2vec_inner import train_batch_fieldembed_0X1
from .word2vec_inner import FAST_VERSION, MAX_WORDS_IN_BATCH

from .fieldembed_inner import train_batch_fieldembed_M0X1

Field_Info = {
    #field: [head-subfield, subfield, subfield]
    'token':['token', 'char', 'basic', 'medical', 'subcomp', 'stroke'],
    'pos'  :['pos'],
    'ner'  :['ner'],
}

Field_Idx = {
    #field: [head-subfield, subfield, subfield]
    'token':0,
    'pos'  :1,
    'ner'  :2,
}

def get_field_info(nlptext, field = 'char', Max_Ngram = 1, end_grain = False):
    GU = nlptext.getGrainUnique(field, Max_Ngram=Max_Ngram, end_grain=end_grain)
    charLookUp, TU = nlptext.getLookUp(field, Max_Ngram=Max_Ngram, end_grain=end_grain)
    # LTU, DTU = TU
    # charLeng = np.array([len(i) for i in charLookUp], dtype = int)
    # charEndIdx = np.cumsum(charLeng, dtype = int) 
    charLeng = np.array([len(i) for i in charLookUp], dtype = np.uint32)
    charLeng_max = np.max(charLeng)
    charEndIdx = np.cumsum(charLeng, dtype = np.uint32) # LESSION: ignoring the np.uint32 wastes me a lot of time
    charLookUp = np.array(list(itertools.chain.from_iterable(charLookUp)), dtype = np.uint32)
    charLeng_inv = 1 / charLeng
    charLeng_inv = charLeng_inv.astype(REAL)
    return GU, charLookUp, charEndIdx, charLeng_inv, charLeng_max, TU

class FieldEmbedding(BaseWordEmbeddingsModel):

    def __init__(self, nlptext = None, Field_Settings = {}, mode = 'sg_nlptext',
                use_merger = 0, size=100, alpha=0.025, window=5, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
                 sg=0, negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=hash, iter=5,
                 neg_init = 0,
                 batch_words=MAX_WORDS_IN_BATCH, compute_loss=False, callbacks=()):

        self.mode = mode
        self.callbacks = callbacks
        self.load = call_on_class_only
        self.Field_Settings = Field_Settings
        self.use_merger = use_merger
        self.neg_init = neg_init

        # here only do initializations for wv, vocabulary, and trainables
        self.wv     = Word2VecKeyedVectors(size)
        self.wv_neg = Word2VecKeyedVectors(size)
        # self.wv_neg = Word2VecKeyedVectors(size)
        self.vocabulary = FieldEmbedVocab(sample=sample, ns_exponent=ns_exponent)
        self.trainables = FieldEmbedTrainables(seed=seed, vector_size=size, hashfxn=hashfxn)

        # self.trainables_dict = {} # will be added
        self.field_info= {}
        self.field_idx = {}
        self.field_sub = []
        self.field_head= []
        # self.meta = {}
        # self.build_vocab makes wv, vocabulary, and trainables rich. see: BaseWordEmbeddingsModel
        super(FieldEmbedding, self).__init__(
                nlptext = nlptext, workers=workers, vector_size=size, epochs=iter, 
                callbacks=callbacks, batch_words=batch_words, sg=sg, alpha=alpha, window=window,
                seed=seed, negative=negative, cbow_mean=cbow_mean, min_alpha=min_alpha, compute_loss=compute_loss,
                fast_version=FAST_VERSION)

    def _get_thread_working_mem(self, proj_num = 1):
        if proj_num == 1: 
            work = matutils.zeros_aligned(self.trainables.layer1_size, dtype=REAL)  # per-thread private work memory
            neu1 = matutils.zeros_aligned(self.trainables.layer1_size, dtype=REAL)
            return work, neu1
        elif proj_num == 2:
            work = matutils.zeros_aligned(self.trainables.layer1_size, dtype=REAL)  # per-thread private work memory
            neu1 = matutils.zeros_aligned(self.trainables.layer1_size, dtype=REAL)
            
            work2 = matutils.zeros_aligned(self.trainables.layer1_size, dtype=REAL)  # per-thread private work memory
            neu2 = matutils.zeros_aligned(self.trainables.layer1_size, dtype=REAL)
        
            return work, neu1, work2, neu2


    def _get_thread_working_mem_for_meager(self):
        # if self.use_merger:
        work_m = matutils.zeros_aligned(self.trainables.layer1_size, dtype=REAL) 
        neu_m  = matutils.zeros_aligned(self.trainables.layer1_size, dtype=REAL) 
        return work_m, neu_m

    def _do_train_job_nlptext(self, indexes, sentence_idx, alpha, inits, merger_private_mem = None):
        tally = 0
        # if len(self.Field_Settings) != 0:
        # if True:
        #     # version 4. On work
        #     work, neu1, work2, neu2 = inits
        #     work_m, neu_m = merger_private_mem
        #     tally += train_batch_fieldembed_M0X1(self, indexes, sentence_idx, alpha, 
        #                                          work, neu1, work2, neu2, work_m, neu_m, self.compute_loss)

        if self.mode == 'fieldembed_0X1':
            work, neu1, work2, neu2 = inits
            tally += train_batch_fieldembed_0X1(self, indexes, sentence_idx, alpha, work, neu1, work2, neu2, self.compute_loss)

        elif self.mode == 'fieldembed_token':
            work, neu1 = inits[0], inits[1]
            tally += train_batch_fieldembed_token(self, indexes, sentence_idx, alpha, work, neu1, self.compute_loss)

        elif self.mode == 'sg_nlptext' and self.sg == 1:
            work, neu1 = inits[0], inits[1]
            tally += train_batch_sg_nlptext(self, indexes, sentence_idx, alpha, work, self.compute_loss)

        elif self.mode == 'cbow_nlptext' and self.sg == 0:
            work, neu1 = inits[0], inits[1]
            tally += train_batch_cbow_nlptext(self, indexes, sentence_idx, alpha, work, neu1, self.compute_loss) 
        return tally, sentence_idx[-1] # sentence_idx[-1] is the length of all tokens form this job sentences.

    def _worker_loop_nlptext(self, job_queue, progress_queue, proj_num = 1):
        thread_private_mem = self._get_thread_working_mem(proj_num = proj_num) # TODO: this is space to store gradients, change this.
        # eturn work, neu1

        merger_private_mem = self._get_thread_working_mem_for_meager()
        jobs_processed = 0
        while True:
            job = job_queue.get()
            if job is None:
                progress_queue.put(None)
                break  # no more jobs => quit this worker
            indexes, sentence_idx, job_parameters = job 

            for callback in self.callbacks:
                callback.on_batch_begin(self)

            tally, raw_tally = self._do_train_job_nlptext(indexes, sentence_idx, job_parameters, thread_private_mem, 
                                                          merger_private_mem = merger_private_mem)

            for callback in self.callbacks:
                callback.on_batch_end(self)

            progress_queue.put((len(sentence_idx), tally, raw_tally))  # report back progress
            jobs_processed += 1
        logger.debug("o----> Worker exiting, processed %i jobs", jobs_processed)
    
    def _job_producer_nlptext(self, 
        sentences_endidx, total_examples, 
        tokens_vocidx, total_words, 
        batch_end_st_idx_list, job_no, job_queue,
        cur_epoch=0):
        
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

            indexes     = tokens_vocidx[token_start:token_end] # dtype = np.uint32
            # sentence_idx= np.array([i-token_start for i in sentences_endidx[start: end]], dtype = np.uint32)
            sentence_idx= [i-token_start for i in sentences_endidx[start: end]]
            # print('The start and end sent loc_id:', start, end)
            # print('The token start and end loc idx in each batch:', token_start, token_end)
            # print(sentence_idx[-1], len(indexes), '\n')
            # assaure that the input is correct
            # TODO
            # print_sentence()
            job_queue.put((indexes, sentence_idx, next_job_params))

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
    ################################################################### 

    def _train_epoch_nlptext(self, nlptext, cur_epoch=0, total_examples=None, total_words=None,queue_factor=2, report_delay=1.0):
        
        ########### preprocess
        sentences_endidx = nlptext.SENT['EndIDXTokens']
        tokens_vocidx    = nlptext.TOKEN['ORIGTokenIndex']
        total_examples  =  len(sentences_endidx)
        total_words =  len(tokens_vocidx)           

        ####################################### get batch_end_st_idx_list and job_no
        print('Start getting batch infos')
        s = datetime.now(); print(s)
        batch_end_st_idx_list, job_no = nlptext.Calculate_Infos(self.batch_words)
        e = datetime.now(); print(e)
        print('The time of finding batch_end_st_idx_list:', e - s)
        print('Total job number is:', job_no)
        ####################################### get batch_end_st_idx_list and job_no

        # sentences_endidx, tokens_vocidx, batch_end_st_idx_list, job_no, 
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [
            threading.Thread(
                target=self._worker_loop_nlptext,
                args=(job_queue, progress_queue, self.proj_num))
            for _ in range(self.workers)
        ]
        logger.info('\n the total_examples is:' + str(total_examples) + '   , the total words is:' + str(total_words) + '\n')
        workers.append(threading.Thread(
            target=self._job_producer_nlptext,
            args=(sentences_endidx, total_examples, tokens_vocidx, total_words, batch_end_st_idx_list, job_no, job_queue,), # data_iterable is sentences
            kwargs={'cur_epoch': cur_epoch,}))

        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        trained_word_count, raw_word_count, job_tally = self._log_epoch_progress(
            progress_queue, job_queue, cur_epoch=cur_epoch, total_examples=total_examples, total_words=total_words,
            report_delay=report_delay, is_corpus_file_mode=False)

        return trained_word_count, raw_word_count, job_tally


    def create_field_embedding(self, size, channel, LGU, DGU):
        # self.wv_neg = Word2VecKeyedVectors(size)
        if channel == 'token':
            # don't need create wv_token for token anymore, we have self.wv already.
            # self.trainables_dict['syn0' ] = self.wv
            return self.wv
        else:
            gw = Word2VecKeyedVectors(size)
            gw.index2word = LGU 
            for gr in DGU:
                gw.vocab[gr] = Vocab(index=DGU[gr])
            self.__setattr__('wv_' + channel, gw)
            # self.trainables_dict['syn0_' + channel] = self.__getattribute__('wv_' + channel)
            return self.__getattribute__('wv_' + channel)

    def prepare_field_info(self, nlptext):

        if len(self.Field_Settings) == 0:
            self.field_idx['token']  = len(self.field_idx)
            self.field_info['token'] = ['token']
            self.field_head.append([])
            self.field_head[self.field_idx['token']] = [1, self.wv] # CB-010 / SN-010
            self.field_sub.append([])
            self.field_sub[self.field_idx['token']]  = []
            self.use_head = 1
            self.use_sub  = 0
            self.proj_num = 2 # change this frequently
        else:
            for channel, f_setting in self.Field_Settings.items():
                # actually, Field_Settings is nlptext's Channel_Settings
                if channel in Field_Info:
                    field = channel
                    LGU, DGU  = nlptext.getGrainUnique(field)
                    wv = self.create_field_embedding(self.vector_size, channel, LGU, DGU)
                    if field in self.field_info:
                        self.field_info[channel].append(channel)
                    else:
                        self.field_info[channel] = [channel]

                    if field not in self.field_idx:
                        self.field_idx[field]  = len(self.field_idx)

                    if self.field_idx[field] == len(self.field_head):
                        self.field_head.append([1, wv])
                    elif self.field_idx[field] < len(self.field_head):
                        self.field_head[self.field_idx[field]] = [1, wv]
                    else:
                        prnt('Error in field ')

                    if self.field_idx[field] == len(self.field_sub):
                        self.field_sub.append([])
                        self.field_sub[self.field_idx['token']] = []

                else:
                    data = get_field_info(nlptext, channel, **f_setting) 
                    GU, LookUp, EndIdx, Leng_Inv, Leng_max, TU = data
                    LGU, DGU = GU
                    wv = self.create_field_embedding(self.vector_size, channel, LGU, DGU)
                    
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
                            # print('subfield:', field)
                            break
            
            self.use_head = sum([i[0] for i in self.field_head])
            self.use_sub  = sum([len(i) for i in self.field_sub])
            self.proj_num = 2
        
        pprint(self.field_idx)
        pprint(self.field_info)
        pprint(self.field_head)
        pprint(self.field_sub)
        print('use_head:', self.use_head, 'use_sub:', self.use_sub)

    def build_vocab(self, nlptext = None, **kwargs):
        # scan_vocab and prepare_vocab
        # build .wv.vocab + .wv.index2word + .wv.cum_table
        update = False

        print('!!!======== Build_vocab based on NLPText....'); s = datetime.now()

        print('-------> Prepare Vocab....')
        total_words, corpus_count,  report_values = self.vocabulary.scan_and_prepare_vocab_from_nlptext(self, nlptext, 
                                                                    self.negative, update = update, **kwargs) 

        self.corpus_count = corpus_count
        self.corpus_total_words = total_words

        print('-------> Prepare Field Info....')

        self.prepare_field_info(nlptext)
        
        # self.create_field_embedding(size = self.size, )
        report_values['memory'] = self.estimate_memory(vocab_size=report_values['num_retained_words'])

        print('-------> Prepare Trainable Weight....')
        self.trainables.prepare_weights_from_nlptext(self, self.negative, self.wv, update=update, vocabulary=self.vocabulary, neg_init = self.neg_init)

        print('======== The Voc and Parameters are Ready!'); e = datetime.now()
        print('======== Total Time: ', e - s)

    def train(self, nlptext = None, total_examples=None, total_words=None,
              epochs=None, start_alpha=None, end_alpha=None, word_count=0,
              queue_factor=2, report_delay=1.0, compute_loss=False, callbacks=(), 
              **kwargs):

        return super(FieldEmbedding, self).train(
            nlptext = nlptext, total_examples=total_examples, total_words=total_words,
            epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha, word_count=word_count,
            queue_factor=queue_factor, report_delay=report_delay, compute_loss=compute_loss, callbacks=callbacks)

    def _clear_post_train(self):
        """Remove all L2-normalized word vectors from the model."""
        self.wv.vectors_norm = None

    def _set_train_params(self, **kwargs):
        if 'compute_loss' in kwargs:
            self.compute_loss = kwargs['compute_loss']
        self.running_training_loss = 0

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

    @deprecated(
        "Method will be removed in 4.0.0, keep just_word_vectors = model.wv to retain just the KeyedVectors instance"
    )
    def _minimize_model(self, save_syn1=False, save_syn1neg=False, save_vectors_lockf=False):
        if save_syn1 and save_syn1neg and save_vectors_lockf:
            return
        if hasattr(self.trainables, 'syn1') and not save_syn1:
            del self.trainables.syn1
        if hasattr(self.trainables, 'syn1neg') and not save_syn1neg:
            del self.trainables.syn1neg
        if hasattr(self.trainables, 'vectors_lockf') and not save_vectors_lockf:
            del self.trainables.vectors_lockf
        self.model_trimmed_post_training = True

    def save(self, *args, **kwargs):
        kwargs['ignore'] = kwargs.get('ignore', ['vectors_norm', 'cum_table'])
        super(Word2Vec, self).save(*args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs):
        model = super(FieldEmbedding, cls).load(*args, **kwargs)
        return model

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

     # my new code
    def scan_and_prepare_vocab_from_nlptext(self, model, nlptext, negative,  update=False, sample=None, **kwargs):
        
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
                word_probability = 0
            else:
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
        # e = datetime.now(); print('\tEnd  : ', e);print('\tTotal Time:', e - s )
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

    def prepare_weights_from_nlptext(self, model, negative, wv, update=False, vocabulary=None, neg_init = 0):
        # reset_weigths only
        # currently, it is the same as self.reset_weights()
        print('o-->', 'Prepare Trainable Parameters')
        s = datetime.now(); print('\tStart: ', s)
        # e = datetime.now(); print('\tEnd  : ', e);print('\tTotal Time:', e - s )
        self.reset_weights(model, negative, neg_init = neg_init)
        e = datetime.now(); print('\tEnd  : ', e);print('\tTotal Time:', e - s )

    ###############################  INIT WEIGHTS HERE ########################## 
    def reset_weights(self, model, negative , neg_init = 0):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting layer weights")
        
        # syn0
        for field, field_idx in model.field_idx.items():
            use, wv = model.field_head[field_idx]
            if use:
                wv.vectors = empty((len(wv.vocab), wv.vector_size), dtype=REAL)
                # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
                for i in range(3):
                    wv.vectors[i] = zeros(self.layer1_size)
                for i in range(3, len(wv.vocab)): 
                    # construct deterministic seed from word AND seed argument
                    wv.vectors[i] = self.seeded_vector(wv.index2word[i] + str(self.seed), self.layer1_size)

            for data in model.field_sub[field_idx]:
                wv = data[0]
                wv.vectors = empty((len(wv.vocab), wv.vector_size), dtype=REAL)
                # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
                for i in range(4):
                    wv.vectors[i] = zeros(self.layer1_size)
                for i in range(4, len(wv.vocab)): 
                    # construct deterministic seed from word AND seed argument
                    wv.vectors[i] = self.seeded_vector(wv.index2word[i] + str(self.seed), self.layer1_size)

        # syn1neg      
        if negative:
            if type(neg_init) == str:
                print('init neg with random:', neg_init)
                model.wv_neg.vectors = empty((len(model.wv.vocab), self.layer1_size), dtype=REAL)
                for i in range(3):
                    model.wv_neg.vectors[i] = zeros(self.layer1_size)
                for i in range(3, len(wv.vocab)): 
                    # construct deterministic seed from word AND seed argument
                    model.wv_neg.vectors[i] = self.seeded_vector(model.wv_neg.index2word[i] + neg_int + str(self.seed), 
                                                                 self.layer1_size)
            else:    
                print('init neg with 0')
                model.wv_neg.vectors = zeros((len(model.wv.vocab), self.layer1_size), dtype=REAL)
            


            self.syn1neg = model.wv_neg.vectors

        # this vectors_lockf is left for trainables
        self.vectors_lockf = ones(len(model.wv_neg.vocab), dtype=REAL)  # zeros suppress learning
