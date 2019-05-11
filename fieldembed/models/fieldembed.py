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

from queue import Queue, Empty

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
from .word2vec_inner import FAST_VERSION, MAX_WORDS_IN_BATCH

class FieldEmbedding(BaseWordEmbeddingsModel):

    def __init__(self, nlptext = None, size=100, alpha=0.025, window=5, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
                 sg=0, negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=hash, iter=5,
                 batch_words=MAX_WORDS_IN_BATCH, compute_loss=False, callbacks=()):
        
        self.callbacks = callbacks
        self.load = call_on_class_only

        # here only do initializations for wv, vocabulary, and trainables
        self.wv = Word2VecKeyedVectors(size)
        self.vocabulary = FieldEmbedVocab(sample=sample, ns_exponent=ns_exponent)
        self.trainables = FieldEmbedTrainables(seed=seed, vector_size=size, hashfxn=hashfxn)

        # self.build_vocab makes wv, vocabulary, and trainables rich. see: BaseWordEmbeddingsModel
        super(FieldEmbedding, self).__init__(
            nlptext = nlptext, workers=workers, vector_size=size, epochs=iter,
            callbacks=callbacks, batch_words=batch_words, sg=sg, alpha=alpha, window=window,
            seed=seed, negative=negative, cbow_mean=cbow_mean, min_alpha=min_alpha, compute_loss=compute_loss,
            fast_version=FAST_VERSION)


    def train(self, nlptext = None, total_examples=None, total_words=None,
              epochs=None, start_alpha=None, end_alpha=None, word_count=0,
              queue_factor=2, report_delay=1.0, compute_loss=False, callbacks=(), 
              sentences=None, corpus_file=None):

        return super(FieldEmbedding, self).train(
            nlptext = nlptext, total_examples=total_examples, total_words=total_words,
            epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha, word_count=word_count,
            queue_factor=queue_factor, report_delay=report_delay, compute_loss=compute_loss, callbacks=callbacks)


    def _do_train_job_nlptext(self, indexes, sentence_idx, alpha, inits):
        work, neu1 = inits
        tally = 0
        
        tally += train_batch_fieldembed_token(self, indexes, sentence_idx, alpha, work, neu1, self.compute_loss)
        # if self.sg:
        #     # print('||--> Use sg..')
        #     tally += train_batch_sg_nlptext(self, indexes, sentence_idx, alpha, work, self.compute_loss)
        # else:
        #     # print('||--> Use cbow..')
        #     tally += train_batch_cbow_nlptext(self, indexes, sentence_idx, alpha, work, neu1, self.compute_loss)
        return tally, sentence_idx[-1] # sentence_idx[-1] is the length of all tokens form this job sentences.

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
    def scan_and_prepare_vocab_from_nlptext(self, nlptext, negative, wv, update=False, sample=None):
        
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
            wv.index2word = LTU # LTU
            wv.vocab = {}       # DTU
            for word, v in iteritems(DTU_freq):
                wv.vocab[word] = Vocab(count=v, index=DTU[word])

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
            wv.vocab[w].sample_int = int(round(word_probability * 2**32))

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
        self.make_cum_table(wv)
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

    def prepare_weights_from_nlptext(self, negative, wv, update=False, vocabulary=None):
        # reset_weigths only
        # currently, it is the same as self.reset_weights()
        print('o-->', 'Prepare Trainable Parameters')
        s = datetime.now(); print('\tStart: ', s)
        # e = datetime.now(); print('\tEnd  : ', e);print('\tTotal Time:', e - s )
        self.reset_weights(negative, wv)
        e = datetime.now(); print('\tEnd  : ', e);print('\tTotal Time:', e - s )

    def seeded_vector(self, seed_string, vector_size):
        """Get a random vector (but deterministic by seed_string)."""
        # Note: built-in hash() may vary by Python version or even (in Py3.x) per launch
        once = random.RandomState(self.hashfxn(seed_string) & 0xffffffff)
        return (once.rand(vector_size) - 0.5) / vector_size

    ###############################  INIT WEIGHTS HERE ########################## 
    def reset_weights(self, negative, wv):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting layer weights")
        # syn0
        wv.vectors = empty((len(wv.vocab), wv.vector_size), dtype=REAL)
        # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once

        for i in range(3):
            wv.vectors[i] = zeros(self.layer1_size)
        for i in range(3, len(wv.vocab)):
            # construct deterministic seed from word AND seed argument
            wv.vectors[i] = self.seeded_vector(wv.index2word[i] + str(self.seed), wv.vector_size)

        # print(wv.vectors[:10])
        
        if negative:
            self.syn1neg = zeros((len(wv.vocab), self.layer1_size), dtype=REAL)
        wv.vectors_norm = None

        self.vectors_lockf = ones(len(wv.vocab), dtype=REAL)  # zeros suppress learning

    def update_weights(self, negative, wv):
        """Copy all the existing weights, and reset the weights for the newly added vocabulary."""
        logger.info("updating layer weights")
        gained_vocab = len(wv.vocab) - len(wv.vectors)
        newvectors = empty((gained_vocab, wv.vector_size), dtype=REAL)

        # randomize the remaining words
        for i in range(len(wv.vectors), len(wv.vocab)):
            # construct deterministic seed from word AND seed argument
            newvectors[i - len(wv.vectors)] = self.seeded_vector(wv.index2word[i] + str(self.seed), wv.vector_size)

        # Raise an error if an online update is run before initial training on a corpus
        if not len(wv.vectors):
            raise RuntimeError(
                "You cannot do an online vocabulary-update of a model which has no prior vocabulary. "
                "First build the vocabulary of your model with a corpus before doing an online update."
            )

        wv.vectors = vstack([wv.vectors, newvectors])

        if hs:
            self.syn1 = vstack([self.syn1, zeros((gained_vocab, self.layer1_size), dtype=REAL)])
        if negative:
            pad = zeros((gained_vocab, self.layer1_size), dtype=REAL)
            self.syn1neg = vstack([self.syn1neg, pad])
        wv.vectors_norm = None

        # do not suppress learning for already learned words
        self.vectors_lockf = ones(len(wv.vocab), dtype=REAL)  # zeros suppress learning

