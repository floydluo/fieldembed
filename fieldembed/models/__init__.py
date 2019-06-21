#     # for backward compatibility
#     @deprecated("Method will be removed in 4.0.0, use self.wv.most_similar() instead")
#     def most_similar(self, positive=None, negative=None, topn=10, restrict_vocab=None, indexer=None):

#         return self.wv.most_similar(positive, negative, topn, restrict_vocab, indexer)

#     @deprecated("Method will be removed in 4.0.0, use self.wv.wmdistance() instead")
#     def wmdistance(self, document1, document2):
#         """Deprecated, use self.wv.wmdistance() instead.

#         Refer to the documentation for :meth:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.wmdistance`.

#         """
#         return self.wv.wmdistance(document1, document2)

#     @deprecated("Method will be removed in 4.0.0, use self.wv.most_similar_cosmul() instead")
#     def most_similar_cosmul(self, positive=None, negative=None, topn=10):
#         """Deprecated, use self.wv.most_similar_cosmul() instead.

#         Refer to the documentation for
#         :meth:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.most_similar_cosmul`.

#         """
#         return self.wv.most_similar_cosmul(positive, negative, topn)

#     @deprecated("Method will be removed in 4.0.0, use self.wv.similar_by_word() instead")
#     def similar_by_word(self, word, topn=10, restrict_vocab=None):
#         """Deprecated, use self.wv.similar_by_word() instead.

#         Refer to the documentation for :meth:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.similar_by_word`.

#         """
#         return self.wv.similar_by_word(word, topn, restrict_vocab)

#     @deprecated("Method will be removed in 4.0.0, use self.wv.similar_by_vector() instead")
#     def similar_by_vector(self, vector, topn=10, restrict_vocab=None):
#         """Deprecated, use self.wv.similar_by_vector() instead.

#         Refer to the documentation for :meth:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.similar_by_vector`.

#         """
#         return self.wv.similar_by_vector(vector, topn, restrict_vocab)

#     @deprecated("Method will be removed in 4.0.0, use self.wv.doesnt_match() instead")
#     def doesnt_match(self, words):
#         """Deprecated, use self.wv.doesnt_match() instead.

#         Refer to the documentation for :meth:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.doesnt_match`.

#         """
#         return self.wv.doesnt_match(words)

#     @deprecated("Method will be removed in 4.0.0, use self.wv.similarity() instead")
#     def similarity(self, w1, w2):
#         """Deprecated, use self.wv.similarity() instead.

#         Refer to the documentation for :meth:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.similarity`.

#         """
#         return self.wv.similarity(w1, w2)

#     @deprecated("Method will be removed in 4.0.0, use self.wv.n_similarity() instead")
#     def n_similarity(self, ws1, ws2):
#         """Deprecated, use self.wv.n_similarity() instead.

#         Refer to the documentation for :meth:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.n_similarity`.

#         """
#         return self.wv.n_similarity(ws1, ws2)

#     @deprecated("Method will be removed in 4.0.0, use self.wv.evaluate_word_pairs() instead")
#     def evaluate_word_pairs(self, pairs, delimiter='\t', restrict_vocab=300000,
#                             case_insensitive=True, dummy4unknown=False):
#         """Deprecated, use self.wv.evaluate_word_pairs() instead.

#         Refer to the documentation for
#         :meth:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.evaluate_word_pairs`.

#         """
#         return self.wv.evaluate_word_pairs(pairs, delimiter, restrict_vocab, case_insensitive, dummy4unknown)


#     # for backward compatibility (aliases pointing to corresponding variables in trainables, vocabulary)
#     @property
#     @deprecated("Attribute will be removed in 4.0.0, use self.epochs instead")
#     def iter(self):
#         return self.epochs

#     @iter.setter
#     @deprecated("Attribute will be removed in 4.0.0, use self.epochs instead")
#     def iter(self, value):
#         self.epochs = value

#     @property
#     @deprecated("Attribute will be removed in 4.0.0, use self.trainables.syn1 instead")
#     def syn1(self):
#         return self.trainables.syn1

#     @syn1.setter
#     @deprecated("Attribute will be removed in 4.0.0, use self.trainables.syn1 instead")
#     def syn1(self, value):
#         self.trainables.syn1 = value

#     @syn1.deleter
#     @deprecated("Attribute will be removed in 4.0.0, use self.trainables.syn1 instead")
#     def syn1(self):
#         del self.trainables.syn1

#     @property
#     @deprecated("Attribute will be removed in 4.0.0, use self.trainables.syn1neg instead")
#     def syn1neg(self):
#         return self.trainables.syn1neg

#     @syn1neg.setter
#     @deprecated("Attribute will be removed in 4.0.0, use self.trainables.syn1neg instead")
#     def syn1neg(self, value):
#         self.trainables.syn1neg = value

#     @syn1neg.deleter
#     @deprecated("Attribute will be removed in 4.0.0, use self.trainables.syn1neg instead")
#     def syn1neg(self):
#         del self.trainables.syn1neg

#     @property
#     @deprecated("Attribute will be removed in 4.0.0, use self.trainables.vectors_lockf instead")
#     def syn0_lockf(self):
#         return self.trainables.vectors_lockf

#     @syn0_lockf.setter
#     @deprecated("Attribute will be removed in 4.0.0, use self.trainables.vectors_lockf instead")
#     def syn0_lockf(self, value):
#         self.trainables.vectors_lockf = value

#     @syn0_lockf.deleter
#     @deprecated("Attribute will be removed in 4.0.0, use self.trainables.vectors_lockf instead")
#     def syn0_lockf(self):
#         del self.trainables.vectors_lockf

#     @property
#     @deprecated("Attribute will be removed in 4.0.0, use self.trainables.layer1_size instead")
#     def layer1_size(self):
#         return self.trainables.layer1_size

#     @layer1_size.setter
#     @deprecated("Attribute will be removed in 4.0.0, use self.trainables.layer1_size instead")
#     def layer1_size(self, value):
#         self.trainables.layer1_size = value

#     @property
#     @deprecated("Attribute will be removed in 4.0.0, use self.trainables.hashfxn instead")
#     def hashfxn(self):
#         return self.trainables.hashfxn

#     @hashfxn.setter
#     @deprecated("Attribute will be removed in 4.0.0, use self.trainables.hashfxn instead")
#     def hashfxn(self, value):
#         self.trainables.hashfxn = value

#     @property
#     @deprecated("Attribute will be removed in 4.0.0, use self.vocabulary.sample instead")
#     def sample(self):
#         return self.vocabulary.sample

#     @sample.setter
#     @deprecated("Attribute will be removed in 4.0.0, use self.vocabulary.sample instead")
#     def sample(self, value):
#         self.vocabulary.sample = value

#     @property
#     @deprecated("Attribute will be removed in 4.0.0, use self.vocabulary.min_count instead")
#     def min_count(self):
#         return self.vocabulary.min_count

#     @min_count.setter
#     @deprecated("Attribute will be removed in 4.0.0, use self.vocabulary.min_count instead")
#     def min_count(self, value):
#         self.vocabulary.min_count = value

#     @property
#     @deprecated("Attribute will be removed in 4.0.0, use self.vocabulary.cum_table instead")
#     def cum_table(self):
#         return self.vocabulary.cum_table

#     @cum_table.setter
#     @deprecated("Attribute will be removed in 4.0.0, use self.vocabulary.cum_table instead")
#     def cum_table(self, value):
#         self.vocabulary.cum_table = value

#     @cum_table.deleter
#     @deprecated("Attribute will be removed in 4.0.0, use self.vocabulary.cum_table instead")
#     def cum_table(self):
#         del self.vocabulary.cum_table