


# import os
# from pprint import pprint
# from nlptext.base import BasicObject
# from datetime import datetime




# BOB = 'data/WikiTotal/word/Token447174/Pyramid/'
# LGU = 'data/WikiTotal/word/Token447174/GrainUnique/'


# BOB = 'data/LuohuCorpus/char/Token3546/Pyramid/'
# LGU = 'data/LuohuCorpus/char/Token3546/GrainUnique/'


# BasicObject.INIT_FROM_PICKLE(BOB, LGU)



# available_fields = ['token', 'char', 'subcomp', 'stroke', 'pinyin', 'pos']





# CHANNEL_SETTINGS_TEMPLATE = {
#     # CTX_IND
#     'token':   {'use': True, 'Max_Ngram': 1,},
#     'char':    {'use': True,'Max_Ngram': 1, 'end_grain': False},
#     'pinyin':   {'use': True,'Max_Ngram': 2, 'end_grain': False},
#     'subcomp': {'use': True,'Max_Ngram': 3, 'end_grain': True},
#     'stroke':  {'use': True,'Max_Ngram': 3, 'end_grain': True},
#     'pos':  {'use': True},
# }

# BasicObject.BUILD_GRAIN_UNI_AND_LOOKUP(CHANNEL_SETTINGS_TEMPLATE)
# # TODO: pretty print the result.
# # BasicObject.CHANNEL_SETTINGS

# from fieldembed.models.fieldembed import FieldEmbedding
# from datetime import datetime

# # import logging
# # logger = logging.getLogger()
# # logger.setLevel(logging.DEBUG)
# # logging.debug('test')


# batch_words = 10000

# s = datetime.now(); print('\n+++++Start++++++', s, '\n')
# # end = datetime.now(); print('+++++End++++++', end, 'Using:',e - s ); 
# model = FieldEmbedding(nlptext = BasicObject,  Field_Settings = BasicObject.CHANNEL_SETTINGS, 
#                         use_merger = 0, neg_init = 0, mode = 'M0XY',
#                         size = 200, alpha = 0.025, 
#                         sg = 0, iter=1, workers = 8, batch_words = batch_words)
# e = datetime.now(); print('\n+++++End++++++', e, 'Using:', e - s )


# # workers = 2, alpha = 0.01
# print('use channel about.')
# print(BasicObject.CHANNEL_SETTINGS)

# self = model
# print()
# print("training model with %i workers on %i vocabulary and %i features, "
#             "using sg=%s hs=%s sample=%s negative=%s window=%s" %(self.workers, len(self.wv.vocab), self.trainables.layer1_size, self.sg,
#             self.hs, self.vocabulary.sample, self.negative, self.window))

# print()

# print("learning rate starts at %i and training has %i epoch."%(self.alpha, self.iter))
# print()

# print("use merger", self.use_merger)


# ######################### do word similariy 
# from evals import get_similarity_score
# # sg_model.wv.vectors

# if 'token' in BasicObject.CHANNEL_SETTINGS:
#     # sg_model.wv.vectors
#     print('\nleft')
#     sim_file = 'sources/240.txt'
#     get_similarity_score(sim_file, token_embedding = model.wv)# wv maybe zero

#     sim_file = 'sources/297.txt'
#     get_similarity_score(sim_file, token_embedding = model.wv)# wv maybe zero



# print('\nright')
# sim_file = 'sources/240.txt'
# get_similarity_score(sim_file, token_embedding = model.wv_neg) # comes at first

# sim_file = 'sources/297.txt'
# get_similarity_score(sim_file, token_embedding = model.wv_neg) # comes at first

# ######################### do word similariy 
# # do word analogy
# #


# # to study:
# model.wv_neg

# model.weights
# # {'token': <fieldembed.models.keyedvectors.Word2VecKeyedVectors at 0x7feb0bd807f0>,
# #  'char': <fieldembed.models.keyedvectors.Word2VecKeyedVectors at 0x7feafa9b6b70>,
# #  'subcomp': <fieldembed.models.keyedvectors.Word2VecKeyedVectors at 0x7feafa8699e8>,
# #  'stroke': <fieldembed.models.keyedvectors.Word2VecKeyedVectors at 0x7feb00b44668>,
# #  'pinyin': <fieldembed.models.keyedvectors.Word2VecKeyedVectors at 0x7fea8db015c0>}


# def iter_char(string):
#     return [i for i in string]

# def string2vec(string, wv, seg_method = jieba.cut): # you can change seg_method
#     # 'unk' s index is 3
#     words = seg_method(string)
#     if hasattr(wv, 'DTU'):
#         # print(wv)
#         vec = []
#         for w in words:
#             word_vocidx = wv.DTU.get(w, 3)
#             grain_start = wv.EndIdx[word_vocidx-1]
#             grain_end   = wv.EndIdx[word_vocidx]
#             grain_vocidx = [i for i in wv.LookUp[grain_start:grain_end]]
#             # print(grain_vocidx)
#             field_word = wv.vectors[grain_vocidx].mean(axis = 0)
#             vec.append(field_word)
#         return np.array(vec).mean(axis = 0)
        
#     else:
#         # words_vocidx = [wv.vocab.get[.index for i in words]
#         words_vocidx = []
#         for w in words:
#             if w in wv.vocab:
#                 words_vocidx.append(wv.vocab[w].index)
#             else:
#                 words_vocidx.append(wv.vocab['</unk>'].index)
                        
#         vec = wv.vectors[words_vocidx]
#     return vec.mean(axis = 0)
    
# string = '扫描二维码登录微信. 登录手机微信. 手机上安装并登录微信'
# v = string2vec(string, model.wv_pinyin)


from pprint import pprint
from nlptext.base import BasicObject
from datetime import datetime


BOB = 'data/LuohuCorpus/char/Token3546/Pyramid/'
LGU = 'data/LuohuCorpus/char/Token3546/GrainUnique/'

BasicObject.INIT_FROM_PICKLE(BOB, LGU)


LGU, DGU = BasicObject.getGrainUnique('pos', tagScheme='BIOES')



# from train import test_model_sent_classification
from train import train_model

sg = 1 
num_field = 1
basic_eva_file = '%dfield_eva.csv' % num_field
standard_grad = 0 
use_merger = 0 
iter = 1 
batch_words = 10000
neg_init = 0
mode = 'M0XY_P'
size = 200
alpha = 0.025
workers = 1


import pickle 

with open('sources/medical_sentence_classification.p', 'rb') as handle:
    D = pickle.load( handle)
    
    
    
from train import prepare_sentence_classificaiton_data

test_data = prepare_sentence_classificaiton_data(D, BasicObject.TokenUnique[1])


# import logging

# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# logging.debug('train')

#  ['token', 'char', 'subcomp', 'stroke', 'pinyin']
fields = ['token', 'char', 'subcomp', 'pos']

model, eval_dict, eval_items = train_model(BasicObject, fields, sg=sg, mode = mode, iter = iter, 
            standard_grad = standard_grad, use_merger = use_merger,  neg_init = neg_init, 
            size = size, alpha =alpha,
            workers = workers, batch_words = batch_words, 
            test_data = test_data, test_lexical = False)