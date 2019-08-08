from pprint import pprint
from nlptext.base import BasicObject
from datetime import datetime

from fieldembed import FieldEmbedding


import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug('test')


Data_Dir = 'data/newsgroup/word/'

# this is not correct
CHANNEL_SETTINGS_TEMPLATE = {
    # CTX_IND
    'token':   {'use': True, 'Max_Ngram': 1,},
#     'char':    {'use': True,'Max_Ngram': 1, 'end_grain': False},
#     'pinyin':  {'use': True,'Max_Ngram': 2, 'end_grain': False},
#     'subcomp': {'use': True,'Max_Ngram': 3, 'end_grain': True},
#     'stroke':  {'use': True,'Max_Ngram': 3, 'end_grain': True},
#     'pos':     {'use': True, 'tagScheme': 'BIOES'},
}

# this is not correct
CHANNEL_SETTINGS_TEMPLATE = {
    # CTX_IND
    'token':   {'use': True, 'Max_Ngram': 1,},
    # 'phoneme':   {'use': True, 'Max_Ngram': 2,},
#     'char':    {'use': True,'Max_Ngram': 1, 'end_grain': False},
#     'pinyin':  {'use': True,'Max_Ngram': 2, 'end_grain': False},
#     'subcomp': {'use': True,'Max_Ngram': 3, 'end_grain': True},
#     'stroke':  {'use': True,'Max_Ngram': 3, 'end_grain': True},
#     'pos':     {'use': True, 'tagScheme': 'BIOES'},
}


min_token_freq = 0

BasicObject.INIT_FROM_PICKLE(Data_Dir, min_token_freq)
# build GV is only workable for embedding training.
BasicObject.BUILD_GV_LKP(CHANNEL_SETTINGS_TEMPLATE)


pprint(BasicObject.CHANNEL_SETTINGS)



mode =  'train_batch_fieldembed_negsamp'
workers = 1
size = 50
batch_words = 10000
alpha = 0.025
sg = 1
iter = 20
train = True
window = 5

sg_or_cbow = 'sg' if sg else 'cbow'

s = datetime.now(); print('+++++Start++++++', s)
# end = datetime.now(); print('+++++End++++++', end, 'Using:',e - s ); 
compute_loss = True
model = FieldEmbedding(nlptext = BasicObject, Field_Settings= BasicObject.CHANNEL_SETTINGS, 
                       size = size, train = train, alpha = alpha, mode = mode,
                       sg = sg, iter=iter, workers = workers, batch_words = batch_words, 
                       window = window, compute_loss = compute_loss)
e = datetime.now(); time = e - s
print('+++++End++++++', e, 'Using:', time); 

# from evals import Evaluation
# wv = model.wv_neg
# evals = Evaluation(wv)

# s = datetime.now()
# d = evals.run_wv_lexical_evals()
# d['time'] = time
# e = datetime.now()
# print(e - s)
# print(d)

# model.save('embeddings/fieldembed/Eng_token_model_' + str(size))
import os
path = 'AAAI-MA/newsgroup/word2vec/' + str(size) 
if not os.path.exists(path):
	os.makedirs(path)
model.wv.save(path + '/' + 'wv')