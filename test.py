

# from pprint import pprint
# from nlptext.base import BasicObject
# from datetime import datetime


# BOB = 'data/LuohuCorpus/char/Token3546/Pyramid/'
# LGU = 'data/LuohuCorpus/char/Token3546/GrainUnique/'

# BasicObject.INIT_FROM_PICKLE(BOB, LGU)


# LGU, DGU = BasicObject.getGrainUnique('pos', tagScheme='BIOES')

# # from train import test_model_sent_classification
# from train import train_model

# sg = 1 
# num_field = 1
# basic_eva_file = '%dfield_eva.csv' % num_field
# standard_grad = 0 
# use_merger = 0 
# iter = 1 
# batch_words = 10000
# neg_init = 0
# mode = 'M0XY_P'
# size = 200
# alpha = 0.025
# workers = 1


# import pickle 

# with open('sources/medical_sentence_classification.p', 'rb') as handle:
#     D = pickle.load( handle)
    
    
    
# from train import prepare_sentence_classificaiton_data

# test_data = prepare_sentence_classificaiton_data(D, BasicObject.TokenUnique[1])


# # import logging

# # logger = logging.getLogger()
# # logger.setLevel(logging.DEBUG)
# # logging.debug('train')

# #  ['token', 'char', 'subcomp', 'stroke', 'pinyin']
# fields = ['token', 'char', 'subcomp', 'pos']

# model, eval_dict, eval_items = train_model(BasicObject, fields, sg=sg, mode = mode, iter = iter, 
#             standard_grad = standard_grad, use_merger = use_merger,  neg_init = neg_init, 
#             size = size, alpha =alpha,
#             workers = workers, batch_words = batch_words, 
#             test_data = test_data, test_lexical = False)


from pprint import pprint
from nlptext.base import BasicObject
from datetime import datetime

from fieldembed.models.fieldembed import FieldEmbedding


BOB = 'data/WikiTotal/word/Token447170/Pyramid/'
LGU = 'data/WikiTotal/word/Token447170/GrainUnique/'

BasicObject.INIT_FROM_PICKLE(BOB, LGU)



CHANNEL_SETTINGS_TEMPLATE = {
    # CTX_IND
    'token':   {'use': True, 'Max_Ngram': 1,},
    'char':    {'use': False,'Max_Ngram': 1, 'end_grain': False},
    'pinyin':  {'use': False,'Max_Ngram': 2, 'end_grain': False},
    'subcomp': {'use': True,'Max_Ngram': 3, 'end_grain': True},
    'stroke':  {'use': False,'Max_Ngram': 3, 'end_grain': True},
    # 'pos':     {'use': True,'Max_Ngram': 3, 'end_grain': True},
}

BasicObject.BUILD_GRAIN_UNI_AND_LOOKUP(CHANNEL_SETTINGS_TEMPLATE)

mode =  'fieldembed_0X1_neat'
workers = 1
size = 200
batch_words = 10000
alpha = 0.025
sg = 1
iter = 1
train = True

sg_or_cbow = 'sg' if sg else 'cbow'

s = datetime.now(); print('+++++Start++++++', s)
# end = datetime.now(); print('+++++End++++++', end, 'Using:',e - s ); 
model = FieldEmbedding(nlptext = BasicObject, Field_Settings= BasicObject.CHANNEL_SETTINGS, 
                       size = size, train = train, alpha = alpha, mode = mode,
                       sg = sg, iter=iter, workers = workers, batch_words = batch_words)
e = datetime.now(); time = e - s
print('+++++End++++++', e, 'Using:', time); 
