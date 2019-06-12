

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