from pprint import pprint
from nlptext.base import BasicObject
from datetime import datetime
import argparse
from fieldembed import FieldEmbedding

import os
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug('test')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-f', '--fields_type', type =int, default = 1,  help="the number of fields used in training")
    parser.add_argument('-s', '--size', default = 100,  type=int, help="the number of iteration")
    args = parser.parse_args()

    # Data_Dir = 'data/ResumeCN/char/'
    Data_Dir = 'data/WikiChinese/word/'

    # this is not correct
    # CHANNEL_SETTINGS_TEMPLATE_FOR_WIKICN = {
    #     # CTX_IND
    #     'token':   {'use': True,},
    #     'char':    {'use': True, 'Min_Ngram': 1, 'Max_Ngram': 3, 'end_grain': False,  'min_grain_freq' : 4480},
    #     'pinyin':  {'use': True, 'Min_Ngram': 1, 'Max_Ngram': 4, 'end_grain': False,  'min_grain_freq' : 15856},
    #     'subcomp': {'use': True, 'Min_Ngram': 1, 'Max_Ngram': 4, 'end_grain': False,  'min_grain_freq' : 21759},
    #     'stroke':  {'use': True, 'Min_Ngram': 3, 'Max_Ngram': 6, 'end_grain': False,  'min_grain_freq' : 19719},
    #     'pos':     {'use': True,},
    # }

    CHANNEL_SETTINGS_TEMPLATE_FOR_WIKICN = {
        # CTX_IND
        'token':   {'use': True, 'Max_Ngram': 1, },
        'char':    {'use': True, 'Max_Ngram': 1, 'end_grain': False},
        'pinyin':  {'use': True, 'Max_Ngram': 3, 'end_grain': True},
        'subcomp': {'use': True, 'Max_Ngram': 3, 'end_grain': True},
        'stroke':  {'use': True, 'Max_Ngram': 3, 'end_grain': True},
        'pos':     {'use': True, },
    }

    CS_TEM = CHANNEL_SETTINGS_TEMPLATE_FOR_WIKICN
    # BasicObject.BUILD_GV_LKP(CHANNEL_SETTINGS_TEMPLATE_FOR_WIKICN)

    # field_selections = [['token'], 
    #                     ['token', 'pinyin'],
    #                     ['token', 'subcomp', 'pinyin']]

    new_field_selections = [#['token'], 
                            #['stroke'], 
                            ['token', 'char'],
                            ['token', 'subcomp'],
                            ['token', 'pinyin'],
                            ['token', 'char', 'pinyin'],
                            ['token', 'stroke', 'pinyin'],
                            ['token', 'subcomp', 'pinyin'],
                            ['token', 'subcomp', 'stroke'],
                            ['subcomp', 'stroke','pinyin']]


    # fields_type = int(args.fields_type)
    # assert fields_type in [1, 2, 3]
    # fields_type = fields_type -1
    # fields_selection = field_selections[fields_type]
    for fields_selection in new_field_selections:
        # print(fields_selection)

        print('\n', '=='*20, '\n')
        print('USE FIELDS:')
        print(fields_selection)
        print(CS_TEM)
        CHANNEL_SETTINGS_TEMPLATE = {k:v for k, v in CS_TEM.copy().items() if k in fields_selection}
        print(CHANNEL_SETTINGS_TEMPLATE)
        
        ###################################################################
        min_token_freq = 10
        BasicObject.INIT_FROM_PICKLE(Data_Dir, min_token_freq)
        # build GV is only workable for embedding training.
        BasicObject.BUILD_GV_LKP(CHANNEL_SETTINGS_TEMPLATE)
        pprint(BasicObject.CHANNEL_SETTINGS)
        ###################################################################

        train = False

        sg = 0 # use cbow or use sg
        iter = 1 # epoch number
        window = 5
        negative = 10
        alpha = 0.025
        sample = 1e-3
        workers = 4

        sample_grain = 1e-4
        use_merger = 1
        size = 200

        compute_loss = True

        s = datetime.now(); print('+++++Start++++++', s)
        # end = datetime.now(); print('+++++End++++++', end, 'Using:',e - s ); 
        model = FieldEmbedding(
            nlptext = BasicObject, Field_Settings = BasicObject.CHANNEL_SETTINGS, train = train,  
            sg=sg,  iter=iter, window=window, negative=negative, alpha=alpha, sample=sample, workers=workers,  
            sample_grain = sample_grain, use_merger = use_merger, size=size, 
            compute_loss = compute_loss)

        e = datetime.now(); time = e - s
        print('+++++End++++++', e, 'Using:', time); 

        field_names = '_'.join([fld for fld in fields_selection])

        EmbeddingPath = os.path.join(Data_Dir.replace('data', 'embeddings/fieldembed'), model.path)
        if not os.path.exists(EmbeddingPath): os.makedirs(EmbeddingPath)
        ModelPath =  os.path.join(EmbeddingPath, str(model.vector_size))
        print('\n', 'Save to:', ModelPath, '\n')
        model.save(ModelPath)


