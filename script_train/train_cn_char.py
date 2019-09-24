from pprint import pprint
from nlptext.base import BasicObject
from datetime import datetime
import argparse
from fieldembed import FieldEmbedding

import os
import logging
from datetime import datetime


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-f', '--fields_type', type =int, default = 1,  help="the number of fields used in training")
    parser.add_argument('-s', '--size', default = 200,  type=int, help="the number of iteration")
    parser.add_argument('-f', '--field_num', default = 1,  type=int, help="the number of iteration")
    args = parser.parse_args()
    field_num = args.field_num

    logging.basicConfig(filename = 'Chinese_Char_log_field_' + str(field_num) + '.txt',
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.debug('test')


    Data_Dir = 'data/WikiChinese/char/'; min_token_freq = 10
    # Data_Dir = 'data/wiki_cn_sample/char/'; min_token_freq = 10
    BasicObject.INIT_FROM_PICKLE(Data_Dir, min_token_freq)

    # # this is not correct
    CHANNEL_SETTINGS_TEMPLATE_FOR_WIKICN = {
        # CTX_IND
        'token':   {'use': True, 'Max_Ngram': 1,},
        'subcomp': {'use': True, 'Min_Ngram': 1, 'Max_Ngram': 4, 'end_grain': False,  'min_grain_freq' : 1152},
        'pinyin':  {'use': True, 'Min_Ngram': 1, 'Max_Ngram': 4, 'end_grain': False,  'min_grain_freq' : 10},
        'pos':     {'use': True, 'tagScheme': 'BIOES'}
    }

    CS_TEM = CHANNEL_SETTINGS_TEMPLATE_FOR_WIKICN

    new_field_selections = [
                            ['token'], 
                            ['token', 'subcomp'],
                            ['token', 'subcomp', 'pinyin'],
                            ['token', 'subcomp', 'pinyin', 'pos'],
                            ]

    # fields_type = int(args.fields_type)
    # assert fields_type in [1, 2, 3]
    # fields_type = fields_type -1
    # fields_selection = field_selections[fields_type]

    # for fields_selection in new_field_selections:
    if field_num in [1,2,3,4]:
        fields_selection = new_field_selections[field_num - 1]
        # print(fields_selection)
        # del BasicObject
        # from nlptext.base import BasicObject
        print('\n')
        print('=='*20)
        print('USE FIELDS:')
        pprint(fields_selection)
        print('=='*20)
        # pprint(CS_TEM)
        CHANNEL_SETTINGS_TEMPLATE = {k:v for k, v in CS_TEM.copy().items() if k in fields_selection}
        pprint(CHANNEL_SETTINGS_TEMPLATE)
        
        ###################################################################
        
        # print('-----'*20)
        # print(len(BasicObject.TokenVocab[0]))
        # print(BasicObject.min_token_freq)
        # print('-----'*20)
        # build GV is only workable for embedding training.
        BasicObject.BUILD_GV_LKP(CHANNEL_SETTINGS_TEMPLATE)

        pprint(BasicObject.CHANNEL_SETTINGS)
        ###################################################################

        train = True

        sg = 0 # use cbow or use sg
        iter = 5 # epoch number
        window = 5
        negative = 10
        alpha = 0.025
        sample = 1e-3
        workers = 4

        sample_grain = 1e-3
        LF = 3
        size = int(args.size)

        compute_loss = True

        s = datetime.now(); print('+++++Start++++++', s)
        # end = datetime.now(); print('+++++End++++++', end, 'Using:',e - s ); 
        model = FieldEmbedding(
            nlptext = BasicObject, Field_Settings = BasicObject.CHANNEL_SETTINGS, train = train,  
            sg=sg,  iter=iter, window=window, negative=negative, alpha=alpha, sample=sample, workers=workers,  
            sample_grain = sample_grain, LF = LF, size=size, 
            compute_loss = compute_loss)

        e = datetime.now(); time = e - s
        print('+++++End++++++', e, 'Using:', time); 

        field_names = '_'.join([fld for fld in fields_selection])

        EmbeddingPath = os.path.join(Data_Dir.replace('data', 'embeddings/fieldembed'), model.path)
        print(EmbeddingPath)
        if not os.path.exists(EmbeddingPath): os.makedirs(EmbeddingPath)
        ModelPath =  os.path.join(EmbeddingPath, str(model.vector_size))
        print('\n')
        print('Save to:', ModelPath, '\n')
        model.save_keyedvectors(ModelPath)
        print('=='*40)
        print('Done!')
        print('=='*40)
