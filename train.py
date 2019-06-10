import argparse
from nlptext.base import BasicObject
from fieldembed.models.fieldembed import FieldEmbedding
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from evals import read_fudan, Evaluation
from itertools import combinations
from datetime import datetime
import numpy as np
import pandas as pd
import pickle 

import logging
import os


def fudan_classification(feat, label, cv=10):
    """
    multi-class logistic classification

    """
    x_train, x_test, y_train, y_test = train_test_split(feat, label, test_size=0.3, random_state=0)
    lr = LogisticRegressionCV(cv=cv, multi_class = 'ovr', solver = 'liblinear')
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    acc = np.mean(np.equal(y_pred, y_test))
    return acc




available_fields = ['token', 'char', 'subcomp', 'stroke', 'pinyin']
# available_fields = ['char', 'subcomp', 'stroke', 'pinyin']

fudan_record = available_fields+['mean_field', 'concat_field', 'right_neg']
basic_eva_record = ['model_name', 'runtime', 'sim1', 'sim2', 'ana_capital', 'ana_state', 'ana_family', 'ana_total',
                    'left_sim1', 'left_sim2']
CHANNEL_SETTINGS_TEMPLATE = {
    # CTX_IND
    'token':   {'Max_Ngram': 1, },
    'char':    {'Max_Ngram': 1, 'end_grain': False},
    'pinyin':  {'Max_Ngram': 3, 'end_grain': True},
    'subcomp': {'Max_Ngram': 3, 'end_grain': True},
    'stroke':  {'Max_Ngram': 3, 'end_grain': True},
    'pos':     {'Max_Ngram': 1, 'tagScheme': 'BIOES'},
}

sim_file1 = 'sources/240.txt'
sim_file2 = 'sources/297.txt'
ana_file  = 'sources/analogy.txt'
fudan_dir = 'sources/fudan'


######################################### don't change the parameters here, change them in main function
sg = 1 
num_field = 1
basic_eva_file = '%dfield_eva.csv' % num_field
standard_grad = 0 
use_merger = 0 
iter = 1 
batch_words = 10000
neg_init = 0
mode = 'M0XY'
size = 200
alpha = 0.025
workers = 4
#########################################

def prepare_sentence_classificaiton_data(sentence_classificaiton_data, DTU):
    test_data = {}
    for st_name in sentence_classificaiton_data:
        print('For validation data:', st_name)
        material = sentence_classificaiton_data[st_name]
        fudan_contents, fudan_labels, label_dict = material
        fudan_contents_idx = [[DTU.get(i, 3) for i in l] for l in fudan_contents]
        test_data[st_name] = [fudan_contents_idx, fudan_labels, label_dict]
        # sentence_classificaiton_data[st_name] = 
    return test_data
#################################################################

def string2vec(words_vocidx, wv):
    vec = wv.merge_vectors[words_vocidx]
    return vec.mean(axis = 0)

def fudan2embed(contents, labels, wv):
    '''
    text to vector
    '''
    embeds = []
    label = []
    for idx, content in enumerate(contents):
        if len(content) == 0:
            continue
        embed = string2vec(content, wv)
        # if embed is not None:
        embeds.append(embed)
        label.append(labels[idx])
    return np.array(embeds), label



def test_model_sent_classification(model, fudan_contents, fudan_labels, label_dict):

    acc = dict(zip(fudan_record, [None] * (len(available_fields) + 2)))

    fields_feat = []
    for field, wv in model.weights.items():
        print('........for', field)
        field_feat, label = fudan2embed(fudan_contents, fudan_labels, wv = wv)
        fields_feat.append(field_feat)
        field_acc = fudan_classification(field_feat, label)
        acc[field] = field_acc

    all_fields_feats = np.array(fields_feat)
    # print(all_fields_feats.shape)
    mean_field_feat = np.array(fields_feat).mean(axis=0)
    mean_field_acc = fudan_classification(mean_field_feat, label)
    print('........for', 'mean')
    acc['mean_field'] = mean_field_acc

    concat_field_feat = np.concatenate(fields_feat, axis = -1)
    # print(concat_field_feat.shape)
    concat_field_acc = fudan_classification(concat_field_feat, label)
    print('........for', 'concat')
    acc['concat_field'] = concat_field_acc

    # right matrix
    right_embed_feat, label = fudan2embed(fudan_contents, fudan_labels, model.wv_neg)
    right_acc = fudan_classification(right_embed_feat, label)
    print('........for', 'right')
    acc['right_neg'] = right_acc
    # d['right_' +  k] = v
    return acc


def train_model(BasicObject, fields, sg=sg, mode = mode, iter = iter, 
    standard_grad = standard_grad, use_merger = use_merger,  neg_init = neg_init, 
    size = size, alpha =alpha,
    workers = workers, batch_words = batch_words, 
    test_data = [],
    test_lexical = True):
    
    print('the fields are:', fields)
    copyed = CHANNEL_SETTINGS_TEMPLATE.copy()
    settings_template = {}
    for field in fields:
        settings_template[field] = copyed[field]

    BasicObject.BUILD_GRAIN_UNI_AND_LOOKUP(settings_template)
    print(BasicObject.CHANNEL_SETTINGS)
    # TODO: pretty print the result.
    start = datetime.now()
    model_name = '+'.join(fields)
    model = FieldEmbedding(nlptext=BasicObject, 
                           Field_Settings=BasicObject.CHANNEL_SETTINGS,
                           sg=sg,
                           mode=mode,
                           size=size,
                           standard_grad = standard_grad,
                           use_merger = use_merger, 
                           neg_init=neg_init, 
                           alpha=alpha,
                           iter=iter, 
                           workers=workers, 
                           batch_words=batch_words)
    end = datetime.now()
    runtime = str(end - start)

    # =============similarity=================#
    # ========================================#
    cols = []
    d = {'model_name':model_name }
    cols.append('model_name')
    # D.append(d)
    

    # d['model_name'] = model_name
    d['runtime'] = runtime
    cols.append('runtime')
    # left_sim1, left_sim2 = -1, -1
    # evals = BasicEva(model.wv)
    if test_lexical:
        if 'token' in BasicObject.CHANNEL_SETTINGS:
            evals = Evaluation(model.wv)
            for k, v in evals.run_wv_lexical_evals().items():
                d[k + '_left'] = v
                cols.append(k + '_left')

        if True:
            evals = Evaluation(model.wv_neg)
            for k, v in evals.run_wv_lexical_evals().items():
                d[k + '_right'] = v
                cols.append(k + '_right')


    # print('=====lexical evaluation done!=====\n')
    # # ================fudan===================#
    # # ========================================#
    # print('=====sentence evaluation starts!=====\n')

    if len(test_data) > 0:
        ########################################### merge grains to token
        for field, wv in model.weights.items():
            print('.....left: ', field)
            wv.set_merge_vectors()
            print('.....left: ', field, 'merge done')

        print('.....right')
        model.wv_neg.set_merge_vectors()
        print('.....right merge done')
        ###########################################

    for st_name in test_data:
        print('For validation data:', st_name)
        material = test_data[st_name]
        fudan_contents, fudan_labels, label_dict = material
        acc = test_model_sent_classification(model, fudan_contents, fudan_labels, label_dict)
        for k, v in acc.items():
            d[st_name + '_'  + k] = v
            cols.append(st_name + '_' + k)

    # info = pd.DataFrame(D)[cols]
    print(cols)
    # print(info)
    # info.to_csv(basic_eva_file, index = False, encoding = 'utf-8')
    print('=====fudan evaluation done!=====')
    eval_dict = d
    eval_items = cols
    return model, eval_dict, eval_items



def train(BasicObject, available_fields, num_field, sg, basic_eva_file, mode = mode, size = size, alpha =alpha,
    iter = iter, workers = workers, standard_grad = standard_grad,
    batch_words = batch_words, use_merger = use_merger, neg_init = neg_init, 
    test_data = [],
    test_lexical = True, model_path = 'models/'):

    D = []
    # cols = []
    eval_items_longest = []
    standard_grad = 1 if sg else 0
    for cnt, fields in enumerate(combinations(available_fields, num_field)):
        if 'pos' not in fields:
            continue
        model_name = '_'.join(fields) + '.mdl'
        model, eval_dict, eval_items = train_model(BasicObject, fields, mode = mode, size = size, alpha =alpha,
                                                   iter = iter, workers = workers, standard_grad = standard_grad,
                                                   batch_words = batch_words, use_merger = use_merger, neg_init = neg_init,
                                                   test_data = test_data, test_lexical = test_lexical)

        if len(eval_items_longest) < len(eval_items):
            eval_items_longest = eval_items
        # fields = ['token', 'stroke']
        D.append(eval_dict)

        info = pd.DataFrame(D)[eval_items_longest]
        print(eval_items_longest)
        print(info)
        print(D)
        info.to_csv(basic_eva_file, index = False, encoding = 'utf-8')


        model.save(model_path + '/' + model_name)
        print('===================> save the model to ' + model_path + '/' + model_name)
    return D


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help="cbow or sg")
    parser.add_argument('-f', '--field_size', type=int, default = 1,  help="the number of fields used in training")
    parser.add_argument('-n', '--iter', type=int, default = 1,  help="the number of iteration")
    parser.add_argument('-s', '--standard_grad', default = 1,  type=int, help="the number of iteration")
    parser.add_argument('-g', '--use_merger', default = 1,  type=int, help="the number of iteration")
    parser.add_argument('-t', '--mode', default = 1,  help="the number of iteration")
    parser.add_argument('-d', '--data', default = 'wiki',  help="the number of iteration")


    args = parser.parse_args()
    
    sg = 1 if args.model == 'sg' else 0
    num_field = args.field_size
    
    standard_grad = 0 if int(args.standard_grad) == 0 else 1
    use_merger = 0 if int(args.use_merger) == 0 else 1
    use_merger_str = '' if use_merger else'noM'
    iter = 1 if int(args.iter) == 1 else int(args.iter)

    batch_words = 10000
    neg_init = 0
    mode = args.mode


    model_path = 'models/'

    data = args.data
    model_path = model_path + data + '2'

    

    folder =  mode  + '_' + args.model  + '_field_%d_'% num_field + use_merger_str + 'iter' + str(iter) 
    basic_eva_file = model_path + '/' + folder + '.csv' 


    final_model_path = model_path + '/' + folder
    if not os.path.exists(final_model_path):
        # os.makedirs(model_path)
        os.makedirs(final_model_path)

    logging.basicConfig(filename = basic_eva_file.split('.')[0] + '_log.txt',
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.debug('train')

    size = 200
    alpha = 0.025
    workers = 8

    if data == 'wiki':
        available_fields = ['token', 'char', 'subcomp', 'stroke', 'pinyin']
        BOB = 'data/WikiTotal/word/Token447174/Pyramid/'
        LGU = 'data/WikiTotal/word/Token447174/GrainUnique/'
        BasicObject.INIT_FROM_PICKLE(BOB, LGU)
        fudan_contents, fudan_labels, label_dict = read_fudan(fudan_dir)
        sentence_classificaiton_data = {'fudan' : (fudan_contents, fudan_labels, label_dict )}
        test_data = []# prepare_sentence_classificaiton_data(sentence_classificaiton_data, BasicObject.TokenUnique[1])
        test_lexical = False

    elif data == 'test':
        available_fields = ['token', 'char', 'subcomp', 'stroke', 'pinyin']
        BOB = 'data/wiki/word/Token6905/Pyramid/'
        LGU = 'data/wiki/word/Token6905/GrainUnique/'
        BasicObject.INIT_FROM_PICKLE(BOB, LGU)
        fudan_contents, fudan_labels, label_dict = read_fudan(fudan_dir)
        sentence_classificaiton_data = {'fudan' : (fudan_contents, fudan_labels, label_dict )}
        test_data = [] # prepare_sentence_classificaiton_data(sentence_classificaiton_data, BasicObject.TokenUnique[1])
        test_lexical = False

    elif data == 'medical':
        print('Use', data)
        available_fields = ['token', 'subcomp', 'stroke', 'pinyin', 'pos']
        BOB = 'data/LuohuCorpus/char/Token3546/Pyramid/'
        LGU = 'data/LuohuCorpus/char/Token3546/GrainUnique/'
        BasicObject.INIT_FROM_PICKLE(BOB, LGU)
        with open('sources/medical_sentence_classification.p', 'rb') as handle:
            D = pickle.load( handle)
        test_data = prepare_sentence_classificaiton_data(D, BasicObject.TokenUnique[1])
        test_lexical = False
    else:
        print("=======================================WRONG DATA INPUT============================")
            
    train(BasicObject, available_fields, num_field, sg, basic_eva_file, mode = mode, size = size, alpha =alpha,
          iter = iter, workers = workers, standard_grad = standard_grad,
          batch_words = batch_words, use_merger = use_merger, neg_init = neg_init,
          test_data = test_data, test_lexical = test_lexical, model_path = final_model_path)
