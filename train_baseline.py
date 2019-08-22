import os
import sys
from pprint import pprint
from datetime import datetime
import subprocess


def dict2list(paramdict):
    resultlist = []
    for k, v in paramdict.items():
        resultlist.append(k)
        if v: resultlist.append(str(v))
    return resultlist


def shell_invoke(args, sinput = None, soutput = None):
    p = subprocess.Popen(args, stdout=subprocess.PIPE)
    
    # while True:
    #     out = p.stdout.readline().rstrip()
    #     # print(out.decode('utf-8'))
    #     if out == b'' and p.poll() != None:
    #         break
    #     if out != b'':
    #         # print(out)
    #         sys.stdout.write(out.decode('utf-8'))
    #         sys.stdout.flush()

    result = p.communicate()
    for robj in result:
        if robj:
            print(robj.decode('utf-8'))
    return None

def generate_para(name, Data_Dir, sg, iter, window, negative, alpha, sample, workers, size, min_count = 10):
    
    LearnPathDict = {
        'word2vec': './baseline/word2vec/word2vec',
        'cwe': './baseline/cwe/src/cwe',
        'jwe': './baseline/jwe/src/jwe',
    }
    
    learn_path = LearnPathDict[name]
    paras = {}
    sg_or_cb = 'sg' if sg else 'cb'; paras['-cbow'] = 0 if sg else 1
    ep  = 'it' + str(iter);          paras['-iter'] = iter
    w   = 'w'  + str(window);        paras['-window'] = window
    neg = 'ng' + str(negative);      paras['-negative'] = negative
    thr = 'th' + str(workers);       paras['-threads'] = workers
    smp = 'smp'+ str(sample);        paras['-sample'] = sample
    alp = 'lr' + str(alpha);         paras['-alpha'] = alpha
    nsexp = 'nsexp' + str(0.75)
    hppara = '-'.join([sg_or_cb, ep, w, neg, alp, smp, nsexp, thr,])
    # print(hppara)
    paras['-train']  = os.path.join(Data_Dir, 'Pyramid/_file/token.txt')
    Path = os.path.join(Data_Dir.replace('data', 'embeddings/baseline'), name, hppara)
    if not os.path.exists(Path): os.makedirs(Path)
        
    if name == 'word2vec':
        paras['-output'] = os.path.join(Path, 'word' + str(size))
        
    elif name == 'cwe':
        paras['-output-word'] = os.path.join(Path, 'word' + str(size))
        paras['-output-char'] = os.path.join(Path, 'char' + str(size))
        paras['-cwe-type'] = 1
        
    elif name =='jwe':
        paras['-output-word'] = os.path.join(Path, 'word' + str(size))
        paras['-output-char'] = os.path.join(Path, 'char' + str(size))
        paras['-output-comp'] = os.path.join(Path, 'comp' + str(size))
        paras['-comp'] = './baseline/jwe/subcharacter/comp.txt'
        paras['-char2comp'] = './baseline/jwe/subcharacter/char2comp.txt'
        paras['-join-type'] = 1
        paras['-pos-type'] = 1
        paras['-average-sum'] = 1
    
    # print(paras['-output'])
    paras['-size'] = size
    paras['-min-count'] = min_count
    return learn_path, paras

def run_baseline_w2v(learn_path, paras):
    part_args = []
    if paras:
        part_args += dict2list(paras)
    shell_invoke([learn_path] + part_args)


Data_Dir = 'data/WikiChinese/word/'
# Data_Dir = 'data/wiki_cn_sample/word/'
min_count = 1
names = ['word2vec', 'cwe', 'jwe']


sg = 0 # use cbow or use sg
iter = 1 # epoch number
window = 5
negative = 10
alpha = 0.025
sample = 1e-3
workers = 4
size = 200
compute_loss = True


for name in names:
    if sg == 1 and name != 'word2vec': continue
    learn_path, paras = generate_para(name, Data_Dir, sg, iter, window, negative, alpha, sample, workers,size,  min_count = min_count)
    print('\n', name,  '=='*30, '\n')
    print(learn_path)
    pprint(paras)
    print('\n')
    s = datetime.now()
    print('Start:', s)
    run_baseline_w2v(learn_path, paras)
    e = datetime.now()
    print('End:  ', e)
    print('Time: ', e -s )