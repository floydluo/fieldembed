from pprint import pprint
from nlptext.base import BasicObject

########### Wiki ###########
CORPUSPath = 'corpus/wiki/'
corpusFileIden = '.txt'
textType   = 'line'
Text2SentMethod  = 're'
Sent2TokenMethod = 'sep- '
TOKENLevel = 'word'
anno = False
annoKW = {}

MaxTextIdx = False
BasicObject.INIT(CORPUSPath, corpusFileIden, textType, 
                 Text2SentMethod, Sent2TokenMethod, TOKENLevel, 
                 anno, annoKW, MaxTextIdx)


from nlptext.corpus import Corpus
corpus = Corpus()
sentences_endidx = BasicObject.SENT['EndIDXTokens']
sentences_endidx[:10]
tokens_vocidx = BasicObject.TOKEN['ORIGTokenIndex']
# the first sentence
tokens_vocidx[0:99]

LTU, DTU = BasicObject.TokenUnique
total_words = len(tokens_vocidx)           
total_examples  = len(sentences_endidx)

import numpy as np


batch_words = 10000

total_words = total_words or len(tokens_vocidx)           
total_examples  = total_examples or len(sentences_endidx)

batch_end_st_idx_list = []
job_no = 0 # job_num
while True:
    job_no = job_no + 1
    batch_token_progress = job_no * batch_words  # 

    if batch_token_progress >= total_words:
        
        # if touch the bottom, go to the end and terminate the loop
        batch_end_st_idx_list.append(total_examples)
        # # This won't work: print('Current batch token number:', sentences_endidx[total_examples]) 
        print("Last sentence's end tk loc:", sentences_endidx[total_examples-1])
        break

    # if not, find the correct end sentence loc_id for this batch
    batch_end_st_idx = np.argmax(sentences_endidx > batch_token_progress)
    batch_end_st_idx_list.append(batch_end_st_idx)
    
    print('Current batch token number:', sentences_endidx[batch_end_st_idx])
    print("Last sentence's end tk loc:", sentences_endidx[batch_end_st_idx-1])

    
print(batch_end_st_idx_list, '\n')

for idx in range(job_no):

    # start and end are batch's start sentence loc_id and end sentence loc_id
    # as python routines, batch is [start, end), left close right open
    start = batch_end_st_idx_list[idx-1] if idx > 0 else 0
    end   = batch_end_st_idx_list[idx]

    # print(start, end)
    # find the start sentence's start token loc_id, and
    # find the end sentence's start token loc_id. (as the end sentence is exluded)
    token_start = sentences_endidx[start-1] if start > 0 else 0
    token_end   = sentences_endidx[end  -1]

    indexes     = tokens_vocidx[token_start:token_end] # dtype = np.uint32
    sentence_idx = np.array([i-token_start for i in sentences_endidx[start: end]], dtype = np.uint32)
    print('The start and end sent loc_id:', start, end)
    print('The token start and end loc idx in each batch:', token_start, token_end)
    print(sentence_idx[-1], len(indexes), '\n')
    
print(end == len(sentences_endidx))
print(token_end == len(tokens_vocidx))

# train_batch_sg_nlptext()
from fieldembed.models.word2vec import Word2Vec, LineSentence
from datetime import datetime
Little_Wiki_Path = 'corpus/wiki/wiki.txt'
Total_Wiki_Path = 'corpus/WikiTotal/WikiTotal7k_v2.txt'

data_input = LineSentence(Little_Wiki_Path)
# data_input = LineSentence(Total_Wiki_Path)
start = datetime.now()

print(start)
sg_model = Word2Vec(data_input, size = 200, alpha = 0.025,  min_count = 0, sg = 1, iter=1, workers = 8)
end = datetime.now()
time = end - start

print(end)
print(time)


from fieldembed.models.word2vec_inner import train_batch_sg_nlptext
from fieldembed.models.word2vec_inner import train_batch_sg

work1, work2 = sg_model._get_thread_working_mem()
work1



L = []
for i in data_input:
    L.append(i)
    
sentences = L[10]

from fieldembed.models.word2vec_inner import train_batch_sg

a = train_batch_sg(sg_model, sentences,  alpha = 0.05, _work = work1, compute_loss = False)
print(a)

a = train_batch_sg_nlptext(sg_model, indexes, sentence_idx, alpha = 0.05, _work = work1, compute_loss = False)
print(a)
