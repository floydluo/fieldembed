from pprint import pprint
from nlptext.base import BasicObject


from nlptext.base import BasicObject

########### Wiki ###########
CORPUSPath = 'corpus/fudan/'

Corpus2GroupMethod = 'Dir'

Group2TextMethod   = 'file'

Text2SentMethod  = 'whole'

Sent2TokenMethod = ' '
TOKENLevel = 'word'
min_token_freq = 1

use_hyper = []

anno = False
anno_keywords = {}

BasicObject.INIT(CORPUSPath, 
                 Corpus2GroupMethod, 
                 Group2TextMethod, 
                 Text2SentMethod, 
                 Sent2TokenMethod, TOKENLevel, min_token_freq = min_token_freq,
                 use_hyper = use_hyper, 
                 anno = False, anno_keywords = anno_keywords)


from nlptext.base import BasicObject

CORPUSPath = 'corpus/newsgroup/'

Corpus2GroupMethod = 'Dir'

Group2TextMethod   = 'file'

Text2SentMethod  = 'whole'

Sent2TokenMethod = 'pos_en'
TOKENLevel = 'word'
min_token_freq = 1

use_hyper = ['pos_en']

anno = False
anno_keywords = {}

BasicObject.INIT(CORPUSPath, 
                 Corpus2GroupMethod, 
                 Group2TextMethod, 
                 Text2SentMethod, 
                 Sent2TokenMethod, TOKENLevel, min_token_freq = min_token_freq,
                 use_hyper = use_hyper, 
                 anno = False, anno_keywords = anno_keywords)
