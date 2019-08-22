from pprint import pprint
from nlptext.base import BasicObject

# CORPUSPath = 'corpus/MSRA/'

# Corpus2GroupMethod = '.txt'

# Group2TextMethod   = 'line'

# Text2SentMethod  = 'whole'

# Sent2TokenMethod = 'iter'
# TOKENLevel = 'char'

# min_token_freq = 1

# use_hyper = []

# # 其实/o 非/o 汉/o 非/o 唐/o ，/o 又是/o 什么/o 与/o 什么/o 呢/o ？/o 
# anno = 'anno_embed_along_token' 
# anno_keywords = {
#     'sep_between_tokens': ' ',
#     'sep_between_token_label': '/', 
# }

# BasicObject.INIT(CORPUSPath, 
#                  Corpus2GroupMethod, 
#                  Group2TextMethod, 
#                  Text2SentMethod, 
#                  Sent2TokenMethod, TOKENLevel, min_token_freq = min_token_freq,
#                  use_hyper = use_hyper, 
#                  anno = anno, anno_keywords = anno_keywords)


# # from pprint import pprint
# # from nlptext.base import BasicObject

# CORPUSPath = 'corpus/ResumeCN/'

# Corpus2GroupMethod = '.bmes'

# Group2TextMethod   = 'block'

# Text2SentMethod  = 'whole'

# Sent2TokenMethod = 'iter'
# TOKENLevel = 'char'

# min_token_freq = 1

# use_hyper = ['pos']

# anno = 'conll_block'
# anno_keywords = {
#     'anno_sep': ' ',
#     'connector': '',
#     'suffix': False,
# }
# BasicObject.INIT(CORPUSPath, 
#                  Corpus2GroupMethod, 
#                  Group2TextMethod, 
#                  Text2SentMethod, 
#                  Sent2TokenMethod, TOKENLevel, min_token_freq = min_token_freq,
#                  use_hyper = use_hyper, 
#                  anno = anno, anno_keywords = anno_keywords)


# # from pprint import pprint
# # from nlptext.base import BasicObject

# CORPUSPath = 'corpus/boson/'

# Corpus2GroupMethod = '.txt'

# Group2TextMethod   = 'line'

# Text2SentMethod  = 're'

# Sent2TokenMethod = 'iter'
# TOKENLevel = 'char'

# min_token_freq = 1

# use_hyper = ['pos']

# anno = 'anno_embed_in_text'
# anno_keywords = {}

# BasicObject.INIT(CORPUSPath, 
#                  Corpus2GroupMethod, 
#                  Group2TextMethod, 
#                  Text2SentMethod, 
#                  Sent2TokenMethod, TOKENLevel, min_token_freq = min_token_freq,
#                  use_hyper = use_hyper, 
#                  anno = anno, anno_keywords = anno_keywords)



# from pprint import pprint
# from nlptext.base import BasicObject

# CORPUSPath = 'corpus/CoNLL-2003/'

# Corpus2GroupMethod = '.openNLP'

# Group2TextMethod   = 'block'

# Text2SentMethod  = 'whole'

# Sent2TokenMethod = ' ' # as this is CoNLL type, we don't need pos_en to seg sentences.
# TOKENLevel = 'word'

# min_token_freq = 1

# use_hyper = ['pos_en']

# anno = 'conll_block'
# anno_keywords = {
#     'anno_sep': ' ', # the seperation
#     'connector': ' ', 
#     'suffix': False,
#     'change_tags': True, # change I-B tags
# }

# BasicObject.INIT(CORPUSPath, 
#                  Corpus2GroupMethod, 
#                  Group2TextMethod, 
#                  Text2SentMethod, 
#                  Sent2TokenMethod, TOKENLevel, min_token_freq = min_token_freq,
#                  use_hyper = use_hyper, 
#                  anno = anno, anno_keywords = anno_keywords)


# from pprint import pprint
# from nlptext.base import BasicObject

# CORPUSPath = 'corpus/NLPBA2004/'

# Corpus2GroupMethod = '.iob2'

# Group2TextMethod   = 'block'

# Text2SentMethod  = 'whole'

# Sent2TokenMethod = ' '
# TOKENLevel = 'word'

# min_token_freq = 1

# use_hyper = ['pos_en']

# anno = 'conll_block'
# anno_keywords = {
#     'anno_sep': '\t',
#     'connector': ' ', 
#     'suffix': False,
#     "change_tags": False, 
# }
# BasicObject.INIT(CORPUSPath, 
#                  Corpus2GroupMethod, 
#                  Group2TextMethod, 
#                  Text2SentMethod, 
#                  Sent2TokenMethod, TOKENLevel, min_token_freq = min_token_freq,
#                  use_hyper = use_hyper, 
#                  anno = anno, anno_keywords = anno_keywords)


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
