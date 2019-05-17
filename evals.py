import numpy as np
import os
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import jieba


def read_wordpair(file):
    """
    load file whose line template is
    token1 token2 score
    """
    pairs = []
    with open(file, 'r') as f:
        for line in f:
            pair = line.split()
            pair[-1] = float(pair[-1])
            pairs.append(pair)
    return pairs


def read_word_analogy(ana_file):
    """
    load analogy.txt in the dataset/evaluation
    """
    capital = []
    state = []
    family = []
    cnt = 0
    with open(ana_file, 'r') as f:
        for line in f:
            pair = line.split()
            if pair[0] == ':':
                cnt = cnt + 1
                continue
            if cnt == 1:
                capital.append(pair)
            elif cnt == 2:
                state.append(pair)
            else:
                family.append(pair)
    return capital, state, family


class BasicEva:
    def __init__(self, wv):

        self.wv = wv
        self.wv.init_sims()

        self.dict_word = {i: v.index for i, v in wv.vocab.items()}
        self.embeddings = wv.vectors_norm

    def word_similarity(self, sim_file):
        """
        compute the cosine of token1 and token2 and use spearman correlation
        as metric.
        :param sim_file's data type should be token1 token2 score
        """
        pairs = read_wordpair(sim_file)
        human_sim = []
        vec_sim = []
        cnt = 0
        total = len(pairs)
        for pair in pairs:
            w1 = pair[0]
            w2 = pair[1]
            if w1 in self.dict_word and w2 in self.dict_word:
                cnt += 1
                id1 = self.dict_word[w1]
                id2 = self.dict_word[w2]
                vsim = self.embeddings[id1].dot(self.embeddings[id2])
                human_sim.append(pair[2])
                vec_sim.append(vsim)
        print(cnt, '/', total, ' word pairs appeared in the training dictionary')
        score = spearmanr(human_sim, vec_sim)
        print(sim_file, ':', score)
        return score

    def word_analogy(self, ana_f):
        """
        work only for analogy.txt
        """
        capital, state, family = read_word_analogy(ana_f)

        capital_total, capital_dict, capital_correct = self.analogy(capital)
        state_total, state_dict, state_correct = self.analogy(state)
        family_total, family_dict, family_correct = self.analogy(family)
        total = capital_total + state_total + family_total
        indict = capital_dict + state_dict + family_dict
        correct = capital_correct + state_correct + family_correct
        print('capital total ', capital_total, ' in dict ', capital_dict, ' correct ', capital_correct, 'perc:',
              capital_correct / capital_dict)
        print('state total ', state_total, ' in dict ', state_dict, ' correct ', state_correct, 'perc:',
              state_correct / state_dict)
        print('family total ', family_total, ' in dict ', family_dict, ' correct ', family_correct, 'perc:',
              family_correct / family_dict)
        print(' total ', total, ' indict ', indict, ' correct ', correct, 'perc:', correct / indict)
        ana_capital = capital_correct / capital_dict
        ana_state = state_correct / state_dict
        ana_family = family_correct / family_dict
        ana_total = correct / indict
        return ana_capital, ana_state, ana_family, ana_total

    def analogy_predict_word(self, pair):
        # return the index of predicted word
        # embeddings have been normed
        id1 = self.dict_word[pair[0]]
        id2 = self.dict_word[pair[1]]
        id3 = self.dict_word[pair[2]]
        id4 = self.dict_word[pair[3]]
        pattern = self.embeddings[id2] - self.embeddings[id1] + self.embeddings[id3]
        pattern = pattern / np.linalg.norm(pattern)
        sim = self.embeddings.dot(pattern)
        sim[id1] = sim[id2] = sim[id3] = -1  # remove the input words
        predict_index = np.argmax(sim)
        if predict_index == id4:
            return 1
        else:
            return 0

    def analogy(self, pairs):
        # embeddings have been normed
        total = len(pairs)
        in_dict_cnt = 0
        predict_cnt = 0
        for pair in pairs:
            in_dict = np.all([p in self.dict_word for p in pair])
            if in_dict:
                in_dict_cnt = in_dict_cnt + 1
                predict_cnt = predict_cnt + self.analogy_predict_word(pair)
        return total, in_dict_cnt, predict_cnt



def string2vec(string, wv, need_seg=False, seg_method=jieba.cut):  # you can change seg_method
    # 'unk' s index is 3
    if need_seg:
        words = seg_method(string)
    else:
        words = string.split()

    words_vocidx = []
    for w in words:
        if w in wv.vocab:
            words_vocidx.append(wv.vocab[w].index)
        else:
            words_vocidx.append(wv.vocab['</unk>'].index)

    vec = wv.merge_vectors[words_vocidx]
    if len(vec) == 0:
        return None
    return vec.mean(axis=0)


def read_fudan(root_dir):
    """
    Fudan Dataset initialization
    Fudan directory contains 5 subdirectories for 5 different classes.

    :return: files-embedding, labels, label-representation
    """
    classes = os.listdir(root_dir)
    contents = []
    label = []
    cnt = 0
    label_dict = dict()
    for cls in classes:
        sub_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(sub_dir):
            continue
        files = os.listdir(sub_dir)
        label_dict[cnt] = cls
        for file in files:
            with open(os.path.join(sub_dir, file)) as f:
                content = f.read()
            if content is not None:
                contents.append(content)
                label.append(cnt)
        cnt += 1
    return contents, label, label_dict


def fudan2embed(contents, labels, wv):
    embeds = []
    label = []
    for idx, content in enumerate(contents):
        embed = string2vec(content, wv, need_seg=False)
        if embed is not None:
            embeds.append(embed)
            label.append(labels[idx])
    return np.array(embeds), label


def fudan_classification(feat, label, cv=10):
    """
    multi-class logistic classification

    """
    x_train, x_test, y_train, y_test = train_test_split(feat, label, test_size=0.3, random_state=0)
    lr = LogisticRegressionCV(cv=cv)
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    acc = np.mean(np.equal(y_pred, y_test))
    return acc



sim_file1 = 'sources/240.txt'
sim_file2 = 'sources/297.txt'
ana_f     = 'sources/analogy.txt'


class Evaluation:
    def __init__(self, wv):

        self.wv = wv
        self.wv.init_sims()

        self.dict_word = {i: v.index for i, v in wv.vocab.items()}
        self.embeddings = wv.vectors_norm

    def run_wv_lexical_evals(self):
        d = {}
        pearson, spearman, oov_ratio = self.wv.evaluate_word_pairs(sim_file1, restrict_vocab=500000, case_insensitive=False)
        d['sim240_spearman'] = spearman.correlation
        pearson, spearman, oov_ratio = self.wv.evaluate_word_pairs(sim_file2, restrict_vocab=500000, case_insensitive=False)
        d['sim297_spearman'] = spearman.correlation
        analogies_score, sections = self.wv.evaluate_word_analogies(ana_f, restrict_vocab=500000, case_insensitive=False)
        
        for section in sections:
            correct = len(section['correct'])
            total = len(section['correct']) + len(section['incorrect'])
            d['ana_' + section['section'] ] = correct/total

        return d

    def run_lexical_evaluation(self):
        d = {}
        
        d['sim240_spearman']  = self.word_similarity(sim_file1).correlation
        d['sim297_spearman']  = self.word_similarity(sim_file2).correlation
        for k, v in self.word_analogy(ana_f):
            d[k] = v
        return d


    def word_similarity(self, sim_file):
        """
        compute the cosine of token1 and token2 and use spearman correlation
        as metric.
        :param sim_file's data type should be token1 token2 score
        """
        pairs = read_wordpair(sim_file)
        human_sim = []
        vec_sim = []
        cnt = 0
        total = len(pairs)
        for pair in pairs:
            w1 = pair[0]
            w2 = pair[1]
            if w1 in self.dict_word and w2 in self.dict_word:
                cnt += 1
                id1 = self.dict_word[w1]
                id2 = self.dict_word[w2]
                # scale = np.linalg.norm(self.embeddings[id1])*np.linalg.norm(self.embeddings[id2])
                vsim = self.embeddings[id1].dot(self.embeddings[id2].T) #/scale
                human_sim.append(pair[2])
                vec_sim.append(vsim)
        print(cnt, '/', total, ' word pairs appeared in the training dictionary')
        score = spearmanr(human_sim, vec_sim)
        print(sim_file, ':', score)
        return score

    def word_analogy(self, ana_f):
        """
        work only for analogy.txt
        """
        d = {}
        capital, state, family = read_word_analogy(ana_f)
    
        capital_total, capital_dict, capital_correct = self.analogy(capital)
        state_total, state_dict, state_correct = self.analogy(state)
        family_total, family_dict, family_correct = self.analogy(family)
        total = capital_total + state_total + family_total
        indict = capital_dict + state_dict + family_dict
        correct = capital_correct + state_correct + family_correct
        print('capital total ', capital_total, ' in dict ', capital_dict, ' correct ', capital_correct, 'perc:',
              capital_correct / capital_dict)
        print('state total ', state_total, ' in dict ', state_dict, ' correct ', state_correct, 'perc:',
              state_correct / state_dict)
        print('family total ', family_total, ' in dict ', family_dict, ' correct ', family_correct, 'perc:',
              family_correct / family_dict)
        print(' total ', total, ' indict ', indict, ' correct ', correct, 'perc:', correct / indict)
        d['ana_capital'] = capital_correct / capital_dict
        d['ana_state']   = state_correct / state_dict
        d['ana_family']  = family_correct / family_dict
        d['ana_total']   = correct / indict
        return d

    def analogy_predict_word(self, pair):
        # return the index of predicted word
        # embeddings have been normed
        id1 = self.dict_word[pair[0]]
        id2 = self.dict_word[pair[1]]
        id3 = self.dict_word[pair[2]]
        id4 = self.dict_word[pair[3]]
        pattern = self.embeddings[id2] - self.embeddings[id1] + self.embeddings[id3]
        pattern = pattern / np.linalg.norm(pattern)
        sim = self.embeddings.dot(pattern)
        sim[id1] = sim[id2] = sim[id3] = -1  # remove the input words
        predict_index = np.argmax(sim)
        if predict_index == id4:
            return 1
        else:
            return 0

    def analogy(self, pairs):
        # embeddings have been normed
        total = len(pairs)
        in_dict_cnt = 0
        predict_cnt = 0
        for pair in pairs:
            in_dict = np.all([p in self.dict_word for p in pair])
            if in_dict:
                in_dict_cnt = in_dict_cnt + 1
                predict_cnt = predict_cnt + self.analogy_predict_word(pair)
        # prec = predict_cnt / in_dict_cnt
        return total, in_dict_cnt, predict_cnt#. prec

    


