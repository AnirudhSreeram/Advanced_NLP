""" 
    Basic feature extractor
"""
from operator import methodcaller
import string
import re
import numpy as np

def tokenize(text):
    # TODO customize to your needs
    text = text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    return text.split()

class Features:

    def __init__(self, data_file):
        with open(data_file) as file:
            data = file.read().splitlines()
        data_split = map(methodcaller("rsplit", "\t", 1), data)
        texts, self.labels = map(list, zip(*data_split))
        self.tokenized_text = [tokenize(text) for text in texts]
        self.labelset = list(set(self.labels))

    @classmethod
    def get_features(cls, tokenized, model):
        # TODO: implement this method by implementing different classes for different features 
        # Hint: try simple general lexical features first before moving to more resource intensive or dataset specific features
        if model == "BOW":
            ####### BOW features ########
            print("BOW featurs")
            _Bag = [item.lower() for sublist in tokenized for item in sublist] # create bags
            Bo = BOW(_Bag)
            Bo.unique()
            clean = Bo.cleaningBag()
            return Bo
        else:
            ####### TFIDF features ########
            print("TFIDF features")
            tfidf = TFIDF(tokenized)
            return tfidf

class BOW:
    def __init__(self, Bg):
        self.Bag = Bg
        self.uniq = None
        self.cleanBag = None

    def unique(self):
        seen = set()
        self.uniq = [x for x in self.Bag if not (x in seen or seen.add(x))]

    def cleaningBag(self):

        stopWord = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                    "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
                    "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
                    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                    "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
                    "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
                    "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
                    "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
                    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
                    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
                    "should",
                    "now"]
        punc =['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '0',
               '1','2','3','4','5','6','7','8','9']
        i = 0
        cleanBag = []
        for word in self.uniq:
            if word not in punc and word not in stopWord :
                word = re.sub(" \d+", " ", word)
                cleanBag.append(word)
        self.cleanBag = cleanBag
        return cleanBag

class TFIDF:
    def __init__(self, token):
        self.bag = [item.lower() for sublist in token for item in sublist] # create bags
        self.cleanBag = self.cleaningBag()
        self.vocab = []
        for i in self.cleanBag:
            if i not in self.vocab:
                self.vocab.append(i)

        self.tdoc = len(token)
        self.vocab = set(self.vocab)  #self.vocab == word_set
        self.idx = {}
        k = 0
        for w in self.vocab:
            self.idx[w] = k
            k += 1
        self.wordCount = []
        for i in token:
            self.wordCount.append(self.get_dict(i))
        print("Created TFIDF object")


    def get_tf(self, doc, w):
        num = len(doc)
        occ = len([tok for tok in doc if tok == w])
        tf = occ/num
        return tf

    def get_idf(self, w):
        try:
            wordOcc = self.wordCount[w] + 1
        except:
            wordOcc = 1
        return np.log(self.tdoc/wordOcc)

    def get_tfidf(self, sent):
        vec = np.zeros((len(self.vocab),))
        for w in sent:
            if w in self.vocab:
                tf = self.get_tf(sent,w)
                idf = self.get_idf(w)
                tfidf = tf*idf
                vec[self.idx[w]] = tfidf
        return vec


    def get_dict(self,sent):
        word_doc_count = {}  # dictionary for keeping count of doc containing the word
        for wo in self.vocab:
            word_doc_count[wo] = 0
            for s in sent:
                if wo in s:
                    word_doc_count[wo] += 1
        return word_doc_count



    def cleaningBag(self):

        stopWord = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                    "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
                    "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
                    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                    "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
                    "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
                    "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
                    "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
                    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
                    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
                    "should",
                    "now"]
        #punc = [",", ":", " ", ";", ".", "?","\\", "'", "!",""]
        punc =['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '0',
               '1','2','3','4','5','6','7','8','9']
        i = 0
        cleanBag = []
        for word in self.bag:
            if word not in punc and word not in stopWord :
                word = re.sub(" \d+", " ", word)
                cleanBag.append(word)
        self.cleanBag = cleanBag
        return cleanBag

    #def get_dictionary(self):




class vectorizer:
    def __init__(self):
        pass

    def vect(self, token,bag):
        vector = []
        for i in range(len(token)):
            vector.append(self.get_vec(token[i],bag))
        return vector

    def get_vec(self,token,bag):
        vec = []
        for word in bag:
             vec.append(token.count(word))
        return vec

    def gettfidf_vec(self,X,tfidf):
        Xtr_vec = []
        for s in X:
            vec = tfidf.get_tfidf(s)
            Xtr_vec.append(vec)
        return Xtr_vec


