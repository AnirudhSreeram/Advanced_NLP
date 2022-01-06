from operator import methodcaller
import string
import re
import numpy as np
from pdb import set_trace as bp

# tokenizer function
def tokenize(text):
    # TODO customize to your needs
    text = text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    return text.split()


#  Class to generate features
class Features:

    def __init__(self, data_file=None):
        if data_file == None:
            pass
        else:
            with open(data_file) as file:
                data = file.read().splitlines()
            data_split = map(methodcaller("rsplit", "\t", 1), data)
            texts, self.labels = map(list, zip(*data_split))
            self.tokenized_text = [tokenize(text) for text in texts]    # get tokenized sentences 
            self.labelset = list(set(self.labels))                      # get labels


    @classmethod
    def get_features(cls, tokenized, model, max_seq_len):
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
        vocab = {}
        # Implementation of word to vec feature extraction
        if model == "datasets/fasttext.wiki.300d.vec":               
            with open("datasets/fasttext.wiki.300d.vec") as f: # load  the wordembedding vector
                for line in f:
                    val = line.split()
                    vocab[val[0]] = val[1:]                     # create a dictionary with {word : vector}
                    vec_len =  len(vocab[val[0]])               # get the vector length
            with open("datasets/unk-odia.vec") as f: # load  the wordembedding vector
                for line in f:
                    val = line.split()
                    vocab[val[0]] = val[1:]                     # create a dictionary with {word : vector}
        else:
            with open("datasets/glove.6B.50d.txt") as f:        # load  the wordembedding vector
                for line in f:
                    val = line.split()
                    vocab[val[0]] = val[1:]                     # create a dictionary with {word : vector}
                    vec_len =  len(vocab[val[0]])               # get the vector length
            with open("datasets/unk.vec") as f: # load  the wordembedding vector
                for line in f:
                    val = line.split()
                    vocab[val[0]] = val[1:]                     # create a dictionary with {word : vector}

        dataset = []                                            # data frame 
        ###### create a data frame for each sentence, where each sentence is a N_words X 300/50 vector
        for sentence in tokenized:
            sent = []
            for word in sentence:
                word = word.lower()
                if word not in stopWord and word not in punc:
                    if word in vocab:
                        sent.append(vocab[word])
                    else:
                        sent.append(vocab["UNK"])   #### handle OOV]
            
            if len(sent) < max_seq_len:                  # check for max length, if lesser append vectors of zero.
                while len(sent) < max_seq_len:
                    sent.append(np.zeros((vec_len)))     # appenderos if the length of sent is short
            else:
                sent = sent[:max_seq_len]                # truncate the length if long sent
            dataset.append(sent)                         # append each sentence to the main data frame
        dataset = np.array(dataset,dtype=np.float64)                                    # convert to numpy array
        dataset = dataset.reshape(dataset.shape[0],dataset.shape[1]*dataset.shape[2])   # reshape np array to (no_sentences, max_seq_len * vector_len)
        return dataset                                   # return the dataset dataframe

            
            

