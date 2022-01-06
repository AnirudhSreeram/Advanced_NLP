import numpy as np
import random
from Features import *

def split_data(inp):
        ########## Questions dataset ############
        if inp.split("/")[-1] ==  "questions.train.txt":
            feat = Features(inp)                                # features object
            token = feat.tokenized_text                         # get tokenized output
            lable = np.asarray(feat.labels, dtype=np.int32)     # convert labels to np
        elif inp.split("/")[-1] ==  "test.train.txt":
            feat = Features(inp)                                # features object
            token = feat.tokenized_text                         # get tokenized output
            lable = np.asarray(feat.labels, dtype=np.int32)     # convert labels to np
        ########## 4dims dataset ##############
        elif inp.split("/")[-1] ==  "4dim.train.txt":
            feat = Features(inp)                                # features object
            token = feat.tokenized_text                         # get tokenized output
            lable = feat.labels
            label =[]
            # Convert labels to integer pos.tru = 0, pos.dec = 1, neg.tru = 2, neg.dec = 3
            for i in lable:
                if i == "pos.tru":
                    label.append(0)
                elif i == "pos.dec":
                    label.append(1)
                elif i == "neg.tru":
                    label.append(2)
                else:
                    label.append(3)
            lable = np.asarray(label, dtype=np.int32)           # convert labels to np
        ########## Products dataset ###############
        elif inp.split("/")[-1] ==  "products.train.txt":
            feat = Features(inp)                                # features object
            token = feat.tokenized_text                         # get tokenized output
            lable = feat.labels
            label = [1  if i == "pos" else 0 for i in lable]    # pos = 1, neg = 0 
            lable = np.asarray(label, dtype=np.int32)           # convert labels to np
        ########## Odiya dataset ##########
        else:
            feat = Features(inp)                                # features object
            token = feat.tokenized_text                         # get tokenized output
            lable = feat.labels
            label =[]
            # business = 0, sports = 1, entertainment = 2
            for i in lable:
                if i == "business":
                    label.append(0)
                elif i == "sports":
                    label.append(1)
                else:
                    label.append(2)
            lable = np.asarray(label, dtype=np.int32)           # convert to np
        #bp()
        # shuffel list and corresponding label randomly
        temp = list(zip(token, lable))                          
        random.shuffle(temp)
        X, Y = zip(*temp)
        
        # get 80:20 split of Xtrain and Xtest with its corresponding ytrain and ytest
        xtrain, xtest = X[0: int(len(X) * 0.8)], X[int(len(X) * 0.8):]
        ytrain, ytest =Y[0: int(len(Y) * 0.8)], Y[int(len(Y) * 0.8):]
        
        return xtrain, xtest, ytrain, ytest                     # return xtrain, xtest, ytrain, ytest