"""
 Refer to Chapter 5 for more details on how to implement a Perceptron
"""
import os

from Model import *
from Features_oov import *
import numpy as np
import random
from pdb import set_trace as bp


class Perceptron(Model):
    def train(self, input_file):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        :return: model: trained model 
        """
        ## TODO write your code here
        ## select features here
        feature = "TFIDF"  # ["BOW", "TFIDF"] select the features required

        ### Split data into train test
        Xtrain, Xtest, Ytrain, Ytest = self.split_data(input_file)

        feat = Features(input_file)  # create feature object
        Vec = vectorizer()  # create vectorizer object

        if feature == "BOW":  # if feature tyoe is bag of words
            bow = feat.get_features(Xtrain, "BOW")  # get the bag of words object with the clean bag
            # Convert data to vectors
            Xtrain = np.asarray(Vec.vect(Xtrain, bow.cleanBag), dtype=np.float32)
            # Xtest = np.asarray(Vec.vect(Xtest,bow.cleanBag), dtype=np.float32)
            Ytrain = np.asarray(Ytrain, dtype=np.float32)
            Ytest = np.asarray(Ytest, dtype=np.float32)
        else:  # else TFIDF features selected
            tfidf = feat.get_features(Xtrain,"TFIDF")  # get TFIDF object from the data with word count and dictionaries
            # Convert data to vectors
            Xtrain = np.asarray(Vec.gettfidf_vec(Xtrain, tfidf), dtype=np.float32)
            # Xtest = np.asarray(Vec.gettfidf_vec(Xtest,tfidf), dtype=np.float32)
            Ytrain = np.asarray(Ytrain, dtype=np.float32)
            Ytest = np.asarray(Ytest, dtype=np.float32)

        # get no. samples , no. features and no. classes
        n_samp, n_feat = Xtrain.shape
        n_classes = len(np.unique(Ytrain))
        # initialize the weights to 0 vector
        W = np.zeros((n_classes, n_feat), dtype=np.float32)
        # number of epochs to train
        epoch = 20
        # Training the model
        for iterations in range(epoch):  # For each epoch
            missclass = 0
            for r in range(len(Ytrain)):  # for each data point (sentense)
                y_pred = self.pred(Xtrain[r], W)  # predict output
                if Ytrain[r] != y_pred:  # check for missclassification
                    missclass += 1
                    W[int(Ytrain[r]), :] = W[int(Ytrain[r])] + Xtrain[r]  # increment the weight for the correct class
                    W[y_pred] = W[y_pred] - Xtrain[r]  # decrement the weight for incorrect class.

            print("epoch = {} | Train ACC = {}% ".format(iterations,
                                                         (1 - (missclass / n_samp)) * 100))  # print training accuracy
        print("----------------------------------------")

        # Save model
        self.model_file = os.fspath("models/Perceptron/tfidf/perceptron.questions.model")
        if feature == "BOW":
            model = {"feature_type": feature, "W": W, "bag": bow.cleanBag}
        else:
            model = {"feature_type": feature, "W": W, "tfidf": tfidf }
        ## Save the model
        self.save_model(model)

        ####### Classification ############ Testing the model ##########
        #y_pred = self.classify(Xtest, self.model_file)
        #acc = self.ACC(y_pred, Ytest)
        #print("Test ACC = {}% ".format(acc * 100))  # print testing accuracy
        #return model

    def ACC(self, y_p, y_t):
        miss = 0
        N = len(y_p)
        for i in range(N):
            if y_p[i] != y_t[i]:
                miss += 1
        return (1 - (miss / N))

    def pred(self, data, w):
        cls = []
        for i in range(w.shape[0]):
            cls.append(np.dot(w[i], data))
        clss = np.asarray(cls)
        return np.argmax(clss)

    def split_data(self, inp):
        if inp.split("/")[-1] == "questions.train.txt":
            feat = Features(inp)
            token = feat.tokenized_text
            lable = np.asarray(feat.labels, dtype=np.int32)
        elif inp.split("/")[-1] == "4dim.train.txt":
            feat = Features(inp)
            token = feat.tokenized_text
            lable = feat.labels
            label = []
            for i in lable:
                if i == "pos.tru":
                    label.append(0)
                elif i == "pos.dec":
                    label.append(1)
                elif i == "neg.tru":
                    label.append(2)
                else:
                    label.append(3)
            lable = np.asarray(label, dtype=np.int32)
        elif inp.split("/")[-1] == "products.train.txt":
            feat = Features(inp)
            token = feat.tokenized_text
            lable = feat.labels
            label = [1 if i == "pos" else 0 for i in lable]
            lable = np.asarray(label, dtype=np.int32)
        else:
            feat = Features(inp)
            token = feat.tokenized_text
            lable = feat.labels
            label = []
            for i in lable:
                if i == "business":
                    label.append(0)
                elif i == "sports":
                    label.append(1)
                else:
                    label.append(2)
            lable = np.asarray(label, dtype=np.int32)

        np.random.seed(1)
        temp = list(zip(token, lable))
        random.shuffle(temp)
        X, Y = zip(*temp)
        xtrain, xtest = X[0: int(len(X) * 0.8)], X[int(len(X) * 0.8):]
        ytrain, ytest = Y[0: int(len(Y) * 0.8)], Y[int(len(Y) * 0.8):]
        return xtrain, xtest, ytrain, ytest

    def classify(self, input_file, model):
        """
        This method will be called by us for the validation stage and or you can call it for evaluating your code 
        on your own splits on top of the training sets seen to you
        :param input_file: path to input file with a text per line without labels
        :param model: the pretrained model
        :return: predictions list
        """
        ## TODO write your code here (and change return)

        mod = self.load_model()
        W = mod["W"]
        f = mod["feature_type"]
        mod_name = self.model_file
        n = mod_name.split("/")[-1].split(".")[1]
        with open(input_file) as file:
            texts = file.read().splitlines()
        tokenized_texts = [tokenize(text) for text in texts]

        Vec = vectorizer()
        if f == "BOW":
            bow = mod["bag"]
            test = np.asarray(Vec.vect(tokenized_texts, bow), dtype=np.float32)
        else:
            #tfidf = mod["bag"]
            tfidf = mod["tfidf"]
            test = np.asarray(Vec.gettfidf_vec(tokenized_texts, tfidf), dtype=np.float32)
        preds = []

        for r in range(len(test)):
            preds.append(self.pred(test[r], W))

        if n ==  "questions":
            predictions=[]
            for i in preds:
                predictions.append(str(i))
        elif n ==  "4dim":
            predictions =[]
            for i in preds:
                if i == 0 :
                    predictions.append("pos.tru")
                elif i == 1:
                    predictions.append("pos.dec")
                elif i == 2:
                    predictions.append("neg.tru")
                else:
                    predictions.append("neg.dec")
        elif n ==  "products":
            predictions = ["pos" if i == 1 else "neg" for i in preds]
        else:
            predictions =[]
            for i in preds:
                if i == 0:
                    predictions.append("business")
                elif i == 1:
                    predictions.append("sports")
                else:
                    predictions.append("entertainment")

        return predictions