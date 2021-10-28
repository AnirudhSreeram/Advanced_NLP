"""
NaiveBayes is a generative classifier based on the Naive assumption that features are independent from each other
P(w1, w2, ..., wn|y) = P(w1|y) P(w2|y) ... P(wn|y)
Thus argmax_{y} (P(y|w1,w2, ... wn)) can be modeled as argmax_{y} P(w1|y) P(w2|y) ... P(wn|y) P(y) using Bayes Rule
and P(w1, w2, ... ,wn) is constant with respect to argmax_{y} 
Please refer to lecture notes Chapter 4 for more details
"""

from Model import *
from Features_oov import *
import numpy as np
import random
import os

class NaiveBayes(Model):
    
    def train(self, input_file):
        """
        This method is used to train your models and generated for a given input_file a trained model
        :param input_file: path to training file with a text and a label per each line
        :return: model: trained model 
        """
        ## TODO write your code here
        feature = "EMPBOW"  #["BOW", "TFIDF"] select the features required
        ## get features here
        ### Split data into train test
        Xtrain, Xtest, Ytrain, Ytest = self.split_data(input_file)

        feat = Features(input_file) # create feature object
        Vec = vectorizer()  # create vectorizer object

        if feature == "BOW": # if feature tyoe is bag of words
            bow = feat.get_features(Xtrain, "BOW")     # get the bag of words object with the clean bag
            # Convert data to vectors
            Xtrain = np.asarray(Vec.vect(Xtrain,bow.cleanBag), dtype=np.float32)
            #Xtest = np.asarray(Vec.vect(Xtest,bow.cleanBag), dtype=np.float32)
            Ytrain = np.asarray(Ytrain, dtype=np.float32)
            Ytest = np.asarray(Ytest, dtype=np.float32)

        elif feature == "EMPBOW":
            empbow, priors, classes = feat.get_features((Xtrain,Ytrain), "EMPBOW")

        else:          # else TFIDF features selected
            tfidf = feat.get_features(Xtrain, "TFIDF")  # get TFIDF object from the data with word count and dictionaries
            # Convert data to vectors
            Xtrain = np.asarray(Vec.gettfidf_vec(Xtrain,tfidf), dtype=np.float32)
            #Xtest = np.asarray(Vec.gettfidf_vec(Xtest,tfidf), dtype=np.float32)
            Ytrain = np.asarray(Ytrain, dtype=np.float32)
            Ytest = np.asarray(Ytest, dtype=np.float32)


       # _Bag = [item.lower() for sublist in feat.tokenized_text for item in sublist] # create bags
        mod = "multi"
        if mod == "gaussian":
            n_samp, n_feat = Xtrain.shape
            self._clas = (np.unique(Ytrain))
            n_clas = len(self._clas)
            self._priors = np.zeros((n_clas), dtype=np.float32)
            self._mean = np.zeros((n_clas, n_feat), dtype=np.float32)
            self._vars = np.zeros((n_clas, n_feat), dtype=np.float32)


            # Compute Prior probabilities, mean and var per class and store these values.
            for i in self._clas:
                i=int(i)
                X_i = Xtrain[ i == Ytrain ]
                # Calculate mean, varience and prior for each class "i"
                self._mean[i, :] = X_i.mean(axis=0)
                self._vars[i, :] = X_i.var(axis=0)
                self._priors[i] = X_i.shape[0]/float(n_samp)
        else:
            print("saving the probabilities")


        # Save model
        self.model_file = os.fspath("models/Naivebayes/nb.products.model")
        if feature == "BOW":
            model = {"feature_type": feature, "mean" : self._mean, "var" :self._vars, "prior": self._priors, "classes": self._clas, "bag": bow.cleanBag}
        elif feature == "EMPBOW":
            model = {"feature_type": feature, "Bag" : empbow, "prior": priors, "classes" : classes }
        else:
            model = {"feature_type": feature, "mean" : self._mean, "var" :self._vars, "prior": self._priors, "classes": self._clas, "bag": tfidf}

        ## Save the model
        self.save_model(model)



        ####### Classification ############ Testing the model ##########
        #y_pred  = self.classify(Xtest, self.model_file)
        #acc = self.ACC(np.array(y_pred),np.array(Ytest))
        #print("Test ACC = {}% ".format(acc*100)) # print testing accuracy
        return model

    def predict_nb(self,sent , empbow, priors, classes):
        posterior = []
        for c in classes:
            wi = len(empbow[c].keys())
            li = []
            for word in sent:
               if word in empbow[c]:
                    li.append(empbow[c][word]/ wi)
            li = np.sum(np.array(li, dtype=float))
            posterior.append( li * priors[c] )
        preds = np.argmax(np.asarray(posterior))
        return preds



    def split_data(self,inp):
        if inp.split("/")[-1] ==  "questions.train.txt":
            feat = Features(inp)
            token = feat.tokenized_text
            lable = np.asarray(feat.labels, dtype=np.int32)
        elif inp.split("/")[-1] ==  "4dim.train.txt":
            feat = Features(inp)
            token = feat.tokenized_text
            lable = feat.labels
            label =[]
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
        elif inp.split("/")[-1] ==  "products.train.txt":
            feat = Features(inp)
            token = feat.tokenized_text
            lable = feat.labels
            label = [1  if i == "pos" else 0 for i in lable]
            lable = np.asarray(label, dtype=np.int32)
        else:
            feat = Features(inp)
            token = feat.tokenized_text
            lable = feat.labels
            label =[]
            for i in lable:
                if i == "business":
                    label.append(0)
                elif i == "sports":
                    label.append(1)
                else:
                    label.append(2)
            lable = np.asarray(label, dtype=np.int32)

        temp = list(zip(token, lable))
        random.shuffle(temp)
        X, Y = zip(*temp)
        xtrain, xtest = X[0: int(len(X) * 0.8)], X[int(len(X) * 0.8):]
        ytrain, ytest =Y[0: int(len(Y) * 0.8)], Y[int(len(Y) * 0.8):]
        return xtrain, xtest, ytrain, ytest

    def predict(self,x):
        return [ self.pred(i) for i in x ]


    def dist(self,inde, inp,mean,var):
        pr = (1/np.sqrt(2*np.pi*var[inde]**2)) * np.exp(-((inp-mean[inde])**2/(2*var[inde])))
        return pr

    def pred(self,input, _priors,_clas,mean,var):
        post = []
        for ind, classes in enumerate(_clas):
            pri = np.log(_priors[ind])
            prr = self.dist(ind,input,mean,var)
            prr = [1 if np.isnan(prr[i]) else prr[i] for i in range(len(prr))]
            likeli = np.sum(np.log(prr))
            post.append(pri + likeli)
        return self._clas[np.argmax(post)]

    def ACC(self, predY, trueY):
        acc = np.sum(trueY == predY)/len(trueY)
        return acc

    def classify(self, input_file, model):
        """
        This method will be called by us for the validation stage and or you can call it for evaluating your code 
        on your own splits on top of the training sets seen to you
        :param input_file: path to input file with a text per line without labels
        :param model: the pretrained model
        :return: predictions list
        """ 
        ## TODO write your code here
        mod = self.load_model()
        pri = mod["prior"]
        classes = mod["classes"]
        f = mod["feature_type"]
        mod_name = self.model_file
        n = mod_name.split("/")[-1].split(".")[1]
        with open(input_file) as file:
            texts = file.read().splitlines()
        tokenized_texts = [tokenize(text) for text in texts]

        Vec = vectorizer()
        if f == "BOW":
            bow = mod["bag"]
            mean = mod["mean"]
            var = mod["var"]
            test = np.asarray(Vec.vect(tokenized_texts,bow), dtype=np.float32)
            preds = []
            for r in range(len(test)):
                preds.append(self.pred(test[r],pri,classes,mean,var))
            return preds

        elif f == "EMPBOW":
            empbow = mod["Bag"]
            preds = []
            for sent in tokenized_texts:
                preds.append(self.predict_nb(sent, empbow, pri, classes))
            return preds
        else:
            tfidf = mod["bag"]
            mean = mod["mean"]
            var = mod["var"]
            test = np.asarray(Vec.gettfidf_vec(tokenized_texts,tfidf), dtype=np.float32)
            preds = []
            for r in range(len(test)):
                preds.append(self.pred(test[r],pri,classes,mean,var))

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







