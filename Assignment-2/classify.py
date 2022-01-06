import pickle
import argparse
from Features import Features
import string
from nn_layers import NNComp
from pdb import set_trace as bp
import numpy as np

def tokenize(text):
    # TODO customize to your needs
    text = text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    return text.split()

def load_model(model_file):
        with open(model_file, "rb") as file:
            model = pickle.load(file)
        return model

def classify(args):
    mod = load_model(args.m)
    mod_name = args.m
    n = mod_name.split("/")[-1].split(".")[1]
    n = "questions"

    with open(args.i) as file:
        texts = file.read().splitlines()
    tokenized_texts = [tokenize(text) for text in texts]

    for k,v in mod.items():
        E = v["E"]
        f = v["f"]

    #bp()
    feats = Features()                                                   # create feature object
    test  = feats.get_features(tokenized_texts, E, f)               # Extract Ytrain feats with given Embeddings
    
    layers = []
    #initialise layers and the model config to run
    for k,v in mod.items():
        if k == "layer1":
            l = NNComp(test.shape[1], v["n"], v["hact"])
            l.W = v["W"]
            l.b = v["b"]
            layers.append(l)
        else:
            l = NNComp(test.shape[1], v["classes"], v["hact"])
            l.W = v["W"]
            l.b = v["b"]
            layers.append(l)


    #initialise layers and the model config to run
    #layers = [ NNComp(test.shape[1], args.u, args.hact), 
              #  NNComp(args.u, n_classes , args.act)]
    
    preds = []
    for j in range(test.shape[0]):
        A_test = test[1 * j : (j * 1 + 1),:].T                 # make mini batches for Xtrain 
        # Perform forward pass without updates (Inference)
        for l in layers:                                                               
            A_test = l.forward(A_test)
        preds.append(np.argmax(A_test))

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

    with open(args.o, "w") as file:
        for pred in predictions:
            file.write(pred+"\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural net inference arguments.')

    parser.add_argument('-m', type=str, help='trained model file')
    parser.add_argument('-i', type=str, help='test file to be read')
    parser.add_argument('-o', type=str, help='output file')

    args = parser.parse_args()

    classify(args)
