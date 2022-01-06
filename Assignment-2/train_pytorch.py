import numpy as np
from numpy.core.fromnumeric import argmax
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from network import Net
from Features import Features
from split_data import split_data
import argparse
from pdb import set_trace as bp
import sys
import pickle

np.random.seed(42)
def ACC(y_p,y_t):
        miss, N = 0 , len(y_p) 
        for i in range(N):
             if torch.argmax(y_p[i]) != y_t[i]:
                 miss += 1
        return (1-(miss/N))

def train(args):
    ############### feature extraction ###############
    Xtrain, Xtest, Ytrain, Ytest = split_data(args.i)                # get test : train :: 20 : 80 splits
    feats = Features(args.i)                                         # create feature object
    Xtrain = feats.get_features(Xtrain, args.E, args.f)              # Extract Xtrain feats with given Embeddings
    Xtest  = feats.get_features(Xtest, args.E, args.f)               # Extract Ytrain feats with given Embeddings

    n_classes = len(np.unique(Ytrain))                               # get unique class labels
    inp_shape   = Xtrain.shape[1]                                      # get the number of training samples

    EPOCHS = args.e                                                  # set number of epochs
    BATCH_SIZE = args.b                                              # set batch size
    LEARNING_RATE = args.l                                           # set learning rate
    net = Net(inp_shape,args.u, n_classes,args.hact,args.act)                                                      # init network
    print("############# DATASET used = ", args.i)
    print(f' EPOCHES:{EPOCHS} | LR:{LEARNING_RATE} | BATCHES: {BATCH_SIZE}, Total batches = {Xtrain.shape[0]//BATCH_SIZE}')

    # set Criterion for loss calc
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCEWithLogitsLoss()                              
    
    # set optimizer SGD or ADAM
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9) 
    #optimizer = optim.Adam(net.parameters(), LEARNING_RATE) #, weight_decay=wd)

    # print the number of parameters
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(pytorch_total_params)

    # set network to train mode
    net.train()
    acc_1 = []
    total_cur_loss, previous_best_loss = 0 , 1000
    plot_loss_tr, plot_loss_tst = [], []  
    plot_acc_tr, plot_acc_tst = [], []
    # iterate through the number of epochs
    for e in range(1, EPOCHS+1):
        epoch_loss = 0      # loss per epoch
        epoch_acc = 0       # acc per epoch
        # iterate through the mini batch
        for i in range(Xtrain.shape[0]//BATCH_SIZE):
            X_batch = Xtrain[BATCH_SIZE * i : (i * BATCH_SIZE + BATCH_SIZE),:]                      # make mini batches for Xtrain
            y_batch = np.array(Ytrain[BATCH_SIZE * i : (i * BATCH_SIZE + BATCH_SIZE)], dtype=np.int16)# make mini batch for Ytrain
            optimizer.zero_grad()                                                                   # grads to zero init
            y_pred = net(torch.from_numpy(X_batch).float())                                         # run the minibatch through the data
            loss = criterion(y_pred,torch.from_numpy(y_batch).long())                               # calc loss
            acc = ACC(y_pred, y_batch)                                                              # calc acc
            loss.backward()                                                                         # perform backward pass
            optimizer.step()                                                                        # optimize
            epoch_loss += loss.item()                                                               # accumilate loss
            epoch_acc += acc                                                                        # accumilate acc
        #print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(Xtrain):.5f} | Acc: {epoch_acc/(Xtrain.shape[0]//BATCH_SIZE) * 100:.3f}')
        acc_1.append(epoch_acc/len(Xtrain))

        # set network to test mode
        net.eval()
        #acc_1 = []
        tst_loss = 0      # loss per epoch
        tst_acc = 0       # acc per epoch
        # iterate through the mini batch
        for i in range(Xtest.shape[0]//BATCH_SIZE):
            X_batch = Xtest[BATCH_SIZE * i : (i * BATCH_SIZE + BATCH_SIZE),:]                      # make mini batches for Xtrain
            y_batch = np.array(Ytest[BATCH_SIZE * i : (i * BATCH_SIZE + BATCH_SIZE)], dtype=np.int16)# make mini batch for Ytrain
            y_pred = net(torch.from_numpy(X_batch).float())                                         # run the minibatch through the data
            loss = criterion(y_pred,torch.from_numpy(y_batch).long())                               # calc loss
            acc = ACC(y_pred, y_batch)                                                              # calc acc
            tst_loss += loss.item()                                                               # accumilate loss
            tst_acc += acc                                                                        # accumilate acc
        print(f'Epoch {e+0:03}: Train | Loss: {epoch_loss/len(Xtrain):.5f} | Acc: {epoch_acc/(Xtrain.shape[0]//BATCH_SIZE) * 100:.3f} || Validation : | Loss: {tst_loss/len(Xtest):.5f} | Acc: {tst_acc/(Xtest.shape[0]//BATCH_SIZE) * 100:.3f}')
        
        # maintain a stack of lists that keep a track of training and testing accuracy and loss per epoch for ploting graphs
        plot_loss_tr.append(epoch_loss/len(Xtrain)) 
        plot_loss_tst.append(tst_loss/len(Xtest))
        plot_acc_tr.append(epoch_acc/(Xtrain.shape[0]//BATCH_SIZE) * 100)
        plot_acc_tst.append(tst_acc/(Xtest.shape[0]//BATCH_SIZE) * 100)
        
        ################## save model #########################
        total_cur_loss = tst_loss/len(Xtest)                    # store the current loss
        if total_cur_loss < previous_best_loss:                 # compare the current loss with the previous loss
            previous_best_loss = total_cur_loss                 # if current loss is lesser than prev save the model
            model_name = args.o +"/lr" +str(LEARNING_RATE) + "_f"+str(args.f)+"_B"+ str(BATCH_SIZE)+"_U"+str(args.u)+"_E30_"+ str(args.E).split("/")[-1]+"SGD."+ str(args.i).split("/")[-1].split(".")[0]+"_pytorch.md"
            d = {}
             # convert each layer of the model to a dictionary and save
            d["model"] = net  
            d["E"] = str(args.E) 
            d["classes"] = n_classes 
            d["f"] = args.f
            save_model(d,model_name)
    
    ################## Plotting figures ############################
    plt.figure(1)
    plt.plot(range(EPOCHS), plot_loss_tr , label = "Training loss")
    plt.plot(range(EPOCHS), plot_loss_tst, label = "Testing loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss" + str(LEARNING_RATE))
    plt.title("LOSS"+ "lr"+ str(LEARNING_RATE) + "_f"+str(args.f)+"_B"+ str(BATCH_SIZE)+"_U"+str(args.u)+"_E30_"+ str(args.E).split("/")[-1]+"SGD."+ str(args.i).split("/")[-1].split(".")[0])
    ls = args.o +"/Figloss.lr" +str(LEARNING_RATE) + "_f"+str(args.f)+"_B"+ str(BATCH_SIZE)+"_U"+str(args.u)+"_E30_"+ str(args.E).split("/")[-1]+"SGD."+ str(args.i).split("/")[-1].split(".")[0] + ".png"
    plt.savefig(ls)

    plt.figure(2)
    plt.plot(range(EPOCHS), plot_acc_tr , label = "Training ACC")
    plt.plot(range(EPOCHS), plot_acc_tst, label = "Testing ACC")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("ACCURACY" + str(LEARNING_RATE))
    plt.title("ACC"+ "lr"+ str(LEARNING_RATE) + "_f"+str(args.f)+"_B"+ str(BATCH_SIZE)+"_U"+str(args.u)+"_E30_"+ str(args.E).split("/")[-1]+"SGD."+ str(args.i).split("/")[-1].split(".")[0])
    ac = args.o +"/FigAcc.lr" +str(LEARNING_RATE) + "_f"+str(args.f)+"_B"+ str(BATCH_SIZE)+"_U"+str(args.u)+"_E30_"+ str(args.E).split("/")[-1]+"SGD."+ str(args.i).split("/")[-1].split(".")[0] + ".png"
    plt.savefig(ac)
    

def save_model(model, model_file):
        with open(model_file, "wb") as file:
            pickle.dump(model, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural net training arguments.')

    parser.add_argument('-u', type=int, help='number of hidden units')
    parser.add_argument('-l', type=float, help='learning rate')
    parser.add_argument('-f', type=int, help='max sequence length')
    parser.add_argument('-b', type=int, help='mini-batch size')
    parser.add_argument('-e', type=int, help='number of epochs to train for')
    parser.add_argument('-E', type=str, help='word embedding file')
    parser.add_argument('-i', type=str, help='training file')
    parser.add_argument('-o', type=str, help='model file to be written')
    parser.add_argument('-hact', type=str, help='activation function used in hidden layers', default='relu')
    parser.add_argument('-act', type=str, help='activation function used in final layers', default='softmax')

    args = parser.parse_args()

    train(args)
