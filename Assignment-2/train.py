import pickle
import argparse
from nn_layers import NNComp
import numpy as np
from split_data import split_data
from Features import Features
import matplotlib.pyplot as plt
from pdb import set_trace as bp

np.random.seed(42)                                                  # Random seed

# function to calculate accuracy
def ACC(y_p,y_t):                                                   
        miss, N = 0 ,len(y_p) 
        for i in range(N):
            if np.argmax(y_p[:,i]) != y_t[i]:
                miss += 1
        return (1-(miss/N))

def train(args):
    ############### feature extraction ###############
    Xtrain, Xtest, Ytrain, Ytest = split_data(args.i)                # get test : train :: 20 : 80 splits
    feats = Features(args.i)                                         # create feature object
    Xtrain = feats.get_features(Xtrain, args.E, args.f)              # Extract Xtrain feats with given Embeddings
    Xtest  = feats.get_features(Xtest, args.E, args.f)               # Extract Ytrain feats with given Embeddings

    ############### set parms ###############
    n_classes = len(np.unique(Ytrain))                               # get unique class labels
    n_train   = Xtrain.shape[0]                                      # get the number of training samples
    EPOCHS = args.e                                                  # set number of epochs
    BATCH_SIZE = args.b                                              # set batch size
    LEARNING_RATE = args.l                                           # set learning rate
    total_batches = (Xtrain.shape[0]//BATCH_SIZE)
    print("############# DATASET used = ", args.i)
    print(f' EPOCHES:{EPOCHS} | LR:{LEARNING_RATE} | BATCHES: {BATCH_SIZE}, Total batches = {Xtrain.shape[0]//BATCH_SIZE}')

    ############## Training ##################
    #initialise layers and the model config to run
    layers = [ NNComp( Xtrain.shape[1], args.u, args.hact, LEARNING_RATE), 
                NNComp(args.u, n_classes , args.act, LEARNING_RATE)]
    
    
    # iterating through 'e' number of epochs
    loss=[]
    acc_1 = []
    total_cur_loss, previous_best_loss = 0 , 1000
    plot_loss_tr, plot_loss_tst = [], []  
    plot_acc_tr, plot_acc_tst = [], []
    for epochs in range(EPOCHS):
        epoch_loss = 0      # loss per epoch
        epoch_acc = 0       # acc per epoch
        costs_batch = []  # cumilative loss function
        ################### Training per mini batch ######################
        for i in range(total_batches):
            A = Xtrain[BATCH_SIZE * i : (i * BATCH_SIZE + BATCH_SIZE),:].T           # make mini batches for Xtrain 
            Yt = np.array(Ytrain[BATCH_SIZE * i : (i * BATCH_SIZE + BATCH_SIZE)])             # make mini batches for Ytrain
            # forward pass of the NN
            for l in layers:
                A = l.forward(A)

            cost = l.cross_entropy_loss(A.T,Yt)                                      # compute cross entropy loss
            acc_tr = ACC(A, Yt)                                                      # calc training acc
            epoch_acc += acc_tr
            costs_batch.append(cost/BATCH_SIZE)                                      # loss per batch
            dA = l.dif_cross_entropy_loss(A.T,Yt)                                    # differentiate the cross entropy loss with softmax
            # backward pass of the NN
            dA = dA.T           
            for l in reversed(layers):                                               # Perform backwardpass
                dA = l.backward(dA)
        loss.append(sum(costs_batch)/(total_batches))                                # append training loss

        ############# Validation per mini batch ###################
        tst_loss = 0                                                                 # loss per epoch
        tst_acc = 0                                                                  # acc per epoch
        for j in range(Xtest.shape[0]//BATCH_SIZE):
             A_test = Xtest[BATCH_SIZE * j : (j * BATCH_SIZE + BATCH_SIZE),:].T                 # make mini batches for Xtrain 
             Ytst = np.array(Ytest[BATCH_SIZE * j : (j * BATCH_SIZE + BATCH_SIZE)])             # make mini batches for Ytrain
             # Perform forward pass without updates (Inference)
             for l in layers:                                                               
                 A_test = l.forward(A_test)
            
             # calculate accuracy
             acc = ACC(A_test, Ytst)
             # calculate test loss
             losses = l.cross_entropy_loss(A_test.T,Ytst)/BATCH_SIZE
             tst_loss += losses                                                               # accumilate loss
             tst_acc += acc                                                                   # accumilate acc
        print(f'Epoch {epochs+0:03}: Train | Loss: {sum(costs_batch)/(total_batches):.5f} | Acc: {epoch_acc/(total_batches) * 100:.3f} || Validation : | Loss: {tst_loss/(Xtest.shape[0]//BATCH_SIZE):.5f} | Acc: {tst_acc/(Xtest.shape[0]//BATCH_SIZE) * 100:.3f}')
        
        # maintain a stack of lists that keep a track of training and testing accuracy and loss per epoch for ploting graphs
        plot_loss_tr.append(sum(costs_batch)/(total_batches)) 
        plot_loss_tst.append(tst_loss/(Xtest.shape[0]//BATCH_SIZE))
        plot_acc_tr.append(epoch_acc/(total_batches) * 100)
        plot_acc_tst.append(tst_acc/(Xtest.shape[0]//BATCH_SIZE) * 100)
        
        ################## save model #########################
        total_cur_loss = tst_loss/(Xtest.shape[0]//BATCH_SIZE)                              # store the current loss
        if total_cur_loss < previous_best_loss:                                             # compare the current loss with the previous loss 
             previous_best_loss = total_cur_loss                                            # if current loss is lesser than prev save the model
             #bp()
             model_name = args.o +"/lr" +str(LEARNING_RATE) + "_f"+str(args.f)+"_B"+ str(BATCH_SIZE)+"_U"+str(args.u)+"_E30_"+ str(args.E).split("/")[-1]+"SGD."+ str(args.i).split("/")[-1].split(".")[0]+".md"
             lar = 1
             d = {}
             # convert each layer of the model to a dictionary and save
             for l in layers:
                d['layer'+str(lar)] = {"W":l.W, "b":l.b, "n":l.neuron, 'hact': l.hact, "E": str(args.E), "classes":n_classes, "f" : args.f}
                lar += 1
             save_model(d,model_name)
    
    ################## Plotting figures ############################
    plt.figure(1)
    plt.plot(range(EPOCHS), plot_loss_tr , label = "Training loss")
    plt.plot(range(EPOCHS), plot_loss_tst, label = "Testing loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss" + str(LEARNING_RATE))
    #bp()
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
    parser.add_argument('-hact', type=str, help='activation function used in hidden layers', default='tanh')
    parser.add_argument('-act', type=str, help='activation function used in final layers', default='sigmoid')

    args = parser.parse_args()

    train(args)
