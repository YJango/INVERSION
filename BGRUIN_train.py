import sys
from BInver import Inver
from M2MGRU import M2MGRU
import numpy as np
import theano

#processing training process
def processing(b,n_samples,times,rmse,loss,cor,rmsevali,lossvali,corvali):
    sys.stdout.write('Epoch:%2.2s(%4.4s) | Best rmse:%6.6s loss:%6.6s cor:%6.6s | Cur rmse:%6.6s loss:%6.6s cor:%6.6s\r' %(times,round((float(b)/n_samples)*100,1),round(float(rmsevali),4),round(float(lossvali),4),round(float(corvali),4),round(float(rmse),4),round(float(loss),4),round(float(cor),4)))
    sys.stdout.flush()
#write log file
def writer(fname,rmse,loss,cor,rmsevali,lossvali,corvali,b,n_samples,times):
    f=open(fname+'.txt','a')
    f.write('Epoch:%2.2s | Best rmse:%6.6s loss:%6.6s cor:%6.6s | Cur rmse:%6.6s loss:%6.6s cor:%6.6s \n' %(times,round(float(rmsevali),4),round(float(lossvali),4),round(float(corvali),4),round(float(rmse),4),round(float(loss),4),round(float(cor),4)))
    f.close()

def trainwithPER(setID,typ='teacher',learning_rate=1e-4, drop_out=0.4,Layers=[2,1,2], N_hidden=2048, N_cell=2048, time_step= 41, L2_lambda=1e-4,patience=3, continue_train=0, evalPER=0,steps=1000):
    N_EPOCHS = 100
    np.random.seed(55)
    typ2=typ
    path="/notebooks/distillation"

    train_data=path+"/%s/LSTMFile%s/%s_train_lstm.npy" %(typ,setID,typ[:3])

    #where to store weights
    fname='%s/%s/LSTMWeight%s/SBGRU_%s_N%s_D%s_L%s_C%s_%s' %(path,typ,setID,typ[:3],N_hidden,drop_out,L2_lambda,N_cell,Layers)
    print(fname)

    #instances
    inver=Inver(path, setID, steps,fname,N_cell)
    #data maker
    inver.total_sentences(train_data)

    ff=M2MGRU(learning_rate=learning_rate, drop_out=drop_out, Layers=Layers, N_hidden=N_hidden, N_cell=N_cell, D_input=39, D_out=48, L2_lambda=L2_lambda, _EPSILON=1e-15)

    #whether to retrain
    if continue_train:
        ff.loader(np.load(fname+'.npy'))
        
    lossvali=100
    rmsevali=100
    corvali=0
    rmse=100
    loss=100
    cor=0
    n_sentences=len(inver.train_data)
    print('n_train:',n_sentences)
    inver.shufflelists()
    # start to train
    for epoch in range(N_EPOCHS):
        for i in range(n_sentences):
            #hidden states
            for l in inver.train_data[i]:
                ff.train(l[0],l[1])
        loss, rmse= inver.vali_loss_rmse(ff.get_out,N_cell)
        cor=inver.cor
        #whether to change learning rate
        if loss<=lossvali:
                #save weights
            ff.saver(fname)
                #update the best rmse and loss
            lossvali=loss
            rmsevali=rmse
            corvali=cor
            writer(fname,rmse,loss,cor,rmsevali,lossvali,corvali,i,n_sentences,epoch)
                #reset patience
            p=1
        else:
            if p>patience:
                writer(fname,rmse,loss,cor,rmsevali,lossvali,corvali,i,n_sentences,epoch)
                f=open(fname+'.txt','a')
                ff.loader(np.load(fname+'.npy'))
                f.write('\nEpoch: %s | Best rmse:%s loss: %s cor: %s\n' %(epoch, round(float(rmsevali),4), round(float(lossvali),4), round(float(corvali),4)))
                f.write('fine-tuning with lr %s\n' %(5e-6))
                f.close()
                break
            p+=1
