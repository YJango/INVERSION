import numpy as np
import os,sys
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import lasagne

class Inver(object):
    """
    path: main path 
    setID: the set ID for cross validation
    time_step: time_step for input 
    batch_size
    """
    def __init__(self, path, setID, steps,fname,N_cell):
        self.path=path
        self.setID=setID
        self.steps=steps
        self.fname=fname
        self.N_cell=N_cell
    def per_sentence(self, seq_X, seq_Y='0',steps=20):
        """M2MGRU data generator 
        Args:
          seq_X: a sequence of vectors for one sentence (len, input_Dim)
          seq_Y: a sequence of vectors for one sentence (len, output_Dim)
          time_step: the length of history
        Returns: a dictionary dic which consists of two/three terms 
          in_present: (1, steps, input_dim)
          in_future:  (1, steps+ahead, input_dim)
          labels: (1, steps, output_dim)
        note:
      
        """
        D_input=seq_X.shape[1]
        D_output=120
        sen_len=len(seq_X)
        lists=[]
        #index
        idx_present=0
        #first part
        #remaining parts
        while idx_present < sen_len:
            if type(seq_Y)!=type('0'):
                lists.append([seq_X[idx_present:idx_present+steps].reshape((1,-1,D_input)).astype("float32"),seq_Y[idx_present:idx_present+steps].astype("float32")])
            else:
                lists.append([seq_X[idx_present:idx_present+steps].reshape((1,-1,D_input)).astype("float32")])
            idx_present+=steps
        return lists

    def total_sentences(self,fname,vali_size=10):
        """lstm data generator 
        return:
          total sentences tensors
        """
        self.train_data=[]
        for x in np.load(fname,encoding='latin1'):
            self.train_data.append(self.per_sentence(x[:,:39],x[:,39:],self.steps))
        self.vali_data=self.train_data[:int(len(self.train_data)/vali_size)]
        self.train_data=self.train_data[int(len(self.train_data)/vali_size):]

    #shuffle several lists together
    def shufflelists(self):
        ri=np.random.permutation(len(self.train_data))
        self.train_data=[self.train_data[i] for i in ri]

    def plots(self,T,P, n=9,length=500):
        plt.figure(figsize=(20,16))
        plt.subplot(211)
        plt.plot(T[:length,n],'--')
        plt.plot(P[:length,n])
        plt.legend(['real','predict'])
        plt.subplot(212)
        plt.plot(T[:length,n+1],'--')
        plt.plot(P[:length,n+1])
        plt.savefig(self.fname+str(n)+'.eps')
        plt.legend(['real','predict'])
        plt.close()


    def vali_loss_rmse(self,get_out,N_cell):
        predicts=[]
        targets=[]
        P=T.matrix('P')
        R=T.matrix('R')
        for i in range(len(self.vali_data)):
            #hidden states
            for l in self.vali_data[i]:
                out=get_out(l[0])
                predicts.append(out)
                targets.append(l[1])

        self.cor=0
        for i in range(48):
            self.cor+=np.corrcoef(np.vstack(predicts)[:,i],np.vstack(targets)[:,i])[1,0]
        self.cor=self.cor/48.0
        loss= T.mean(lasagne.objectives.squared_error(P, R))
        rmse=T.sqrt(loss)
        performance=theano.function([P,R],[loss,rmse])
        self.plots(np.vstack(targets),np.vstack(predicts), n=9,length=500)
        return performance(np.vstack(predicts),np.vstack(targets))

