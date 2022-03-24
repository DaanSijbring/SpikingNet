import numpy as np
from SpikingNet import ensemble
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
import os

class Connection:
    
    learning_options = ["PES_Weights", "Voja_Weights"]
    
    def __init__(self, pre: ensemble.Ensemble, post: ensemble.Ensemble, dim, learning = True):
        self.pre = pre
        self.post = post
        self.measureStep = pre.measureStep
        self.dt = pre.dt
        self.resetConn()
        
        
    def zeroWeights(self):
        self.weights = np.zeros((self.pre.num_N, self.post.num_N))
        
    def resetConn(self):
        self.weights = np.dot(self.pre.decoder, np.array(self.post.encoder).T)
        self.saveTimes = []
        self.trackDict = {}
        self.tempTrack = {}
        for l in self.learning_options:
            self.tempTrack[l] = 0.0
            self.trackDict[l] = []
        self.nPES = []
        self.nPESind = [0.0] * self.pre.num_N
    def updateWeights(self, learningWeights, prevFilt, t, error = None, outputCont = None):
        if "PES_Weights" in learningWeights:
            if error is None:
                sys.exit('Error: can not apply PES learning without error!')
            delta = np.transpose(-self.post.learning_rate_PES*self.post.dt/self.pre.num_N * 
                                 np.outer(np.reshape(np.dot(self.post.encoder, error), (-1,1)), np.reshape(prevFilt, (-1,1))))
            self.nPESind += np.sum(np.absolute(delta), axis = 1)
            self.tempTrack["PES_Weights"] += np.sum(np.absolute(delta))
            self.weights += delta
            
        if "Voja_Weights" in learningWeights:
            if outputCont is None:
                sys.exit('Error: can not apply Voja learning without continuous output from post!')
            for i in range(self.post.num_N):
                delta = self.post.learning_rate_voja * outputCont[i] * (prevFilt[:,0] - (
                                                                              self.weights[:,i]))
                self.tempTrack["Voja_Weights"] += np.sum(np.absolute(delta))
                self.weights[:,i] += delta
    
    def plotWeightChanges(self, show = True, save = False, path = None):
        if save and not path:
            sys.exit('Error: a path must be supplied')
        plt.figure()
        for l in self.learning_options:
            if self.trackDict[l]:
                plt.plot(self.saveTimes, self.trackDict[l], label = l + ' changes')
        plt.title('Weight changes over time')
        plt.legend(loc = 'upper right')
        file_name = 'Weight_changes'
        saveLoc = path + file_name
        if os.path.isfile(saveLoc):
            os.remove(saveLoc)
        plt.savefig(saveLoc)
        if show:
            plt.show()
        plt.close()
        
    def plotIndWeightChanges(self, wordList, dur, show = True, save = False, path = None):
        if save and not path:
            sys.exit('Error: a path must be supplied')
        changes = self.nPES[1:]
        times = self.saveTimes[1:]
        enc = self.pre.encoder
        for i in range(len(wordList)):
            cols = cm.RdYlGn([np.dot(wordList[i], e) for e in enc])
            ind_start = int((i * dur) / self.measureStep)
            ind_stop = int(((i+1) * dur)/ self.measureStep)
            plt.figure()
            for y, c in zip(np.transpose(changes[ind_start:ind_stop]), cols):
                plt.plot(times[ind_start:ind_stop], y, color = c)
            plt.title("Weight changes during presentation of " + str(wordList[i]))
            plt.xlabel('Time (s)')
            plt.ylabel('Absolute weight change')
            file_name = 'Individual_weight_changes-' + str(i*dur) + '-' + str((i+1) * dur) + '-seconds.jpg'
            saveLoc = path + file_name
            if os.path.isfile(saveLoc):
                os.remove(saveLoc)
            plt.savefig(saveLoc)
            if show:
                plt.show()
            plt.close()
