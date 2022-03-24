import random
import math
from sklearn import preprocessing
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from SpikingNet import functions
from matplotlib.animation import FuncAnimation


class Ensemble:
    t_rc = 0.02  # membrane RC time constant
    t_ref = 0.002  # refractory period
    t_pstc = 0.1  # post-synaptic time constant
    rates = [25, 75]
    spikeCap = 10
    intercepts = [-1, 1]
    dt = 0.001
    measureStep = 0.05
    learning_rate_PES = 1000
    learning_rate_voja = 0.005
    learning_rate_voja2 = 0.000005
    max_out = 0.001
    bias2 = 0.001
    learning_rate_voja2rf = 0.00005
    learning_rate_voja2rfh = 0.005
    kf = 1000
    learning_rate_voja2rfb = 0.005
    percBias = 0.5
    forgettingRate = 0.05
    pstc_scale = 1.0 - math.exp(-dt / t_pstc)
    learning_options = ["PES_Decoder", "Voja_Encoder", "Voja2_Encoder", "Voja2RF_Encoder", 
                        "Voja2RF-H_Encoder",  "Voja2RF-H-B_Encoder"]
    
    def __init__(self, num_N, dim, normalize_encoders = False, optimize_decoders = True, intercept = None):
        self.name = 'Ensemble 0'
        self.num_N = num_N
        self.dim = dim
        if intercept is not None:
            self.intercepts = [intercept, intercept]
        self.gain, self.bias = functions.generate_gain_and_bias(num_N, self.intercepts[0], self.intercepts[1], 
                                                      self.rates[0], self.rates[1], self.t_ref, self.t_rc)
        self.encoder = np.array([[random.uniform(-1, 1) for d in range(dim)] for i in range(num_N)])
        if normalize_encoders:
            self.encoder = preprocessing.normalize(self.encoder, norm = "l2")
        self.orig_encoder = self.encoder
        self.resetEns()
        self.decoder = np.zeros((num_N, dim))
        if optimize_decoders:
            self.decOption = 'Optimized'
            self.setDecoderLinearly()
        else:
            self.decOption = 'Not optimized'
        self.mask = np.flip(np.exp(-np.arange(self.spikesCont.shape[1])*self.dt/self.t_pstc) * self.pstc_scale,0)
        self.mask = np.tile(self.mask, [self.spikesCont.shape[0],1])
        
    def resetEns(self):
        self.t = 0.0
        self.spikesCont = np.zeros((self.num_N, int(self.t_pstc*1/self.dt*10)))
        self.v = [0.0] * self.num_N
        self.ref = [0.0] * self.num_N  
        self.inputEns = [0.0] * self.num_N
        self.filtered = [0.0] * self.num_N
        self.spikeMem = [[] for i in range(self.num_N)]
        self.time_total = 0.0
        self.total_input = []
        self.plotInputs = []
        self.times = []
        self.outputs = []
        self.sumChangeTotals = [0.0] * self.dim
        self.sumChangesPerNeuron = [[] for i in range(self.num_N)]
        self.measurePoints = []
        self.errors = []
        self.encoderLog = []
        self.temptrack = {}
        self.trackDict = {}
        self.tempTrack = {}
        self.saveTimes = []
        for l in self.learning_options:
            self.tempTrack[l] = 0.0
            self.trackDict[l] = []
        self.encoder = self.orig_encoder
        
    def setRates(self, newRates):
        self.rates = newRates
        self.gain, self.bias = functions.generate_gain_and_bias(self.num_N, self.intercepts[0], self.intercepts[1], 
                                                      self.rates[0], self.rates[1], self.t_ref, self.t_rc)
        
    def setTimestep(self, newdt):
        self.dt = newdt
        
    def setSpikeCap(self, newSpikeCap):
        self.spikeCap = newSpikeCap
        
    def setIntercepts(self, newIntercepts):
        self.intercepts = newIntercepts
        self.gain, self.bias = functions.generate_gain_and_bias(self.num_N, self.intercepts[0], self.intercepts[1], 
                                                      self.rates[0], self.rates[1], self.t_ref, self.t_rc)
    
    def setMeasureStep(self, newMeasureStep):
        self.measureStep = newMeasureStep
        
    def setTau(self, newTau):
        self.t_pstc = newTau;
        self.pstc_scale = 1.0 - math.exp(-self.dt / newTau)
        self.spikesCont = np.zeros((self.num_N, int(newTau*1/self.dt*10)))
        self.mask = np.flip(np.exp(-np.arange(self.spikesCont.shape[1])*self.dt/newTau) * self.pstc_scale,0)
        
    def setDecoderLinearly(self):
        self.decoder = functions.compute_decoder_nengo(self.encoder, self.gain, self.bias, self.dt)
        
    def setVoja2Rate(self, newVoja2Rate):
        self.learning_rate_voja2 = newVoja2Rate
        self.voja2_alpha = self.learning_rate_voja2 * self.dt
    
    def zeroDecoders(self):
        self.decoder = np.zeros((self.num_N, self.dim))
        
    def updateEnsemble(self, learningEnsembles, t, error = None, outputCont = None, inputNext = None):
        if "PES_Decoder" in learningEnsembles:
            if error is None:
                sys.exit('Error: can not apply PES learning without error!')
            pass
        
        if "Voja_Encoder" in learningEnsembles:
            if outputCont is None or inputNext is None:
                sys.exit('Error: too little information supplied to run Voja on the encoders')
            delta = self.learning_rate_voja * outputCont * (inputNext - self.encoder)
            self.tempTrack["Voja_Encoder"] += np.sum(np.absolute(delta))
            self.encoder += delta
            self.encoder = preprocessing.normalize(self.encoder, norm = "l2")
            
        if "Voja2_Encoder" in learningEnsembles:
            if max(outputCont) > self.max_out:
                self.bias2 = max(outputCont) * self.percBias
                self.max_out = max(outputCont)
            shifted_post = 1/(outputCont-self.bias2)
            delta = self.learning_rate_voja2 *  shifted_post * (inputNext - self.encoder)
            self.tempTrack["Voja2_Encoder"] += np.sum(np.absolute(delta))
            self.encoder += delta
            self.encoder = preprocessing.normalize(self.encoder, norm = "l2")
            
        if "Voja2RF_Encoder" in learningEnsembles:
            if max(outputCont) > self.max_out:
                self.bias2 = max(outputCont) * self.percBias
                self.max_out = max(outputCont)
            shifted_post = 1/(outputCont-self.bias2)
            delta = self.learning_rate_voja2rf * shifted_post * outputCont * (inputNext - self.encoder)
            self.tempTrack["Voja2RF_Encoder"] += np.sum(np.absolute(delta))
            self.encoder += delta
            self.encoder = preprocessing.normalize(self.encoder, norm = "l2")
            
        if "Voja2RF-H_Encoder" in learningEnsembles:
            if max(outputCont) > self.max_out:
                self.bias2 = max(outputCont) * self.percBias
                self.max_out = max(outputCont)
            factor = np.array([[2 / (1 + math.exp(-2 * self.kf * (x - self.bias2))) - 1] for x in outputCont])
            delta = self.learning_rate_voja2rfh * factor * outputCont * (inputNext - self.encoder)
            self.tempTrack["Voja2RF-H_Encoder"] += np.sum(np.absolute(delta))
            self.encoder += delta
            self.encoder = preprocessing.normalize(self.encoder, norm = "l2")
            
        if "Voja2RF-H-B_Encoder" in learningEnsembles:            
            if max(outputCont) > self.max_out:
                self.bias2 = max(outputCont) * self.percBias
                self.max_out = max(outputCont)
            factor = np.array([[2 / (1 + math.exp(-2 * self.kf * (x - self.bias2))) - 1] for x in outputCont])
            dist = math.sqrt(math.pow(1 - math.cos(math.acos(self.intercepts[0])), 2) + math.pow(0 - math.sin(math.acos(self.intercepts[0])), 2))
            enc_dist = [np.linalg.norm(inputNext - e) for e in self.encoder]
            bumpf = np.array([[math.exp(-1 / (1 - math.pow((2/dist) * ed - 1, 2)))] if -dist < ed < dist else [0] for ed in enc_dist])
            delta = self.learning_rate_voja2rfh * factor * bumpf * outputCont * (inputNext - self.encoder)
            self.tempTrack["Voja2RF-H-B_Encoder"] += np.sum(np.absolute(delta))
            self.encoder += delta
            self.encoder = preprocessing.normalize(self.encoder, norm = "l2")
            
    def run(self, time, inputs, learning = None, ideal = None):
        self.time_total += time
        self.total_input.extend(inputs)
        output = [0.0] * self.dim
        filtered = [0.0] * self.num_N
        error = [0.0] * self.dim
        sumChangePerNeuron = [[0.0] * self.dim for i in range(self.num_N)]
        sumChangeTotal = [0.0] * self.dim
        steps = int(time/self.dt)
        print("Running ensemble for " + str(time) + " seconds...")
        for ti in tqdm(range(steps)): 
            x = inputs[ti]
            
            for i in range(self.num_N):
                self.inputEns[i] = np.dot(x,self.encoder[i]) * self.gain[i] + self.bias[i]
                
            spikes, self.v, self.ref = functions.run_neurons2(self.inputEns, self.v, self.ref, self.dt,  self.t_rc, self.t_ref)
            
            for i, s in enumerate(spikes):
                if s:
                    if len(self.spikeMem[i]) == self.spikeCap:
                        self.spikeMem[i].pop(0)
                    self.spikeMem[i].append(self.t)
                filtered[i] = sum([math.exp(-(self.t - si)/self.t_pstc) for si in self.spikeMem[i]])
                
            for d in range(self.dim):
                output[d] = sum([x*y[d] for x,y in zip(filtered,self.decoder)])
                if learning != None:
                    error[d] =  ideal[ti][d] - output[d]
            
            if learning != None:
                for i in range(self.num_N):
                    for d in range(self.dim):
                        if learning == "PES":
                            deltaDec = (self.learning_rate_PES / self.num_N) * error[d] * filtered[i]
                            self.decoder[i][d] += deltaDec
                            sumChangePerNeuron[i][d] += abs(deltaDec)
                            sumChangeTotal[d] += abs(deltaDec)
                    
            self.times.append(self.t)
            self.outputs.append(output[:])
            if learning != None:
                self.errors.append(error[:])
            
            if self.t % self.measureStep <= self.dt:
                for i in range(self.num_N):
                    self.sumChangesPerNeuron[i].append(sumChangePerNeuron[i])
                    self.sumChangesPerNeuron[i] = [0.0] * self.dim
                self.sumChangeTotals.append(sumChangeTotal[:])
                self.sumChangeTotals = [0.0] * self.dim
                self.measurePoints.append(self.t)
                
            self.t += self.dt
            
    def plotOutput(self, show = True, save = False, path = None, smoothFactor = 100):
        ints = int(self.time_total / 10)
        if (save) & (path == None):
            sys.exit('Error: Assign a folder for the plots to be saved.')
        for i in range(ints):
            blim = i * 10000
            tlim = ((i + 1) * 10000) - 1
            blimName = int(i*10)
            tlimName = int((i+1)*10)
            subTimes = self.times[blim:tlim]
            subInputs = self.total_input[blim:tlim]
            subOutputs = self.outputs[blim:tlim]
            subErrors = [[abs(y) for y in x] for x in self.errors[blim:tlim]]
            
            if self.dim == 1:
                fig = plt.figure()
                plt.hlines(0, subTimes[0], subTimes[-1], color = 'black', linewidth = 1)
                plt.plot(subTimes, subInputs, label = 'input')
                plt.plot(subTimes, subOutputs, label = 'output')
                plt.plot(subTimes, functions.smooth([err[0] for err in subErrors], smoothFactor), label = 'smoothed error', color = 'red', linestyle = '--')
                plt.title(self.name + ': simulation results ' + str(self.dim) + 'D (' + str(blimName) + ' - ' + str(tlimName) + ' seconds)')
                plt.legend()
            else:
                fig, axes = plt.subplots(nrows=self.dim, ncols=1, figsize=(11, 2*self.dim))
                fig.suptitle(self.name + ': simulation results ' + str(self.dim) + 'D (' + str(blimName) + ' - ' + str(tlimName) + ' seconds)')
                for d in range(self.dim):
                    axes[d].set_ylim([-1.1, 1.1])
                    axes[d].axhline(0, color = 'black', linewidth = 1)
                    axes[d].plot(subTimes, [i[d] for i in subInputs], label = 'input')
                    axes[d].plot(subTimes, [i[d] for i in subOutputs], label = 'output')
                    axes[d].plot(subTimes, functions.smooth([err[d] for err in subErrors], smoothFactor), label = 'smoothed error', color = 'red', linestyle = '--')
                    title = 'Dimension ' + str(d + 1) 
                    axes[d].set_title(title)
                plt.subplots_adjust(hspace = 0.5)
                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="right", title="Legend")
            if show:
                plt.show()
            file_name = 'Output-' + str(self.dim) + 'D-' + str(blimName) + '-' + str(tlimName) + 'seconds.png'
            saveLoc = path + file_name
            if os.path.isfile(saveLoc):
                os.remove(saveLoc)
            fig.savefig(saveLoc)
            plt.close()
        
    def plot2DEncoders(self, save = False, path = None, words = None, intercept = None, alg = None):
        if self.dim != 2:
            sys.exit('Error: can only animate encoder changes in 2 dimensions')
        else:
            angle = math.acos(intercept)
            bound = [math.cos(angle), math.sin(angle)]
            dist = math.sqrt(math.pow(1 - bound[0], 2) + math.pow(0 - bound[1], 2))
            
            fig = plt.figure(figsize = (10,10))
            ax = plt.axes(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
            line, = ax.plot([], [], 'r.', label = 'Encoders')
            
            for word in words:
                ax.add_artist(plt.Circle(word, dist, color = "blue", fill = False, ls = "--"))         

            def init():
                line.set_data([], [])
                return line,
            def animate(i):
                dat = self.encoderLog[i]
                line.set_data(dat.T[0], dat.T[1])
                time = self.time_total / len(self.encoderLog)
                plt.title('Learning rule = ' + alg + ', Simulation time = ' + str(round(i * time, 1)) + ', Voja learning rate = ' + str(self.learning_rate_voja) + ', Input = ' + str(self.plotInputs[i]))
                return line,
            
            anim = FuncAnimation(fig, animate, init_func=init,
                                           frames=len(self.encoderLog), interval=50, blit=True)
            if words is not None:
                plt.plot(np.transpose(words)[0], np.transpose(words)[1], 'bo', label = 'Word vectors')
            ax.legend(loc='upper left', frameon=False)
            file_name = '2DEncoders_Animation.gif'
            saveLoc = path + file_name
            if os.path.isfile(saveLoc):
                    os.remove(saveLoc)
            anim.save(saveLoc, dpi=100)
            
    
    
    def plotEnsembleChanges(self, show = True, save = False, path = None):
        if save and not path:
            sys.exit('Error: a path must be supplied to save plots!')
        plt.figure()
        for l in self.learning_options:
            if self.trackDict[l]:
                plt.plot(self.saveTimes, self.trackDict[l], label = l + ' changes')
        plt.title('Ensemble changes over time')
        plt.legend(loc = 'upper right')
        file_name = 'Ensemble_Changes'
        saveLoc = path + file_name
        if os.path.isfile(saveLoc):
            os.remove(saveLoc)
        if save: plt.savefig(saveLoc)
        if show: 
            plt.show()
        else:
            plt.close()