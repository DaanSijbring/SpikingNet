from SpikingNet import connection
from SpikingNet import ensemble
from SpikingNet import functions
import numpy as np
import cupy as cp
import os
import sys
import csv
import time
import matplotlib.pyplot as plt
import datetime
from sklearn import preprocessing
import scipy.io

class Network:
    
    dt = 0.001
    measureStep = 0.05
    updateStep = 1.0
    
    def __init__(self, ensembles: [ensemble.Ensemble], GPU = False):
        self.date = time.strftime("%d-%m-%Y")
        self.time = time.strftime("%H:%M")
        self.startTime = datetime.datetime.now()
        self.endTime = datetime.datetime.now()
        self.GPU = GPU
        self.ensembles = ensembles
        self.num_ens = len(ensembles)
        for e in range(self.num_ens):
            self.ensembles[e].name = 'Ensemble ' + str(e + 1)
            self.ensembles[e].ensPath = self.ensembles[e].name + '/'
        self.connections = []
        for i in range(self.num_ens - 1):
            self.connections.append(connection.Connection(ensembles[i], ensembles[i+1], ensembles[i].dim))
        
        self.resetNet()
        if GPU: self.convGPU()
    
    def resetNet(self):
        self.t = 0.0
        self.total_time = 0.0
        self.totalInput = []
        self.totalOutput = [[] for e in range(self.num_ens)]
        for e in range(self.num_ens):
            self.ensembles[e].resetEns()
        for c in range(len(self.connections)):
            self.connections[c].resetConn()
            self.connections[c].zeroWeights()
            
    def convGPU(self):
        for ens in self.ensembles:
            ens.spikesCont = cp.array(ens.spikesCont)
            ens.mask = cp.array(ens.mask)
            ens.decoder = cp.array(ens.decoder)
            
    
    def runNetwork(self, inputs, learningEnsembles = None, learningWeights = None, transferFunctionsDecoders = None, 
                   transferFunctionsWeights = None, reset = False, GPU = False):
        if reset:
            self.resetNet()
        if transferFunctionsWeights == None:
            transferFunctionsWeights = [lambda x: x for i in range(self.num_ens)]
        self.total_time += len(inputs) * self.dt
        self.totalInput.extend(inputs)
        for e in range(self.num_ens):
            self.ensembles[e].total_input.extend(inputs)
            self.ensembles[e].time_total += len(inputs) * self.dt
        if learningEnsembles and "PES_Decoder" in learningEnsembles:
            for e in range(self.num_ens):
                self.ensembles[e].zeroDecoders()
        
        steps = int(len(inputs))
        for ti in range(steps):
        #for ti in tqdm(range(steps)):
            inputNext = inputs[ti]
            if self.t % self.measureStep <= self.dt:
                self.ensembles[0].plotInputs.append(inputNext)
            for e in range(self.num_ens):
                ens = self.ensembles[e]
                ens.encoder = preprocessing.normalize(ens.encoder, norm = "l2")
                for i in range(ens.num_N):
                    if e == 0:
                        ens.inputEns[i] = np.dot(np.array(inputNext), ens.encoder[i]) * ens.gain[i] + ens.bias[i]
                    else:
                        ens.inputEns[i] = inputNext[0,i] * ens.gain[i] + ens.bias[i]
                spikes, ens.v, ens.ref = functions.run_neurons2(ens.inputEns, ens.v, ens.ref, ens.dt, ens.t_rc, ens.t_ref)
                if GPU:
                    ens.spikesCont = cp.hstack((ens.spikesCont[:,1:],cp.array(spikes,ndmin=2).T))
                    outputCont = cp.sum(ens.mask * ens.spikesCont,1).reshape((ens.num_N,1))
                    outputContDecoded = cp.sum(outputCont * ens.decoder,0)
                    outputCont = cp.asnumpy(outputCont)
                    outputContDecoded = cp.asnumpy(outputContDecoded)
                    cp.cuda.Stream.null.synchronize()
                else:
                    ens.spikesCont = np.hstack((ens.spikesCont[:,1:],np.array(spikes,ndmin=2).T))
                    outputCont = np.sum(ens.mask * ens.spikesCont,1).reshape((ens.num_N,1))
                    outputContDecoded = np.sum(outputCont * ens.decoder,0)
                if e == 0:
                    error = outputContDecoded - np.array(inputNext)
                    prevFilt = np.array([[0.0] for i in range(ens.num_N)])
                else:
                    if learningWeights:
                        # for running words
                        error = outputContDecoded - np.array(inputs[ti])
                        #error = outputContDecoded - transferFunctionsWeights[e-1](self.ensembles[e-1].outputs[-1])
                        self.connections[e-1].updateWeights(learningWeights, prevFilt, self.t, error, outputCont)
                    else:
                        error = outputContDecoded - self.ensembles[e-1].outputs[-1]
                if learningEnsembles and e == 0:
                    ens.updateEnsemble(learningEnsembles, self.t, error, outputCont, inputNext)
    
                prevFilt = outputCont
                if e != (self.num_ens - 1): 
                    inputNext = np.dot(outputCont.T, self.connections[e].weights)
                #storing
                if self.t % self.measureStep <= self.dt:
                    ens.saveTimes.append(self.t)
                    ens.encoder = preprocessing.normalize(ens.encoder, norm = "l2")
                    ens.encoderLog.append(ens.encoder)
                    for l in ens.learning_options:
                        ens.trackDict[l].append(ens.tempTrack[l])
                        ens.tempTrack[l] = 0.0
                    if e > 0:
                        self.connections[e-1].saveTimes.append(self.t)
                        self.connections[e-1].nPES.append(self.connections[e-1].nPESind)
                        self.connections[e-1].nPESind = [0.0] * self.connections[e-1].pre.num_N
                        for l in self.connections[e-1].learning_options:
                            self.connections[e-1].trackDict[l].append(self.connections[e-1].tempTrack[l])
                            self.connections[e-1].tempTrack[l] = 0.0
                ens.times.append(self.t)
                ens.outputs.append(outputContDecoded[:])
                ens.errors.append(error)
# =============================================================================
#                 if self.t % self.updateStep <= self.dt and e == 0:
#                         ens.setDecoderLinearly()
#                         if GPU: ens.decoder = cp.array(ens.decoder)
# =============================================================================
            self.t += self.dt
        self.endTime = datetime.datetime.now()
            
            
    def plotNetworkSeperate(self, show = True, save = False, path = None):
        for e in range(self.num_ens):
            subPath = path + self.ensembles[e].ensPath
            if not os.path.exists(subPath):
                os.makedirs(subPath)
            self.ensembles[e].plotOutput(show, save, subPath)
            
    def plot2DEncoders(self, save = False, path = None):
        for e in range(self.num_ens):
            subPath = path + self.ensembles[e].ensPath
            if not os.path.exists(subPath):
                os.makedirs(subPath)
            if self.ensembles[e].encoderLog:
                self.ensembles[e].plot2DEncoders(save, subPath)
            
    def plotWeightChanges(self, show = True, save = False, path = None):
        if save and not path:
            sys.exit('Error: a path must be supplied')
        for c in range(len(self.connections)):
            subPath = path + 'Connection-' + self.connections[c].pre.name + '-' + self.connections[c].post.name + '/'
            if not os.path.exists(subPath):
                os.makedirs(subPath)
            self.connections[c].plotWeightChanges(show, save, subPath)
            
    def plotEnsembleChanges(self, show = True, save = False, path = None):
        if save and not path:
            sys.exit('Error: a path must be supplied')
        for e in range(self.num_ens):
            subPath = path + self.ensembles[e].ensPath
            if not os.path.exists(subPath):
                os.makedirs(subPath)
            self.ensembles[e].plotEnsembleChanges(show, save, subPath)
            
    def plotIndWeightChanges(self, wordList, dur, show = True, save = False, path = None):
        if save and not path:
            sys.exit('Error: a path must be supplied')
        for c in range(len(self.connections)):
            subPath = path + 'Connection-' + self.connections[c].pre.name + '-' + self.connections[c].post.name + '/'
            if not os.path.exists(subPath):
                os.makedirs(subPath)
            self.connections[c].plotIndWeightChanges(wordList, dur, show, save, subPath)
            
    def writeInformation(self, path, words, intercept, algs):
        lastEns = self.ensembles[-1]
        #data = {'Times' : lastEns.times}
        data = []
        data.append(lastEns.times)
        for dim in range(0,lastEns.dim):
            #data['Dimension ' + str(dim)] = np.transpose(lastEns.outputs)[dim]
            data.append(np.transpose(lastEns.outputs)[dim])
        FrameStack = np.empty((len(data),), dtype=np.object)
        for i in range(len(data)):
            FrameStack[i] = data[i]
        scipy.io.savemat(path + 'data.mat', {"FrameStack":FrameStack})
        
        with open(path + 'simulation_information.csv', 'w', newline = '') as f:
            writer = csv.writer(f, delimiter = ';')
            writer.writerow(['Simulation Data'])
            writer.writerow(['Date', self.date])
            writer.writerow(['Time', self.time])
            writer.writerow(['Runtime', str(self.endTime - self.startTime)])
            writer.writerow(['GPU', str(self.GPU)])
            writer.writerow([''])
            writer.writerow(['Number of ensembles', str(self.num_ens)])
            writer.writerow(['Total simulation time', str(self.total_time)])
            writer.writerow(['Timestep', str(self.dt)])
            writer.writerow(['Intercept', str(intercept)])
            writer.writerow([''])         
            writer.writerow(['Ensemble information'])
            writer.writerow(['Ensemble name', '# of neurons', '# of dimensions', 'Optimized decoders', 'Membrane RC time constant',
                             'Refractory period', 'Post-synaptic time constant', 'Encoder learning', 'Weight learning',
                             'PES learning rate', 'Voja learning rate',
                             'Voja2 learning rate', 'Voja2 bias percentage', 'Voja2RF learning rate', 'Voja 2RF-B learning rate'])
            for ens in self.ensembles:
                writer.writerow([ens.name, str(ens.num_N), str(ens.dim), ens.decOption, str(ens.t_rc),
                                 str(ens.t_ref), str(ens.t_pstc), algs[0], algs[1],
                                 str(ens.learning_rate_PES), str(ens.learning_rate_voja),
                                str(ens.learning_rate_voja2), str(ens.percBias), str(ens.learning_rate_voja2rf),
                                str(ens.learning_rate_voja2rfb)])
            writer.writerow([''])
            writer.writerow(['Vocabulary'])
            writer.writerow(words)
        return data
                
    def wordResults(self, path, d, testDesign, start, test_dur, pause_dur, step = 0.005, show = True, 
                    labelThreshold = -1.0, avgBracket = 0.1):
        subPath = path + "Results/"
        if not os.path.exists(subPath):
                os.makedirs(subPath)
        words = list(d.keys())
        wDict = {}
        sDict = {}
        for w in words:
            wDict[w] = 0.0
            sDict[w] = 0.0
        vectors = list(d.values())
        ivl = test_dur + pause_dur
        des = []
        avgDot = []
        voteCorrs = []
        avgInt = np.arange(int((test_dur - avgBracket) / step), int(test_dur / step) , 1)
        for w in testDesign:
            for w2, v in d.items():
                if np.sum(np.subtract(w,v)) == 0: 
                    des.append(w2)
                    break
        # plots
        for it in range(len(des)):
            inds = np.arange(int(start / self.dt) + int((it * ivl) / self.dt), int(start / self.dt) + int(((it+1) * ivl)/ self.dt) - 1, 
                             int(step/self.dt))
            outVals = [self.ensembles[-1].outputs[ix] for ix in inds]
            times = [self.ensembles[-1].times[ix] for ix in inds]
            dots = [[np.dot(v, out) for v in vectors] for out in outVals]
            avgDots = [sum(d[avgInt]) / len(avgInt) for d in np.transpose(dots)]
            if avgDots.index(max(avgDots)) == words.index(des[it]):
                voteCorrs.append(1)
                sDict[des[it]] += 1
            else:
                voteCorrs.append(0)
            wDots = np.transpose(dots)[words.index(des[it])]
            avg = sum(wDots[avgInt]) / len(avgInt)
            avgDot.append(avg)
            wDict[des[it]] += avg
            
            #plot           
            if len(words) < 4:
                ind = np.arange(0, len(words), 1)
            else: 
                ind = [avgDots.index(x) for x in sorted(avgDots)[-3:]]
            fig, ax = plt.subplots()
            ax.set_ylim((-1.1,1.1))
            for w in range(len(words)):
                lab = words[w] if w in ind else ""
                #lab = words[w] if any(x > labelThreshold for x in np.transpose(dots)[w]) else ""
                ax.plot(times, np.transpose(dots)[w], label = lab)
            ax.axhline(color = "black", ls = "--")
            ax.axvline(times[0] + test_dur, color = "red", ls = "--", label = "Stop learning")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Dot product between word and output vector")
            ax.set_title("Output similarity when presented with " + des[it])        
            ax.legend()
            file_name = 'Output-Similarity-' + des[it] + '-' + str(it+1)
            saveLoc = subPath + file_name
            if os.path.isfile(saveLoc):
                os.remove(saveLoc)
            plt.savefig(saveLoc)
            if show:
                plt.show()
            plt.close()
        # csv
        with open(subPath + 'results.csv', 'w', newline = '') as f:
            writer = csv.writer(f, delimiter = ';')
            writer.writerow(['Word learning results'])
            writer.writerow(['Test position', 'Word', 'Output similarity', 'Highest'])
            for it in range(len(des)):
                writer.writerow([it+1, des[it], "{:.6f}".format(avgDot[it]), voteCorrs[it]])
            writer.writerow([''])         
            writer.writerow(['Word', 'Average output similarity', 'Accuracy'])
            for w, avg in wDict.items():
                writer.writerow([w, str(avg / (len(testDesign) / len(wDict))), str(sDict[w] / (len(testDesign) / len(wDict)))])

    def wordPairResults(self, path, d, testDesign, testNames, vecs, start, test_dur, pause_dur, step = 0.005, show = True, 
                    labelThreshold = -1.0, avgBracket = 0.1):
        subPath = path + "Results/"
        if not os.path.exists(subPath):
                os.makedirs(subPath)
        words = list(d.keys())
        wDict = {}
        sDict = {}
        for w in testNames:
            wDict[w] = 0.0
            sDict[w] = 0.0
        vectors = list(d.values())
        ivl = test_dur + pause_dur
        avgDot = []
        voteCorrs = []
        avgInt = np.arange(int((test_dur - avgBracket) / step), int(test_dur / step) , 1)
        testSeqNames = []
        # plots
        for it in range(len(testDesign)):
            inds = np.arange(int(start / self.dt) + int((it * ivl) / self.dt), int(start / self.dt) + int(((it+1) * ivl)/ self.dt) - 1, 
                             int(step/self.dt))
            outVals = [self.ensembles[-1].outputs[ix] for ix in inds]
            times = [self.ensembles[-1].times[ix] for ix in inds]
            dots = [[np.dot(v, out) for v in vectors] for out in outVals]
            avgDots = [sum(d[avgInt]) / len(avgInt) for d in np.transpose(dots)]
        
            for i, x in enumerate(vecs):
                if np.array_equal(x, testDesign[it]):
                    pres = testNames[i]
                    ind = i
                    testSeqNames.append(pres)
                    break
            wpDots = [[np.dot(v, out) for v in vecs] for out in outVals]
            wpDot = np.transpose(wpDots)[ind]
            avg = sum(wpDot[avgInt]) / len(avgInt)
            avgDot.append(avg)
            if avgDots.index(max(avgDots)) == ind:
                voteCorrs.append(1)
                sDict[testDesign[it]] += 1
            else:
                voteCorrs.append(0)
            #wDict[testDesign[it]] += avg
            #plot           
            if len(words) < 6:
                ind = np.arange(0, len(words), 1)
            else: 
                ind = [avgDots.index(x) for x in sorted(avgDots)[-3:]]
            fig, ax = plt.subplots()
            ax.set_ylim((-1.1,1.1))
            for w in range(len(words)):
                lab = words[w] if w in ind else ""
                ax.plot(times, np.transpose(dots)[w], label = lab)
            ax.axhline(color = "black", ls = "--")
            ax.axvline(times[0] + test_dur, color = "red", ls = "--", label = "Stop learning")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Dot product between word pair and output vector")
            ax.set_title("Output similarity when presented with " + pres)        
            ax.legend()
            file_name = 'Output-Similarity-' + str(it+1)
            saveLoc = subPath + file_name
            if os.path.isfile(saveLoc):
                os.remove(saveLoc)
            plt.savefig(saveLoc)
            if show:
                plt.show()
            plt.close()
        # csv
        with open(subPath + 'results.csv', 'w', newline = '') as f:
            writer = csv.writer(f, delimiter = ';')
            writer.writerow(['Word pair learning results'])
            writer.writerow(['Test position', 'Word Pair', 'Output similarity', 'Highest'])
            for it in range(len(testDesign)):
                writer.writerow([it+1, testSeqNames[it], "{:.6f}".format(avgDot[it]), voteCorrs[it]])
            writer.writerow([''])         
            writer.writerow(['Word', 'Average output similarity', 'Accuracy'])
            for w, avg in wDict.items():
                writer.writerow([w, str(avg / (len(testDesign) / len(wDict))), str(sDict[w] / (len(testDesign) / len(wDict)))])
                
                
    def wpResults(self, path, vocab, w1, w2, wps, fans, testKeys, testVecs, 
                  start, test_dur, pause_dur, step = 0.005, avgBracket = 0.1):
        subPath = path + "Results/"
        if not os.path.exists(subPath):
            os.makedirs(subPath)
        ivl = test_dur + pause_dur
        avgInt = np.arange(int((test_dur - avgBracket) / step), int(test_dur / step) , 1)
        testAveragesPair = []
        testFans = []
        avgDotFans = []
        # remove ITEM1 and ITEM2 from vocab, not relevant, original vocab is still callable
        vocabEx = vocab.create_subset([x for x in list(vocab.keys()) if x not in ['ITEM1', 'ITEM2']])
        # vocab with unique words, unconvolved
        vocabWords = vocab.create_subset(list(set(w1 + w2))) 
        # vocab with unique words, convolved
        vocabConvWords = vocab.create_subset([x for x in list(vocabEx.keys()) if any(char.isdigit() for char in x)])
        # vocab with pairs of convolved words
        vocabPairs = vocab.create_subset([x + '_' + y for x, y in zip(w1, w2)])
        pairComparisons = {}
        for p in list(vocabPairs.keys()):
            pairComparisons[p] = []
        for i in range(len(testKeys)):
            pairName = testKeys[i]
            pairVec = testVecs[i]
            fan = fans[wps.index(pairName)]
            # indices of interest, times of interest, outputs of interest
            inds = np.arange(int(start / self.dt) + int((i * ivl) / self.dt), int(start / self.dt) + int(((i+1) * ivl)/ self.dt) - 1, 
                             int(step/self.dt))
            times = [self.ensembles[-1].times[ix] for ix in inds]
            out = [self.ensembles[-1].outputs[ix] for ix in inds]
            # dot product over entire interval with output and unconvolved unique words
            dotWordUnconvolved = [[np.dot(v, o) for v in vocabWords.vectors] for o in out]
            avgDotsWordUnconvolved = [sum(d[avgInt]) / len(avgInt) for d in np.transpose(dotWordUnconvolved)]
            self.plotDots(path = subPath + "Words_Unconvolved/", times = times, dots = dotWordUnconvolved,
                     avgDots = avgDotsWordUnconvolved, pairName = pairName, nameList = list(vocabWords.keys()), 
                     legCount = 4, testCount = i + 1, test_dur = test_dur)
            # dot product over entire interval with output and convolved unique words
            dotWordConvolved = [[np.dot(v, o) for v in vocabConvWords.vectors] for o in out]
            avgDotsWordConvolved = [sum(d[avgInt]) / len(avgInt) for d in np.transpose(dotWordConvolved)]
            self.plotDots(path = subPath + "Words_Convolved/", times = times, dots = dotWordConvolved,
                     avgDots = avgDotsWordConvolved, pairName = pairName, nameList = list(vocabConvWords.keys()), 
                     legCount = 4, testCount = i + 1, test_dur = test_dur)
            # dot product over entire interval with output and pairs of convolved words
            dotPair = [[np.dot(v, o) for v in vocabPairs.vectors] for o in out]
            avgDotsPair = [sum(d[avgInt]) / len(avgInt) for d in np.transpose(dotPair)]
            self.plotDots(path = subPath + "Pairs/", times = times, dots = dotPair,
                     avgDots = avgDotsPair, pairName = pairName, nameList = list(vocabPairs.keys()), 
                     legCount = 4, testCount = i + 1, test_dur = test_dur)
            avgTarget = sum([np.dot(pairVec, o) for o in out][min(avgInt):max(avgInt)]) / len(avgInt)
            testAveragesPair.append(avgTarget)
            
            # fan part
            testFans.append(fan)
            if fan == 2:
                otherPairs = [pair for pair in wps if any(w in pairName for w in pair)]
                otherPairs.remove(pairName)
                otherVecs = [vocab[p[0] + '_' + p[1]].v for p in otherPairs]
                otherDots = [sum([np.dot(v, o) for o in out][min(avgInt):max(avgInt)]) / len(avgInt) for v in otherVecs]
                avgDotFan = sum(otherDots) / len(otherDots)
                avgDotFans.append(avgDotFan)
            else:
                avgDotFans.append(0)       
            
            # collect dot comparisons to other pairs
            pairComparisons[pairName[0] + '_' + pairName[1]].append(avgDotsPair)
        
        # plot dot comparison
        for ic, curp in enumerate(list(vocabPairs.keys())):
            if fans[ic] == 2:
                otherPairs = [pair for pair in wps if any(w in wps[ic] for w in pair)]
                otherPairs.remove(wps[ic])
                fanNames = [p[0] + '_' + p[1] for p in otherPairs]
            else:
                fanNames = []
            fig, ax = plt.subplots()
            ax.set_ylim((-1.1, 1.1))
            for i, p in enumerate(list(vocabPairs.keys())):
                lineColor = 'red'
                if curp == p:
                    lineColor = 'green'
                elif p in fanNames:
                    lineColor = 'orange'
                ax.plot(np.arange(0,len(testKeys)/len(vocabPairs), 1), np.transpose(pairComparisons[curp])[i], label = p, color = lineColor)
            ax.set_xlabel("Test position")
            ax.set_ylabel("Average output similarity over last 100 msec of presentation")
            ax.set_title("Output similarities when presented with " + curp)
            ax.legend()
            file_name = 'Comparison-output-similarities-' + curp
            saveLoc = subPath + file_name
            if os.path.isfile(saveLoc):
                os.remove(saveLoc)
            plt.savefig(saveLoc)
            plt.close()
            
        # write csv
        with open(subPath + 'results.csv', 'w', newline = '') as f:
            writer = csv.writer(f, delimiter = ';')
            writer.writerow(['Word pair learning results'])
            writer.writerow([''])
            writer.writerow(['Tests'])
            writer.writerow(['Test position', 'Word Pair', 'Output similarity', 'Fan', 'Output similarity to fan pair'])
            for i, w in enumerate(testKeys):
                writer.writerow([i + 1, w[0] + '_' + w[1], testAveragesPair[i], testFans[i], avgDotFans[i]])
            
            
    def plotDots(self, path, times, dots, avgDots, pairName, nameList, legCount, testCount, test_dur):
        if not os.path.exists(path):
            os.makedirs(path)
        count = len(avgDots)
        if count < legCount - 1:
            leg_ind = np.arange(0, count, 1)
        else: 
            leg_ind = [avgDots.index(x) for x in sorted(avgDots)[-legCount:]]
        fig, ax = plt.subplots()
        ax.set_ylim((-1.1,1.1))
        for w in range(count):
            lab = nameList[w] if w in leg_ind else ""
            ax.plot(times, np.transpose(dots)[w], label = lab)
        ax.axhline(color = "black", ls = "--")
        ax.axvline(times[0] + test_dur, color = "red", ls = "--", label = "Stop presenting")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Dot product")
        ax.set_title("Output similarity when presented with " + pairName[0] + '_' + pairName[1])        
        ax.legend()
        file_name = 'Output-Similarity-' + str(testCount)
        saveLoc = path + file_name
        if os.path.isfile(saveLoc):
            os.remove(saveLoc)
        plt.savefig(saveLoc)
        plt.close()