import random
import math
import numpy as np
import nengo
import pandas as pd

def generate_gain_and_bias(count, intercept_low, intercept_high, 
                           rate_low, rate_high, t_ref, t_rc):
    gain = []
    bias = []
    for i in range(count):
        intercept = random.uniform(intercept_low, intercept_high)
        rate = random.uniform(rate_low, rate_high)
        z = 1.0 / (1 - math.exp((t_ref - (1.0 / rate)) / t_rc))
        g = (1 - z) / (intercept - 1.0)
        b = 1 - g * intercept
        gain.append(g)
        bias.append(b)
    return gain, bias

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def run_neurons2(input, v, ref, dt, t_rc = 0.02, t_ref = 0.002):
    spikes = []
    for i, _ in enumerate(v):
        dV =  (input[i] - v[i])*(1-np.exp(-dt / t_rc))  # the LIF voltage change equation #without input this gets smaller
        #print(dV)
        v[i] += dV
        if v[i] < 0:
            v[i] = 0  # don't allow voltage to go below 0

        if ref[i] > 0:  # if we are in our refractory period
            v[i] = 0   # keep voltage at zero and
            ref[i] -= dt  # decrease the refractory period

        if v[i] > 1:  # if we have hit threshold
            spikes.append(True)  # spike
            v[i] = 0  # reset the voltage
            ref[i] = t_ref  # and set the refractory period
        else:
            spikes.append(False)
    return spikes, v, ref


def run_neurons(input, v, ref, dt, t_rc = 0.02, t_ref = 0.002):
    spikes = []
    for i, _ in enumerate(v):
        dV =  (input[i] - v[i])*(1-np.exp(-dt / t_rc))  # the LIF voltage change equation #without input this gets smaller
        v[i] += dV
        if v[i] < 0:
            v[i] = 0  # don't allow voltage to go below 0

        if ref[i] > 0:  # if we are in our refractory period
            v[i] = 0   # keep voltage at zero and
            ref[i] -= dt  # decrease the refractory period

        if v[i] > 1:  # if we have hit threshold
            spikes.append(True)  # spike
            v[i] = 0  # reset the voltage
            ref[i] = t_ref  # and set the refractory period
        else:
            spikes.append(False)
    return spikes


def compute_response(x, encoder, gain, bias, dt, v, time_limit=0.5):
    N = len(encoder)  # number of neurons
    #v = [0] * N  # voltage
    ref = [0] * N  # refractory period

    input = []
    for i in range(N):
        input.append(np.dot(x, encoder[i]) * gain[i] + bias[i])

    count = [0] * N  # spike count for each neuron
    
    t = 0
    while t < time_limit:
        spikes = run_neurons(input, v, ref, dt) 
        for i, s in enumerate(spikes):
            if s:
                count[i] += 1
        t += dt
    return [c / float(time_limit) for c in count]  # return the spike rate (in Hz)

def compute_tuning_curves(encoder, gain, bias, dt, N_samples = 25):
    x_values = [i * 2.0 / N_samples - 1.0 for i in range(N_samples)]

    # build up a matrix of neural responses to each input (i.e. tuning curves)
    A = []
    v = np.random.uniform(0, 1, len(encoder))  # randomize the initial voltage level

    for x in x_values:
        for y in x_values:
            response = compute_response(np.array((x,y)), encoder, gain, bias, dt, v)
            A.append(response)
    return np.array((x_values,x_values)), np.array(A)

def compute_decoder(encoder, gain, bias, dt, function=lambda x: x):
    x_values, A = compute_tuning_curves(encoder, gain, bias, dt)

    value_x = []
    for i in x_values[0]:
        for j in x_values[0]:
            value_x.append(np.array([function(i), function(j)]))
    # find the optimal linear decoder
    A = np.array(A).T
    Gamma = np.dot(A, A.T)
    Upsilon = np.dot(A, value_x)
    Ginv = np.linalg.pinv(Gamma)
    decoder = np.dot(Ginv.T, Upsilon) / dt
    return decoder

def compute_decoder_nengo(encoder, gain, bias, dt, seed = 0, rcond = 0.01):
    nengo.rc.set('progress', 'progress_bar', 'nengo.utils.progress.NoProgressBar')
    n_neurons, D = np.shape(encoder)
    with nengo.Network(seed = seed) as model:
        pre = nengo.Ensemble(n_neurons, D, seed = seed)
    with nengo.Simulator(model) as sim:
        pass
    pre_built  = sim.data[pre]
    Y = pre_built.eval_points
    x = np.dot(Y, encoder.T)
    A = pre.neuron_type.rates(x, gain, bias)
    decoder, res, rank, s = np.linalg.lstsq(A, Y, rcond = 0.01)
    return decoder / dt

def saveVectors(vocab, path, name):
    data = {}
    for key, value in vocab.items():
        data[key] = value.v
    df = pd.DataFrame(data, columns = list(data.keys()))
    df.to_csv(path + name + '.csv', sep = ';', index = False)
    
def saveOrder(order, path, name):
    df = pd.DataFrame(order)
    df.to_csv(path + name + '.csv', sep = ';', index = False, header = False)