# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 14:04:35 2017

@author: cranial
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 10:54:26 2017

@author: cranial
"""
import numpy
from numpy import genfromtxt
import pandas as pd
from sklearn import svm
import csv
import os

"""extracts file path"""

rootmkt = '/Users/amansinghthakur/Flask/login/templates/csvfiles'
filemkt = os.listdir(rootmkt)
dataroot = os.path.join(rootmkt, filemkt[1])
print dataroot
# """ Function will be given sliced arrays"""
data = pd.read_csv('/Users/amansinghthakur/Flask/login/templates/csvfiles/accNSE.csv')
dataset = genfromtxt(dataroot, delimiter=',')
siz = dataset.shape[0] - 150
print dataset.shape

"""Moving Average"""


def CalcMA(x):
    return numpy.average(x)


"""Weighted 10 day Moving average calculator"""


def CalcWavg(x):
    w = numpy.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    w = w.reshape((10, 1))
    x = w * x
    return numpy.average(x)


"""Calculates momentum"""


def CalcMom(x):
    return (x[0, 0] - x[9, 0])


"""Stochastic K%"""


def CalcK(x):
    x = numpy.reshape(x, (x.shape[0], 1))
    m = numpy.amin(x)
    M = numpy.amax(x)
    K = (x[0, 0] - m) / float((M - m))
    return K * 100


"""Stochastic D%"""


def CalcD(K, end=siz):
    D = numpy.zeros((0, 0))
    for i in range(K.shape[0]):
        if i < end:
            k = K[i:i + 10, 0]
            D = numpy.append(D, numpy.average(k))
    return D


"""William R%"""


def CalcR(x, period=10):
    m = numpy.amin(x)
    M = numpy.amax(x)
    return ((M - x[0, 0]) / (M - m)) * (-100)


"""Commodity Channel Index"""


# calculates typical price

def Truncate(x, beg=siz + 1):
    x = numpy.delete(x, numpy.s_[beg:], 0)
    x = numpy.reshape(x, (beg, 1))
    return x


def CalcDt(x, y):
    Dt = abs(x - y)
    Dt = numpy.reshape(Dt, (Dt.shape[0], 1))
    return Dt


def CalcCCI(x, period=10):
    TP = numpy.zeros((0, 0))
    Tavg = numpy.zeros((0, 0))
    avg = numpy.zeros((0, 0))
    Tsma = numpy.zeros((0, 0))
    prices = x[1:, 2:5]
    """Calculating TP"""
    for i in range(prices.shape[0]):
        Tavg = (prices[i, 0] + prices[i, 1] + prices[i, 2]) / 3
        TP = numpy.append(TP, Tavg)
    TP = numpy.reshape(TP, (TP.shape[0], 1))
    """Calculating TPSma"""
    for i in range(TP.shape[0]):
        sma = TP[i:i + period, 0]
        avg = numpy.average(sma)
        Tsma = numpy.append(Tsma, avg)
    Tsma = numpy.reshape(Tsma, (Tsma.shape[0], 1))
    """Calculating CCI"""
    Dt = 0.015 * CalcDt(TP, Tsma)
    Truncate(TP)
    Truncate(Tsma)
    Truncate(Dt)
    diff = TP - Tsma
    return diff / Dt


"""Accumulation/Distribution Oscillator (Inumpyut:High,Low,Current)"""


def CalcAD(x):
    prices = x[1:, 2:5]
    ADO = numpy.zeros((0, 0))
    for i in range(prices.shape[0]):
        if i < siz + 1:
            # if (prices[i,0] - prices[i,1]) != 0:
            AD = (prices[i, 0] - prices[i + 1, 2]) / (prices[i, 0] - prices[i, 1])
            # else:
            #    AD = (prices[i,0] - prices[i+1,2])/(0.00000001)
            ADO = numpy.append(ADO, AD)
    return ADO


"""Relative Strength index RSI"""


def CalcAvgs(x):
    change = numpy.zeros((x.shape[0], 1))
    gain = numpy.zeros((0, 0))
    loss = numpy.zeros((0, 0))
    for i in range(x.shape[0]):
        if i < x.shape[0] - 1:
            change[i, 0] = x[i, 0] - x[i + 1, 0];
        if change[i, 0] >= 0:
            gain = numpy.append(gain, change[i, 0])
            gain = gain.reshape((gain.shape[0], 1))
        else:
            loss = numpy.append(loss, change[i, 0])
            loss = loss.reshape((loss.shape[0], 1))
    return gain, loss


def CalcRSI(x, period=14):
    RSI = numpy.zeros((0, 0))
    for i in range(x.shape[0]):
        if i < 1844:
            param = x[i:i + period, 0, None]
            g, l = CalcAvgs(param)
            avgG = numpy.average(g)
            avgL = numpy.average(l)
            RS = avgG / avgL
            mid = 100 - (100 / RS)
            RSI = numpy.append(RSI, mid)
            RSI = numpy.reshape(RSI, (RSI.shape[0], 1))
    return RSI


"""Creating Dataset for the predictive model"""

ClosingP = dataset[1:, 4, None]
prices = dataset[1:, 2:5]
MA = numpy.zeros((0, 0))
WMA = numpy.zeros((0, 0))
K = numpy.zeros((0, 0))
R = numpy.zeros((0, 0))
for i in range(ClosingP.shape[0]):
    Inumpyut = ClosingP[i:i + 10, 0]
    MA = numpy.append(MA, CalcMA(Inumpyut))
    WMA = numpy.append(WMA, CalcWavg(Inumpyut))
    K = numpy.append(K, CalcK(Inumpyut))
    R = numpy.append(R, Inumpyut)
    # TP = CalcTP(Inumpyut)
    # TPsma = CalcTPsma(Inumpyut)
    # Dt = CalcDt(Inumpyut)
Rsi = CalcRSI(ClosingP)
"""Reshaping the returned arrays"""
Inumpyut = numpy.reshape(Inumpyut, (Inumpyut.shape[0], 1))
MA = numpy.reshape(MA, (MA.shape[0], 1))
WMA = numpy.reshape(WMA, (WMA.shape[0], 1))
K = numpy.reshape(K, (K.shape[0], 1))
R = numpy.reshape(R, (R.shape[0], 1))
"""-----------------------------------------------------"""
CCI = CalcCCI(dataset)
ADO = CalcAD(dataset)
D = CalcD(K, end=1855)
"""-----------Truncating every parameter to same size---------------"""
MAf = Truncate(MA)
WMAf = Truncate(WMA)
Kf = Truncate(K)
Rf = Truncate(R)
ADOf = Truncate(ADO)
Df = Truncate(D)
RSIf = Truncate(Rsi)

Y = numpy.zeros((0, 0))
for i in range(ClosingP.shape[0] - 1):
    if ClosingP[i, 0] >= ClosingP[i + 1, 0]:
        Y = numpy.append(Y, 1)
    else:
        Y = numpy.append(Y, 0)
Y = numpy.reshape(Y, (Y.shape[0], 1))
print MAf.shape, WMAf.shape, Kf.shape, Rf.shape, ADO.shape, Df.shape, RSIf.shape
features = numpy.concatenate([MAf, WMAf, Kf, Rf, ADOf, Df, RSIf], 1)
Y = Truncate(Y, beg=siz)
features = numpy.delete(features, (0), axis=0)
print features.shape, Y.shape
FinalData = numpy.concatenate([features, Y], 1)

numpy.savetxt('finalData.csv', FinalData, delimiter=',')

X_train = features
Y_train = Y
X_test = features[0, :]
Y_test = Y[0, :]
