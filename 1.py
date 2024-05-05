# -*- coding: utf-8 -*-
import numpy as np
import random

#opponents behavior
#110000000000111111111101010101
opponent_action=[1]*5+[0]*10+[1]*10+[0,1]*5

#first number is whether self cooperate or cheat, second is opponent's

#evolution of trust version
m11=(1,1)
m01=(2,-1)
m10=(-1,2)
m00=(0,0)

#evolution of cooperation version
# m11=(3,3)
# m10=(0,5)
# m01=(5,0)
# m00=(1,1)

#zero sum
# m11=(0,0)
# m01=(1,-1)
# m10=(-1,1)
# m00=(0,0)

#deterrence
# m11=(0,0)
# m01=(10,-100)
# m10=(-100,-100)
# m00=(-100,-100)

#in exponent, -1 means evaluate other as 1/e of self
#experimental value is about -3 to -5
altruism=-3


#memory decay rate per round, to simulate the forgiveness for past
#correspond to the active towards change
#should be about 0.5 to 1
decay=0.4

#decay rate=1 and altruism=-infinity is tit-for-tat strategy

#input a list and decay rate, return the probability for next decision is cooperate
def next_probability(l,decay):
    if len(l)==0:
        return 1
    a=0
    b=0
    factor=1
    for i in l[::-1]:
        a+=i*factor
        b+=factor
        factor*=(1-decay)
    return a/b
    
#variable naming
#p: probability
#e: expect
#h: history
#a: self
#b: opponent
#1: cooperate
#0: cheat

ha=[]#history of self decisions
hb=[]#history of opponent decisions
pa1s=[]#probability of self cooperate
pb1s=[]#probability of estimated opponent cooperate
i=0
for b in opponent_action:
    i+=1
    # print(i)
    hb.append(b)
    
    pb1=next_probability(hb,decay)
    pb0=1-pb1
    pb1s.append(pb1)
    pa1=next_probability(ha,decay)
    pa0=1-pa1
    
    #expected gain by changing from cooperate to cheat in both situation
    ea=(m01[0]-m11[0])*pb1+(m00[0]-m10[0])*pb0
    #expected gain of opponent
    pb0=max(0.01,pb0)#still set a lower boundary to avoid divided by 0, and more than 0.001 to avoid overflow exponent
    cooperate_factor=pb1/pb0
    #consider opponent gain by changing from cheat to cooperate, with altruism and probability variable
    eb=(m11[1]-m01[1])*pb0+(m10[1]-m00[1])*pb1+altruism+cooperate_factor
    print(ea,eb)
    #softmax to acquire probability
    pa1=np.exp(eb)/(np.exp(eb)+np.exp(ea))
    pa0=1-pa1
    pa1s.append(pa1)
    print(pa1)
    
    a=1
    if random.random()>pa1:
        a=0
    ha.append(a)
    

import matplotlib.pyplot as plt
plt.plot(pa1s,"k-")
plt.plot(pb1s,"--")
for i in range(len(ha)):
    if ha[i]:
        plt.plot(i,-0.05,"g.")
    else:
        plt.plot(i,-0.05,"r.")
for i in range(len(hb)):
    if hb[i]:
        plt.plot(i-0.5,pa1s[i],"g.")
    else:
        plt.plot(i-0.5,pa1s[i],"r.")
# plt.plot()
plt.ylim(-0.1,1.1)
plt.show()
    