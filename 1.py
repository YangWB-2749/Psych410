# -*- coding: utf-8 -*-
import numpy as np
import random

#opponents behavior
opponent_action=[1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,1,0,1]

#first number is whether self cooperate or cheat, second is opponent's

m11=(1,1)
m01=(2,-1)
m10=(-1,2)
m00=(0,0)

# m11=(3,3)
# m10=(0,5)
# m01=(5,0)
# m00=(1,1)


altruism=0.2
egoist=1-altruism


decay=0.25#memory decay rate per round, to simulate the forgiveness for past

#list to store all cooperate probabilities
p_cooperates=[]

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

ha=[]
hb=[]
i=0
for b in opponent_action:
    i+=1
    # print(i)
    hb.append(b)
    
    pb1=next_probability(hb,decay)
    pb0=1-pb1
    
    pa1=next_probability(ha,decay)
    pa0=1-pa1
    
    #expected gain by changing from cooperate to cheat in both situation
    ea=abs((m10[0]-m11[0])*pb1+(m00[0]-m01[0])*pb0)+egoist
    #expected gain of opponent
    pb0=max(0.01,pb0)#still set a lower boundary to avoid divided by 0
    et=abs((m10[1]-m11[1])*pb1+(m00[1]-m01[1])*pb0)+altruism+(pb1/pb0)
    # print(ea,et)
    #softmax to acquire probability
    pa1=np.exp(et)/(np.exp(et)+np.exp(ea))
    pa0=1-pa1
    p_cooperates.append(pa1)
    print(pa1)
    
    a=1
    if random.random()>pa1:
        a=0
    ha.append(a)
    
print(p_cooperates)

import matplotlib.pyplot as plt
plt.plot(p_cooperates)
plt.show()
    