# -*- coding: utf-8 -*-
import numpy as np
import random

m11=(1,1)
m01=(2,-1)
m10=(-1,2)
m00=(0,0)

altruism1=-3
altruism2=-3

decay1=0.4
decay2=0.4

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

def decision(altruism,decay,side,hb):
    pb1=next_probability(hb,decay)
    pb0=1-pb1
    
    #expected gain by changing from cooperate to cheat in both situation
    ea=(m01[0]-m11[0])*pb1+(m00[0]-m10[0])*pb0
    #expected gain of opponent
    pb0=max(0.15,pb0)#still set a lower boundary to avoid divided by 0, and more than 0.001 to avoid overflow exponent
    cooperate_factor=pb1/pb0
    #consider opponent gain by changing from cheat to cooperate, with altruism and probability variable
    eb=(m11[1]-m01[1])*pb0+(m10[1]-m00[1])*pb1+altruism+cooperate_factor
    #softmax to acquire probability
    pa1=np.exp(eb)/(np.exp(eb)+np.exp(ea))
    a=1
    if random.random()>pa1:
        a=0
    return a,pa1
    
    

h1=[]
h2=[]
p1s=[]
p2s=[]

n=400

for i in range(n):
    decision1,p1=decision(altruism1,decay1,0,h2)
    decision2,p2=decision(altruism2,decay2,1,h1)
    h1.append(decision1)
    h2.append(decision2)
    p1s.append(p1)
    p2s.append(p2)
print(h1,h2)

import matplotlib.pyplot as plt
plt.plot(h1,"r")
plt.plot(h2,"b")
plt.plot(p1s,"r--")
plt.plot(p2s,"b--")
plt.ylim(-0.1,1.1)
plt.show()

