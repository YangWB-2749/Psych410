# -*- coding: utf-8 -*-
import numpy as np
import random

#opponents behavior
#1100000000001111111111010101010101010101
opponent_action=[1]*5+[0]*10+[1]*10+[0,1]*10

# opponent_action=[1]*10+[0]+[1]*5+[0,0,0]+[1]*10+[0]*5

#first number is whether self cooperate or cheat, second is opponent's

#evolution of trust version
# m11=(1,1)
# m01=(2,-1)
# m10=(-1,2)
# m00=(0,0)

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

#deterrer
# m11=(0,0)
# m01=(-100,-100)
# m10=(-100,10)
# m00=(-100,-100)

#deteree
# m11=(0,0)
# m01=(10,-100)
# m10=(-100,-100)
# m00=(-100,-100)

#another version
#INFERRING STRATEGIES FROM OBSERVATIONS IN LONG ITERATED PRISONER’S DILEMMA EXPERIMENTS
#https://arxiv.org/pdf/2202.04171v1
m11=(3,3)
m01=(4,0)
m10=(0,4)
m00=(1,1)

#standardize the matrix
matrix=np.array([[m00,m01],[m10,m11]])
norm=(matrix-matrix.mean())/matrix.std()

m11=norm[1,1]
m01=norm[0,1]
m10=norm[1,0]
m00=norm[0,0]


#in exponent, -1 means evaluate other as 1/e of self
#experimental value is about -3 to -5
altruism=-3#-3


#memory decay rate per round, to simulate the forgiveness for past
#correspond to the active towards change
#should be about 0.3 to 0.8
decay=0.3

#decay rate=1 and altruism=-infinity is tit-for-tat strategy

#input a list and decay rate, return the probability for next decision is cooperate
def next_probability(l,decay):
    l=[1]+l
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

for b in opponent_action:
    hb.append(b)
    
    pb1=next_probability(hb,decay)
    pb0=1-pb1
    pb1s.append(pb1)
    
    #expected gain by changing from cooperate to cheat in both situation
    ea=(m01[0]-m11[0])*pb1+(m00[0]-m10[0])*pb0
    
    #expected gain of opponent
    pb0=max(0.25,pb0)#avoid overflow exponent
    #trust between 0 and 3 with 0.25 above
    trust=pb1/pb0#pb1*4#
    #consider opponent gain by changing from cheat to cooperate, with altruism and probability variable
    eb=(m11[1]-m01[1])*pb0+(m10[1]-m00[1])*pb1+altruism+trust
    
    #softmax to acquire probability
    pa1=np.exp(eb)/(np.exp(eb)+np.exp(ea))
    pa1s.append(pa1)
    
    #decide action based on random value
    ha.append(random.random()<pa1)
    
import matplotlib.pyplot as plt
#black line, probability to cooperate
plt.plot(pa1s,"k-")
#dash line, expected opponent cooperate probability
plt.plot(pb1s,"--")

for i in range(len(opponent_action)):
    if ha[i]:
        plt.plot(i,pa1s[i],"gx")
    else:
        plt.plot(i,pa1s[i],"rx")
        
    if hb[i]:
        plt.plot(i-0.5,pa1s[i],"g2")
    else:
        plt.plot(i-0.5,pa1s[i],"r1")
plt.ylim(-0.1,1.1)
plt.show()
