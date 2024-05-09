import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#INFERRING STRATEGIES FROM OBSERVATIONS IN LONG ITERATED PRISONER’S DILEMMA EXPERIMENTS
#https://arxiv.org/pdf/2202.04171v1
#https://datadryad.org/stash/dataset/doi:10.5061/dryad.37pvmcvmk
m11=(3,3)
m01=(4,0)
m10=(0,4)
m00=(1,1)

df=pd.read_csv("./fix.csv",sep=";")

competitions=[]
competition_num=df.shape[0]//100
for i in range(competition_num):
    competitions.append(df[100*i:100*i+100])
# print(competitions)

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
def decision(altruism,decay,side,hb):
    pb1=next_probability(hb,decay)
    pb0=1-pb1
    
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
    return pa1

for competition in competitions:
    action_player=[i=="C" for i in competition["action_player"]]
    action_opponent=[i=="C" for i in competition["action_opponent"]]
    print(action_player)
    print(action_opponent)
    for i in range(100):
        print(i,action_player[i],action_opponent[i])
    plt.plot(action_player[1:])
    #change variables for better fit
    expected_action=[decision(-3,0.3,0,action_opponent[1:i]) for i in range(1,100)]
    plt.plot(expected_action)
    plt.show()
    #ways to evaluate prediction accuracy
        
    break
