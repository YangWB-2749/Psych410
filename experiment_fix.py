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

#the opponent action history length that will be used to predict next. 
history_length=4
#minimum number of pattern that would be considered as valid
#for example, if 0000 appears for 9 times, and 0001 appears for 10 times, and 10 is threshold
#then only that for 0001 would be considered
threshold=5

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
def decision(altruism,decay,hb):
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

#Find the best result by tuning hyperparameter of the model to fit in the input distribution
def model_best(response,action):
    altruism=0
    decay=0
    min_error=10000
    #perform grid search on altruism
    for d in range(1,10):
        d/=10
        for a in range(-100,50):
            a/=10
            acc_error=0
            for i in range(2**history_length):
                #if such pattern appear less than threshold in the action, go next loop
                if action[i]<threshold:
                    continue
                h=[int(j) for j in bin(i)[2:].zfill(history_length)]
                p=decision(a,d,h)
                
                acc_error+=action[i]*abs(p-(response[i]/action[i]))
                distribution.append(p)
            #compare distribution to real response and action
            if min_error>acc_error:
                min_error=acc_error
                decay=d
                altruism=a
    return altruism,decay,min_error



for competition in competitions:
    action_player=[i=="C" for i in competition["action_player"]]
    action_opponent=[i=="C" for i in competition["action_opponent"]]
    
    #count the occurance of each pattern in history_length
    #number for 0000, 0001, 0010, 0011, ... , 1110, 1111
    #Given self action, the probability the opponent would cooperate
    frequency_action_opponent_cooperate=[0]*(2**history_length)
    frequency_action_self=[0]*(2**history_length)
    #jump over several initial warm up rounds
    for i in range(10,100):
        action=action_player[i-history_length:i-1]
        #convert bit 8421 to integer
        index=0
        for a in action:
            index*=2
            index+=a
        frequency_action_self[index]+=1
        if action_opponent[i]==1:
            frequency_action_opponent_cooperate[index]+=1
    distribution=[]
    for i in range(2**history_length):
        print(bin(i)[2:].zfill(history_length),frequency_action_opponent_cooperate[i],"/",frequency_action_self[i])
        if frequency_action_self[i]:
            distribution.append(frequency_action_opponent_cooperate[i]/frequency_action_self[i])
        else:
            distribution.append(-1)
    print(distribution)
    accuracy=model_best(frequency_action_opponent_cooperate,frequency_action_self)
    print(accuracy)
    
    break
    
    
    # print(action_player)
    # print(action_opponent)
    # for i in range(100):
    #     print(i,action_player[i],action_opponent[i])
    # plt.plot(action_player[1:])
    # # change variables for better fit
    # expected_action=[decision(-3,0.3,action_opponent[1:i]) for i in range(1,100)]
    # plt.plot(expected_action)
    # plt.show()
    # # ways to evaluate prediction accuracy
    # break
