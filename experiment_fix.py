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

#standardize the matrix
matrix=np.array([[m00,m01],[m10,m11]])
norm=(matrix-matrix.mean())/matrix.std()

m11=norm[1,1]
m01=norm[0,1]
m10=norm[1,0]
m00=norm[0,0]

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

#input a list and decay rate, return the probability for next decision is cooperate
def next_probability(l,decay):
    # l=[1]+l
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
    # pb0=max(0.25,pb0)#avoid overflow exponent
    #trust between 0 and 3 with 0.25 above
    trust=pb1*4#pb1/pb0#
    #consider opponent gain by changing from cheat to cooperate, with altruism and probability variable
    eb=(m11[1]-m01[1])*pb0+(m10[1]-m00[1])*pb1+altruism+trust
    
    #softmax to acquire probability
    pa1=np.exp(eb)/(np.exp(eb)+np.exp(ea))
    return pa1

#Find the best result by tuning hyperparameter of the model to fit in the input distribution
def model_best(response,action):
    altruism=0
    decay=0
    min_error=sum(action)
    #perform grid search on altruism
    for d in range(0,21):#search from 0 to 1 with interval of 0.05
        d/=20
        for a in range(-100,60):#search from -10 to 5 with interval of 0.1
            a/=10
            acc_error=0
            for i in range(2**history_length):
                #if such pattern appear less than threshold in the action, go next loop
                if action[i]==0:#<threshold:
                    continue
                h=[int(j) for j in bin(i)[2:].zfill(history_length)]
                p=decision(a,d,h)
                
                acc_error+=action[i]*abs(p-(response[i]/action[i]))
            #compare distribution to real response and action
            if acc_error<min_error:
                min_error=acc_error
                decay=d
                altruism=a
    # print(altruism,decay)
    return altruism,decay,min_error

def analyze(action_player,action_opponent):
    #count the occurance of each pattern in history_length
    #number for 0000, 0001, 0010, 0011, ... , 1110, 1111
    #Given self action, the probability the opponent would cooperate
    frequency_action_self_cooperate=[0]*(2**history_length)
    frequency_action_self_received=[0]*(2**history_length)
    #jump over several initial warm up rounds
    for i in range(history_length,100):
        action=action_opponent[i-history_length-1:i-1]
        #convert bit 8421 to integer
        index=0
        for a in action:
            index*=2
            index+=a
        frequency_action_self_received[index]+=1
        #if player select cooperate
        if action_player[i]==1:
            frequency_action_self_cooperate[index]+=1
    distribution=[]
    for i in range(2**history_length):
        # print(bin(i)[2:].zfill(history_length),frequency_action_self_cooperate[i],"/",frequency_action_self_received[i])
        if frequency_action_self_received[i]:
            distribution.append(frequency_action_self_cooperate[i]/frequency_action_self_received[i])
        else:
            distribution.append(-1)
    # print(distribution)
    estimated_altruism,estimated_decay,estimated_error=model_best(frequency_action_self_cooperate,frequency_action_self_received)
    error=estimated_error/(100-history_length)
    # print(estimated_altruism,estimated_decay,estimated_error/90)
    return error


#running all entries in competition
errors=[]
counter=0
for competition in competitions:
    # if counter>50:
    #     break
    action_player1=[i=="C" for i in competition["action_player"]]
    action_player2=[i=="C" for i in competition["action_opponent"]]
    
    errors.append(analyze(action_player1,action_player2))
    errors.append(analyze(action_player2,action_player1))
    # if errors[-2]>0.25 or errors[-1]>0.25:
    #     print(counter)
    counter+=1
    
print("average error",sum(errors)/len(errors))
#display error distribution
plt.hist(errors)
plt.xlabel("error")
plt.ylabel("number")
plt.show()
#print max error index
# print(max(errors),errors.index(min(errors)))


'''
#display the worst case in index 22
competition=competitions[22]
action_player1=[i=="C" for i in competition["action_player"]]
action_player2=[i=="C" for i in competition["action_opponent"]]

frequency_action_self_cooperate=[0]*(2**history_length)
frequency_action_self_received=[0]*(2**history_length)
#jump over several initial warm up rounds
for i in range(history_length,100):
    action=action_player2[i-history_length-1:i-1]
    #convert bit 8421 to integer
    index=0
    for a in action:
        index*=2
        index+=a
    frequency_action_self_received[index]+=1
    #if player select cooperate
    if action_player1[i]==1:
        frequency_action_self_cooperate[index]+=1
distribution=[]
for i in range(2**history_length):
    print(bin(i)[2:].zfill(history_length),frequency_action_self_cooperate[i],"/",frequency_action_self_received[i])
    if frequency_action_self_received[i]:
        distribution.append(frequency_action_self_cooperate[i]/frequency_action_self_received[i])
    else:
        distribution.append(-1)
print(distribution)
# plt.imshow([distribution[i:i+4] for i in range(0,16,4)])
estimated_altruism,estimated_decay,estimated_error=model_best(frequency_action_self_cooperate,frequency_action_self_received)
error=estimated_error/(100-history_length)
print(error)
altruism,decay,min_error=model_best(distribution,[100]*16)
print(altruism,decay)
model_p=[]
print("{:<16s}    {:<6}{:<6s}    {:<6s}    {:<6}".format("opponent action","n","actual p","model p","error n"))
for i in range(2**history_length):
    h=[int(j) for j in bin(i)[2:].zfill(history_length)]
    p=decision(altruism,decay,h)
    model_p.append(p)
    n=frequency_action_self_received[i]
    if frequency_action_self_received[i]==0:
        ac=0
    else:
        ac=frequency_action_self_cooperate[i]/frequency_action_self_received[i]
    error1=n*abs(ac-p)
    print("{:<16s}    {:<6}{:<6f}    {:<6f}    {:<6f}".format(str(h),n,ac,p,error1))
# plt.imshow([model_p[i:i+4] for i in range(0,16,4)])
'''

'''
#self experimental data
actual=[0,25,25,50,25,50,50,75,25,50,50,75,50,75,75,100]
actual=[1, 40, 20, 40, 10, 20, 45, 80, 5, 40, 60, 30, 10, 80, 90, 99]#not mine
# actual=[5,60,40,80,30,60,45,90,20,55,45,70,35,65,50,99]#mine
# plt.imshow([actual[i:i+4] for i in range(0,16,4)])

altruism,decay,min_error=model_best(actual,[100]*16)
print(altruism,decay)
ps=[]
print("{:<16s}    {:<6}      {:<6}".format("opponent action","data","model"))
for i in range(2**history_length):
    h=[int(j) for j in bin(i)[2:].zfill(history_length)]
    p=decision(altruism,decay,h)
    ps.append(p)
    print("{:<16s}    {:<6}    {:<6f}".format(str(h),actual[i]/100,p))
plt.imshow([ps[i:i+4] for i in range(0,16,4)])
print(sum(ps)/1600)
'''