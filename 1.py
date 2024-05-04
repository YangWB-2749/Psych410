# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:57:14 2024

@author: caiy
"""

import matplotlib.pyplot as plt

history=[0,1]
history_len=10

feedbacks=[0,0,0,0,0,1,1,1,0,0,0,0]#opponents behavior


matrix=[[[1,1],[2,-1]],[[-1,2],[0,0]]]
matrix=[[[3,3],[0,5]],[[5,0],[1,1]]]
matrix=[[[0,0],[1,-1]],[[-1,1],[0,0]]]

egoist=0.2
altruism=0.8


for i in range(len(feedbacks)):
    feedback=feedbacks[i]
    history.append(feedback)
    print(history)
    p=0
    power=-1
    for h in history[::-1]:
        p+=h*2**power
        power-=1
    print(feedback,p)


ls=[]
initial=0.5
for f in feedbacks:
    if f:
        initial=1-(1-initial)/2
    else:
        initial/=2
    print(f,initial)
    ls.append(initial)
plt.plot(ls)
plt.show()
    