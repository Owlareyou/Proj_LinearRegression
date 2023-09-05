# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #data habdling tool


#%% import data
#data (number, years of experience, salary)

salarydata = pd.read_csv('Salary_dataset.csv') #(30,3)



#%% data visualization
#salarydata.plot()
salary = salarydata.iloc[:,2]
yoe = salarydata.iloc[:,1]
candidate = salarydata.iloc[:,0]
#can comback and make it look prettier, but this is not the objective
#we can learn how to beutify plots with plt later



#%% functions and algorithms

# 0.initialization
    #initialize w and b, f() = wx + b
    #random; np.random.default_rng(seed=100)
init_w = 10
init_b = 10
iteration = 100
alpha = 0.001

w = init_w
b = init_b

# 1.lost function, squared error loss
# 2.cost function, empircal risk
    #vector calcultaion
def model_prediction(w,x,b):
    return w*x + b

def square_error_loss(w,x,b,y):
    prediction_value = model_prediction(w, x, b)
    sqe = np.square(y - prediction_value)
    
    return sqe

def cost_function(w,x,b,y, N):
    sqe = square_error_loss(w, x, b,y)
    cost = (1/N) * sum(sqe)
    
    return cost
#prediction = yoe * init_w + init_b
#meansquareerror = (1/candidate.shape[0])*(sum(np.square(salary - prediction)))
#print(meansquareerror)

# 3.Gradient Descent
w = init_w
b = init_b

#this works because i only have one feature, need adjustmesnt moving forward
#dl_dw = (1/candidate.shape[0]) * sum((-2)*(yoe)*(salary - prediction))
#dl_db = (1/candidate.shape[0]) * sum((-2)*(salary - prediction))
def update_wb(w,b,alpha):
    salary = salarydata.iloc[:,2]
    yoe = salarydata.iloc[:,1]
    candidate = salarydata.iloc[:,0]
    
    prediction = model_prediction(w, yoe, b)
    
    dl_dw = (1/candidate.shape[0]) * np.sum((-2)*(yoe)*(salary - prediction))
    dl_db = (1/candidate.shape[0]) * np.sum((-2)*(salary - prediction))
    
    w = w - alpha * dl_dw
    b = b - alpha * dl_db
    
    return w, b

#challenger = update_wb(w, b, alpha)
def train():
    return



cont_cost = []
cont_i = []
for i in range(iteration):
    print("\niteration:", i)
    w, b = update_wb(w, b, alpha)
    modely = model_prediction(w, yoe, b)
    if i % 2 == 0:
        #iteration 1~100
        cost = cost_function(w, yoe, b, salary, 30)
        plt.figure()
        plt.xlim([-5,101])
        plt.ylim([10,30])
        
        cost = np.log(cost)
        cont_cost.append(cost)
        cont_i.append(i) 
        
        plt.suptitle('average cost through time', fontsize=16, style = "italic",color='orange')
        plt.xlabel('Iterations')
        plt.ylabel('log(cost)')
        plt.plot(cont_i,cont_cost, marker = 'o', color = 'orange', markersize = 3.5)
        
        
        '''
        plt.figure()
        plt.ylim([5000,140000])
        plt.plot(yoe, salary, 'r')
        plt.plot(yoe, modely, 'b')    
        '''
    print(modely)

#pltx = np.linspace(2, 100, num=50)
#plt.plot(pltx,cont_cost)

#plt.plot(10,10, marker="o", markersize=20, markeredgecolor="red", markerfacecolor="green")
    






#%%
#testing cell




#%% data visualization

spyder console color
current directory
"no such file found, salary dataset"
pd.read_csv()
- 10 minutes of pandas

ipython vs spyder
ipython is a shell

change shell color
ipython profile for color configuration
need to set profile and use them