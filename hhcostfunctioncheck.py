import torch
import numpy as np
import matplotlib.pyplot as plt
import utilities
import pickle
import math

def customDerivative (x, t):
    x_dt = []
    for i in range(len(x) - 1):
        x_dt.append([(x[i]-x[i+1])/(t[i]-t[i+1])])
    x_dt.append([(x[-1]-x[-2])/(t[-1]-t[-2])])
    return x_dt

infile = open("hhdata.pkl", "rb")

data = pickle.load(infile)
t_all = data["time"]
V_all = data["Varr"]
h_all = data["harr"]
n_all = data["narr"]
m_all = data["marr"]
V_dt_all = customDerivative(V_all, t_all)
h_dt_all = customDerivative(h_all, t_all)
n_dt_all = customDerivative(n_all, t_all)
m_dt_all = customDerivative(m_all, t_all)

V_dt_less = []
h_dt_less = []
n_dt_less = []
m_dt_less = []

V_less = []
m_less = []
h_less = []
n_less = []
t_less = []

def alphaM(V) :
    return ((2.5 - 0.1*(V+65)) / (math.exp(2.5 - 0.1*(V+65)) - 1))

def alphaN(V) :
    return ((0.1 - 0.01*(V+65)) / (math.exp(1 - 0.1*(V+65)) - 1))

def alphaH(V) :
    return 0.07 * (math.exp(-(V+65)/20))

def betaM(V) :
    return 4.0*(math.exp(-(V+65)/18))

def betaN(V) :
    return 0.125*(math.exp(-(V+65)/80))

def betaH(V) :
    return 1.0/(math.exp(3.0-0.1*(V+65))+1)

alphaM_less = []
alphaN_less = []
alphaH_less = []
betaM_less = []
betaN_less = []
betaH_less = []

for i in range(0, len(V_all), 10):
    V_less.append(V_all[i])
    m_less.append(m_all[i])
    n_less.append(n_all[i])
    h_less.append(h_all[i])
    t_less.append(t_all[i]/10)
    V_dt_less.append(V_dt_all[i])
    h_dt_less.append(h_dt_all[i])
    n_dt_less.append(n_dt_all[i])
    m_dt_less.append(m_dt_all[i])
    
    alphaM_less.append(alphaM(V_all[i]))
    alphaN_less.append(alphaN(V_all[i]))
    alphaH_less.append(alphaH(V_all[i]))
    betaM_less.append(betaM(V_all[i]))
    betaN_less.append(betaN(V_all[i]))
    betaH_less.append(betaH(V_all[i]))

V_less = (np.array(V_less)+80)/110
# V_all = ((np.array(V_all) + 80)/ 110)

# plt.plot(t_less, V_less)
# plt.show()
# plt.plot(t_less, m_less)
# plt.show()
# plt.plot(t_less, h_less)
# plt.show()
# plt.plot(t_less, n_less)
# plt.show()
# plt.plot(t_less, V_dt_less)
# plt.show()
# plt.plot(t_less, m_dt_less)
# plt.show()
# plt.plot(t_less, h_dt_less)
# plt.show()
# plt.plot(t_less, n_dt_less)
# plt.show()

# plt.plot(alpham_less)
# plt.show()
# plt.plot(alphan_less)
# plt.show()
# plt.plot(alphah_less)
# plt.show()
# plt.plot(betam_less)
# plt.show()
# plt.plot(betan_less)
# plt.show()
# plt.plot(betah_less)
# plt.show()

def costFunction (V_pred_norm, h_pred, n_pred, m_pred):
    
    gNa = 120
    eNa = 115
    gK = 36  
    eK = -12
    gL = 0.3  
    eL = 10.6
        
    V_pred = (V_pred_norm * 110) - 80
    
    alphaM = ((2.5 - 0.1*(V_pred+65)) / (torch.exp(2.5 - 0.1*(V_pred+65)) - 1))
        
    alphaN = ((0.1 - 0.01*(V_pred+65)) / (torch.exp(1 - 0.1*(V_pred+65)) - 1))
    
    alphaH = 0.07 * (torch.exp(-(V_pred+65)/20))
        
    betaM = 4.0*(torch.exp(-(V_pred+65)/18))
        
    betaN = 0.125*(torch.exp(-(V_pred+65)/80))
        
    betaH = 1.0/(torch.exp(3.0-0.1*(V_pred+65))+1)
    

    # V_pred_dt = utilities.get_derivative(V_pred, self.t, 1)
    # h_pred_dt = utilities.get_derivative(h_pred, self.t, 1)
    # m_pred_dt = utilities.get_derivative(m_pred, self.t, 1)
    # n_pred_dt = utilities.get_derivative(n_pred, self.t, 1)
    I = 10

    V_pred_dt = torch.reshape(torch.tensor(V_dt_less)*10, [len(V_pred)])
    h_pred_dt = torch.reshape(torch.tensor(h_dt_less)*10, [len(V_pred)])
    n_pred_dt = torch.reshape(torch.tensor(n_dt_less)*10, [len(V_pred)])
    m_pred_dt = torch.reshape(torch.tensor(m_dt_less)*10, [len(V_pred)])

    plt.plot(t_less, V_pred_dt)
    # plt.show()
    # plt.plot(t_less, m_pred_dt)
    # plt.show()
    # plt.plot(t_less, h_pred_dt)
    # plt.show()
    # plt.plot(t_less, n_pred_dt)
    # plt.show()

    # plt.plot(t_less, alphaM)
    # plt.show()
    # plt.plot(t_less, alphaN)
    # plt.show()
    # plt.plot(t_less, alphaH)
    # plt.show()
    # plt.plot(t_less, betaM)
    # plt.show()
    # plt.plot(t_less, betaN)
    # plt.show()
    # plt.plot(t_less, betaH)
    # plt.show()

    V_dt_get = ((gK * (n_pred ** 4) * (eK - (V_pred + 65))) + (gNa * (m_pred ** 3) * h_pred * (eNa - (V_pred + 65))) + (gL * (eL - (V_pred + 65))) + I)
    m_dt_get = (alphaM * (1 - m_pred) - betaM * m_pred)
    n_dt_get = (alphaN * (1 - n_pred) - betaN * n_pred)
    h_dt_get = (alphaH * (1 - h_pred) - betaH * h_pred)

    # plt.plot(t_less, V_dt_get)
    # plt.show()
    # plt.plot(t_less, m_dt_get)
    # plt.show()
    # plt.plot(t_less, h_dt_get)
    # plt.show()
    # plt.plot(t_less, n_dt_get)
    # plt.show()
    # print(V_pred_dt.shape, V_dt_get.shape)
    # v_diff = (v_pred_dt - v_dt_get)
    # print(v_diff.shape)
    # v_diff_plt = np.array(v_diff)
    # plt.plot(t_less, v_diff_plt)
    
    plt.show()

    print(V_dt_get.shape, V_pred_dt.shape)
    
    print(torch.sum((V_pred_dt - \
                    ((gK * (n_pred ** 4) * (eK - (V_pred + 65))) + \
                     (gNa * (m_pred ** 3) * h_pred * (eNa - (V_pred + 65))) + \
                     (gL * (eL - (V_pred + 65))) + \
                     I)) ** 2))
    print(torch.sum((m_pred_dt - \
                    (alphaM * (1 - m_pred) - \
                     betaM * m_pred)) ** 2))
    print(torch.sum((n_pred_dt - \
                    (alphaN * (1 - n_pred) - \
                     betaN * n_pred)) ** 2))
    print(torch.sum((h_pred_dt - \
                    (alphaH * (1 - h_pred) - \
                    betaH * h_pred)) ** 2))
               

    
    equation_loss = \
        (torch.sum((V_pred_dt - \
                    ((gK * (n_pred ** 4) * (eK - (V_pred + 65))) + \
                     (gNa * (m_pred ** 3) * h_pred * (eNa - (V_pred + 65))) + \
                     (gL * (eL - (V_pred + 65))) + \
                     I)) ** 2) + \
         torch.sum((m_pred_dt - \
                    (alphaM * (1 - m_pred) - \
                     betaM * m_pred)) ** 2) + \
         torch.sum((n_pred_dt - \
                    (alphaN * (1 - n_pred) - \
                     betaN * n_pred)) ** 2) + \
         torch.sum((h_pred_dt - \
                    (alphaH * (1 - h_pred) - \
                    betaH * h_pred)) ** 2))
               
    return equation_loss

print(costFunction(torch.tensor(V_less), torch.tensor(h_less), torch.tensor(n_less), torch.tensor(m_less)))
