
import matplotlib.pylab as plt
import math
import pickle

def step (I, V, m, h, n, t) :

    if (t*1000-int(t*1000) != 0):
        print("enter values upto 4 decimal places")

    dt = 0.001 # resolution
    
    gNa = 120
    eNa=115
    gK = 36  
    eK=-12
    gL=0.3  
    eL=10.6

    looptimes = t / dt
    for i in range(int(looptimes)) :
        V1 = V + dt*((gNa * pow(m, 3) * h * (eNa - (V+65))) + (gK * pow(n, 4) * (eK - (V+65))) + (gL * (eL - (V + 65))) + I)
        m1 = m + dt*(alphaM(V)*(1-m) - betaM(V)*m)
        h1 = h + dt*(alphaH(V)*(1-h) - betaH(V)*h)
        n1 = n + dt*(alphaN(V)*(1-n) - betaN(V)*n)
        V = V1
        m = m1
        h = h1
        n = n1
        
    return (V, m, h, n)

    
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

V = -65
m = 0.5
h = 0.06
n = 0.5
Varr = []
marr = []
harr = []
narr = []
x = []
for i in range(10000) :
    harr.append(h)
    marr.append(m)
    narr.append(n)
    Varr.append(V)
    x.append(float(i/500))
    V, m, h, n = step(10, V, m, h, n, 0.002)
plt.plot(x, Varr)
plt.show()
V_plt = []
t_plt = []
data = {"time": x, "Varr": Varr, "marr": marr, "narr": narr, "harr": harr}
outfile = open("hhdata.pkl", "wb")
pickle.dump(data, outfile)
outfile.close()
# for i in range(100):
#     print(i*2, Varr[i*20], harr[i*20], marr[i*20], narr[i*20])
#     V_plt.append(Varr[i*20])
#     t_plt.append(i*2)
# # plt.plot(t_plt[0:20], V_plt[0:20])
# plt.plot(x, Varr)
# plt.show()
