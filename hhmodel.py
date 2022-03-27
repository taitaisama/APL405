
import matplotlib.pylab as plt
import math
import pickle
import time

def step (I, V, m, h, n, t) :

    if (t*1000-int(t*1000) != 0):
        print("enter values upto 4 decimal places")

    dt = 0.01 # resolution
    
    gNa = 120
    eNa = 115
    gK = 36  
    eK = -12
    gL = 0.3  
    eL = 10.6

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

print(alphaM(10))
print(alphaN(10))
print(alphaH(10))
print(betaM(10))
print(betaN(10))
print(betaH(10))

V = -75
m = 0.0255
h = 0.13
n = 0.68

Varr = []
marr = []
harr = []
narr = []
x = []
I = 10
time = 200000
for i in range(time) :
    harr.append(h)
    marr.append(m)
    narr.append(n)
    Varr.append(V)
    x.append(float(i/100))
    V, m, h, n = step(I, V, m, h, n, 0.01)
l3 = int((time*2)/3)

x_plt = x[0:1464]
V_plt = Varr[197562:197562+1464]
m_plt = marr[197562:197562+1464]
n_plt = narr[197562:197562+1464]
h_plt = harr[197562:197562+1464]
plt.plot(x_plt, V_plt)
plt.show()
plt.plot(x_plt, m_plt)
plt.show()
plt.plot(x_plt, h_plt)
plt.show()
plt.plot(x_plt, n_plt)
plt.show()
count = 0
maxi = Varr[l3]
mini = Varr[l3]
Vmins = []
mmins = []
nmins = []
hmins = []
for i in range(l3,time-1):
    if (Varr[i] < mini):
        mini = Varr[i]
    if (Varr[i] > maxi):
        maxi = Varr[i]
    if (Varr[i-1] > Varr[i] and Varr[i] < Varr[i+1]):
        print("V: ", i)
        Vmins.append(i)
        print("    V ", Varr[i])
        print("    m ", marr[i])
        print("    n ", narr[i])
        print("    h ", harr[i])
    if (marr[i-1] > marr[i] and marr[i] < marr[i+1]):
        print("m: ", i)
        mmins.append(i)
    if (narr[i-1] > narr[i] and narr[i] < narr[i+1]):
        print("n: ", i)
        nmins.append(i)
    if (harr[i-1] > harr[i] and harr[i] < harr[i+1]):
        print("h: ", i)
        hmins.append(i)

for i in range(len(Vmins)-1):
    print(Vmins[i+1] - Vmins[i])

for i in range(len(mmins)-1):
    print(mmins[i+1] - mmins[i])

for i in range(len(hmins)-1):
    print(hmins[i+1] - hmins[i])

for i in range(len(nmins)-1):
    print(nmins[i+1] - nmins[i])




print(count)

# print(Varr[197628], marr[197628], narr[197628], harr[197628])
# print(Varr[199091], marr[199091], narr[199091], harr[199091])


# file1 = open("Current-Amplitude.txt", "a")
# file1.write("I = " + str(I) + " max = " + str(maxi) + " min = " + str(mini) + "\n")
# file1.close()
# data = {"time": x, "Varr": Varr, "marr": marr, "narr": narr, "harr": harr}
# outfile = open("hhdata.pkl", "wb")
# pickle.dump(data, outfile)
# outfile.close()
# for i in range(100):
#     print(i*2, Varr[i*20], harr[i*20], marr[i*20], narr[i*20])
#     V_plt.append(Varr[i*20])
#     t_plt.append(i*2)
# # plt.plot(t_plt[0:20], V_plt[0:20])
# plt.plot(x, Varr)
# plt.show()

# V generally ranges from -80 to 30, so we will normalise it by (V+90)/110


