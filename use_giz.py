# python -i use_giz.py

import pickle

with open('sim_giz.bin', 'rb') as fp:
    data = pickle.load(fp)




# add code here

giz_list = [item['giz'] for item in data]

rates_exc = [item['popRateG_exc'][-100:].mean() for item in data]
rates_u = [item['Pu'][-100:].mean() for item in data]
rates_inh = [item['popRateG_inh'][-100:].mean() for item in data]
#for giz, re,ri,ru  in zip(giz_list, rates_exc,rates_inh, rates_u):
   # print(f"for giz={giz}: rate={rate}")


import matplotlib.pylab as plt

if 1:
    x = data[0]['TimBinned']
    plt.plot(x, data[0]['Pu'])
    plt.plot(x, data[0]['popRateG_inh'])
    plt.plot(x, data[0]['popRateG_exc'])
else:
    plt.plot(giz_list, rates_u, color= 'r')
    plt.plot(giz_list, rates_exc, color='b')
    plt.plot(giz_list, rates_inh, color='g')
plt.show()
