import pickle

import matplotlib.pylab as plt
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as pl


with open('MF_sim.bin', 'rb') as fp:
    data = pickle.load(fp)

item = data[0]
t = item['t']
LSfe = item['LSfe']
LSfi = item['LSfi']
LSw = item['LSw']


if 1:
	plt.figure()

	plt.plot(t, LSfe)
	plt.plot(t, LSfi)
	plt.plot(t, LSw)
elif 0:
	plt.plot(LSfe, LSfi)
else:
	fig=plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection = '3d')
	ax.plot(LSfe, LSfi, LSw)
	plt.figure()
	plt.plot(LSfe,LSfi)
	plt.figure()
	plt.plot(t, LSw)
plt.show()
