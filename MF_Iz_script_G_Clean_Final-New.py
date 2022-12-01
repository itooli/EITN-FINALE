import numpy as np
import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
import sys
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as pl
from mpmath import *
#import plotly.graph_objects as go
from typing import Tuple, Iterable
#from plotly.subplots import make_subplots
#import plotly.express as px
import pandas as pd
import scipy.special as sp_spec




gizi=0.04 
giz=0.01
Eiz=-65.
Eizi=-60.
Ee=0.0
Ei=-80.0
Qi=5.0
Qe=1.5
tauize=1
tauizi=1
tau_adap=1
a_adap=1
d_adap=15
b_adap=0
tause=5e-3
tausi=5e-3
Ke=8000*0.05
Ki=2000*0.05
ue=0
T=0.005

def sv(ne,ni,ue,giz,Eiz):
    muge=Qe*ne*tause*Ke
    mugi=Qi*ni*tause*Ki

    ui=0
 

    muv=((2*giz*Eiz+muge+mugi+b_adap)-np.sqrt((2*giz*Eiz+muge+mugi+b_adap)**2-4*giz*(giz*Eiz**2+muge*Ee+mugi*Ei-ue)))/(2*giz)
    

    ae=giz*(muv-Eiz)**2
    ai=giz*(muv-Eiz)**2
 
    be=Qe*(Ee-muv)#/mug  Note that we are not using mug!!!
    bi=Qi*(Ei-muv)#/mug
    ce=tause
    ci=tausi
    
    argsv=Ke*ne*(2*ae*be*ce**3/tauize**2+ce**3*be**2/(8*tauize)+be**2*ce**3/(8*tauize**2)) + Ki*ni*(2*ai*bi*ci**3/tauizi**2+ci**3*bi**2/(8*tauizi)   +bi**2*ci**3/(8*tauizi**2))

    if argsv>0:
    	sv=np.sqrt(Ke*ne*(2*ae*be*ce**3/tauize**2+ce**3*be**2/(8*tauize)+be**2*ce**3/(8*tauize**2)) + Ki*ni*(2*ai*bi*ci**3/tauizi**2+ci**3*bi**2/(8*tauizi)+bi**2*ci**3/(8*tauizi**2)))

    else:
    	sv=1e-9
    
    
    return sv
    
def tauv(ne,ni,ue,giz,Eiz):
    
    ne=ne+1e-9
    ni=ni+1e-9
    muge=Qe*ne*tause*Ke
    mugi=Qi*ni*tause*Ki
    ui=0
    
    muv=((2*giz*Eiz+muge+mugi+b_adap)-np.sqrt((2*giz*Eiz+muge+mugi+b_adap)**2-4*giz*(giz*Eiz**2+muge*Ee+mugi*Ei-ue)))/(2*giz)
     

    ae=giz*(muv-Eiz)**2
    ai=giz*(muv-Eiz)**2
    be=Qe*(Ee-muv)
    bi=Qi*(Ei-muv)
    ce=tause
    ci=tausi
  
    
    argv=Ke*ne*(2*ae*be*ce**3/tauize**2+ce**3*be**2/(8*tauize)+be**2*ce**3/(8*tauize**2)) + Ki*ni*(2*ai*bi*ci**3/tauizi**2+ci**3*bi**2/(8*tauizi)+bi**2*ci**3/(8*tauizi**2))
    if argv >0:
    	svv2=Ke*ne*(2*ae*be*ce**3/tauize**2+ce**3*be**2/(8*tauize)+be**2*ce**3/(8*tauize**2)) + Ki*ni*(2*ai*bi*ci**3/tauizi**2+ci**3*bi**2/(8*tauizi)+bi**2*ci**3/(8*tauizi**2))
    
    	tauv=0.5*(Ke*ne*(be**2*ce**4/(2*np.pi*tauize**2))+Ki*ni*(bi**2*ci**4/(2*np.pi*tauizi**2)))/(svv2+1e-9)
        
    else:
      tauv=1
      
    return tauv



def TF(nu,P,ue,giz,Eiz):

    ne,ni=nu
    Po,Pmuv,Psv,Ptauv,Pvsv,Pvtauv,Psvtauv,Pvv,Ptt,Pss=P
    

 
    muge=Qe*ne*tause*Ke
    mugi=Qi*ni*tause*Ki
    mug=muge+mugi+giz
    ui=0

    ### Please note that the following parameters have changed
    muvo=-35
    dmuvo=25
    svo=0.7
    dsvo=1
    tauvo=0.0015
    dtauvo=0.003

    
    muv=((2*giz*Eiz+muge+mugi)-np.sqrt((2*giz*Eiz+muge+mugi+b_adap)**2-4*giz*(giz*Eiz**2+muge*Ee+mugi*Ei-ue)))/(2*giz)
    #muv=muve

 
    Pscale=1.
    noutf=Pscale*(sp_spec.erfc(((Po + Pmuv*(muv-muvo)/dmuvo + Psv*(sv(ne,ni,ue,giz,Eiz)-svo)/dsvo + Ptauv*(tauv(ne,ni,ue,giz,Eiz)-tauvo)/dtauvo \
            +Pvsv*(muv-muvo)*(sv(ne,ni,ue,giz,Eiz)-svo)/(dsvo*dmuvo) + Pvtauv*(muv-muvo)*(tauv(ne,ni,ue,giz,Eiz)-tauvo)/(dtauvo*dmuvo)\
            +Psvtauv*(sv(ne,ni,ue,giz,Eiz)-svo)*(tauv(ne,ni,ue,giz,Eiz)-tauvo)/(dtauvo*dsvo) + Pvv*(muv-muvo)**2/(dmuvo*dmuvo) \
            +Ptt*(tauv(ne,ni,ue,giz,Eiz)-tauvo)**2/(dtauvo*dtauvo) + Pss*(sv(ne,ni,ue,giz,Eiz)-svo)**2/(dsvo*dsvo))-muv)/(np.sqrt(2)*sv(ne,ni,ue,giz,Eiz)) ))/(2*tauv(ne,ni,ue,giz,Eiz))
 
        
    return noutf













f = open('RS_fit_new.txt', 'r')
lines = f.readlines()


PRS=np.zeros(10)

for i in range(0,len(PRS)):
    PRS[i]=lines[i]

f.close()


f = open('FS_fit_new.txt', 'r')
lines = f.readlines()


PFS=np.zeros(10)


for i in range(0,len(PRS)):
    PFS[i]=lines[i]

f.close()




tfinal=5.
dt=0.0001

t = np.linspace(0, tfinal, int(tfinal/dt))

f = open('time.txt', 'wb')

external_input=10.
fecont=2;
ficont=5;
dRS=15
aRS=1
w=fecont*dRS/aRS


LSw=[]
LSfe=[]
LSfi=[]




for i in range(len(t)):
    
    fecontold=fecont
    fecont+=dt/T*(TF((fecont+external_input,ficont), PRS,w,giz,Eiz)-fecont) 
    w+=dt*( -aRS*w+(dRS)*fecontold)
    ficont+=dt/T*(TF((fecont+external_input,ficont),PFS,0,gizi,Eizi) - ficont) 
    LSfe.append(float(fecont))
    LSfi.append(float(ficont))
    LSw.append(float(w))
    


print('fe=',fecont)
print('fi=',ficont)

plt.figure()

plt.plot(t, LSfe)
plt.plot(t, LSfi)
plt.plot(t, LSw)

#fig=plt.figure()
#plt.plot(LSfe, LSfi)
'''
ax = fig.add_subplot(1, 1, 1, projection = '3d')
ax.plot(LSfe, LSfi, LSw)
plt.figure()
plt.plot(LSfe,LSfi)
plt.figure()
plt.plot(t, LSw)
'''
plt.show()

#f.close()
