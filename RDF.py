#from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))
#%matplotlib inline
#%matplotlib auto
#from IPython.display import display, Math, Latex

import os
import re
import csv
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.pyplot import figure
from matplotlib.backends.backend_pdf import PdfPages
import seaborn
import pandas as pd    #used for exporting array to csv file

from scipy.spatial import ConvexHull
from   scipy.interpolate import interp1d
import random

import ase.io
import ase.neighborlist
from   ase import atom

print(os.getcwd())

ESO='./PM/ESO_PM1.vasp'
RS='./PM/rocksalt1.vasp'
TN='./PM/CuO.tenorite.vasp'
WU='./PM/ZnO.wurtzite.vasp'
figOut='./RDF'
legends=['ESO-PM1','Rock salt','Tenorite','Wurtzite']
factor=[1,16,4,4]

searchDist=10
nBins=1000
olddistance=2.4
sigma=2*(searchDist/nBins) ##for gaussian smoothing

SMALL_SIZE = 20
MEDIUM_SIZE = 24
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
  

rcParams['axes.linewidth']   = 3
rcParams['figure.figsize']   = 6,6
rcParams['figure.dpi']       = 300
#rcParams['mathtext.fontset'] = 'custom'
#rcParams['mathtext.rm'] = "Times New Roman"
#rcParams['mathtext.it'] = "Times New Roman"

rcParams['xtick.major.size']  = 8
rcParams['xtick.major.width'] = 3
rcParams['xtick.minor.size']  = 3
rcParams['ytick.major.size']  = 8
rcParams['ytick.major.width'] = 3
rcParams['ytick.minor.size']  = 3
#rcParams['text.latex.preamble'] = ["\\usepackage{amsmath}"] 


rcParams['text.usetex']      = 'false'
rcParams['font.family']      = 'serif'
#rcParams['font.serif']       = 'Times New Roman'

rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm']      = 'serif'
rcParams['mathtext.it']      = 'serif:italic'
rcParams['mathtext.bf']      = 'serif:bold'
rcParams['mathtext.default'] = 'it'

markers=['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
colors=['b','g','r','c','m','y','k','b','g','r','c','m']
lines=['solid',(0, (1,1)),(0, (5,1)),(0, (3, 2, 1, 2)),(0, (3, 2, 1, 2, 1, 2))]

def gaussianSmoothing(x_vals,y_vals):
    y_smth = np.zeros(y_vals.shape) 
    for it in range(0,len(x_vals)):
        x_position      = x_vals[it]
        gaussian_kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
        gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
        y_smth[it]      = np.sum(y_vals * gaussian_kernel)
    return(y_smth)
    
def getRDF(CONTCARfileName):
    system             = ase.io.read(CONTCARfileName,format="vasp")
    firstatom          = ase.neighborlist.neighbor_list('i', system, searchDist)
    secondatom         = ase.neighborlist.neighbor_list('j', system, searchDist)
    distance           = ase.neighborlist.neighbor_list('d', system, searchDist)
    zeroAppendDistance = np.append(distance,[0])
    totalH, bin_edges  = np.histogram(zeroAppendDistance, bins=nBins)
    bin_centres        = (bin_edges[1:] + bin_edges[:-1]) / 2
    totalRDF           = totalH/(4*np.pi/3*(bin_edges[1:]**3 - bin_edges[:-1]**3)) * system.get_volume()/len(system)
    totalRDF[0]        = 0
    return(bin_centres,totalRDF)
    
for it, val in enumerate([ESO,RS,TN,WU]):
    bin_centres,totalRDF=getRDF(val)
    smoothTotalRDF=gaussianSmoothing(bin_centres,totalRDF)
    #plt.plot(np.transpose(bin_centres),np.transpose(totalRDF))
    plt.plot(np.transpose(bin_centres),np.transpose(smoothTotalRDF)*factor[it],label=legends[it],color=colors[it],linestyle=lines[it])
plt.tick_params(labeltop=False, labelright=False, bottom=True, top=True, left=True, right=True, axis='both',direction='in')
plt.legend(frameon=0)
plt.xlim(1.5,5)
plt.ylim(-100,1500)
plt.ylabel('$g(r)$ [arb. units]');
plt.xlabel('$\mathit{r}$ [$\mathrm{\AA}$]')
#plt.show()
plt.savefig(figOut+'.pdf',transparent=True, bbox_inches='tight', pad_inches=0)
plt.savefig(figOut+'.png',transparent=True, bbox_inches='tight', pad_inches=0)
