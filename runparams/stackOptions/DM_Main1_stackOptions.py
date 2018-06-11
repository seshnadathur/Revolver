import os
import numpy as np
from scipy.stats.mstats import mquantiles

#specify the sample
sHandle = "MDR1_DM_Main1"
useDM = True
dmFile = "/Users/seshadri/Workspace/Sims/MultiDark/MDR1_Delta_512.npy"
DMres = 512
usePhi = False
PhiFile = ""
Phires = 0

#structure catalogue to use
prefix = "Voids"

#stack selection and binning choices
MaxDist = 3.0  #in units of the void radius, max distance for which profile is required
maxrange = 60.0
Nbins = 50	#number of bins
differential = True  #set True for differential density profile, False for cumulative
useVTFE = False

catFile = "/Users/seshadri/Workspace/structures/MDR1/"+sHandle+"/"+prefix+"_info.txt"
catalogue = np.loadtxt(catFile,skiprows=2)
#shortprefix = prefix.replace("Voids",'V')
shortprefix = "Voids"

#profiles are calculated for the voids in the stacks specified by stackList
#set stackList equal to one of the examples below or choose your own criteria

#custom specification   
CustomStack = [
  {'stackName'  : shortprefix+"_all", 
   'Rmin'	: 0, 'Rmax' : max(catalogue[:,6])+0.1,
   'CentDensMin': 0,'CentDensMax' : 1.0,
   'AvgDensMin'	: 0, 'AvgDensMax' : max(catalogue[:,5])+0.1,
   'zoneMax'	: 0,
  },
]

#set stackList here
stackList = CustomStack
