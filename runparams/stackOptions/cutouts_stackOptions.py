import os
import numpy as np
from scipy.stats.mstats import mquantiles

#specify the sample
sHandle = "CMASSlike_pp"
useDM = True
dmFile = "/store/erebos/snadathur/BigMDPl_Delta2350_flat_z052.bin"
DMres = 2350
usePhi = True
PhiFile = "/store/erebos/snadathur/BigMDPl_Phi1175_flat_z052.bin"
Phires = 1175

#structure catalogue to use
prefix = "Voids"

#stack selection and binning choices
MaxDist = 3.0  #in units of the void radius, max distance for which profile is required
maxrange = 180.0
Nbins = 30	#number of bins
differential = True  #set True for differential density profile, False for cumulative
useVTFE = False

catFile = "/z/snadathur/structures/BigMDPl/"+sHandle+"/"+prefix+"_info.txt"
catalogue = np.loadtxt(catFile,skiprows=2)
#shortprefix = prefix.replace("Voids",'V')
shortprefix = sHandle.replace("CMASSlike_","")+"_"+prefix

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
