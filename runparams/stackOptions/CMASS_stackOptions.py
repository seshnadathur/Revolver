import os
import numpy as np
from scipy.stats.mstats import mquantiles

#specify the sample
sHandle = "CMASS"
useDM = True
dmFile = "/store/erebos/snadathur/BigMDPl_Delta2350_flat_g1_z052.bin"
DMres = 2350
usePhi = True
PhiFile = "/store/erebos/snadathur/BigMDPl_Phi1175_flat_g1_z052.bin"
Phires = 1175

#structure catalogue to use
prefix = "IsolatedVoids"

#stack selection and binning choices
MaxDist = 3.0  #in units of the void radius, max distance for which profile is required
maxrange = 160.0
Nbins = 30	#number of bins
differential = True  #set True for differential density profile, False for cumulative
useVTFE = True

catFile = "/z/snadathur/structures/BigMDPl/"+sHandle+"/"+prefix+"_info.txt"
catalogue = np.loadtxt(catFile,skiprows=2)
shortprefix = prefix.replace("Voids",'V')

#profiles are calculated for the voids in the stacks specified by stackList
#set stackList equal to one of the examples below or choose your own criteria

#custom specification   
CustomStack = [
  {'stackName'  : shortprefix+"_root_0.0A0.8", 
   'Rmin'	: 0, 'Rmax' : max(catalogue[:,6])+0.1,
   'CentDensMin': 0,'CentDensMax' : 1.0,
   'AvgDensMin'	: 0, 'AvgDensMax' : 0.8,
   'zoneMax'	: 0,
  },
  {'stackName'  : shortprefix+"_root_0.8A0.9", 
   'Rmin'	: 0, 'Rmax' : max(catalogue[:,6])+0.1,
   'CentDensMin': 0,'CentDensMax' : 1.0,
   'AvgDensMin'	: 0.8, 'AvgDensMax' : 0.9,
   'zoneMax'	: 0,
  },
  {'stackName'  : shortprefix+"_root_0.9A1.0", 
   'Rmin'	: 0, 'Rmax' : max(catalogue[:,6])+0.1,
   'CentDensMin': 0,'CentDensMax' : 1.0,
   'AvgDensMin'	: 0.9, 'AvgDensMax' : 1.0,
   'zoneMax'	: 0,
  },
  {'stackName'  : shortprefix+"_root_1.0A1.1", 
   'Rmin'	: 0, 'Rmax' : max(catalogue[:,6])+0.1,
   'CentDensMin': 0,'CentDensMax' : 1.0,
   'AvgDensMin'	: 1.0, 'AvgDensMax' : 1.1,
   'zoneMax'	: 0,
  },
  {'stackName'  : shortprefix+"_root_0.0A1.1", 
   'Rmin'	: 0, 'Rmax' : max(catalogue[:,6])+0.1,
   'CentDensMin': 0,'CentDensMax' : 1.0,
   'AvgDensMin'	: 0.0, 'AvgDensMax' : 1.1,
   'zoneMax'	: 0,
  },
  {'stackName'  : shortprefix+"_root_0.0A1.0", 
   'Rmin'	: 0, 'Rmax' : max(catalogue[:,6])+0.1,
   'CentDensMin': 0,'CentDensMax' : 1.0,
   'AvgDensMin'	: 0.0, 'AvgDensMax' : 1.0,
   'zoneMax'	: 0,
  },
  {'stackName'  : shortprefix+"_root_1.1A5.0", 
   'Rmin'	: 0, 'Rmax' : max(catalogue[:,6])+0.1,
   'CentDensMin': 0,'CentDensMax' : 1.0,
   'AvgDensMin'	: 1.1, 'AvgDensMax' : 5,
   'zoneMax'	: 0,
  }
]

#set stackList here
stackList = CustomStack
