import os

#specify the sample
sHandle = "MDR1_LRGD"
#tracerFile = os.getenv('HOME')+"/Workspace/Sims/MultiDark/HOD/"+sHandle+"/"+sHandle+".mock" #currently not used!
useDM = False
dmFile = os.getenv('HOME')+""

#structure catalogue to use
prefix = "ParisVoids"

#stack selection and binning choices
MaxDist = 2	#in units of the void radius, max distance for which profile is required
Nbins = 20
stackList = [
  {'stackName'  : "ParisV_dp_0D03_RQ1",
   'differential': True,  
   'Rmin'	: 37.7,
   'Rmax'	: 60.65,
   'CentDensMin': 0.,
   'CentDensMax': 0.3,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
   'zoneMax'	: 0,
  },

  {'stackName'  : "ParisV_dp_0D03_RQ2",
   'differential': True,  
   'Rmin'	: 60.65,
   'Rmax'	: 69.48,
   'CentDensMin': 0.,
   'CentDensMax': 0.3,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
   'zoneMax'	: 0,
  },

  {'stackName'  : "ParisV_dp_0D03_RQ3",
   'differential': True,  
   'Rmin'	: 69.48,
   'Rmax'	: 81.13,
   'CentDensMin': 0.,
   'CentDensMax': 0.3,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
   'zoneMax'	: 0,
  },

  {'stackName'  : "ParisV_dp_0D03_RQ4",
   'differential': True,  
   'Rmin'	: 81.13,
   'Rmax'	: 128.5,
   'CentDensMin': 0.,
   'CentDensMax': 0.3,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
   'zoneMax'	: 0,
  },

]
