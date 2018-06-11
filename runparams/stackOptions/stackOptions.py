import os

#specify the sample
sHandle = "MDR1_Main2"
#tracerFile = os.getenv('HOME')+"/Workspace/Sims/MultiDark/HOD/"+sHandle+"/"+sHandle+".mock" #currently not used!
useDM = False
dmFile = os.getenv('HOME')+"/Workspace/Sims/MultiDark/DarkMatter/MDR1_DM_sub512_z0.dat"

#structure catalogue to use
prefix = "ParisVoids"

#stack selection and binning choices
MaxDist = 2	#in units of the void radius, max distance for which profile is required
Nbins = 20
stackList = [
  {'stackName'  : "ParisV_cp_015D025_6R12",
   'differential': False,  #set True for differential density profile, False for cumulative
   'Rmin'	: 6.0,
   'Rmax'	: 12.0,
   'CentDensMin': 0.15,
   'CentDensMax': 0.25,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
  },

  {'stackName'  : "ParisV_cp_015D025_25R30",
   'differential': False,  #set True for differential density profile, False for cumulative
   'Rmin'	: 25.0,
   'Rmax'	: 30.0,
   'CentDensMin': 0.15,
   'CentDensMax': 0.25,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
  },

  {'stackName'  : "ParisV_cp_035D045_6R12",
   'differential': False,  #set True for differential density profile, False for cumulative
   'Rmin'	: 6.0,
   'Rmax'	: 12.0,
   'CentDensMin': 0.35,
   'CentDensMax': 0.45,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
  },

  {'stackName'  : "ParisV_cp_055D065_6R12",
   'differential': False,  #set True for differential density profile, False for cumulative
   'Rmin'	: 6.0,
   'Rmax'	: 12.0,
   'CentDensMin': 0.55,
   'CentDensMax': 0.65,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
  },

  {'stackName'  : "ParisV_cp_0075D0175_10R15",
   'differential': False,  #set True for differential density profile, False for cumulative
   'Rmin'	: 10.0,
   'Rmax'	: 15.0,
   'CentDensMin': 0.075,
   'CentDensMax': 0.175,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
  },

  {'stackName'  : "ParisV_cp_0075D0175_15R20",
   'differential': False,  #set True for differential density profile, False for cumulative
   'Rmin'	: 15.0,
   'Rmax'	: 20.0,
   'CentDensMin': 0.075,
   'CentDensMax': 0.175,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
  },

  {'stackName'  : "ParisV_cp_0075D0175_30R35",
   'differential': False,  #set True for differential density profile, False for cumulative
   'Rmin'	: 30.0,
   'Rmax'	: 35.0,
   'CentDensMin': 0.075,
   'CentDensMax': 0.175,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
  },

  {'stackName'  : "ParisV_cp_0075D0175_45R60",
   'differential': False,  #set True for differential density profile, False for cumulative
   'Rmin'	: 45.0,
   'Rmax'	: 60.0,
   'CentDensMin': 0.075,
   'CentDensMax': 0.175,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
  },

  {'stackName'  : "ParisV_cp_0D03_RQ1",
   'differential': False,  #set True for differential density profile, False for cumulative
   'Rmin'	: 6.3,
   'Rmax'	: 17.5,
   'CentDensMin': 0.0,
   'CentDensMax': 0.3,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
  },

  {'stackName'  : "ParisV_cp_0D03_RQ2",
   'differential': False,  #set True for differential density profile, False for cumulative
   'Rmin'	: 17.5,
   'Rmax'	: 22.2,
   'CentDensMin': 0.0,
   'CentDensMax': 0.3,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
  },

  {'stackName'  : "ParisV_cp_0D03_RQ1",
   'differential': False,  #set True for differential density profile, False for cumulative
   'Rmin'	: 22.2,
   'Rmax'	: 29.0,
   'CentDensMin': 0.0,
   'CentDensMax': 0.3,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
  },

  {'stackName'  : "ParisV_cp_0D03_RQ1",
   'differential': False,  #set True for differential density profile, False for cumulative
   'Rmin'	: 29.0,
   'Rmax'	: 548.0,
   'CentDensMin': 0.0,
   'CentDensMax': 0.3,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
  },

  {'stackName'  : "ParisV_dp_015D025_6R12",
   'differential': True,  #set True for differential density profile, False for cumulative
   'Rmin'	: 6.0,
   'Rmax'	: 12.0,
   'CentDensMin': 0.15,
   'CentDensMax': 0.25,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
  },

  {'stackName'  : "ParisV_dp_015D025_25R30",
   'differential': True,  #set True for differential density profile, False for cumulative
   'Rmin'	: 25.0,
   'Rmax'	: 30.0,
   'CentDensMin': 0.15,
   'CentDensMax': 0.25,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
  },

  {'stackName'  : "ParisV_dp_035D045_6R12",
   'differential': True,  #set True for differential density profile, False for cumulative
   'Rmin'	: 6.0,
   'Rmax'	: 12.0,
   'CentDensMin': 0.35,
   'CentDensMax': 0.45,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
  },

  {'stackName'  : "ParisV_dp_055D065_6R12",
   'differential': True,  #set True for differential density profile, False for cumulative
   'Rmin'	: 6.0,
   'Rmax'	: 12.0,
   'CentDensMin': 0.55,
   'CentDensMax': 0.65,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
  },

  {'stackName'  : "ParisV_dp_0075D0175_10R15",
   'differential': True,  #set True for differential density profile, False for cumulative
   'Rmin'	: 10.0,
   'Rmax'	: 15.0,
   'CentDensMin': 0.075,
   'CentDensMax': 0.175,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
  },

  {'stackName'  : "ParisV_dp_0075D0175_15R20",
   'differential': True,  #set True for differential density profile, False for cumulative
   'Rmin'	: 15.0,
   'Rmax'	: 20.0,
   'CentDensMin': 0.075,
   'CentDensMax': 0.175,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
  },

  {'stackName'  : "ParisV_dp_0075D0175_30R35",
   'differential': True,  #set True for differential density profile, False for cumulative
   'Rmin'	: 30.0,
   'Rmax'	: 35.0,
   'CentDensMin': 0.075,
   'CentDensMax': 0.175,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
  },

  {'stackName'  : "ParisV_dp_0075D0175_45R60",
   'differential': True,  #set True for differential density profile, False for cumulative
   'Rmin'	: 45.0,
   'Rmax'	: 60.0,
   'CentDensMin': 0.075,
   'CentDensMax': 0.175,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
  },

  {'stackName'  : "ParisV_dp_0D03_RQ1",
   'differential': True,  #set True for differential density profile, False for cumulative
   'Rmin'	: 6.3,
   'Rmax'	: 17.5,
   'CentDensMin': 0.0,
   'CentDensMax': 0.3,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
  },

  {'stackName'  : "ParisV_dp_0D03_RQ2",
   'differential': True,  #set True for differential density profile, False for cumulative
   'Rmin'	: 17.5,
   'Rmax'	: 22.2,
   'CentDensMin': 0.0,
   'CentDensMax': 0.3,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
  },

  {'stackName'  : "ParisV_dp_0D03_RQ1",
   'differential': True,  #set True for differential density profile, False for cumulative
   'Rmin'	: 22.2,
   'Rmax'	: 29.0,
   'CentDensMin': 0.0,
   'CentDensMax': 0.3,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
  },

  {'stackName'  : "ParisV_dp_0D03_RQ1",
   'differential': True,  #set True for differential density profile, False for cumulative
   'Rmin'	: 29.0,
   'Rmax'	: 548.0,
   'CentDensMin': 0.0,
   'CentDensMax': 0.3,
   'AvgDensMin'	: 0,
   'AvgDensMax'	: 100.0,
  },
]
