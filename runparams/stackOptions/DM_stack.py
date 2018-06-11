import os
import numpy as np
import glob

# specify paths and simulation box properties
Ngrid = 512
boxLen = 100.
dmFile = '/Users/seshadri/Workspace/Sims/MultiDark/MDR1/MDR1_Delta_512.npy'

# stack selection and binning choices
scaled_profiles = False
MaxDist = 120.0  # if using scaled profiles, max units of void radius; else, max distance in absolute units
Nbins = 50	#number of bins

# list of structures to stack
filelist = glob.glob('/Users/seshadri/Workspace/structures/MDR1/RSD/catalogues/*_Voids_*')
profilePath = '/Users/seshadri/Workspace/structures/MDR1/RSD/profiles/'
stackList = []
for File in filelist:
    handle = File.replace('/Users/seshadri/Workspace/structures/MDR1/RSD/catalogues/','')
    handle = handle.replace('_cat.txt','')
    stackList.append({'catPath' : File,
                      'stackname' : handle+'_all',
                      })
