import os
import numpy as np
import glob

# specify paths and simulation box properties
Ngrid = 2350
boxLen = 2500.
vxFile = "/store/erebos/snadathur/BigMDPl_vx2350_z052_flat.bin"
vyFile = "/store/erebos/snadathur/BigMDPl_vy2350_z052_flat.bin"
vzFile = "/store/erebos/snadathur/BigMDPl_vz2350_z052_flat.bin"

# stack selection and binning choices
scaled_profiles = False
MaxDist = 120.0  # if using scaled profiles, max units of void radius; else, max distance in absolute units
Nbins = 50	#number of bins

# list of structures to stack
filelist = glob.glob('/z/snadathur/structures/BigMDPl/CMASS_RSD/catalogues/*_Voids_*')
profilePath = '/z/snadathur/structures/BigMDPl/CMASS_RSD/profiles/velocity/'
stackList = []
for File in filelist:
    handle = File.replace('/z/snadathur/structures/BigMDPl/CMASS_RSD/catalogues/','')
    handle = handle.replace('_cat.txt','')
    stackList.append({'catPath' : File,
                      'stackname' : handle+'_all',
                      })
