import sys
import os
import fileinput
import pandas as pd
import numpy as np
import healpy as hp
from astropy.table import Table, vstack
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from vast.voidfinder.distance import z_to_comoving_dist
import shutil
from astropy.cosmology import FlatLambdaCDM

csm0 = FlatLambdaCDM(Om0=.315, H0=100)

p = ArgumentParser(description='Voronoi Voids (V^2) void finder.',
                   formatter_class=ArgumentDefaultsHelpFormatter)

p.add_argument('-y', '--year', type=str, default='Y5',
               help='DESI release year (e.g. Y1, Y3, Y5)')
p.add_argument('-t', '--tracer', type=str, default='',
               help='DESI tracer ("LRG" or "ELG") (defaults to invalid value)')
p.add_argument('-t2', '--tracer2', type=str, default='',
               help='Secondary DESI tracer name used when tracer has subtypes')
p.add_argument('-pl', '--pipeline', type=str, default='',
               help='DESI pipeline ("Loa" or "Loa_blinded" or "Uchuu") ')
p.add_argument('-a', '--algorithm', type=str, default='',
               help='voidfidner (revolver of voxel)')
p.add_argument('-c', '--cap', type=str, default='',
               help='DESI galactic cap (NGC or SGC)')

args = p.parse_args()

z_lims = {'BGS':[0.1,0.5],
          'LRG':[.4,1.1],
          'ELG':[.8, 1.6],
          'QSO':[.8,3.5],
}

# set up directories
mask_file = f'mask_{args.year}_{args.pipeline}_{args.tracer}_{args.cap}.fits'
void_prefix = f'Voids_{args.year}_{args.pipeline}_{args.tracer}'
handle = f'{args.year}_{args.pipeline}_{args.tracer}'
if args.pipeline == 'Iron':

    output_folder = f'/pscratch/sd/h/hrincon/revolver/{args.tracer}_Iron_{args.cap}/'

    tracer_path = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.2/unblinded/{args.tracer2}_clustering.dat.fits'
    rand_path = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.2/unblinded/{args.tracer2}_0_clustering.ran.fits'
    
elif args.pipeline == 'Iron_blinded':

    output_folder = f'/pscratch/sd/h/hrincon/revolver/{args.tracer}_Iron_blinded_{args.cap}/'

    tracer_path = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.2/blinded/{args.tracer2}_clustering.dat.fits'
    rand_path = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.2/blinded/{args.tracer2}_0_clustering.ran.fits'

if args.pipeline == 'Loa':
    output_folder = f'/pscratch/sd/h/hrincon/revolver/{args.tracer}_Loa_{args.cap}/'
    tracer_path = f'/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/nonKP/{args.tracer2}_clustering.dat.fits'
    
    rand_path = f'/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/nonKP/{args.tracer2}_0_clustering.ran.fits'

elif args.pipeline == 'Loa_blinded':
    raise ValueError('Loa_blinded pipeline not yet implemented')

elif args.pipeline == 'Uchuu':
    output_folder = f'/pscratch/sd/h/hrincon/revolver/{args.tracer}/{args.year}/Uchuu_{args.cap}/'
    tracer_path = f'/global/cfs/cdirs/desi/mocks/cai/Uchuu-SHAM/Y3-v2.0/0000/Uchuu-SHAM_{args.tracer}_Y3-v2.0_0000_clustering.dat.fits'
    void_prefix = f'Voids_{args.year}_{args.pipeline}_{args.tracer}'
    #rand_path = # for BGS insert 0 as v2.0_0_0000 and change dat to ran, wheras for LRG ans QSO insert as v2.0_0000_0 and chacne dat to ran 
    z_lims['QSO'] = [.8,2.1]
    
elif args.pipeline == 'Abacus_altmtl':
    output_folder = f'/pscratch/sd/h/hrincon/revolver/{args.tracer}/Y3/Abacus_altmtl_{args.cap}/'
    tracer_path = f'/global/cfs/cdirs/desi/survey/catalogs/DA2/mocks/SecondGenMocks/AbacusSummit_v4_1/altmtl0/kibo-v1/mock0/LSScats/{args.tracer2}_clustering.dat.fits'

elif args.pipeline == 'Abacus_altmtl_unblinded':
    output_folder = f'/pscratch/sd/h/hrincon/revolver/{args.tracer}/Y3/Abacus_altmtl_unblinded_{args.cap}/'
    tracer_path = f'/pscratch/sd/h/hrincon/LSScats/{args.tracer2}_clustering.dat.fits'
elif args.pipeline.startswith("Abacus_altmtl_blinded_s"):
    output_folder = f'/pscratch/sd/h/hrincon/revolver/{args.tracer}/Y3/{args.pipeline}/'
    token=args.pipeline.split("Abacus_altmtl_blinded_s",1)[1]
    tracer_path = f'/pscratch/sd/h/hrincon/AbacusBlinded/specified{token}/blinded_mock/{args.tracer2}_clustering.dat.fits'
    if 'm' in token:
        tracer_path = f'/pscratch/sd/h/hrincon/AbacusBlinded/specified{token}/blinded_mock/{args.tracer2}_clustering.dat.fits'
        

elif args.pipeline == 'Abacus_complete':
    output_folder = f'/pscratch/sd/h/hrincon/revolver/{args.tracer}/Y3/Abacus_complete_{args.cap}/'
    tracer_path = f'/global/cfs/cdirs/desi/survey/catalogs/DA2/mocks/SecondGenMocks/AbacusSummit_v4_1/mock0/{args.tracer2}_complete_NGC_clustering.dat.fits'
    tracers_ngc = Table.read(tracer_path)
    tracer_path = f'/global/cfs/cdirs/desi/survey/catalogs/DA2/mocks/SecondGenMocks/AbacusSummit_v4_1/mock0/{args.tracer2}_complete_SGC_clustering.dat.fits'
    tracers_sgc = Table.read(tracer_path)
    # the random files don't have Z columns for some reason ...
    #rand_path = f'/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/{args.tracer2}_0_full_HPmapcut.ran.fits'

    mock = vstack([tracers_ngc,  tracers_sgc])
    tracer_path = f'/pscratch/sd/h/hrincon/desigroup/voids/tmp/Abacus_complete_{args.tracer}.fits'
    mock.write(tracer_path, overwrite=True)  
    
    
    #rand_path = f'/pscratch/sd/h/hrincon/desigroup/voids/tmp/Loa_blinded_{args.tracer}_ran.fits

# select cap

if args.cap == 'SGC':

    mock = Table.read(f'{tracer_path}')
    mock = mock[(mock['RA']<85)+(mock['RA']>305)]
    mock.write(f'/pscratch/sd/h/hrincon/revolver/revolver_input/galaxy_input.fits', overwrite=True)  
    tracer_path = f'/pscratch/sd/h/hrincon/revolver/revolver_input/galaxy_input.fits'
    mock = Table.read(f'{rand_path}')
    mock = mock[(mock['RA']<85)+(mock['RA']>305)]
    mock.write(f'/pscratch/sd/h/hrincon/revolver/revolver_input/randoms_input.fits', overwrite=True)  
    rand_path = f'/pscratch/sd/h/hrincon/revolver/revolver_input/randoms_input.fits'

elif args.cap == 'NGC':
    
    mock = Table.read(f'{tracer_path}')
    mock = mock[(mock['RA']>85)*(mock['RA']<305)]
    mock.write(f'/pscratch/sd/h/hrincon/revolver/revolver_input/galaxy_input.fits', overwrite=True)  
    tracer_path = f'/pscratch/sd/h/hrincon/revolver/revolver_input/galaxy_input.fits'
    mock = Table.read(f'{rand_path}')
    mock = mock[(mock['RA']>85)*(mock['RA']<305)]
    mock.write(f'/pscratch/sd/h/hrincon/revolver/revolver_input/randoms_input.fits', overwrite=True)  
    rand_path = f'/pscratch/sd/h/hrincon/revolver/revolver_input/randoms_input.fits'
elif args.cap == 'Full':
    mock = Table.read(f'{tracer_path}')
    mock.write(f'/pscratch/sd/h/hrincon/revolver/revolver_input/galaxy_input.fits', overwrite=True)  
    tracer_path = f'/pscratch/sd/h/hrincon/revolver/revolver_input/galaxy_input.fits'
    mock = Table.read(f'{rand_path}')
    mock.write(f'/pscratch/sd/h/hrincon/revolver/revolver_input/randoms_input.fits', overwrite=True)  
    rand_path = f'/pscratch/sd/h/hrincon/revolver/revolver_input/randoms_input.fits'
else:
    message =f'Invalid cap: {args.cap}'
    raise ValueError(message)

# we run one or the other because revolver is paralelized (run in jobs) and voxel is not (run on login node)
if args.algorithm == 'zobov':
    run_zob = True
    run_vox = False
elif args.algorithm == 'voxel':
    run_zob = False
    run_vox = True
else:
    message =f'Invalid algorithm: {args.algorithm}'
    raise ValueError(message)

os.makedirs(output_folder, exist_ok=True)

#set up param file for REVOLVER
param_file = f'./params_cutsky_{args.year}_{args.tracer}_{args.pipeline}.py'
shutil.copyfile('./params_cutsky.py', param_file)
for line in fileinput.input(param_file, inplace=1):
    if line.startswith('output_folder'):
        print(f"output_folder = '{output_folder}'",end='\n')
    elif line.startswith('tracer_file'):
        print(f"tracer_file = '{tracer_path}'",end='\n')
    elif line.startswith('mask_file'):
        print(f"mask_file = '/pscratch/sd/h/hrincon/revolver/revolver_masks/{mask_file}'", end='\n')
    elif line.startswith('handle'):
        print(f"handle = '{handle}'",end='\n')
    elif line.startswith('void_prefix'):
        print(f"void_prefix = '{void_prefix}'",end='\n')
    elif line.startswith('z_low_cut'):
        print(f"z_low_cut = {z_lims[args.tracer][0]}",end='\n')
    elif line.startswith('z_high_cut'):
        print(f"z_high_cut = {z_lims[args.tracer][1]}",end='\n')
    elif line.startswith('z_min'):
        print(f"z_min = {z_lims[args.tracer][0]}",end='\n')
    elif line.startswith('z_max'):
        print(f"z_max = {z_lims[args.tracer][1]}",end='\n')
    elif line.startswith('run_voxelvoids'):
        print(f"run_voxelvoids = {run_vox}",end='\n')
    elif line.startswith('run_zobov'):
        print(f"run_zobov = {run_zob}",end='\n')
    elif line.startswith('random_file') and args.pipeline != 'Loa_blinded':
        print(f"random_file = '{rand_path}'",end='\n')
    elif line.startswith('zobov_box_div'):
        #survey_diameter = 2*csm0.comoving_distance(z_lims[args.tracer][1]).value
        #num_1000_boxes = int(survey_diameter / 1000)
        
        #print(f"zobov_box_div = {num_1000_boxes}",end='\n')
        if 'Iron' in args.pipeline:
            print(f"zobov_box_div = {3}",end='\n')
        else:
            print(f"zobov_box_div = {4}",end='\n')
    elif line.startswith('nthreads'):
        #survey_diameter = 2*csm0.comoving_distance(z_lims[args.tracer][1]).value
        #num_1000_boxes = int(survey_diameter / 1000)
        
        #print(f"zobov_box_div = {num_1000_boxes}",end='\n')
        if 'Iron' in args.pipeline:
            print(f"nthreads = {3**3 + 2}",end='\n')
        else:
            print(f"nthreads = {4**3 + 2}",end='\n')
    elif line.startswith('zobov_buffer') and 'Iron' in args.pipeline:
        print(f"zobov_buffer = {0.16}",end='\n')
    else:
        print(line,end='')

#print(num_1000_boxes, 'box divisions:', num_1000_boxes**3,'boxes')        
        
# create mask


maskra = 360
maskdec = 180
dec_offset = -90

D2R = np.pi/180.0


def generate_mask(gal_data, 
                  z_max = 1.1, 
                  min_maximal_radius=10.0,
                  Omega_M=0.315,
                  h=1.0):
    
    
    ############################################################################
    # First, extract the ra (Right Ascension) and dec (Declination) coordinates
    # of our galaxies from the astropy table.  Make a big (N,2) numpy array
    # where each row of the array is the (ra, dec) pair corresponding to a 
    # galaxy in the survey.
    #
    # Also make sure the ra values are in the range [0,360) instead of 
    # [-180, 180)
    #---------------------------------------------------------------------------
    ra  = gal_data['RA'].data
    
    dec = gal_data['DEC'].data
    

    r_max = z_to_comoving_dist(np.array([z_max], dtype=np.float32), 
                               Omega_M, 
                               h)

    
    num_galaxies = ra.shape[0]

    #ang = np.array(list(zip(ra,dec)))
    
    ang = np.concatenate((ra.reshape(num_galaxies, 1), 
                          dec.reshape(num_galaxies,1)), 
                         axis=1)

    mask_resolution = 1 + int(D2R*r_max/min_maximal_radius) # scalar value despite value of r_max
    
    scaled_converted_ang = (mask_resolution*ang).astype(int)
    
    num_px = maskra * maskdec * mask_resolution ** 2
    nside_max = round(np.sqrt(num_px / 12))
    nside = 1
    while nside * 2 <= nside_max:
        nside *= 2
    healpix_mask = np.zeros(hp.nside2npix(nside), dtype = bool)
    galaxy_pixels = hp.ang2pix(nside, ra, dec, lonlat = True)
    healpix_mask[galaxy_pixels] = 1
    
    return healpix_mask

if args.pipeline != 'Loa_blinded':
    mock = Table.read(f'{tracer_path}')
mask = generate_mask(mock)

hp.write_map(f"/pscratch/sd/h/hrincon/revolver/revolver_masks/{mask_file}", mask, overwrite=True,  dtype=np.float32) 
