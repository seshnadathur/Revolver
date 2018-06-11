import os

# name for this sample
sampleHandle = "MDR1_LRGB"

# path to input tracer data
tracerFile = os.getenv('HOME')+"/Workspace/Sims/MultiDark/HOD/MDR1_LRGB/MDR1_LRGB.mock"
# folder in which to store all output files
outputFolder = os.getenv('HOME')+"/Workspace/structures/"+sampleHandle+"/"
 
# path to all ZOBOV executable files
ZOBOVdir = os.getenv('HOME')+"/Workspace/ZOBOV/bin/"

# ZOBOV run-time options
runZobov = False
useIsol = False
numZobovDivisions = 2
ZobovBufferSize = 0.2

# cosmology
OmegaM = 0.27

# do the tracers occupy a full cubic box? if yes, assume coordinates (x,y,z) and periodic BC
isBox = True
# if isBox == True, the box side in Mpc/h
boxLen = 1000.0

# if tracers don't occupy the full box, they are assumed to have coordinates specified by (RA,Dec,redshift);
# these will be converted into box coordinates and appropriate boxLen determined automatically based on -
# HEALPix file giving the sky mask 
maskFile = ""
# redshift extent of tracers
zMin = 0
zMax = 0
# should we correct for redshift-dependent selection fn.? 
useSelFn = False
# if yes, should the selection fn be determined by the code?
genSelFn = False
# if useSelFn == True and genSelFn == False, specify a predetermined selection fn. file
selFnFile = ""
# the code will place mock 'buffer' particles around the boundaries to avoid contamination; 
# the mock particle density to use can be specified as a ratio to tracer density or an absolute value
# set bufferDensRat < 0 to use absolute bufferDens value
mockDensRat = 10.0  
# if bufferDensRat > 0, bufferDens automatically reset by the code
mockDens = 0.0

runPostProcess = False

useDM = False
DMfile = ''
DM_resolution = 1175
usePhi = False
Phifile = ''
Phi_resolution = 1175

#------------------------------------
# postprocessing parameters for voids
#------------------------------------
# if False, will only count top-level voids; otherwise double counts subvoids too
countAllV = True
# discard voids with minimum density > minDensCut
minDensCut = 1.0
# prevent zone mergers if minimum linking density is > linkDensThreshV
linkDensThreshV = 0.3
# prevent zone mergers if sub-void has 'density contrast' > rThreshV (set rThreshV = 0 to deactivate)
rThreshV = 2
# "strip" voids to include only Voronoi cells with density < stripDensV (set stripDensV = 0 to deactivate)
stripDensV = 0   #(if stripDensV > 0, MUST satisfy: minDensCut, linkDensThreshV <= stripDensV)
# use volume-weighted barycentre definition of void centre? if useBaryC = False, will use min dens circumcentre method
useBaryC = True
# discard voids if minimum number of particles < NminV
NminV = 1
# postproc prefix
prefixV = "HelsinkiVoids"
#------------------------------------

#--------------------------------------------
# postprocessing parameters for superclusters
#--------------------------------------------
# analogous to countAllV
countAllC = True
# discard superclusters with maximum density < maxDensCut
maxDensCut = 22.0
# prevent zone mergers if maximum linking density is < linkDensThreshC
linkDensThreshC = 22.0
# "strip" superclusters to include only Voronoi cells with density > stripDensC (set stripDensC = 0 to deactivate)
stripDensC = 0   #(if stripDens > 0, MUST satisfy: maxDensCut, linkDensThreshC >= stripDensC)
# prevent zone mergers if sub-cluster has 'density contrast' > rThreshC (set rThreshC = 0 to deactivate)
# prevent zone mergers if sub-cluster has 'density contrast' > rThreshC (set rThreshC = 0 to deactivate)
rThreshC = 0
# discard superclusters if minimum number of particles < NminC
NminC = 1
# postproc prefix
prefixC = "SafeClusters"
#--------------------------------------------


