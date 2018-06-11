import os

sampleHandle = "LOWZ_speczZRecon"

tracerFile = os.getenv('HOME')+"/Workspace/Sims/MultiDark/BigMDPl/newHOD/LOWZ_speczZRecon.npy"
outputFolder = os.getenv('HOME')+"/Workspace/structures/BigMDPl/"+sampleHandle+"/"

useDM = False
DMfile = ""
DM_resolution = 0
usePhi = False
Phifile = ""
Phi_resolution = 0

ZOBOVdir = os.getenv('HOME')+"/Workspace/ZOBOV/bin/"

runZobov = True
useIsol = False
numZobovDivisions = 2
ZobovBufferSize = 0.05

posnCols = [1,2,3]
massCut = False
redshiftSpace = False
if redshiftSpace:
    velocityCols = [4,5,6]
    LOSaxis = 2

OmegaM = 0.307115
redshift = 0.52

isBox = True
boxLen = 2500.

angCoords = True
genMask = False
maskFile = os.getenv('HOME')+"/Workspace/Sims/MultiDark/BigMDPl/HOD/RSD/mask_sample2_n128.fits"
zMin = 0.43
zMax = 0.70
useSelFn = True
genSelFn = True
selFnFile = ""
mockDensRat = 10.0  
mockDens = 0.0
useGuards = True

postprocessVoids = True
countAllV = False
minDensCut = 1.0
dontMergeV = True #if True, no void merging and next two lines are ignored
linkDensThreshV = 1.0
rThreshV = 2
stripDensV = 0   #(if stripDensV > 0, MUST satisfy: minDensCut, linkDensThreshV <= stripDensV)
useBaryC = True
NminV = 1
prefixV = "LOWZ_speczZReconVoids"

postprocessClusters = False
countAllC = False
maxDensCut = 1.0
dontMergeC = True #if True, no cluster merging and next two lines are ignored
linkDensThreshC = 22.0
rThreshC = 16.3
stripDensC = 0   #(if stripDens > 0, MUST satisfy: maxDensCut, linkDensThreshC >= stripDensC)
NminC = 1
#useBaryC = False
prefixC = "Clusters"
 
