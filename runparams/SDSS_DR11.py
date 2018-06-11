import os

sampleHandle = "LOWZ_North"

tracerFile = os.getenv('HOME')+"/Workspace/Surveys/SDSS_DR11/LOWZ/galaxy_DR11v1cut_LOWZ_North.txt"
outputFolder = os.getenv('HOME')+"/Workspace/structures/SDSS_DR11/"+sampleHandle+"/"

useDM = False
DMfile = ""
DM_resolution = 0
usePhi = False
Phifile = ""
Phi_resolution = 0

ZOBOVdir = os.getenv('HOME')+"/Workspace/ZOBOV/bin/"

runZobov = True
useIsol = True
numZobovDivisions = 2
ZobovBufferSize = 0.2

OmegaM = 0.308

isBox = False

angCoords = True
genMask = False
maskFile = os.getenv('HOME')+"/Workspace/Surveys/SDSS_DR11/unified_DR11v1_LOWZ_North_completeness_n128.fits"
zMin = 0.15
zMax = 0.43
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
linkDensThreshV = 0
rThreshV = 2
stripDensV = 0   #(if stripDensV > 0, MUST satisfy: minDensCut, linkDensThreshV <= stripDensV)
useBaryC = True
NminV = 1
prefixV = "Voids"

postprocessClusters = True
countAllC = False
maxDensCut = 1.0
dontMergeC = True #if True, no void merging and next two lines are ignored
linkDensThreshC = 22.0
rThreshC = 16.3
stripDensC = 0   #(if stripDens > 0, MUST satisfy: maxDensCut, linkDensThreshC >= stripDensC)
useBaryC = True
NminC = 1
prefixC = "Clusters"
 
