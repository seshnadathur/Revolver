import os

sampleHandle = "North_r2"

tracerFile = os.getenv('HOME')+"/Workspace/Surveys/DES-SDSS/CMASS_North_photoz_r2.txt"
outputFolder = os.getenv('HOME')+"/Workspace/structures/DES-SDSS/"+sampleHandle+"/"

useDM = False
DMfile = ""
DM_resolution = 0
usePhi = False
Phifile = ""
Phi_resolution = 0

ZOBOVdir = os.getenv('HOME')+"/Workspace/ZOBOV/bin/"

runZobov = False
useIsol = True
numZobovDivisions = 2
ZobovBufferSize = 0.2

OmegaM = 0.3156

isBox = False

angCoords = True
genMask = False
maskFile = os.getenv('HOME')+"/Workspace/Surveys/SDSS_DR11/CMASS/mask_DR11v1_CMASS_North_n128.fits"
zMin = 0.43
zMax = 0.7
useSelFn = True
genSelFn = True
selFnFile = ""
mockDensRat = 5.0  
mockDens = 0.0
useGuards = True

postprocessVoids = True
countAllV = False
minDensCut = 1.0
linkDensThreshV = 0
rThreshV = 2
stripDensV = 0   #(if stripDensV > 0, MUST satisfy: minDensCut, linkDensThreshV <= stripDensV)
useBaryC = False
NminV = 1
prefixV = "IsolatedVoids"

postprocessClusters = False
countAllC = False
maxDensCut = 22.0
linkDensThreshC = 22.0
rThreshC = 16.3
stripDensC = 0   #(if stripDens > 0, MUST satisfy: maxDensCut, linkDensThreshC >= stripDensC)
NminC = 1
prefixC = "Clusters"
 
