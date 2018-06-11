import os

sampleHandle = "CMASS"

tracerFile = os.getenv('HOME')+"/Workspace/Surveys/SDSS_uDR12/galaxy_uDR12_CMASS.txt"
angCoords = True
outputFolder = os.getenv('HOME')+"/Workspace/structures/SDSS_uDR12/"+sampleHandle+"/"

useDM = False
DMfile = ""
DM_resolution = 0
usePhi = False
Phifile = ""
Phi_resolution = 0

ZOBOVdir = os.getenv('HOME')+"/Workspace/ZOBOV/bin/"

runZobov = False
useIsol = True
useGuards = True
numZobovDivisions = 2
ZobovBufferSize = 0.2

OmegaM = 0.27

isBox = False
boxLen = 1000.0

genMask = False
maskFile = os.getenv('HOME')+"/Workspace/Surveys/SDSS_uDR12/mask_uDR12_CMASS_n128.fits"
zMin = 0.43
zMax = 0.70
useSelFn = True
genSelFn = True
selFnFile = ""
mockDensRat = 5.0  
mockDens = 0.0

postprocessVoids = True
countAllV = False
minDensCut = 1.0
linkDensThreshV = 1.0
rThreshV = 2
stripDensV = 0   #(if stripDensV > 0, MUST satisfy: minDensCut, linkDensThreshV <= stripDensV)
useBaryC = True
NminV = 1
prefixV = "Minimal_baryC_Voids"

postprocessClusters = False
countAllC = False
maxDensCut = 22.0
linkDensThreshC = 22.0
rThreshC = 16.3
stripDensC = 0   #(if stripDens > 0, MUST satisfy: maxDensCut, linkDensThreshC >= stripDensC)
NminC = 1
prefixC = "Clusters"
 
