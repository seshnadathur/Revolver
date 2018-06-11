import os

sampleHandle = "dim1"

tracerFile = os.getenv('HOME')+"/Workspace/Surveys/SDSS_DR7/dim1.txt"
angCoords = True
outputFolder = os.getenv('HOME')+"/Workspace/structures/SDSS_DR7/"+sampleHandle+"/"

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

OmegaM = 0.3082

isBox = False
boxLen = 1000.0

genMask = False
maskFile = os.getenv('HOME')+"/Workspace/Surveys/SDSS_DR7/sdss_dr72safe0_workingmask.fits"
zMin = 0.0
zMax = 0.05
useSelFn = True
genSelFn = True
selFnFile = ""
mockDensRat = 10.0  
mockDens = 0.0

postprocessVoids = True
countAllV = False
minDensCut = 1.0
linkDensThreshV = 1.0
rThreshV = 2
stripDensV = 0   #(if stripDensV > 0, MUST satisfy: minDensCut, linkDensThreshV <= stripDensV)
useBaryC = True
NminV = 1
prefixV = "MinimalVoids"

postprocessClusters = False
countAllC = False
maxDensCut = 22.0
linkDensThreshC = 22.0
rThreshC = 16.3
stripDensC = 0   #(if stripDens > 0, MUST satisfy: maxDensCut, linkDensThreshC >= stripDensC)
NminC = 1
prefixC = "Clusters"
 
