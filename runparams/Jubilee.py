import os

sampleHandle = "Box0_mr_gals_magcuts"

tracerFile = os.getenv('HOME')+"/Workspace/Sims/Magneticum/Box0_mr_bao_gals_magcuts_cat.txt"
outputFolder = os.getenv('HOME')+"/Workspace/structures/Magneticum/"+sampleHandle+"/"

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

OmegaM = 0.272

isBox = False

angCoords = True
genMask = False
maskFile = os.getenv('HOME')+"/Workspace/Sims/Jubilee1/mask.fits"
zMin = 0.16
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
dontMergeV = True
linkDensThreshV = 1.0
rThreshV = 2
stripDensV = 0   #(if stripDensV > 0, MUST satisfy: minDensCut, linkDensThreshV <= stripDensV)
useBaryC = True
NminV = 1
prefixV = "Voids"

postprocessClusters = False
countAllC = False
maxDensCut = 1.0
dontMergeC = True
linkDensThreshC = 22.0
rThreshC = 16.3
stripDensC = 0   #(if stripDens > 0, MUST satisfy: maxDensCut, linkDensThreshC >= stripDensC)
NminC = 1
prefixC = "Clusters"
 
