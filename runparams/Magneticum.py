import os

sampleHandle = "Box0_mr_box_gal_magcuts"

tracerFile = os.getenv('HOME')+"/Workspace/Sims/Magneticum/Box0_mr_bao_033_box_gal_magcuts_cat.txt"
outputFolder = os.getenv('HOME')+"/Workspace/structures/Magneticum/"+sampleHandle+"/"

useDM = False
DMfile = ""
DM_resolution = 0
usePhi = False
Phifile = ""
Phi_resolution = 0

ZOBOVdir = os.getenv('HOME')+"/Workspace/ZOBOV/bin/"

posnCols = [0,1,2]
massCut = False
redshiftSpace = True
if redshiftSpace:
    velocityCols = [3,4,5]
    LOSaxis = 2

runZobov = True
useIsol = False
numZobovDivisions = 2
ZobovBufferSize = 0.05

OmegaM = 0.272
redshift = 0.14

isBox = True
boxLen = 2688.0

angCoords = True
genMask = False
maskFile = os.getenv('HOME')+"/Workspace/Sims/Jubilee1/mask.fits"
zMin = 0.16
zMax = 0.43
useSelFn = False
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
 
