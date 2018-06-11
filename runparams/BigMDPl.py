import os

sampleHandle = "CMASS"

tracerFile = os.getenv('HOME')+"/Workspace/Sims/MultiDark/BigMDPl/HOD/BigMDPl_CMASS.dat"
outputFolder = os.getenv('HOME')+"/Workspace/structures/BigMDPl/"+sampleHandle+"/"

useDM = False
DMfile = ''
DM_resolution = 1175
usePhi = False
Phifile = ''
Phi_resolution = 1175

ZOBOVdir = os.getenv('HOME')+"/Workspace/ZOBOV/bin/"

posnCols = [1,2,3]
massCut = False
redshiftSpace = False

runZobov = False
useIsol = False
numZobovDivisions = 2
ZobovBufferSize = 0.05

OmegaM = 0.307115

isBox = True
boxLen = 2500.0

maskFile = ""
zMin = 0
zMax = 0
useSelFn = False
genSelFn = False
selFnFile = ""
mockDensRat = 10.0  
mockDens = 0.0
useGuards = False

postprocessVoids = False
countAllV = False
minDensCut = 1.0
dontMergeV = True #if True, no void merging and next two lines are ignored
linkDensThreshV = 1.0
rThreshV = 2
stripDensV = 0   #(if stripDensV > 0, MUST satisfy: minDensCut, linkDensThreshV <= stripDensV)
useBaryC = True
NminV = 1
prefixV = "Voids"

postprocessClusters = True
countAllC = False
maxDensCut = 1.0
dontMergeC = True #if True, no cluster merging and next two lines are ignored
linkDensThreshC = 22.0
rThreshC = 16.3
stripDensC = 0   #(if stripDens > 0, MUST satisfy: maxDensCut, linkDensThreshC >= stripDensC)
NminC = 1
#useBaryC = False
prefixC = "Clusters"
 
