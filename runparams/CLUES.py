import os

sampleHandle = "AHF_s1"

tracerFile = os.getenv('HOME')+"/Workspace/Sims/CLUES/Random_Halos/AHF_s1.halos"
outputFolder = os.getenv('HOME')+"/Workspace/structures/CLUES/Random_"+sampleHandle+"/"

useDM = False
DMfile = ''
DM_resolution = 0
usePhi = False
Phifile = ''
Phi_resolution = 0

ZOBOVdir = os.getenv('HOME')+"/Workspace/ZOBOV/bin/"

runZobov = False
useIsol = False
numZobovDivisions = 2
ZobovBufferSize = 0.1

OmegaM = 0.307

isBox = True
boxLen = 500.0

maskFile = ""
zMin = 0
zMax = 0
useSelFn = False
genSelFn = False
selFnFile = ""
mockDensRat = 10.0  
mockDens = 0.0

postprocessVoids = True
countAllV = False
minDensCut = 1.0
uselinkDensV = True
linkDensThreshV = 0.0
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
 
