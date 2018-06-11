import os

sampleHandle = "Poiss_Main2"

tracerFile = os.getenv('HOME')+"/Workspace/Sims/Poisson/Poiss_Main2/Poiss_Main2.mock"
outputFolder = os.getenv('HOME')+"/Workspace/structures/"+sampleHandle+"/"

useDM = False
DMfile = os.getenv('HOME')+"/Workspace/Sims/MultiDark/MDR1_Dens_1024.npy"
DM_resolution = 1024
usePhi = False
Phifile = os.getenv('HOME')+"/Workspace/Sims/MultiDark/MDR1_Phi_1024.npy"
Phi_resolution = 1024

ZOBOVdir = os.getenv('HOME')+"/Workspace/ZOBOV/bin/"

runZobov = False
useIsol = False
numZobovDivisions = 2
ZobovBufferSize = 0.1

OmegaM = 0.27

isBox = True
boxLen = 1000.0

maskFile = ""
zMin = 0
zMax = 0
useSelFn = False
genSelFn = False
selFnFile = ""
mockDensRat = 10.0  
mockDens = 0.0

runPostProcess = True

countAllV = True
minDensCut = 1.0
uselinkDensV = True
linkDensThreshV = 1.0
rThreshV = 2
stripDensV = 0   #(if stripDensV > 0, MUST satisfy: minDensCut, linkDensThreshV <= stripDensV)
useBaryC = False
NminV = 1
prefixV = "MinimalVoids"

countAllC = False
maxDensCut = 22.0
linkDensThreshC = 22.0
rThreshC = 16.3
stripDensC = 0   #(if stripDens > 0, MUST satisfy: maxDensCut, linkDensThreshC >= stripDensC)
NminC = 1
prefixC = "testClusters"

