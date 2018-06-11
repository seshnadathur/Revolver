import os

sampleHandle = "DM_z053_base_res"

tracerFile = os.getenv('HOME')+"/Workspace/Sims/MultiDark/MDR1/DarkMatter/DM_parts_z053_n5e-4.npy"
outputFolder = os.getenv('HOME')+"/Workspace/structures/MDR1/"+sampleHandle+"/"

useDM = False
DMfile = os.getenv('HOME')+"/Workspace/Sims/MultiDark/MDR1/MDR1_Dens_1024.npy"
DM_resolution = 1024
usePhi = False
Phifile = os.getenv('HOME')+"/Workspace/Sims/MultiDark/MDR1/MDR1_Phi_1024.npy"
Phi_resolution = 1024

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

OmegaM = 0.27
redshift = 0.53

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

postprocessVoids = True
countAllV = False
minDensCut = 1.0
dontMergeV = True #if True, no void merging and next two lines are ignored
linkDensThreshV = 1.0
rThreshV = 2
stripDensV = 0   #(if stripDensV > 0, MUST satisfy: minDensCut, linkDensThreshV <= stripDensV)
useBaryC = True
NminV = 1
prefixV = "DM_z053_base_res_trueposVoids"

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
