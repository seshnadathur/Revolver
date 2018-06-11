import os

sampleHandle = "DESMICEv1_photoz_r2"

tracerFile = os.getenv('HOME')+"/Workspace/Sims/DESMICE/DMv1.0_photozr2_0.3z0.7.dat"
outputFolder = os.getenv('HOME')+"/Workspace/structures/DES-MICE/"+sampleHandle+"/"

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

OmegaM = 0.25

isBox = False

angCoords = True
genMask = False
maskFile = os.getenv('HOME')+"/Workspace/Sims/DESMICE/mask_DMv1.0_n128.fits"
zMin = 0.3
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
 
