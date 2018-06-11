import os

sampleHandle = "realspace"

tracerFile = "/Users/seshadri/Workspace/Sims/MultiDark/BigMDPl/newHOD/CMASS_mocks_for_kSZ_realspace.npy"
outputFolder = "/Users/seshadri/Workspace/structures/CMASS_mocks_for_kSZ/"+sampleHandle+"/"

ZOBOVdir = os.getenv('HOME')+"/Workspace/ZOBOV/bin/"

posnCols = [1,2,3]  # columns in tracerFile containing tracer x,y,z positions (start from 0)
massCut = False     # set True to select halo tracers by some mass threshold
if massCut:
    massCol = 0
    targetNumDens = 2.0e-4  # automatically determine mass threshold to match target density
    minDeltaMh = 8.5e9
redshiftSpace = False # if True, will convert simulation tracer positions into redshift space before running void-finding (not recommended)
if redshiftSpace:
    velocityCols = [4,5,6]
    LOSaxis = 2

isBox = True  # True if data covers cubic box with PBC, False if running for a sky survey
boxLen = 2500. # in same units as tracer position coords; ignored if isBox=False

OmegaM = 0.307115 # required if isBox=False or if redshiftSpace=True
redshift = 0.0   # used only if isBox=True and redshiftSpace=True

runZobov = True   # set to False if only re-processing a previous ZOBOV run
doSC = False      # set to True if supercluster catalogues are required
useIsol = False   # set to True if isBox=False, otherwise True is recommended
# if tessellation fails (see log), changing next 2 lines may help; otherwise leave as is 
numZobovDivisions = 2
ZobovBufferSize = 0.05

# these lines required only for survey samples, i.e. isBox=False
angCoords = True
genMask = False
maskFile = ""
zMin = 0.43
zMax = 0.70
useSelFn = True
genSelFn = True
selFnFile = ""
mockDensRat = 10.0  
mockDens = 0.0
useGuards = True

# void post processing options: do not change
postprocessVoids = True
countAllV = False
minDensCut = 1.0
dontMergeV = True #if True, no void merging and next two lines are ignored
linkDensThreshV = 1.0
rThreshV = 2
stripDensV = 0   #(if stripDensV > 0, MUST satisfy: minDensCut, linkDensThreshV <= stripDensV)
useBaryC = True
NminV = 1
prefixV = sampleHandle+"Voids"

# cluster post-processing options
postprocessClusters = False # don't set to True unless doSC=True or ZOBOV has previously been run with doSC=True
countAllC = False
maxDensCut = 1.0
dontMergeC = True #if True, no cluster merging and next two lines are ignored
linkDensThreshC = 22.0
rThreshC = 16.3
stripDensC = 0   #(if stripDens > 0, MUST satisfy: maxDensCut, linkDensThreshC >= stripDensC)
NminC = 1
#useBaryC = False
prefixC = sampleHandle+"Clusters"
 
# vestigial options: ignore
useDM = False
DMfile = ""
DM_resolution = 0
usePhi = False
Phifile = ""
Phi_resolution = 0

