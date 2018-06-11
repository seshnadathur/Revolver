import os

sampleHandle = "F4_r06"

ZOBOVdir = os.getenv('HOME')+"/Workspace/ZOBOV/bin/"

tracerFile = os.getenv('HOME')+"/Workspace/Sims/fR/a0.6/b1500_F4_r06_a0.60.mxyzvxvyvz.dat"
outputFolder = os.getenv('HOME')+"/Workspace/structures/fR/a0.6/"+sampleHandle+"/"

posnCols = [1,2,3]

runZobov = False
useIsol = False
numZobovDivisions = 2
ZobovBufferSize = 0.05

OmegaM = 0.24
redshift = 1/0.6-1

isBox = True
boxLen = 1500.0

massCut = True
if massCut:
    massCol = 0
    targetNumDens = 2.0e-4
    minDeltaMh = 8.5e9

redshiftSpace = True
if redshiftSpace:
    velocityCols = [4,5,6]
    LOSaxis = 2

postprocessVoids = True
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
 
