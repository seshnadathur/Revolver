import os

#------------Sample class definition------------#
class Sample:
  tracerFile = os.getenv('HOME')+"/Workspace/Sims/MultiDark/HOD/MDR1_LRGB/MDR1_LRGB.mock"
  sampleHandle = "MDR1_LRGB"
  outputFolder = os.getenv('HOME')+"/Workspace/structures/"+sampleHandle+"/"
  posnFile = ""
  useDM = False
  DMfile = ""
  DM_resolution = 1024
  usePhi = False
  Phifile = ""
  Phi_resolution = 1024
  isBox = True
  OmegaM = 0.27
  boxLen = 1000.0
  numTracer = 20000
  numMock = 0
  numPartTot = numTracer + numMock
  numNonEdge = 0
  tracerDens = 0.00002
  maskFile = ""
  fSky = 1.0
  useSelFn = False
  genSelFn = False
  selFnFile = ""
  zMin = 0.16
  zMax = 0.44
  mockDensRat = 10.0
  mockDens = 0.0002
  countAllV = True
  minDensCut = 1.0
  dontMergeV = True
  uselinkDensV = True
  linkDensThreshV = 1.0
  stripDensV = 0
  rThreshV = 0
  NminV = 1
  prefixV = ""
  countAllC = True
  maxDensCut = 1.0
  linkDensThreshC = 1.0
  stripDensC = 0
  rThreshC = 0
  NminC = 1
  prefixC = ""
  useBaryC = True
  dontMergeC = True
 
  def __init__(self, tracerFile = "", sampleHandle = "", outputFolder = "", 
		posnFile = "", useDM = False, DMfile = "", DM_resolution = 1024, usePhi = False, Phifile = "",
		Phi_resolution = 1024, isBox = True, OmegaM = 0.27, boxLen = 1000.0, numTracer = 20000, 
		numMock = 0, numNonEdge = 20000, tracerDens = 0.00002, maskFile = "", fSky = 1.0, useSelFn = False, 
		genSelFn = False, selFnFile = "", zMin = 0.16, zMax = 0.44,
		mockDensRat = 10, mockDens = 0.0002, countAllV = True, minDensCut = 1.0, dontMergeV = True, 
		uselinkDensV = True, linkDensThreshV = 1.0, stripDensV = 0, rThreshV = 0, NminV = 1, useBaryC = True, 
		prefixV = "", countAllC = True, maxDensCut = 1.0, dontMergeC = True, linkDensThreshC = 1.0, 
		stripDensC = 0, rThreshC = 0, NminC = 1, prefixC = ""):
    self.tracerFile = tracerFile
    self.sampleHandle = sampleHandle
    self.outputFolder = outputFolder
    self.posnFile = posnFile
    self.useDM = useDM
    self.DMfile = DMfile
    self.DM_resolution = DM_resolution
    self.usePhi = usePhi
    self.Phifile = Phifile
    self.Phi_resolution = Phi_resolution
    self.isBox = isBox
    self.OmegaM = OmegaM 
    self.boxLen = boxLen
    self.numTracer = numTracer
    self.numMock = numMock
    self.numPartTot = numTracer + numMock
    self.numNonEdge = numNonEdge
    self.tracerDens = tracerDens
    self.maskFile = maskFile
    self.fSky = fSky
    self.useSelFn = useSelFn
    self.genSelFn = genSelFn
    self.zMin = zMin
    self.zMax = zMax 
    self.mockDensRat = mockDensRat 
    self.mockDens = mockDens
    self.countAllV = countAllV
    self.minDensCut = minDensCut
    self.dontMergeV = dontMergeV
    self.uselinkDensV = uselinkDensV
    self.linkDensThreshV = linkDensThreshV
    self.stripDensV = stripDensV
    self.rThreshV = rThreshV
    self.NminV = NminV
    self.useBaryC = useBaryC
    self.prefixV = prefixV
    self.countAllC = countAllC
    self.maxDensCut = maxDensCut
    self.dontMergeC = dontMergeC
    self.linkDensThreshC = linkDensThreshC
    self.stripDensC = stripDensC
    self.rThreshC = rThreshC
    self.NminC = NminC
    self.prefixC = prefixC
#---------------------------------------------#

#--------Catalogue class definition-----------#
class Catalogue:
  sHandle = ""
  prefix = ""
  stackName = ""
  Rmin = 0.0
  Rmax = 1000.0
  CentDensMin = 0
  CentDensMax = 1.0
  AvgDensMin = 0
  AvgDensMax = 5.0
  zoneMax = 0
  Nbins = 10
  MaxDist = 2
  isBox = True
  boxLen = 1000.0
  tracerDens = 0

  def __init__(self, sHandle, prefix, stackName, Rmin, Rmax, CentDensMin, 
		CentDensMax, AvgDensMin, AvgDensMax, zoneMax = 0, Nbins = 20, 
		MaxDist = 2, isBox = True, boxLen = 1000.0, tracerDens = 0):
    self.sHandle = sHandle
    self.prefix = prefix
    self.stackName = stackName
    self.Rmin = Rmin
    self.Rmax = Rmax
    self.CentDensMin = CentDensMin
    self.CentDensMax = CentDensMax
    self.AvgDensMin = AvgDensMin
    self.AvgDensMax = AvgDensMax
    self.zoneMax = zoneMax
    self.Nbins = Nbins
    self.MaxDist = MaxDist
    self.isBox = isBox 
    self.boxLen = boxLen
    self.tracerDens = tracerDens
#---------------------------------------------#
