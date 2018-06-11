
import argparse
import imp
from python_tools import *
from scipy.stats.mstats import mquantiles

#----------Main wrapper code-------------#
#----------------------------------------#

# Read in default values and special settings
parser = argparse.ArgumentParser(description='options')
parser.add_argument('--par', dest='par',default="",help='path to parameter file')
args = parser.parse_args()

defaultsFile = "runparams/defaults.py"
parms = imp.load_source("name", defaultsFile)
globals().update(vars(parms))

filename = args.par
if os.access(filename, os.F_OK):
	print "Loading parameters from", filename
	parms = imp.load_source("name", filename)
	globals().update(vars(parms))
else:
	print "\nDid not find settings file %s, proceeding with default settings\n" % filename

if not os.access(outputFolder, os.F_OK):
	os.makedirs(outputFolder)

# get the number of tracer particles
if not os.access(tracerFile, os.F_OK):
	print "Can't find tracer file %s" % tracerFile
	sys.exit(-1)
if '.npy' in tracerFile:
    tracers = np.load(tracerFile)
else:
    tracers = np.loadtxt(tracerFile)
numTracer = len(tracers)
print "%i tracer particles in file\n" %numTracer

posnFile = outputFolder+sampleHandle+"_pos.dat"
if runZobov:
	if isBox: #we are dealing with full cubic box - no mocks, no coordinate shift
		if boxLen <= 0:
			print "Box length %f does not make sense, aborting", boxLen
			exit(-1)

		tracerDens = numTracer/boxLen**3.0

		if massCut:		# drop low-mass halos to achieve desired number densities
			targetNum = parms.targetNumDens*(boxLen**3)
			print "Tracer number density %0.3e, target number density %0.3e" %(tracerDens,parms.targetNumDens)
			if targetNum<numTracer:
				keepFrac = targetNum/numTracer
				cutMass = mquantiles(tracers[:,parms.massCol],1-keepFrac)
				if np.sum(tracers[:,parms.massCol]>cutMass)<targetNum: 
					cutMass -= parms.minDeltaMh
				print "\tKeeping only halos with Mhalo>%0.3e to achieve ~ target density" %cutMass
				tracers = tracers[tracers[:,parms.massCol]>cutMass]
				numTracer = tracers.shape[0]
				tracerDens = numTracer/(boxLen**3)
			else:
				print "/tNot enough tracers to achieve target density. Ignoring."
		
		if redshiftSpace:	# convert velocities into RS displacement along chosen axis (plane-parallel approx.)
			print "Applying redshift-space effect along axis %d using plane-parallel approximation" %LOSaxis
			LOSvelCol = velocityCols[LOSaxis]
			LOSposCol = posnCols[LOSaxis]
			
			LOSvel = tracers[:,LOSvelCol] 	# velocity along the specified LOS direction
			DeltaZ = LOSvel*(1+redshift)/LIGHT_SPEED  # convert the LOS velocity to a change to observed redshift
			tracers[:,LOSposCol] += 0.01*LIGHT_SPEED / np.sqrt(OmegaM*(1+redshift)**3 + 1 - OmegaM)*DeltaZ 
			tracers[tracers[:,LOSposCol]<0,LOSposCol] += boxLen
			tracers[tracers[:,LOSposCol]>boxLen,LOSposCol] -= boxLen

		sample = Sample(tracerFile = tracerFile, sampleHandle = sampleHandle, 
			outputFolder = outputFolder, posnFile = posnFile, useDM = useDM, 
			DMfile = DMfile, DM_resolution = DM_resolution, usePhi = usePhi, 
			Phifile = Phifile, OmegaM = OmegaM, boxLen = boxLen, 
			numTracer = numTracer, numMock = 0, numNonEdge = numTracer, 
			tracerDens = tracerDens, countAllV = countAllV, minDensCut = minDensCut, 
			dontMergeV = dontMergeV, linkDensThreshV = linkDensThreshV, 
			stripDensV = stripDensV, rThreshV = rThreshV, NminV = NminV, 
			useBaryC = useBaryC, prefixV = prefixV, countAllC = countAllC, 
			maxDensCut = maxDensCut, dontMergeC = dontMergeC, 
			linkDensThreshC = linkDensThreshC, stripDensC = stripDensC, 
			rThreshC = rThreshC, NminC = NminC, prefixC = prefixC)

		# write to file in ZOBOV-readable format
		with open(sample.posnFile,'w') as F:
			Npart = np.array(sample.numTracer,dtype=np.int32)
			Npart.tofile(F,format='%d')
			data = tracers[:,posnCols[0]]
			data.tofile(F,format='%f')
			data = tracers[:,posnCols[1]]
			data.tofile(F,format='%f')
			data = tracers[:,posnCols[2]]
			data.tofile(F,format='%f')
		
		# run ZOBOV
		runZOBOV(sample, useIsol=useIsol, numZobovDivisions=numZobovDivisions, ZobovBufferSize=ZobovBufferSize, doSC=doSC)

		# write important info about the sample to file for future reference
		sampleInfo = "sampleHandle = '%s'\nisBox = %s\nnumTracer = %d\n" %(sampleHandle, sample.isBox, sample.numTracer)
		sampleInfo += "numMock = %d\nnumNonEdge = %d\nboxLen = %f\n" %(sample.numMock, sample.numNonEdge, sample.boxLen)
		sampleInfo += "tracerDens = %e" %sample.tracerDens
		infoFile = outputFolder+"sample_info.txt"
		with open(infoFile,'w') as F:
			F.write(sampleInfo)

	else:	#data does not cover a full cubic box, must include buffer mocks

		if genMask:	#no survey mask provided, generate one
			maskFile  = outputFolder+sampleHandle+'_mask.fits'
			fSky = generate_mask(tracerFile,maskFile)
		else:
			if not os.access(maskFile,os.F_OK):
				print 'Could not find mask file %s!' %maskFile
				sys.exit(-1)
			mask = hp.read_map(maskFile,verbose=False)
			fSky = 1.0*sum(mask)/len(mask)

		tracerDens = numTracer/(fSky*full_volume_z([zMin,zMax],OmegaM))

		sample = Sample(tracerFile = tracerFile, sampleHandle = sampleHandle, 
			outputFolder = outputFolder, posnFile = posnFile, useDM = useDM, 
			DMfile = DMfile, DM_resolution = DM_resolution, usePhi = usePhi, 
			Phifile = Phifile, isBox = isBox, OmegaM = OmegaM, numTracer = numTracer,
			maskFile = maskFile, fSky = fSky, zMin = zMin, zMax = zMax, 
			useSelFn = useSelFn, genSelFn = genSelFn, selFnFile = selFnFile,
			tracerDens = tracerDens, mockDensRat = mockDensRat, mockDens = mockDens,
			countAllV = countAllV, minDensCut = minDensCut, dontMergeV = dontMergeV, 
			linkDensThreshV = linkDensThreshV, stripDensV = stripDensV, 
			rThreshV = rThreshV, NminV = NminV, useBaryC = useBaryC, prefixV = prefixV, 
			countAllC = countAllC, maxDensCut = maxDensCut, dontMergeC = dontMergeC, 
			linkDensThreshC = linkDensThreshC, stripDensC = stripDensC, 
			rThreshC = rThreshC, NminC = NminC, prefixC = prefixC)

		#first convert tracer file to required format
		if angCoords:
			coords_ang2std(sample)
		else:
			coords_Cartesian2std(sample)

		# selection function reqd but not provided, generate it
		if useSelFn and genSelFn: generate_selFn(sample,nbins=15)

		#now add buffer mocks
		generate_buffer(sample, useGuards=useGuards)
		print "Added total %d mocks" %sample.numMock
		sample.numPartTot = sample.numTracer + sample.numMock
		
		# call C code to write tracer positions into ZOBOV-readable format	
		conf = "%s %s %s %d %d %lf %d" %(sample.tracerFile, sample.posnFile, "Survey", 4, sample.numPartTot, sample.boxLen, 1)
		parFile = os.getcwd()+"/writebox.par"
		file(parFile,mode="w").write(conf)
		cmd = [ZOBOVdir+"writeBox", parFile]
		subprocess.call(cmd)
		
		# then remove the intermediate file used
		if os.access(parFile, os.F_OK): os.unlink(parFile)

		# run ZOBOV
		runZOBOV(sample, useIsol=useIsol, numZobovDivisions=numZobovDivisions, ZobovBufferSize=ZobovBufferSize, doSC=doSC)

		# and remove the temporary ascii tracer file
		if os.access(sample.tracerFile, os.F_OK): os.unlink(sample.tracerFile)

		# write important info about the sample to file for future reference
		sampleInfo = "sampleHandle = '%s'\nisBox = %s\nnumTracer = %d\n" %(sampleHandle, sample.isBox, sample.numTracer)
		sampleInfo += "numMock = %d\nnumNonEdge = %d\nboxLen = %f\n" %(sample.numMock, sample.numNonEdge, sample.boxLen)
		sampleInfo += "tracerDens = %e\nmaskFile = '%s'\nfSky = %f\n" %(sample.tracerDens, sample.maskFile, sample.fSky)
		sampleInfo += "OmegaM = %0.3f\nzMin = %0.3f\nzMax = %0.3f" %(sample.OmegaM, sample.zMin, sample.zMax)
		infoFile = outputFolder+"sample_info.txt"
		with open(infoFile,'w') as F:
			F.write(sampleInfo)
else:
	infoFile = outputFolder+"sample_info.txt"
	parms = imp.load_source("name", infoFile)
	globals().update(vars(parms))

	sample = Sample(tracerFile = tracerFile, sampleHandle = sampleHandle, 
		outputFolder = outputFolder, posnFile = posnFile, useDM = useDM, 
		DMfile = DMfile, DM_resolution = DM_resolution, usePhi = usePhi, 
		Phifile = Phifile, isBox = isBox, boxLen = boxLen, OmegaM = OmegaM, 
		numTracer = numTracer, numMock = numMock, numNonEdge = numNonEdge,
		zMin = zMin, zMax = zMax, useSelFn = useSelFn, genSelFn = genSelFn, 
		selFnFile = selFnFile, tracerDens = tracerDens, mockDensRat = mockDensRat, 
		mockDens = mockDens, countAllV = countAllV, minDensCut = minDensCut, 
		dontMergeV = dontMergeV, linkDensThreshV = linkDensThreshV, 
		stripDensV = stripDensV, rThreshV = rThreshV, NminV = NminV, 
		useBaryC = useBaryC, prefixV = prefixV, countAllC = countAllC, 
		maxDensCut = maxDensCut, dontMergeC = dontMergeC, linkDensThreshC = linkDensThreshC, 
		stripDensC = stripDensC, rThreshC = rThreshC, NminC = NminC, prefixC = prefixC)

	if not sample.isBox:
		sample.maskFile = maskFile
		sample.fSky = fSky

if postprocessVoids:
	postprocVoids(sample)
	if sample.countAllV: get_rootVoids(sample)
if postprocessClusters:
	postprocClusters(sample)
print "Done post-processing\n"


