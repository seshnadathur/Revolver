from tools import *
import fileinput

#------------postprocess voids according to specified merging and selection criteria-----------#
#------------------call routines to find void centres and record _cat.txt files---------------#
def postprocVoids(sample):
	"""Postprocess raw ZOBOV output to get a final set of voids according to specified criteria.

	Argument sample is of class Sample.
	"""
	
	print "Post-processing voids ...\n"
	
	if sample.stripDensV>0 and sample.minDensCut>sample.stripDensV:
		print "ERROR: If using the stripDensV functionality, minDensCut<=stripDensV is required"
		sys.exit(-1)
	if sample.stripDensV>0 and sample.linkDensThreshV>sample.stripDensV:
		print "ERROR: If using the stripDensV functionality, linkDensThreshV<=stripDensV is required"
		sys.exit(-1)
	
	# the files with ZOBOV output
	zoneFile = sample.outputFolder+"rawZOBOV/"+sample.sampleHandle+".zone"
	voidFile = sample.outputFolder+"rawZOBOV/"+sample.sampleHandle+".void"
	listFile = sample.outputFolder+"rawZOBOV/"+sample.sampleHandle+".txt"
	volFile = sample.outputFolder+"rawZOBOV/"+sample.sampleHandle+".trvol"
	densFile = sample.outputFolder+"rawZOBOV/"+sample.sampleHandle+".vol"

	# new files after post-processing
	newVoidFile = sample.outputFolder+sample.prefixV+".void"
	newListFile = sample.outputFolder+sample.prefixV+"_list.txt"

	#load the list of void candidates 
	voidsread = np.loadtxt(listFile,skiprows=2)
	#sort in asc order of min dens
	sorted_order = np.argsort(voidsread[:,3])
	voidsread = voidsread[sorted_order]

	num_voids = len(voidsread[:,0])
	vid = np.asarray(voidsread[:,0],dtype=int)
	edgelist = np.asarray(voidsread[:,1],dtype=int)
	vollist = voidsread[:,4]
	numpartlist = np.asarray(voidsread[:,5],dtype=int)
	rlist = voidsread[:,9]

	# load up the void hierarchy
	with open(voidFile,'r') as Fvoid:
		hierarchy = Fvoid.readlines()
	#sanity check		
	nvoids = int(hierarchy[0])
	if nvoids != num_voids: 
		print "Unequal void numbers in voidfile and listfile, %d and %d!" %(nvoids, num_voids)
		sys.exit(-1)
	hierarchy = hierarchy[1:]

	#load up the particle-zone info
	zonedata = np.loadtxt(zoneFile,dtype='int',skiprows=1)
	
	#load the VTFE volume information
	with open(volFile,'r') as File:
		Np = np.fromfile(File, dtype=np.int32,count=1)[0]
		if not Np==sample.numTracer: #sanity check
			print "Np = %d in %s.trvol file does not match numTracer = %d!" %(Np,sample.sampleHandle,sample.numTracer)
			sys.exit(-1)
		vols = np.fromfile(File, dtype=np.float64,count=Np)

	#load the VTFE density information
	with open(densFile, 'r') as File:
		Np = np.fromfile(File, dtype=np.int32,count=1)[0]
		if not Np==sample.numTracer: #sanity check
			print "Np = %d in %s.vol file does not match numTracer = %d!" %(Np,sample.sampleHandle,sample.numTracer)
			sys.exit(-1)
		densities = np.fromfile(File, dtype=np.float64,count=Np)
		densities = 1./densities

	# mean volume per particle in box (including all buffer mocks)
	meanvol_trc = (sample.boxLen**3.)/sample.numPartTot

	# parse the list of structures, performing minimal pruning
	with open(newVoidFile,'w') as Fnewvoid:
		with open(newListFile,'w') as Fnewlist:
				
			#initialize variables
			counted_zones = np.empty(0,dtype=int)
			edgeFlag = np.empty(0,dtype=int)
			wtdAvgDens = np.empty(0,dtype=int)
			num_Acc = 0
			
			for i in range(num_voids):
				coredens = voidsread[i,3]
				voidline = hierarchy[sorted_order[i]].split()
				pos = 1
				numzonestoadd = int(voidline[pos])
				finalpos = pos + numzonestoadd + 1;
				rval = float(voidline[pos+1])
				rstopadd = rlist[i]
				StartGrow = False
				num_adds = 0
				if rval >= 1 and coredens < sample.minDensCut and numpartlist[i] >= sample.NminV \
					and (sample.countAllV or vid[i] not in counted_zones):
					#this zone qualifies as a seed zone
					StartGrow = True
					AddMore = True
					num_Acc += 1
					zonelist = vid[i]
					totalVol = vollist[i]
					totalNumParts = numpartlist[i]
					while numzonestoadd > 0 and AddMore:	#more zones can potentially be added
						zonestoadd = np.asarray(voidline[pos+2:pos+numzonestoadd+2],dtype=int)
						dens = rval * coredens
						rsublist = rlist[np.in1d(vid,zonestoadd)]
						volsublist = vollist[np.in1d(vid,zonestoadd)]
						partsublist = numpartlist[np.in1d(vid,zonestoadd)]				
						if sample.dontMergeV or (sample.uselinkDensV and dens > sample.linkDensThreshV) or  \
							(sample.rThreshV>0 and max(rsublist) > sample.rThreshV):
							#cannot add these zones
							rstopadd = rval
							AddMore = False
							finalpos -= (numzonestoadd+1)
						else:
							#keep adding zones
							zonelist = np.append(zonelist,zonestoadd)
							num_adds += numzonestoadd
							totalVol += np.sum(volsublist)		# 
							totalNumParts += np.sum(partsublist)	# 
						pos += numzonestoadd + 2
						numzonestoadd = int(voidline[pos])
						rval = float(voidline[pos+1])
						if AddMore:
							finalpos = pos + numzonestoadd + 1
					
					if StartGrow:
						counted_zones = np.append(counted_zones,zonelist)
						MemberIDs = np.logical_and(np.logical_or(sample.stripDensV==0,densities[:]<sample.stripDensV), \
							np.in1d(zonedata,zonelist))
		
						#if using void "stripping" functionality, recalculate void volume and number of particles
						if sample.stripDensV>0:		
							totalVol = np.sum(vols[MemberIDs])
							totalNumParts = len(vols[MemberIDs])
		
						#check if the void is edge-contaminated (useful for observational surveys only)
						if 1 in edgelist[np.in1d(vid,zonestoadd)]: 
							edgeFlag = np.append(edgeFlag,1)
						else: edgeFlag = np.append(edgeFlag,0)
		
						#average density of member cells weighted by cell volumes
						wAD = np.sum(vols[MemberIDs]*densities[MemberIDs])/np.sum(vols[MemberIDs])
						wtdAvgDens = np.append(wtdAvgDens, wAD)
		
						#set the new line for the .void file
						newvoidline = voidline[:finalpos] 
						if not AddMore:
							newvoidline.append(str(0))
						newvoidline.append(str(rstopadd))
						#write line to the output .void file				
						for j in range(len(newvoidline)):
							Fnewvoid.write(newvoidline[j]+'\t')
						Fnewvoid.write('\n')
						if rstopadd>10**20: rstopadd = -1 #will be true for structures entirely surrounded by edge particles
						#write line to the output _list.txt file
						Fnewlist.write("%d\t%d\t%f\t%d\t%d\t%d\t%f\t%f\n" %(vid[i], int(voidsread[i,2]), coredens, \
							int(voidsread[i,5]), num_adds+1, totalNumParts, totalVol*meanvol_trc, rstopadd))				 
						
	# tidy up the files
	#insert first line with number of voids to the new .void file
	with open(newVoidFile,'r+') as Fnewvoid:
		old = Fnewvoid.read()
		Fnewvoid.seek(0)
		topline = "%d\n" %num_Acc
		Fnewvoid.write(topline+old)
	
   	#insert header to the _list.txt file
	listdata = np.loadtxt(newListFile)
	np.savetxt(newListFile,listdata,fmt='%d %d %0.6f %d %d %d %0.6f %0.6f',\
		header="%d non-edge tracers in %s, %d %s voids\nVoidID CoreParticle CoreDens Zone#Parts Void#Zones Void#Parts VoidVol(Mpc/h^3) VoidDensRatio"\
		%(sample.numNonEdge, sample.sampleHandle, num_Acc, sample.prefixV))

	#now find void centres and create the _cat.txt files
	edgeFlag = Circumcentres(sample, num_Acc, wtdAvgDens, edgeFlag)
	if sample.useBaryC:
		if not os.access(sample.outputFolder+"barycentres/", os.F_OK):
			os.makedirs(sample.outputFolder+"barycentres/")
		Barycentres(sample, num_Acc, edgeFlag)

	if sample.isBox:
		#tracer and DM densities around void centres
		print "Calculating structure metrics ...\n"
		StructMetrics(sample)
	
	#find and record void boundaries
	#print "Finding void boundaries ..."
	#StructureBoundary(sample,True)
#----------------------------------------------------------------------------------------------#


#------------postprocess superclusters according to specified merging and selection criteria-----------#
#------------------call routines to find supercluster centres and record _cat.txt files---------------#
def postprocClusters(sample):
	"""Postprocess raw ZOBOV output to get a final set of superclusters according to specified criteria.

	Argument sample is of class Sample.
	"""
	
	print "Post-processing superclusters ...\n"

	if sample.stripDensC>0 and sample.maxDensCut<sample.stripDensC:
		print "ERROR: If using the stripDensC functionality, maxDensCut>=stripDensC is required"
		sys.exit(-1)
	if sample.stripDensC>0 and sample.linkDensThreshC<sample.stripDensC:
		print "ERROR: If using the stripDensC functionality, linkDensThreshC>=stripDensC is required"
		sys.exit(-1)
		
	# the files with ZOBOV output
	zoneFile = sample.outputFolder+"rawZOBOV/"+sample.sampleHandle+"c.zone"
	clustFile = sample.outputFolder+"rawZOBOV/"+sample.sampleHandle+"c.void"
	posnFile = sample.outputFolder+sample.sampleHandle+"_pos.dat"
	listFile = sample.outputFolder+"rawZOBOV/"+sample.sampleHandle+"c.txt"
	volFile = sample.outputFolder+"rawZOBOV/"+sample.sampleHandle+".trvol"
	densFile = sample.outputFolder+"rawZOBOV/"+sample.sampleHandle+".vol"
	InfoFile = sample.outputFolder+sample.prefixC+"_cat.txt"
	SkyPosFile = sample.outputFolder+sample.prefixC+"_skypos.txt"

	# new files after post-processing
	newClustFile = sample.outputFolder+sample.prefixC+".void"
	newListFile = sample.outputFolder+sample.prefixC+"_list.txt"

	#load the list of supercluster candidates 
	clustersread = np.loadtxt(listFile,skiprows=2)
	# sort in desc order of max dens
	sorted_order = np.argsort(1./clustersread[:,3])
	clustersread = clustersread[sorted_order]

	num_clusters = len(clustersread[:,0])
	vid = np.asarray(clustersread[:,0],dtype=int)
	edgelist = np.asarray(clustersread[:,1],dtype=int)
	vollist = clustersread[:,4]
	numpartlist = np.asarray(clustersread[:,5],dtype=int)
	rlist = clustersread[:,9]

	# load up the cluster hierarchy
	with open(clustFile,'r') as Fclust:
		hierarchy = Fclust.readlines()
	#sanity check		
	nclusters = int(hierarchy[0])
	if nclusters != num_clusters: 
		print "Unequal cluster numbers in clustfile and listfile, %d and %d!" %(nclusters, num_clusters)
		sys.exit(-1)
	hierarchy = hierarchy[1:]

	#load up the particle-zone info
	zonedata = np.loadtxt(zoneFile,dtype='int',skiprows=1)
	
	#load the VTFE volume information
	with open(volFile,'r') as File:
		Np = np.fromfile(File, dtype=np.int32,count=1)[0]
		if not Np==sample.numTracer: #sanity check
			print "Np = %d in %s.trvol file does not match numTracer = %d!" %(Np,sample.sampleHandle,sample.numTracer)
			sys.exit(-1)
		vols = np.fromfile(File, dtype=np.float64,count=Np)

	#load the VTFE density information
	with open(densFile, 'r') as File:
		Np = np.fromfile(File, dtype=np.int32,count=1)[0]
		if not Np==sample.numTracer: #sanity check
			print "Np = %d in %s.vol file does not match numTracer = %d!" %(Np,sample.sampleHandle,sample.numTracer)
			sys.exit(-1)
		densities = np.fromfile(File, dtype=np.float64,count=Np)
		densities = 1./densities

	#load the tracer particle positions
	with open(posnFile,'r') as File:
		Np = np.fromfile(File, dtype=np.int32,count=1)[0]
		if not Np==sample.numPartTot: #sanity check
			print "Np = %d in %s_pos.dat file does not match numPartTot = %d!" %(Np, sample.sampleHandle, sample.numPartTot)
			sys.exit(-1)
		Posns = np.empty([Np,3])
		Posns[:,0] = np.fromfile(File, dtype=np.float64,count=Np)
		Posns[:,1] = np.fromfile(File, dtype=np.float64,count=Np)
		Posns[:,2] = np.fromfile(File, dtype=np.float64,count=Np)
	Posns = Posns[:sample.numTracer] #only care about real tracers, not buffer mocks

	# mean volume per tracer particle
	meanvol_trc = (sample.boxLen**3.)/sample.numPartTot

	with open(newClustFile,'w') as Fnewclust:
		with open(newListFile,'w') as Fnewlist:
	
			#initialize variables
			counted_zones = np.empty(0,dtype=int)
			edgeFlag = np.empty(0,dtype=int)
			wtdAvgDens = np.empty(0,dtype=int)
			num_Acc = 0
			
			for i in range(num_clusters):
				coredens = clustersread[i,3]
				#clustline = sorted_hierarchy[i].split()
				clustline = hierarchy[sorted_order[i]].split()
				pos = 1
				numzonestoadd = int(clustline[pos])
				finalpos = pos + numzonestoadd + 1;
				rval = float(clustline[pos+1])
				rstopadd = rlist[i]
				StartGrow = False
				num_adds = 0
				if rval >= 1 and coredens > sample.maxDensCut and numpartlist[i] >= sample.NminC \
					and (sample.countAllC or vid[i] not in counted_zones):
					#this zone qualifies as a seed zone
					StartGrow = True
					AddMore = True
					num_Acc += 1
					zonelist = [vid[i]]
					totalVol = vollist[i]
					totalNumParts = numpartlist[i]
					
					while numzonestoadd > 0 and AddMore:
						zonestoadd = np.asarray(clustline[pos+2:pos+numzonestoadd+2],dtype=int)
						dens = coredens / rval
						rsublist = rlist[np.in1d(vid,zonestoadd)]
						volsublist = vollist[np.in1d(vid,zonestoadd)]
						partsublist = numpartlist[np.in1d(vid,zonestoadd)]				
						if dens < sample.linkDensThreshC or sample.dontMergeC or\
							(sample.rThreshC>0 and max(rsublist) > sample.rThreshC):
							#cannot add these zones
							rstopadd = rval
							AddMore = False
							finalpos -= (numzonestoadd+1)
						else:
							#keep adding zones
							zonelist = np.append(zonelist,zonestoadd)
							num_adds += numzonestoadd
							totalVol += np.sum(volsublist)		
							totalNumParts += np.sum(partsublist)	 
						pos += numzonestoadd + 2
						numzonestoadd = int(clustline[pos])
						rval = float(clustline[pos+1])
						if AddMore:
							finalpos = pos + numzonestoadd + 1
					
					if StartGrow:
						counted_zones = np.append(counted_zones,zonelist)
						MemberIDs = np.logical_and(np.logical_or(sample.stripDensC==0,densities[:]>sample.stripDensC) \
							,np.in1d(zonedata,zonelist))
						if sample.stripDensC>0:		#need to recalculate totalVol and totalNumParts after stripping
							totalVol = np.sum(vols[MemberIDs])
							totalNumParts = len(vols[MemberIDs])
						counted_zones = np.append(counted_zones,zonelist)
						if 1 in edgelist[np.in1d(vid,zonestoadd)]: 
							edgeFlag = np.append(edgeFlag,1)
						else: edgeFlag = np.append(edgeFlag,0)
						#average density of member cells weighted by cell volumes
						wAD = np.sum(vols[MemberIDs]*densities[MemberIDs])/np.sum(vols[MemberIDs])
						wtdAvgDens = np.append(wtdAvgDens, wAD)
		
						newclustline = clustline[:finalpos] 
						if not AddMore:
							newclustline.append(str(0))
						newclustline.append(str(rstopadd))
		
						#write line to the output .void file				
						for j in range(len(newclustline)):
							Fnewclust.write(newclustline[j]+'\t')
						Fnewclust.write('\n')
		
						if rstopadd>10**20: rstopadd = -1 #will be true for structures entirely surrounded by edge particles
						#write line to the output _list.txt file
						Fnewlist.write("%d\t%d\t%f\t%d\t%d\t%d\t%f\t%f\n" %(vid[i], int(clustersread[i,2]), coredens, \
							int(clustersread[i,5]), num_adds+1, totalNumParts, totalVol*meanvol_trc, rstopadd))
						
	# tidy up the files
	#insert first line with number of clusters to the new .void file
	with open(newClustFile,'r+') as Fnewclust:
		old = Fnewclust.read()
		Fnewclust.seek(0)
		topline = "%d\n" %num_Acc
		Fnewclust.write(topline+old)
	
   	#insert header to the output _list.txt file
	listdata = np.loadtxt(newListFile)
	np.savetxt(newListFile,listdata,fmt='%d %d %0.6f %d %d %d %0.6f %0.6f',\
		header="%d non-edge tracers in %s, %d %s clusters\nClustID CoreParticle CoreDens Zone#Parts Clust#Zones Clust#Parts ClustVol(Mpc/h^3) ClustDensRatio"\
		%(sample.numNonEdge, sample.sampleHandle, num_Acc, sample.prefixV))

	# now find the maximum density centre locations of the superclusters
	ListArray = np.loadtxt(newListFile) 
	info_output = np.zeros((num_Acc,11))
	with open(newClustFile,'r') as FHierarchy:
		FHierarchy.readline() #skip the first line, contains total number of structures
		for i in range(num_Acc):
			#get the member zones of the structure
			structline = (FHierarchy.readline()).split()
			pos = 1
			AddZones = int(structline[pos])>0
			MemberZones = np.asarray(structline[0],dtype=int)
			while AddZones:
				numzonestoadd = int(structline[pos])
				zonestoadd = np.asarray(structline[pos+2:pos+numzonestoadd+2],dtype=int)
				MemberZones = np.append(MemberZones,zonestoadd)					
				pos +=numzonestoadd + 2
				AddZones = int(structline[pos])>0

			#get the member particles for these zones
			if sample.stripDensC>0:
				MemberIDs = np.logical_and(densities[:]>sample.stripDensC,np.in1d(zonedata,MemberZones))
			else:	#stripDens functionality disabled
				MemberIDs = np.in1d(zonedata,MemberZones)
			MemberVol = vols[MemberIDs]
			MemberDens = densities[MemberIDs]
 				
			# centre location is position of max. density member particle
			CorePartID = int(ListArray[i,1])
			Centre = Posns[CorePartID]
				
			#total volume of structure in Mpc/h, and effective radius
			VoidVol = np.sum(MemberVol) * meanvol_trc  
			EffRad = (3.0*VoidVol/(4*np.pi))**(1.0/3)

			# average density of member cells weighted by cell volumes
			WtdAvgDens = np.sum(MemberDens * MemberVol) / np.sum(MemberVol)

			if sample.isBox:
				info_output[i] = [ListArray[i,0], Centre[0], Centre[1], Centre[2], EffRad, ListArray[i,2], WtdAvgDens,
									(WtdAvgDens-1)*EffRad**1.6, ListArray[i,7], 0, 0]
			else:
				CentreObs = Centre - sample.boxLen/2.0	# move back into observer coordinates
				CentreDist = np.linalg.norm(CentreObs)
				CentreRed = brentq(lambda x: comovr(x,sample.OmegaM) - CentreDist, 0.0, 1.0)
				CentreDec = 90 - np.degrees(np.arccos(CentreObs[2] / CentreDist))
				CentreRA = np.degrees(np.arctan2(CentreObs[1],CentreObs[0]))
				if CentreRA < 0:
					CentreRA += 360  #to get RA in the range 0 to 360
				EffAngRad = np.degrees(EffRad/CentreDist)			
				info_output[i] = [ListArray[i,0], CentreRA, CentreDec, CentreRed, EffRad, ListArray[i,2], WtdAvgDens, 
							(WtdAvgDens-1)*EffRad**1.6, ListArray[i,7], EffAngRad, edgeFlag[i]]
	
	# save output data to file
	header = "%d superclusters from %s\n" %(num_Acc, sample.sampleHandle)
	if sample.isBox:
		header = header + "ClustID CentreXYZ[3](Mpc/h) R_eff(Mpc/h) MaxDens WtdAvgDens lambda_c DensRatio ignore ignore"
		np.savetxt(InfoFile,info_output,fmt='%d %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %d %d',header=header)
	else:
		header = header + "ClustID CentreRA(deg) CentreDec(deg) redshift R_eff(Mpc/h) MaxDens WtdAvgDens lambda_c DensRatio Theta_eff(deg) EdgeFlag"
		np.savetxt(InfoFile,info_output,fmt='%d %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %d',header=header)
		
#	if sample.useBaryC: 	#also find cluster barycentres
#		if not os.access(sample.outputFolder+"barycentres/", os.F_OK):
#			os.makedirs(sample.outputFolder+"barycentres/")
#		Barycentres(sample, num_Acc, edgeFlag, Voids=False)
	
	if sample.isBox:
		#tracer and DM densities around cluster centres
		print "Calculating densities around supercluster centres ..."
		StructMetrics(sample,voids=False)

#------------------------------------------#
#------------------------------------------#

#---------find barycentres and related info for a processed list of structures---------#
#---------------------------record output in _cat.txt file----------------------------#
def Barycentres(sample, num_struct, edgeFlag, Voids=True):
	"""Goes through a processed list of structures, gathers member particles and find barycentres and related information.
	Writes output to <prefixV/C>_cat.txt file and, if reqd., <prefixV/C>_skypos.txt file.

	Arguments:
	   sample -- object of type Sample
	   num_struct -- integer, the number of structures in the processed list
	   edgeFlag -- integer array of shape (num_struct,); values are 1 if structure is edge-contaminated, 0 if not
	   Voids -- boolean, True if structures are voids, False if superclusters
	"""
	
	print "Identified %d structures. Now extracting barycentre info ...\n" %num_struct

	# set the filenames
	volFile = sample.outputFolder+"rawZOBOV/"+sample.sampleHandle+".trvol"
	densFile = sample.outputFolder+"rawZOBOV/"+sample.sampleHandle+".vol"
	posnFile = sample.outputFolder+sample.sampleHandle+"_pos.dat"
	if Voids:
		zoneFile = sample.outputFolder+"rawZOBOV/"+sample.sampleHandle+".zone"
		HierarchyFile = sample.outputFolder+sample.prefixV+".void"
		ListFile = sample.outputFolder+sample.prefixV+"_list.txt"
#		InfoFile = sample.outputFolder+"barycentres/"+sample.prefixV.replace("Voids","_baryC_Voids")+"_cat.txt"
#		SkyPosFile = sample.outputFolder+"barycentres/"+sample.prefixV.replace("Voids","_baryC_Voids")+"_skypos.txt"
		InfoFile = sample.outputFolder+"barycentres/"+sample.prefixV+"_baryC_cat.txt"
		SkyPosFile = sample.outputFolder+"barycentres/"+sample.prefixV+"_baryC_Voids_skypos.txt"
	else:	#i.e. for superclusters
		zoneFile = sample.outputFolder+"rawZOBOV/"+sample.sampleHandle+"c.zone"
		HierarchyFile = sample.outputFolder+sample.prefixC+".void"
		ListFile = sample.outputFolder+sample.prefixC+"_list.txt"
		InfoFile = sample.outputFolder+sample.prefixC+"_cat.txt"
		SkyPosFile = sample.outputFolder+sample.prefixC+"_skypos.txt"
	
	#load up the particle-zone info
	zonedata = np.loadtxt(zoneFile,dtype='int',skiprows=1)
	
	#load the VTFE volume information
	with open(volFile,'r') as File:
		Np = np.fromfile(File, dtype=np.int32,count=1)[0]
		if not Np==sample.numTracer: #sanity check
			print "Np = %d in %s.trvol file does not match numTracer = %d!" %(Np,sample.sampleHandle,sample.numTracer)
			sys.exit(-1)
		vols = np.fromfile(File, dtype=np.float64,count=Np)

	#load the VTFE density information
	with open(densFile, 'r') as File:
		Np = np.fromfile(File, dtype=np.int32,count=1)[0]
		if not Np==sample.numTracer: #sanity check
			print "Np = %d in %s.vol file does not match numTracer = %d!" %(Np,sample.sampleHandle,sample.numTracer)
			sys.exit(-1)
		densities = np.fromfile(File, dtype=np.float64,count=Np)
		densities = 1./densities

	with open(posnFile,'r') as File:
		Np = np.fromfile(File, dtype=np.int32,count=1)[0]
		if not Np==sample.numPartTot: #sanity check
			print "Np = %d in %s_pos.dat file does not match numPartTot = %d!" %(Np, sample.sampleHandle, sample.numPartTot)
			sys.exit(-1)
		Posns = np.empty([Np,3])
		Posns[:,0] = np.fromfile(File, dtype=np.float64,count=Np)
		Posns[:,1] = np.fromfile(File, dtype=np.float64,count=Np)
		Posns[:,2] = np.fromfile(File, dtype=np.float64,count=Np)
	Posns = Posns[:sample.numTracer] #only care about real tracers, not buffer mocks
	
	# get the mean volume per tracer
	meanvol_trc = (sample.boxLen**3.)/sample.numPartTot
	
	ListArray = np.loadtxt(ListFile,skiprows=2)
	if sample.isBox:
        	info_output = np.zeros((num_struct,9))
    	else:
        	info_output = np.zeros((num_struct,11))
	with open(HierarchyFile,'r') as FHierarchy:
		FHierarchy.readline() #skip the first line, contains total number of structures
		for i in range(num_struct):
			#get the member zones of the structure
			structline = (FHierarchy.readline()).split()
			pos = 1
			AddZones = int(structline[pos])>0
			MemberZones = np.asarray(structline[0],dtype=int)
			while AddZones:
				numzonestoadd = int(structline[pos])
				zonestoadd = np.asarray(structline[pos+2:pos+numzonestoadd+2],dtype=int)
				MemberZones = np.append(MemberZones,zonestoadd)					
				pos +=numzonestoadd + 2
				AddZones = int(structline[pos])>0

			#get the member particles for these zones
			if sample.stripDensV>0:
				MemberIDs = np.logical_and(densities[:]<sample.stripDensV,np.in1d(zonedata,MemberZones))
			else:	#stripDens functionality disabled
				MemberIDs = np.in1d(zonedata,MemberZones)
			MemberX = Posns[MemberIDs,0] - Posns[int(ListArray[i,1]),0]	#position relative to core particle
			MemberY = Posns[MemberIDs,1] - Posns[int(ListArray[i,1]),1]
			MemberZ = Posns[MemberIDs,2] - Posns[int(ListArray[i,1]),2]
			MemberVol = vols[MemberIDs]
			MemberDens = densities[MemberIDs]
 
			if sample.isBox:
				#must account for periodic boundary conditions, assume box coordinates in range [0,boxLen]!
				shiftVec = np.zeros((len(MemberX),3))
				shiftXIDs = abs(MemberX)>sample.boxLen/2.0
				shiftYIDs = abs(MemberY)>sample.boxLen/2.0
				shiftZIDs = abs(MemberZ)>sample.boxLen/2.0
				shiftVec[shiftXIDs,0] = -np.copysign(sample.boxLen,MemberX[shiftXIDs])
				shiftVec[shiftYIDs,1] = -np.copysign(sample.boxLen,MemberY[shiftYIDs])
				shiftVec[shiftZIDs,2] = -np.copysign(sample.boxLen,MemberZ[shiftZIDs])
				MemberX += shiftVec[:,0]
				MemberY += shiftVec[:,1]
				MemberZ += shiftVec[:,2]
			
			#volume-weighted barycentre of the structure
			Centre = np.empty(3)
			Centre[0] = np.sum(MemberX * MemberVol / np.sum(MemberVol)) + Posns[int(ListArray[i,1]),0]
			Centre[1] = np.sum(MemberY * MemberVol / np.sum(MemberVol)) + Posns[int(ListArray[i,1]),1]
			Centre[2] = np.sum(MemberZ * MemberVol / np.sum(MemberVol)) + Posns[int(ListArray[i,1]),2]

			#put centre coords back within the fiducial box if they have leaked out
			if sample.isBox:
				if Centre[0]<0 or Centre[0]>sample.boxLen: 
					Centre[0] -= sample.boxLen*np.sign(Centre[0])
				if Centre[1]<0 or Centre[1]>sample.boxLen: 
					Centre[1] -= sample.boxLen*np.sign(Centre[1])
				if Centre[2]<0 or Centre[2]>sample.boxLen: 
					Centre[2] -= sample.boxLen*np.sign(Centre[2])
				
			#total volume of structure in Mpc/h, and effective radius
			VoidVol = np.sum(MemberVol) * meanvol_trc  
			EffRad = (3.0*VoidVol/(4*np.pi))**(1.0/3)

			# average density of member cells weighted by cell volumes
			WtdAvgDens = np.sum(MemberDens * MemberVol) / np.sum(MemberVol)

			if Voids: 
				Lambda = (WtdAvgDens-1)*EffRad**1.2
			else:
				Lambda = (WtdAvgDens-1)*EffRad**1.6

			#if required, write sky positions to file
			if sample.isBox:
				info_output[i] = [ListArray[i,0], Centre[0], Centre[1], Centre[2], EffRad, (ListArray[i,2]-1.), (WtdAvgDens-1.),
									Lambda, ListArray[i,7]]
			else:
				CentreObs = Centre - sample.boxLen/2.0	# move back into observer coordinates
				CentreDist = np.linalg.norm(CentreObs)
				if (CentreDist>=comovr(sample.zMin,sample.OmegaM)) and (CentreDist<=comovr(sample.zMax,sample.OmegaM)):
					CentreRed = brentq(lambda x: comovr(x,sample.OmegaM) - CentreDist, 0.0, 1.0)
					CentreDec = 90 - np.degrees(np.arccos(CentreObs[2] / CentreDist))
					CentreRA = np.degrees(np.arctan2(CentreObs[1],CentreObs[0]))
					if CentreRA < 0:
						CentreRA += 360  #to get RA in the range 0 to 360
					EffAngRad = np.degrees(EffRad/CentreDist)
					mask = hp.read_map(sample.maskFile,verbose=False)
					nside = hp.get_nside(mask)
					pixel = hp.ang2pix(nside,np.deg2rad(90-CentreDec),np.deg2rad(CentreRA))
					if mask[pixel]==0:	#something has gone wrong at tessellation stage
						CentreRed = -1
						CentreDec = -60; CentreRA = -60
						EffAngRad = 0
						edgeFlag[i] = 2						
				else:		#something has gone wrong at tessellation stage
					CentreRed = -1
					CentreDec = -60; CentreRA = -60
					EffAngRad = 0
					edgeFlag[i] = 2
				info_output[i] = [ListArray[i,0], CentreRA, CentreDec, CentreRed, EffRad, (ListArray[i,2]-1.), (WtdAvgDens-1.),
							Lambda, ListArray[i,7], EffAngRad, edgeFlag[i]]
							
	# save output data to file
	if Voids:
		header = "%d voids from %s\n" %(num_struct, sample.sampleHandle) 
		if sample.isBox:
			header = header + "VoidID CentreXYZ[3](Mpc/h) R_eff(Mpc/h) delta_min delta_avg lambda_v DensRatio"
			np.savetxt(InfoFile,info_output,fmt='%d %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f',header=header)
		else: 
			header = header + "VoidID CentreRA(deg) CentreDec(deg) redshift R_eff(Mpc/h) delta_min delta_avg lambda_v DensRatio Theta_eff(deg) EdgeFlag"
			np.savetxt(InfoFile,info_output,fmt='%d %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %d',header=header)
	else:
		header = "%d superclusters from %s\n" %(num_struct, sample.sampleHandle) 
		if sample.isBox:
			header = header + "ClustID CentreXYZ[3](Mpc/h) R_eff(Mpc/h) delta_max delta_avg lambda_c DensRatio"
			np.savetxt(InfoFile,info_output,fmt='%d %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f',header=header)
		else:
			header = header + "ClustID CentreRA(deg) CentreDec(deg) redshift R_eff(Mpc/h) delta_max delta_avg lambda_c DensRatio Theta_eff(deg) EdgeFlag"
			np.savetxt(InfoFile,info_output,fmt='%d %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %d',header=header)

#----------------------------------------------#

#---------find "circumcentres" and related info for a processed list of structures---------#
#-----------------------------record output in _cat.txt file------------------------------#
def Circumcentres(sample, num_struct, wtdAvgDens, edgeFlag):
	"""Goes through a processed list of voids, finds minimum density centres ('circumcentres') and related information.
	Writes output to <prefixV>_cat.txt file and, if reqd., <prefixV>_skypos.txt file.

	Arguments:
	   sample -- object of type Sample
	   num_struct -- integer, the number of voids in the processed list
	   wtdAvgDens -- float array of shape (num_struct,); values are (weighted) average densities of each void
	   edgeFlag -- integer array of shape (num_struct,); values are 1 if void is edge-contaminated, 0 if not
	"""

	print "Identified %d structures. Now extracting circumcentre info ...\n" %num_struct

	# set the filenames
	densFile = sample.outputFolder+"rawZOBOV/"+sample.sampleHandle+".vol"
	adjacencyFile = sample.outputFolder+"rawZOBOV/"+sample.sampleHandle+".adj"
	posnFile = sample.outputFolder+sample.sampleHandle+"_pos.dat"
	ListFile = sample.outputFolder+sample.prefixV+"_list.txt"
	InfoFile = sample.outputFolder+sample.prefixV+"_cat.txt"
	SkyPosFile = sample.outputFolder+sample.prefixV+"_skypos.txt"
	
	#load the VTFE density information
	with open(densFile, 'r') as File:
		Np = np.fromfile(File, dtype=np.int32,count=1)[0]
		if not Np==sample.numTracer: #sanity check
			print "Np = %d in %s.vol file does not match numTracer = %d!" %(Np,sample.sampleHandle,sample.numTracer)
			sys.exit(-1)
		densities = np.fromfile(File, dtype=np.float64,count=Np)
		densities = 1./densities

	with open(posnFile,'r') as File:
		Np = np.fromfile(File, dtype=np.int32,count=1)[0]
		if not Np==sample.numPartTot: #sanity check
			print "Np = %d in %s_pos.dat file does not match numPartTot = %d!" %(Np, sample.sampleHandle, sample.numPartTot)
			sys.exit(-1)
		Posns = np.empty([Np,3])
		Posns[:,0] = np.fromfile(File, dtype=np.float64,count=Np)
		Posns[:,1] = np.fromfile(File, dtype=np.float64,count=Np)
		Posns[:,2] = np.fromfile(File, dtype=np.float64,count=Np)
	Posns = Posns[:sample.numTracer] #only care about real tracers, not buffer mocks

	ListArray = np.loadtxt(ListFile) #,skiprows=2)
	vID = np.asarray(ListArray[:,0],dtype=int) 
	corepart = np.asarray(ListArray[:,1],dtype=int)

	#read and assign adjacencies from ZOBOV output
	with open(adjacencyFile,'r') as AdjFile:
		Npfromadj = np.fromfile(AdjFile,dtype=np.int32,count=1)
		if not Npfromadj == sample.numTracer:
			# sanity check
			print "Np = %d from adjacency file does not match numTracer = %d!" %(Npfromadj, sample.numTracer)
			sys.exit(-1)
		partadjs = [[] for i in range(Npfromadj)]	#list of lists to record adjacencies - is there a better way?
		partadjcount = np.zeros(Npfromadj,dtype=np.int32) #counter to monitor adjacencies
		nadj= np.fromfile(AdjFile,dtype=np.int32,count=Npfromadj) #number of adjacencies for each particle
		#load up the adjacencies from ZOBOV output
		for i in range(Npfromadj):	    
			numtomatch = np.fromfile(AdjFile,dtype=np.int32,count=1) 
			if numtomatch > 0: 
				# particle numbers of adjacent particles
				adjpartnumbers = np.fromfile(AdjFile,dtype=np.int32,count=numtomatch) 
				# keep track of how many adjacencies had already been assigned
				oldcount = partadjcount[i] 
				newcount = oldcount+len(adjpartnumbers) 
				partadjcount[i] = newcount
				# and now assign new adjacencies
				partadjs[i][oldcount:newcount] = adjpartnumbers 
				# now also assign the reverse adjacencies
				# (ZOBOV records only (i adj j) or (j adj i), not both)
				for index in adjpartnumbers: partadjs[index].append(i)	
				partadjcount[adjpartnumbers] += 1

	if sample.isBox:
	        info_output = np.zeros((num_struct,9))
    	else:
        	info_output = np.zeros((num_struct,11))
	CircCentre = np.empty(3)
	# now loop over void cores, calculating circumcentres and writing to file
	for i in range(num_struct):

		#get adjacencies of the core particle
		coreadjs = partadjs[corepart[i]]
		adjDens = densities[coreadjs]

		#get the 3 lowest density mutually adjacent neighbours of the core particle
		firstNbr = coreadjs[np.argmin(adjDens)]
		mutualadjs = np.intersect1d(coreadjs,partadjs[firstNbr])
		if len(mutualadjs)==0:
			CircCentre = np.asarray([0,0,0])
			edgeFlag[i] =2
		else:
			mutualadjDens = densities[mutualadjs]
			secondNbr = mutualadjs[np.argmin(mutualadjDens)]
			finaladjs = np.intersect1d(mutualadjs,partadjs[secondNbr])
			if len(finaladjs)==0:		#something has gone wrong at tessellation stage!
				CircCentre = np.asarray([0,0,0])
				edgeFlag[i] = 2
			else:			#can calculate circumcentre position
				finaladjDens = densities[finaladjs]
				thirdNbr = finaladjs[np.argmin(finaladjDens)]

				#collect positions of the vertices
				VertexPos = np.zeros((4,3))
				VertexPos[0,:] = Posns[corepart[i],:]
				VertexPos[1,:] = Posns[firstNbr,:]
				VertexPos[2,:] = Posns[secondNbr,:]
				VertexPos[3,:] = Posns[thirdNbr,:]
				if sample.isBox:	#need to adjust for periodic BC
					shiftInds = abs(VertexPos[0,0]-VertexPos[:,0])>sample.boxLen/2.0
					VertexPos[shiftInds,0] += sample.boxLen*np.sign(VertexPos[0,0]-VertexPos[shiftInds,0])					
					shiftInds = abs(VertexPos[0,1]-VertexPos[:,1])>sample.boxLen/2.0
					VertexPos[shiftInds,1] += sample.boxLen*np.sign(VertexPos[0,1]-VertexPos[shiftInds,1])					
					shiftInds = abs(VertexPos[0,2]-VertexPos[:,2])>sample.boxLen/2.0
					VertexPos[shiftInds,2] += sample.boxLen*np.sign(VertexPos[0,2]-VertexPos[shiftInds,2])					

				#solve for the circumcentre
				#for more details on this method and its stability, see http://www.ics.uci.edu/~eppstein/junkyard/circumcentre.html
				A = np.bmat([[2*np.dot(VertexPos,VertexPos.T), np.ones((4,1))],
						[np.ones((1,4)), np.zeros((1,1))]])
				b = np.hstack((np.sum(VertexPos*VertexPos, axis=1), np.ones((1))))
				x = np.linalg.solve(A,b)
				bary_coords = x[:-1]
				CircCentre[:] = np.dot(bary_coords,VertexPos)

		if sample.isBox:	
			#put centre coords back within the fiducial box if they have leaked out
			if CircCentre[0]<0 or CircCentre[0]>sample.boxLen:
				CircCentre[0] -= sample.boxLen*np.sign(CircCentre[0])
			if CircCentre[1]<0 or CircCentre[1]>sample.boxLen:
				CircCentre[1] -= sample.boxLen*np.sign(CircCentre[1])
			if CircCentre[2]<0 or CircCentre[2]>sample.boxLen:
				CircCentre[2] -= sample.boxLen*np.sign(CircCentre[2])

		#calculate void effective radius
		EffRad = (3.0*ListArray[i,6]/(4*np.pi))**(1.0/3)

		#if required, write sky positions to file
		if sample.isBox:
			info_output[i] = [vID[i], CircCentre[0], CircCentre[1], CircCentre[2], EffRad, (ListArray[i,2]-1.), (wtdAvgDens[i]-1.),
								(wtdAvgDens[i]-1)*EffRad**1.2, ListArray[i,7]]
		else:
			CentreObs = CircCentre - sample.boxLen/2.0	# move back into observer coordinates
			CentreDist = np.linalg.norm(CentreObs)
			if (CentreDist>=comovr(sample.zMin,sample.OmegaM)) and (CentreDist<=comovr(sample.zMax,sample.OmegaM)):
				CentreRed = brentq(lambda x: comovr(x,sample.OmegaM) - CentreDist, 0.0, 1.0)
				CentreDec = 90 - np.degrees(np.arccos(CentreObs[2] / CentreDist))
				CentreRA = np.degrees(np.arctan2(CentreObs[1],CentreObs[0]))
				if CentreRA < 0:
					CentreRA += 360  #to get RA in the range 0 to 360
				EffAngRad = np.degrees(EffRad/CentreDist)
				mask = hp.read_map(sample.maskFile,verbose=False)
				nside = hp.get_nside(mask)
				pixel = hp.ang2pix(nside,np.deg2rad(90-CentreDec),np.deg2rad(CentreRA))
				if mask[pixel]==0:	#something has gone wrong at tessellation stage
					CentreRed = -1
					CentreDec = -60; CentreRA = -60
					EffAngRad = 0
					edgeFlag[i] = 2						
			else:		#something has gone wrong at tessellation stage
				CentreRed = -1
				CentreDec = -60; CentreRA = -60
				EffAngRad = 0
				edgeFlag[i] = 2
			info_output[i] = [vID[i], CentreRA, CentreDec, CentreRed, EffRad, (ListArray[i,2]-1.), (wtdAvgDens[i]-1.),
						(wtdAvgDens[i]-1)*EffRad**1.2, ListArray[i,7], EffAngRad, edgeFlag[i]]

	# save output data to file
	header = "%d voids from %s\n" %(num_struct, sample.sampleHandle) 
	if sample.isBox:
		header = header + "VoidID CentreXYZ[3](Mpc/h) R_eff(Mpc/h) delta_min delta_avg lambda_v DensRatio"
		np.savetxt(InfoFile,info_output,fmt='%d %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f',header=header)
	else: 
		header = header + "VoidID CentreRA(deg) CentreDec(deg) redshift R_eff(Mpc/h) delta_min delta_avg lambda_v DensRatio Theta_eff(deg) EdgeFlag"
		np.savetxt(InfoFile,info_output,fmt='%d %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %d',header=header)

	return edgeFlag
#------------------------------------------------------------------------------------------#

#-----------find positions defining boundaries for a processed list of structures--------------#
#---------------------------record output in _boundaries.txt file------------------------------#
def StructureBoundary(sample,Voids=True):
	"""Goes through a processed list of structures, finds the tracer particles defining the boundary
	Writes output to <sampleHandle>_boundaries.dat file

	Arguments:
	   sample -- object of type Sample
	   Voids -- True if finding boundary of voids, False for clusters
	"""

	# set the filenames
	densFile = sample.outputFolder+"rawZOBOV/"+sample.sampleHandle+".vol"
	adjacencyFile = sample.outputFolder+"rawZOBOV/"+sample.sampleHandle+".adj"
	posnFile = sample.outputFolder+sample.sampleHandle+"_pos.dat"
	if Voids:
		zoneFile = sample.outputFolder+"rawZOBOV/"+sample.sampleHandle+".zone"
		HierarchyFile = sample.outputFolder+sample.prefixV+".void"
		ListFile = sample.outputFolder+sample.prefixV+"_list.txt"
		boundaryFile = sample.outputFolder+sample.prefixV+"_boundaries.dat"
	else:	#i.e. for superclusters
		zoneFile = sample.outputFolder+"rawZOBOV/"+sample.sampleHandle+"c.zone"
		HierarchyFile = sample.outputFolder+sample.prefixC+".void"
		ListFile = sample.outputFolder+sample.prefixC+"_list.txt"
		boundaryFile = sample.outputFolder+sample.prefixC+"_boundaries.dat"
	
	ListArray = np.loadtxt(ListFile) #,skiprows=2)
	num_struct = len(ListArray)
	print "Identifying boundaries for %d structures ..." %num_struct

	#load up the particle-zone info
	zonedata = np.loadtxt(zoneFile,dtype='int',skiprows=1)

	print "	Loading adjacency information ..."
	#read and assign adjacencies from ZOBOV output
	with open(adjacencyFile,'r') as AdjFile:
		Npfromadj = np.fromfile(AdjFile,dtype=np.int32,count=1)
		if not Npfromadj == sample.numTracer:
			# sanity check
			print "Np = %d from adjacency file does not match numTracer = %d!" %(Npfromadj, sample.numTracer)
			sys.exit(-1)
		partadjs = [[] for i in range(Npfromadj)]	#list of lists to record adjacencies - is there a better way?
		partadjcount = np.zeros(Npfromadj,dtype=np.int32) #counter to monitor adjacencies
		nadj= np.fromfile(AdjFile,dtype=np.int32,count=Npfromadj) #number of adjacencies for each particle
		#load up the adjacencies from ZOBOV output
		for i in range(Npfromadj):	    
			numtomatch = np.fromfile(AdjFile,dtype=np.int32,count=1) 
			if numtomatch > 0: 
				# particle numbers of adjacent particles
				adjpartnumbers = np.fromfile(AdjFile,dtype=np.int32,count=numtomatch) 
				# keep track of how many adjacencies had already been assigned
				oldcount = partadjcount[i] 
				newcount = oldcount+len(adjpartnumbers) 
				partadjcount[i] = newcount
				# and now assign new adjacencies
				partadjs[i][oldcount:newcount] = adjpartnumbers 
				# now also assign the reverse adjacencies
				# (ZOBOV records only (i adj j) or (j adj i), not both)
				for index in adjpartnumbers: partadjs[index].append(i)	
				partadjcount[adjpartnumbers] += 1

	#load the VTFE density information
	File = file(densFile)
	Np = np.fromfile(File, dtype=np.int32,count=1)[0]
	if not Np==sample.numTracer: #sanity check
		print "Np = %d in %s.vol file does not match numTracer = %d!" %(Np,sample.sampleHandle,sample.numTracer)
		sys.exit(-1)
	densities = np.fromfile(File, dtype=np.float64,count=Np)
	densities = 1./densities
	File.close()

	#load the tracer particle positions
	File = file(posnFile)
	Np = np.fromfile(File, dtype=np.int32,count=1)[0]
	if not Np==sample.numPartTot: #sanity check
		print "Np = %d in %s_pos.dat file does not match numPartTot = %d!" %(Np, sample.sampleHandle, sample.numPartTot)
		sys.exit(-1)
	Posns = np.empty([Np,3])
	Posns[:,0] = np.fromfile(File, dtype=np.float64,count=Np)
	Posns[:,1] = np.fromfile(File, dtype=np.float64,count=Np)
	Posns[:,2] = np.fromfile(File, dtype=np.float64,count=Np)
	File.close()	
	Posns = Posns[:sample.numTracer] #only care about real tracers, not buffer mocks
	
	#list of indices for boundary particles for each structure
	BoundaryParts = [[] for i in range(num_struct)]
	#arrays of points defining boundary for each structure
	BoundaryPosn = [np.zeros((1,3)) for i in range(num_struct)]
#	BoundaryPosn = [np.zeros((1,3)) for i in range(10)]	#to speed things up for DR11 batch processing

	with open(boundaryFile,'w') as Fbound:
		np.asarray(num_struct,dtype=np.int32).tofile(Fbound,format="%d")
		with open(HierarchyFile,'r') as FHierarchy:
			FHierarchy.readline() #skip the first line, contains total number of structures
			for i in range(num_struct):  
#			for i in range(10):	#to speed things up for DR11 batch processing
				if i%1000==0 and i>0: print "	Done %d structures..." %i
				#get the member zones of the structure
				structline = (FHierarchy.readline()).split()
				pos = 1
				AddZones = int(structline[pos])>0
				MemberZones = np.asarray(structline[0],dtype=int)
				while AddZones:
					numzonestoadd = int(structline[pos])
					zonestoadd = np.asarray(structline[pos+2:pos+numzonestoadd+2],dtype=int)
					MemberZones = np.append(MemberZones,zonestoadd)					
					pos +=numzonestoadd + 2
					AddZones = int(structline[pos])>0

				#get the member particles for these zones
				if sample.stripDensV>0:
					MemberIDs = np.logical_and(densities[:]<sample.stripDensV,np.in1d(zonedata,MemberZones))
				else:	#stripDens functionality disabled
					MemberIDs = np.in1d(zonedata,MemberZones)

				MemberParts = np.arange(sample.numTracer)[MemberIDs]

				#check adjacencies of member particles to find those on boundary
				for j in range(len(MemberParts)):
					neighbours = np.asarray(partadjs[MemberParts[j]],dtype=int)
					out_inds = np.in1d(neighbours,MemberParts,invert=True)
					out_neighbours = neighbours[out_inds]
					if len(out_neighbours)>0:
						# jth member particle is on the void boundary
						BoundaryParts[i].append(MemberParts[j])
						neigh_Posns = Posns[out_neighbours]					
						if sample.isBox: #calculate mid-point accounting for PBC
							delta = Posns[MemberParts[j]] - neigh_Posns
							Xleak = np.abs(delta[:,0])>sample.boxLen/2.
							Yleak = np.abs(delta[:,1])>sample.boxLen/2.
							Zleak = np.abs(delta[:,2])>sample.boxLen/2.
							neigh_Posns[Xleak,0] += sample.boxLen*np.sign(delta[Xleak,0])
							neigh_Posns[Yleak,1] += sample.boxLen*np.sign(delta[Yleak,1])
							neigh_Posns[Zleak,2] += sample.boxLen*np.sign(delta[Zleak,2])
							halfway = (Posns[MemberParts[j]] + neigh_Posns)/2.0
							#put midpoints back into the box if necessary
							Xleak = np.logical_or(halfway[:,0]<0,halfway[:,0]>sample.boxLen)
							Yleak = np.logical_or(halfway[:,1]<0,halfway[:,1]>sample.boxLen)
							Zleak = np.logical_or(halfway[:,2]<0,halfway[:,2]>sample.boxLen)
							halfway[Xleak,0] -= sample.boxLen*np.sign(halfway[Xleak,0])
							halfway[Yleak,1] -= sample.boxLen*np.sign(halfway[Yleak,1])
							halfway[Zleak,2] -= sample.boxLen*np.sign(halfway[Zleak,2])
						else:
							halfway = (Posns[MemberParts[j]] + neigh_Posns)/2.0
						BoundaryPosn[i] = np.append(BoundaryPosn[i],halfway,axis=0)
				#remove the leading zeros
				BoundaryPosn[i] = BoundaryPosn[i][1:]

				# write the boundary data to file
				# first the structure ID
				np.asarray(ListArray[i,0],dtype=np.int32).tofile(Fbound,format="%d")
				# then the number of boundary coordinates to be written
				np.asarray(len(BoundaryPosn[i]),dtype=np.int32).tofile(Fbound,format="%d")
				# then the actual coordinates, in standard ZOBOV format
				filler = np.asarray(BoundaryPosn[i][:,0],dtype=np.float64)
				filler.tofile(Fbound,format="%f")
				filler = np.asarray(BoundaryPosn[i][:,1],dtype=np.float64)
				filler.tofile(Fbound,format="%f")
				filler = np.asarray(BoundaryPosn[i][:,2],dtype=np.float64)
				filler.tofile(Fbound,format="%f")

	return BoundaryParts, BoundaryPosn

#------------------------------------------------------------------------------------------#

#------------read in the structure boundary positions stored using StructureBoundary()--------#
#---------------------------------return in a list of arrays----------------------------------#
def read_boundaries(filename):
	"""reads in the boundary positions as written to file by StructureBoundary function
	returns as a list of arrays for further analysis	

	Arguments:
	   filename -- filename in which boundary is stored

	Returns:
	   structIDs -- structure ID numbers
	   BoundaryPosns -- a list of arrays; list length is equal to length of structIDs, each 
			    array contains boundary positions for corresponding structure
	"""

	with open(filename,'r') as Fread:
		num_struct = np.fromfile(Fread,dtype=np.int32,count=1)
		structIDs = np.zeros((num_struct))
		part_numbers = np.zeros((num_struct),dtype=int)
		BoundaryPosns = [np.zeros((1,3)) for i in range(num_struct)]
		for i in range(num_struct):
			structIDs[i] = np.fromfile(Fread,dtype=np.int32,count=1)
			part_numbers[i] = np.fromfile(Fread,dtype=np.int32,count=1)
			BoundaryPosns[i] = np.empty((part_numbers[i],3))
			BoundaryPosns[i][:,0] = np.fromfile(Fread,dtype=np.float64,count=part_numbers[i])
			BoundaryPosns[i][:,1] = np.fromfile(Fread,dtype=np.float64,count=part_numbers[i])
			BoundaryPosns[i][:,2] = np.fromfile(Fread,dtype=np.float64,count=part_numbers[i])

	return structIDs, BoundaryPosns
#-------------------------------------------------------------------------------------#

#------------------------------------------------------------------------------------------#
#------------------compute various density and potential metrics for structures-------------------#
def StructMetrics(sample,voids=True):
	"""Goes through a processed list of structures, calculates various density- and potential-based metrics
	Writes output to <sampleHandle>_metrics.txt file

	Arguments:
	   sample -- object of type Sample
	   voids -- boolean, True if structures are voids, False if superclusters
	"""

	if voids:
		InfoFile = sample.outputFolder + sample.prefixV + "_cat.txt"
		ListFile = sample.outputFolder + sample.prefixV + "_list.txt"
		OutFile = sample.outputFolder + sample.prefixV + "_metrics.txt"
	else:
		InfoFile = sample.outputFolder + sample.prefixC + "_cat.txt"
		ListFile = sample.outputFolder + sample.prefixC + "_list.txt"
		OutFile = sample.outputFolder + sample.prefixC + "_metrics.txt"
	CatArray = np.loadtxt(InfoFile) #,skiprows=2)
	ListArray = np.loadtxt(ListFile) #,skiprows=2)
	centres = CatArray[:,1:4]
	radii = CatArray[:,4]
	vid = CatArray[:,0].astype(int)
	
	print "Loading tracer particle data..."
	posnFile = sample.outputFolder + sample.sampleHandle + "_pos.dat"
	File = file(posnFile)
	Np = np.fromfile(File, dtype=np.int32,count=1)[0]
	Posns = np.empty([Np,3])
	Posns[:,0] = np.fromfile(File, dtype=np.float64,count=Np)
	Posns[:,1] = np.fromfile(File, dtype=np.float64,count=Np)
	Posns[:,2] = np.fromfile(File, dtype=np.float64,count=Np)
	File.close()
	Posns = Posns[:sample.numTracer]	# drop mocks, if there are any
	CentPart_inds = ListArray[:,1].astype(int)	#indices of central particles (min/max density particles for voids/clusters)
		
	if sample.isBox:
		# full cubic box, periodic boundary conditions
		print "Building the tracer kd-tree ..."
		Tree = cKDTree(Posns,boxsize=sample.boxLen)
		#also make sure that dist from barycentre to centre particle is correctly calculated
		shiftXInds = abs(CatArray[:,1]-Posns[CentPart_inds[:],0])>sample.boxLen/2.0
		shiftYInds = abs(CatArray[:,2]-Posns[CentPart_inds[:],1])>sample.boxLen/2.0
		shiftZInds = abs(CatArray[:,3]-Posns[CentPart_inds[:],2])>sample.boxLen/2.0
		shiftVec = np.zeros((len(CatArray),3))
		shiftVec[shiftXInds,0] = -np.copysign(sample.boxLen,(CatArray[:,1]-Posns[CentPart_inds[:],0])[shiftXInds])
		shiftVec[shiftYInds,1] = -np.copysign(sample.boxLen,(CatArray[:,2]-Posns[CentPart_inds[:],1])[shiftYInds])
		shiftVec[shiftZInds,2] = -np.copysign(sample.boxLen,(CatArray[:,3]-Posns[CentPart_inds[:],2])[shiftZInds])
	else:
		# no periodic boundary conditions
		print "Building the tracer kd-tree ..."
		Tree = cKDTree(Posns)
	print "\t done\n"
	
	#calculate enclosed tracer densities
	DeltaN = np.zeros((len(radii),2))
	print "Calculating enclosed tracer number densities ...\n"
	for i in range(DeltaN.shape[0]):
		small_vol = (4*np.pi*(radii[i])**3)/3.0
		big_vol = (4*np.pi*(3.0*radii[i])**3)/3.0
		small_nums = len(getBall(Tree,centres[i],radii[i]))
		big_nums = len(getBall(Tree,centres[i],3.0*radii[i]))
		DeltaN[i,0] = (small_nums+1)/(small_vol*sample.tracerDens) - 1
		DeltaN[i,1] = (big_nums+1)/(big_vol*sample.tracerDens) - 1

	if sample.usePhi:	#extract the Phi value at the structure centre
		print "Loading the Phi data from file ...\n"
		Phi = np.load(sample.Phifile)
		#get the grid indices for the structure centres
		xind = np.mod(np.floor(centres[:,0]*sample.Phi_resolution/sample.boxLen),sample.Phi_resolution)
		yind = np.mod(np.floor(centres[:,1]*sample.Phi_resolution/sample.boxLen),sample.Phi_resolution)
		zind = np.mod(np.floor(centres[:,2]*sample.Phi_resolution/sample.boxLen),sample.Phi_resolution)
		#get the Phi values at these grid cells
		Phi_cent_vals = Phi[xind.astype(int),yind.astype(int),zind.astype(int)]*10**5
		#clean up Phi to save memory
		del Phi
	else:		#no Phi data, so set to zero
		Phi_cent_vals = np.zeros_like(vid)

	if sample.useDM:	
		print "Loading the DM density data from file ...\n"
		DMDens = np.load(sample.DMfile)
		DMDens+=1
		#get the grid indices for the structure centres
		xind = np.mod(np.floor(centres[:,0]*sample.DM_resolution/sample.boxLen),sample.DM_resolution)
		yind = np.mod(np.floor(centres[:,1]*sample.DM_resolution/sample.boxLen),sample.DM_resolution)
		zind = np.mod(np.floor(centres[:,2]*sample.DM_resolution/sample.boxLen),sample.DM_resolution)
		#get the DM density values at these grid cells
		DM_cent_vals = DMDens[xind.astype(int),yind.astype(int),zind.astype(int)]
		#now get the enclosed density contrast within Rv and 3*Rv
		print "Calculating enclosed density contrast Delta ..."
		Delta = np.zeros((len(radii),2))
		for i in range(Delta.shape[0]):
			small_vol = (4*np.pi*(radii[i])**3)/3.0
			big_vol = (4*np.pi*(3.0*radii[i])**3)/3.0
			small_mass = annular_DM_healpy(DMDens,centres[i],0,radii[i],sample.DM_resolution)
			big_mass = small_mass + annular_DM_healpy(DMDens,centres[i],radii[i],3.0*radii[i],sample.DM_resolution)
			Delta[i,0] = small_mass/small_vol - 1
			Delta[i,1] = big_mass/big_vol - 1
			if i%10000==0: print "	Done %d structures ..." %i
		#clean up DMDens to save memory
		del DMDens
	else:		#no DM data, so set all DM density measures to zero
		DM_cent_vals = np.zeros_like(vid)
		Delta = np.zeros((len(radii),2))
	
	with open(OutFile,'w') as Fout:
		Fout.write("StructID R_eff(Mpc/h) CentNumDens WtdAvgNumDens CentDMDens Phi_cent*10^5 DeltaN(Rv) Delta(Rv) DeltaN(3Rv) Delta(3Rv)\n")
		for i in range(CatArray.shape[0]):
			Fout.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" %(vid[i], CatArray[i,6], CatArray[i,4], CatArray[i,5], \
				DM_cent_vals[i], Phi_cent_vals[i], DeltaN[i,0], Delta[i,0], DeltaN[i,1], Delta[i,1]))

#----------------------------------------------------------------------------#

#--------------determine the no. of structure hierarchy levels---------------#
def hierarchy_rec(OldLevel,BaseLevel,level_count,voidfile):
	"""Recursively determine the maximum hierarchy depth of a set of structures.

	Arguments:
	  OldLevel -- sequence of central zone numbers for structures at previous hierarchy level
	  BaseLevel -- sequence of central zone numbers for structures at first hierarchy level
	  level_count -- the hierarchy level being determined
	  voidfile -- filename to the .void file in which the full hierarchy is stored
	The first call to hierarchy_rec should be made with OldLevel and BaseLevel both containing the 
	list of central zone numbers for all structures containing sub-structures, and level_count=1.
	"""
	NewLevel = np.empty(0,dtype=int)
	for i in range(len(OldLevel)):
		for line in fileinput.input([voidfile]):
			if int(line.split()[0])==OldLevel[i]:
				voidline = line.split()
				break
		fileinput.close()
		pos = 1
		AddZones = int(voidline[pos])>0
		SubZones = np.empty(0,dtype=int)
		while AddZones:
			numzonestoadd = int(voidline[pos])
			zonestoadd = np.asarray(voidline[pos+2:pos+numzonestoadd+2],dtype=int)
			SubZones = np.append(SubZones,zonestoadd)
			pos += numzonestoadd + 2
			AddZones = int(voidline[pos])>0
		NewLevel = np.append(NewLevel,BaseLevel[np.in1d(BaseLevel,SubZones)])
	NewLevel = np.asarray(list(set(NewLevel)),dtype=int)
	print "Hierarchy level %d: %d sub-structures have further sub-structure" %(level_count, NewLevel.shape[0])
	if NewLevel.shape[0]>0:
		hierarchy_rec(NewLevel,BaseLevel,level_count+1,voidfile)
#----------------------------------------------------------------------------#

#----------------obtain a list of all root-level voids-----------------------#
def get_rootVoids(sample):
	"""Goes through a processed list of voids and finds IDs of those which are root-level voids. 
	Writes output to <sampleHandle>_rootIDs.txt file.

	Arguments:
	   sample -- object of type Sample
	"""

	print "Obtaining list of root-level voids ..."

	voidFile = sample.outputFolder+"rawZOBOV/"+sample.sampleHandle+".void"
	listFile = sample.outputFolder+"rawZOBOV/"+sample.sampleHandle+".txt"

	#load the list of void candidates 
	voidsread = np.loadtxt(listFile,skiprows=2)
	#sort in asc order of min dens
	sorted_order = np.argsort(voidsread[:,3])
	voidsread = voidsread[sorted_order]

	num_voids = len(voidsread[:,0])
	vid = np.asarray(voidsread[:,0],dtype=int)
	numpartlist = np.asarray(voidsread[:,5],dtype=int)
	rlist = voidsread[:,9]

	# load up the void hierarchy
	with open(voidFile,'r') as Fvoid:
		hierarchy = Fvoid.readlines()
	#sanity check		
	nvoids = int(hierarchy[0])
	if nvoids != num_voids: 
		print "Unequal void numbers in voidfile and listfile, %d and %d!" %(nvoids, num_voids)
		sys.exit(-1)
	hierarchy = hierarchy[1:]

	#initialise variables
	counted_zones = np.empty(0,dtype=int)
	root_IDs = np.empty(0,dtype=int)
	num_Acc = 0

	for i in range(num_voids):
		coredens = voidsread[i,3]
		voidline = hierarchy[sorted_order[i]].split()
		pos = 1
		numzonestoadd = int(voidline[pos])
		finalpos = pos + numzonestoadd + 1;
		rval = float(voidline[pos+1])
		StartGrow = False
		num_adds = 0
		if rval >= 1 and coredens <= sample.minDensCut and numpartlist[i] >= sample.NminV \
			and vid[i] not in counted_zones:
			#this zone qualifies as a seed zone
			StartGrow = True
			AddMore = True
			num_Acc += 1
			zonelist = vid[i]
			while numzonestoadd > 0 and AddMore:	#more zones can potentially be added
				zonestoadd = np.asarray(voidline[pos+2:pos+numzonestoadd+2],dtype=int)
				dens = rval * coredens
				rsublist = rlist[np.in1d(vid,zonestoadd)]
				if (sample.uselinkDensV and dens > sample.linkDensThreshV) or \
					(sample.rThreshV>0 and max(rsublist) > sample.rThreshV):
					#cannot add these zones
					AddMore = False
					finalpos -= (numzonestoadd+1)
				else:
					#keep adding zones
					zonelist = np.append(zonelist,zonestoadd)
					num_adds += numzonestoadd
				pos += numzonestoadd + 2
				numzonestoadd = int(voidline[pos])
				rval = float(voidline[pos+1])
				if AddMore:
					finalpos = pos + numzonestoadd + 1
			
			if StartGrow:
				counted_zones = np.append(counted_zones,zonelist)
				root_IDs = np.append(root_IDs,vid[i])

	np.savetxt(sample.outputFolder+sample.prefixV+"_rootIDs.txt",root_IDs,fmt='%d')
#----------------------------------------------------------------------------#
