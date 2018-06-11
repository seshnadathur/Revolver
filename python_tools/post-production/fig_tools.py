import numpy as np
import os
import sys
import imp
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats.mstats import mquantiles
from scipy.stats import gaussian_kde, linregress
from scipy.optimize import curve_fit
from scipy.interpolate import interp2d
from tools import test_bin, bin_mean_val, binner

structdir = os.getenv('HOME')+'/Workspace/structures/'
figdir = os.getenv('HOME')+'/Workspace/figures/'

wad_colours = ['#2A4BD7','#1D6914','#814A19','#8126C0',
	      '#9DAFFF','#81C57A','#E9DEBB','#AD2323',
	      '#29D0D0','#FFEE1F','#FF9233','#FFCDF3',
	      '#000000','#575757','#A0A0A0']

kelly_colours = [
	'#FFB300', #0 Vivid Yellow
	'#803E75', #1 Strong Purple
	'#FF6800', #2 Vivid Orange
	'#A6BDD7', #3 Very Light Blue
	'#C10020', #4 Vivid Red
	'#CEA262', #5 Grayish Yellow
	'#817066', #6 Medium Gray
	'#007D34', #7 Vivid Green
	'#F6768E', #8 Strong Purplish Pink
	'#00538A', #9 Strong Blue
	'#FF7A5C', #10 Strong Yellowish Pink
	'#53377A', #11 Strong Violet
	'#FF8E00', #12 Vivid Orange Yellow
	'#B32851', #13 Strong Purplish Red
	'#F4C800', #14 Vivid Greenish Yellow
	'#7F180D', #15 Strong Reddish Brown
	'#93AA00', #16 Vivid Yellowish Green
	'#593315', #17 Deep Yellowish Brown
	'#F13A13', #18 Vivid Reddish Orange
#	'#232C16', #19 Dark Olive Green
	'#556B2F', #19 my alternative Dark Olive Green (previous was too dark)
]
kelly_RdYlGn = np.asarray(kelly_colours)[[7,16,14,0,12,2,18,4,15]]
kelly_RdYlBu = np.asarray(kelly_colours)[[11,9,3,0,12,2,18,4,15]]

pointstyles = ['o','^','v','s','D','<','>','p','d','*','h','H','8']

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
	'''
	Function to offset the "center" of a colormap. Useful for
	data with a negative min and positive max and you want the
	middle of the colormap's dynamic range to be at zero
	
	Input
	-----
	  cmap : The matplotlib colormap to be altered
	  start : Offset from lowest point in the colormap's range.
	          Defaults to 0.0 (no lower ofset). Should be between
	          0.0 and `midpoint`.
	  midpoint : The new center of the colormap. Defaults to 
	          0.5 (no shift). Should be between 0.0 and 1.0. In
	          general, this should be  1 - vmax/(vmax + abs(vmin))
	          For example if your data range from -15.0 to +5.0 and
	          you want the center of the colormap at 0.0, `midpoint`
	          should be set to  1 - 5/(5 + 15)) or 0.75
	  stop : Offset from highets point in the colormap's range.
	          Defaults to 1.0 (no upper ofset). Should be between
	          `midpoint` and 1.0.
	'''
	cdict = {
		'red': [],
		'green': [],
		'blue': [],
		'alpha': []
	}

	# regular index to compute the colors
	reg_index = np.linspace(start, stop, 257)

	# shifted index to match the data
	shift_index = np.hstack([
		np.linspace(0.0, midpoint, 128, endpoint=False), 
		np.linspace(midpoint, 1.0, 129, endpoint=True)
	])

	for ri, si in zip(reg_index, shift_index):
		r, g, b, a = cmap(ri)

		cdict['red'].append((si, r, r))
		cdict['green'].append((si, g, g))
		cdict['blue'].append((si, b, b))
		cdict['alpha'].append((si, a, a))

	newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
	plt.register_cmap(cmap=newcmap)

	return newcmap

def linf(x,A,B):
	return A+B*x

def const_linf(x,B):
	return B*x-B

def HSW_profile(r,d0,s,alpha,beta):
	return 1 + d0*(1-(r/s)**alpha)/(1+r**beta)

def twop_HSW(r,d0,s):
	alpha = -2.0*s + 4.0
	beta = 17.5*s -6.5 if s<0.91 else -9.8*s + 18.4
	return 1 + d0*(1-(r/s)**alpha)/(1+r**beta)

def NH_profile(r,d0,s,alpha,beta):
	return 1 + d0*(1-(r/s)**alpha)/(1+(r/s)**beta)

def new_profile(r,d0,s1,s2,alpha,beta):
	return 1 + d0*(1-(r/s1)**alpha)/(1+(r/s2)**beta)

def c1_profile(r,d0,s,alpha,beta):
	return 1 + d0*(1-(r/s)**alpha)/(1+(r/0.9)**beta)

def pol_profile(r,a0,a1,a2,a3,a4,a5):
	return a0 + a1*r + a2*r**2. + a3*r**3. + a4*r**4. + a5*r**5.

def bin2d_mean_val(xmin,xmax,ymin,ymax,xvector,yvector,values):
	
	bin_vals = values[np.logical_and(test_bin(xvector,xmin,xmax),test_bin(yvector,ymin,ymax))]
	if len(bin_vals)>0:
		return np.mean(bin_vals), np.std(bin_vals)/np.sqrt(len(bin_vals))
	else:
		return np.nan, np.nan

def binner_2d(xvector,yvector,values,nbins):

	H, xedges, yedges = np.histogram2d(xvector,yvector,bins=nbins)
	bin_means, bin_err = np.empty(H.shape),np.empty(H.shape)
	for i in range(H.shape[0]):
		for j in range(H.shape[1]):
			bin_means[i,j],bin_err[i,j] = bin2d_mean_val(xedges[i],xedges[i+1],yedges[j],yedges[j+1],xvector,yvector,values)
	masked_bin_means = np.ma.masked_where(np.isnan(bin_means),bin_means)
	masked_bin_err = np.ma.masked_where(np.isnan(bin_err),bin_err)
	return xedges, yedges, masked_bin_means, masked_bin_err

def nmin_Rv_hist2d(sampleHandle,vPrefix,Rmin=3,Rmax=120,usePoisson=True,nbins=60):

	#load the void data
	filename = structdir+sampleHandle+'/'+vPrefix+'_info.txt'
	if not os.access(filename, os.F_OK):
		print "%s not found for %s, check the file paths" %(vPrefix, sampleHandle)
		exit(-1)
	VoidsInfo = np.loadtxt(filename,skiprows=2)

	#get some info on the sample
	sinfoFile = structdir+sampleHandle+'/'+'sample_info.dat'
	parms = imp.load_source("name",sinfoFile)
	meanNNsep = parms.tracerDens**(-1.0/3)
	
	#make the histogram: 7th column is radius, 5th column is minimum tracer density
	xbins = np.linspace(Rmin,Rmax,nbins)
	ybins = np.linspace(0,1,nbins)
	H, xedges, yedges = np.histogram2d(VoidsInfo[:,6],VoidsInfo[:,4],bins=[xbins,ybins])
	max_scale = np.max([np.max(H),100])
	#mask the bins which have no voids, for clarity in plotting
	masked_hist = np.ma.masked_where(H==0,H)

	#now plot the histogram
	plt.figure(figsize=(10,6))
	plt.pcolormesh(xedges,yedges,masked_hist.transpose(),cmap='YlOrRd',norm=LogNorm(),vmax=max_scale)
	cbar = plt.colorbar()
	#plt.xlim([0,np.max(VoidsInfo[:,6])])
	plt.xlim([0,Rmax])
	plt.ylim([0,1])
	if usePoisson:
		PoissHandle = sampleHandle.replace('MDR1_','').replace('DM_','')
		contour_levels = np.array((0.05,0.01))	#2-sigma and 3-sigma equivalent contours	
		#build up the Poisson data contours
		if vPrefix == 'MinimalVoids':
			filename = structdir+'Poiss_'+PoissHandle+'/MinimalInterp.txt'
			Poisson = np.loadtxt(filename,skiprows=2)
			f = interp2d(Poisson[:,0],Poisson[:,1],Poisson[:,2])
			x = np.arange(0.5*meanNNsep,Rmax,2.5)
			y = np.arange(0,1,0.05)
			CS = plt.contour(x,y,f(x,y),contour_levels,colors='k')
		else:
			filename = structdir+'Poiss_'+PoissHandle+'/Poisson_'+PoissHandle+'.txt'
			Poisson = np.loadtxt(filename,skiprows=2)

			Pxmin = np.min(Poisson[:,6])		
			Pxmax = np.max(Poisson[:,6])		
			Pymin = np.min(Poisson[:,4])		
			Pymax = np.max(Poisson[:,4])
			values = np.vstack((Poisson[:,6],Poisson[:,4]))	
			k = gaussian_kde(values)
			xi, yi = np.mgrid[Pxmin:Pxmax:30j,Pymin:Pymax:30j]
			zi = k(np.vstack([xi.flatten(),yi.flatten()]))	
			CS = plt.contour(xi,yi,zi.reshape(xi.shape),contour_levels,colors='k')
		for c in CS.collections:
			c.set_dashes([(0, (2.0, 2.0))])

	#add a minimum size line
	x = np.linspace(0.6*meanNNsep,1.8*meanNNsep,25)
	plt.plot(x,(3/(4*np.pi))*(meanNNsep/x)**3,'k--')

	#make the plot prettier
 	ax = plt.gca()
 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)
	#add a little arrow to show the NN separation
	ax.annotate('',xy=(meanNNsep,0.1),xytext=(meanNNsep,0),arrowprops=dict(facecolor='black',width=0.5,headwidth=2.5))
	
	#put the axes labels in
	plt.xlabel("$R_v\,[h^{-1}\mathrm{Mpc}]$",fontsize=24,fontweight='extra bold')
	plt.ylabel(r'$n_\mathrm{min}/\overline{n}$',fontsize=24,fontweight='extra bold')
	cbar.solids.set_edgecolor("face")
	cbar.ax.get_yaxis().labelpad=20
	cbar.set_label("$\mathrm{counts}$",fontsize=24,fontweight='extra bold',rotation=270)
	plt.tight_layout()

	#save the figure
	fig_filename = figdir+'/'+sampleHandle+'_'+vPrefix+'_nmin-rad-hist'+'.pdf'
	fig_filename = figdir+'/slide versions/DMMain1_nmin_Rv_hist.jpg'
	plt.savefig(fig_filename)

def navg_Rv_hist2d(sampleHandle,vPrefix,nbins=40,usePoisson=False):

	PoissHandle = sampleHandle.replace('MDR1_','').replace('DM_','')

	#load the void data
	filename = structdir+sampleHandle+'/'+vPrefix+'_info.txt'
	if not os.access(filename, os.F_OK):
		print "%s not found for %s, check the file paths" %(vPrefix, sampleHandle)
		exit(-1)
	VoidsInfo = np.loadtxt(filename,skiprows=2)

	#get some info on the sample
	sinfoFile = structdir+sampleHandle+'/'+'sample_info.dat'
	parms = imp.load_source("name",sinfoFile)
	meanNNsep = parms.tracerDens**(-1.0/3)

	#make the histogram: 7th column is radius, 5th column is minimum tracer density
	H, xedges, yedges = np.histogram2d(VoidsInfo[:,4],VoidsInfo[:,5],bins=nbins)
	max_scale = np.max([np.max(H),100])
	#mask the bins which have no voids, for clarity in plotting
	masked_hist = np.ma.masked_where(H==0,H)

	#now plot the histogram
	plt.figure(figsize=(10,6))
	plt.pcolormesh(xedges,yedges,masked_hist.transpose(),cmap='YlOrRd',norm=LogNorm(),vmax=max_scale)
	cbar = plt.colorbar()
	plt.xlim([0,np.max(VoidsInfo[:,4])])
	plt.ylim([0,4])

	if usePoisson:
		#get the corresponding Poisson data
		filename = structdir+'Poiss_'+PoissHandle+'/Poisson_'+PoissHandle+'.txt'
		Poisson = np.loadtxt(filename,skiprows=2)

		#build up the Poisson data contours
		Pxmin = np.min(Poisson[:,6])		
		Pxmax = np.max(Poisson[:,6])		
		Pymin = np.min(Poisson[:,5])		
		Pymax = np.max(Poisson[:,5])
		values = np.vstack((Poisson[:,6],Poisson[:,5]))	
		k = gaussian_kde(values)
		xi, yi = np.mgrid[Pxmin:Pxmax:30j,Pymin:Pymax:30j]
		zi = k(np.vstack([xi.flatten(),yi.flatten()]))
		contour_levels = np.array((0.05,0.01))	#2-sigma and 3-sigma equivalent contours	
	
		#add in the contours on top
		CS = plt.contour(xi,yi,zi.reshape(xi.shape),contour_levels,colors='k')
		for c in CS.collections:
			c.set_dashes([(0, (2.0, 2.0))])

	#make the plot prettier
 	ax = plt.gca()
 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)
	#add a little arrow to show the NN separation
	ax.annotate('',xy=(meanNNsep,0.3),xytext=(meanNNsep,0),arrowprops=dict(facecolor='black',width=0.5,headwidth=2.5))
	
	#put the axes labels in
	#plt.xlabel("$R_v\,[h^{-1}\mathrm{Mpc}]$",fontsize=24,fontweight='extra bold')
	plt.ylabel(r'$n_\mathrm{avg}/\overline{n}$',fontsize=24,fontweight='extra bold')
	cbar.solids.set_edgecolor("face")
	cbar.ax.get_yaxis().labelpad=20
	cbar.set_label("$\mathrm{counts}$",fontsize=24,fontweight='extra bold',rotation=270)
	plt.tight_layout()

	#save the figure
#	fig_filename = figdir+'/'+sampleHandle+'_'+vPrefix+'_navg-rad-hist.pdf'
#	plt.savefig(fig_filename)
	
def nmin_Rv_hist2d_multiple(vPrefix,Rmin=3,Rmax=115,nbins=60):

	#the samples for which to plot the void histograms
	sampleHandles = ("MDR1_DM_Main1","MDR1_Main1","MDR1_DM_Main2","MDR1_Main2","MDR1_DM_LOWZ","MDR1_LOWZ")
	textstr = ('$\overline{n}=3.18\cdot10^{-3}$','$\overline{n}=3.18\cdot10^{-3}$','$\overline{n}=1.16\cdot10^{-3}$',\
			'$\overline{n}=1.16\cdot10^{-3}$','$\overline{n}=2.98\cdot10^{-4}$','$\overline{n}=2.98\cdot10^{-4}$')
	props = dict(edgecolor='none',facecolor='none')

	#set up the figure with subplots
	fig,axes = plt.subplots(3,2,sharex=False,sharey=False,figsize=(14,12))

	fPrefix = ''
	for name in sampleHandles:	

		ind = sampleHandles.index(name)
		fPrefix += name.replace('MDR1_','')+'-'

		#load the void and sample info
		filename = structdir+'MDR1/'+name+'/'+vPrefix+'_info.txt'
		if not os.access(filename, os.F_OK):
			print "%s not found for %s, check the file paths" %(vPrefix, sampleHandles[0])
			sys.exit(-1)
		VoidsInfo = np.loadtxt(filename,skiprows=2)
		sinfoFile = structdir+'MDR1/'+name+'/'+'sample_info.dat'
		parms = imp.load_source("name",sinfoFile)
		meanNNsep = parms.tracerDens**(-1.0/3)

		#make the histograms: 7th column is radius, 5th column is minimum tracer density
		xbins = np.linspace(Rmin,Rmax,nbins)
		ybins = np.linspace(0,1,nbins)
		H, xedges, yedges = np.histogram2d(VoidsInfo[:,6],VoidsInfo[:,4],bins=[xbins,ybins])
		#mask the bins which have no voids, for clarity in plotting
		masked_hist = np.ma.masked_where(H==0,H)
		if ind==0: max_scale = np.max(H) 

		#plot the histogram
		ax = axes.flat[ind]
		im = ax.pcolormesh(xedges,yedges,masked_hist.transpose(),cmap='YlOrRd',norm=LogNorm(),vmax=max_scale)

		PoissHandle = name.replace('MDR1_','').replace('DM_','')
		contour_levels = np.array((0.05,0.01))	#2-sigma and 3-sigma equivalent contours	
		#build up the Poisson data contours
		if vPrefix == 'MinimalVoids':
			filename = structdir+'Poisson/Poiss_'+PoissHandle+'/MinimalInterp.txt'
			Poisson = np.loadtxt(filename,skiprows=2)
			f = interp2d(Poisson[:,0],Poisson[:,1],Poisson[:,2])
			x = np.arange(0.5*meanNNsep,Rmax,2.5)
			y = np.arange(0,1,0.05)
			CS = ax.contour(x,y,f(x,y),contour_levels,colors='k')
		else:
			filename = structdir+'Poisson/Poiss_'+PoissHandle+'/Poisson_'+PoissHandle+'.txt'
			Poisson = np.loadtxt(filename,skiprows=2)

			Pxmin = np.min(Poisson[:,6])		
			Pxmax = np.max(Poisson[:,6])		
			Pymin = np.min(Poisson[:,4])		
			Pymax = np.max(Poisson[:,4])
			values = np.vstack((Poisson[:,6],Poisson[:,4]))	
			k = gaussian_kde(values)
			xi, yi = np.mgrid[Pxmin:Pxmax:30j,Pymin:Pymax:30j]
			zi = k(np.vstack([xi.flatten(),yi.flatten()]))	
			CS = ax.contour(xi,yi,zi.reshape(xi.shape),contour_levels,colors='k')
		for c in CS.collections:
			c.set_dashes([(0, (2.0, 2.0))])

		#add a minimum size line
		x = np.linspace(0.6*meanNNsep,1.8*meanNNsep,25)
		ax.plot(x,(3/(4*np.pi))*(meanNNsep/x)**3,'k--')
		
		#set standard x-y extents for all subplots
		ax.set_xlim([0,Rmax])
		ax.set_ylim([0,1])
		ax.set_xticks([0,20,40,60,80])

		ax.text(0.6,0.8,textstr[ind],transform=ax.transAxes,fontsize=16,fontweight='extra bold',bbox=props)
		ax.set_xlabel("$R_v\,[h^{-1}\mathrm{Mpc}]$",fontsize=24,fontweight='extra bold')
		ax.set_ylabel(r'$n_\mathrm{min}/\overline{n}$',fontsize=24,fontweight='extra bold')
		ax.set_title(name.replace('MDR1_','').replace('_',' '),fontsize=20)
	 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
	 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
		ax.tick_params(axis='both', labelsize=16)
		
		#add a little arrow to show the NN separation
		ax.annotate('',xy=(meanNNsep,0.1),xytext=(meanNNsep,0),arrowprops=dict(facecolor='black',width=0.5,headwidth=2.5))
	
	plt.tight_layout(w_pad=2)

	#add the colorbar
	cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
	cbar = plt.colorbar(im,cax=cax,ticks=[1,10,100])
	cbar.solids.set_edgecolor("face")
	cbar.ax.get_yaxis().labelpad=20
	cbar.ax.set_yticklabels(['$10^0$','$10^1$','$10^2$'],fontsize=16,fontweight='extra bold')
	cbar.set_label("$\mathrm{counts}$",fontsize=24,fontweight='extra bold',rotation=270)

	#save the figure
	fig_filename = figdir+'nmin-rad-hist.png'
	plt.savefig(fig_filename,bbox_inches='tight')

def nmin_Rv_Phi2d_multiple(vPrefix='MinimalVoids',Rmin=3,Rmax=120,nbins=60):

	#the samples for which to plot the void histograms
	sampleHandles = ("MDR1_Main1","MDR1_Main2","MDR1_LOWZ")

	#set up the figure with subplots
	fig,axes = plt.subplots(1,3,sharex=False,sharey=True,figsize=(22,8))

	for name in sampleHandles:	

		ind = sampleHandles.index(name)
		VoidsInfo = np.loadtxt(structdir+name+'/'+vPrefix+'_metrics.txt',skiprows=1)
		xbins = np.linspace(Rmin,Rmax,nbins)
		ybins = np.linspace(0,1,nbins)
		xedges, yedges, binned_mean, binned_err = binner_2d(VoidsInfo[:,1],VoidsInfo[:,2],VoidsInfo[:,5],[xbins,ybins])
		if ind==0: max_scale = np.max(np.abs(binned_mean)) 

		#plot the histogram
		ax = axes.flat[ind]
		im = ax.pcolormesh(xedges, yedges, binned_mean.transpose(),cmap='RdYlGn_r',vmin=-max_scale,vmax=max_scale)
		#set standard x-y extents for all subplots
		ax.set_xlim([0,Rmax])
		ax.set_ylim([0,1])
		ax.set_title(name.replace('MDR1_',''),fontsize=24)

		ax.set_xlabel("$R_v\,[h^{-1}\mathrm{Mpc}]$",fontsize=24,fontweight='extra bold')
		if ind==0: ax.set_ylabel(r'$n_\mathrm{min}/\overline{n}$',fontsize=24,fontweight='extra bold')
	 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
	 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
		ax.tick_params(axis='both', labelsize=16)
			
	#add the colorbar
	divider = make_axes_locatable(plt.gca())
	cax = divider.append_axes("right","5%",pad="3%")
	cbar = plt.colorbar(im,cax=cax)
	cbar.solids.set_edgecolor("face")
	cbar.ax.get_yaxis().labelpad=30
	cbar.ax.tick_params(axis='y', labelsize=16)
	cbar.set_label("$10^5\,\Phi_c$",fontsize=24,fontweight='extra bold',rotation=270)
	plt.tight_layout(w_pad=1)

	#save the figure
	#fig_filename = figdir+'compensation.png'
	#plt.savefig(fig_filename,bbox_inches='tight')

def nmin_Rv_Delta(vPrefix='MinimalVoids',ycol=9,useRoot=False,Rmin=3,Rmax=120,nbins=60):

	#the samples for which to plot the void histograms
	sampleHandles = ("MDR1_Main1","MDR1_Main2","MDR1_LOWZ")

	#set up the figure with subplots
	plt.figure(figsize=(24,8))
	for name in sampleHandles:	

		ind = sampleHandles.index(name)

		#load the void and sample info
		VoidsInfo = np.loadtxt(structdir+name+'/'+vPrefix+'_metrics.txt',skiprows=1)
		if useRoot: #use only root-level voids
			rootIDs = np.loadtxt(structdir+name+'/'+vPrefix+'_rootIDs.txt')
			VoidsInfo = VoidsInfo[np.in1d(VoidsInfo[:,0],rootIDs)]
		xbins = np.linspace(Rmin,Rmax,nbins)
		ybins = np.linspace(0,1,nbins)
		xedges, yedges, binned_mean, binned_err = binner_2d(VoidsInfo[:,1],VoidsInfo[:,2],VoidsInfo[:,ycol],[xbins,ybins])
		#max_scale = np.max(np.abs(binned_mean)) 
		max_scale = np.abs(np.min(binned_mean)) #this option saturates the overcompensated pixels

		#set the colormap to display the whole range efficiently		
		#vmax = np.max(binned_mean)
		#vmin = np.abs(np.min(binned_mean))
		#midpoint = 1 - vmax/(vmax+vmin)
		#orig_cmap = matplotlib.cm.RdYlGn
		#shifted_cmap = shiftedColorMap(orig_cmap, midpoint=midpoint, name='shifted')

		#plot the histogram
		plt.subplot(1,3,ind+1)
		#im = plt.pcolormesh(xedges, yedges, binned_mean.transpose(),cmap=shifted_cmap,linewidth=0)
		im = plt.pcolormesh(xedges, yedges, binned_mean.transpose(),cmap='RdYlGn',linewidth=0,vmin=-max_scale,vmax=max_scale)
		#set standard x-y extents for all subplots
		plt.xlim([0,Rmax])
		plt.ylim([0,1])
		plt.title(name.replace('MDR1_',''),fontsize=24)

		cbar = plt.colorbar(im)
		cbar.solids.set_edgecolor("face")
		cbar.ax.get_yaxis().labelpad=30
		cbar.ax.tick_params(axis='y', labelsize=16)
		if ycol==9:
			cbar.set_label("$\Delta(r=3R_v)$",fontsize=24,fontweight='extra bold',rotation=270)
		elif ycol==7:
			cbar.set_label("$\Delta(R_v)$",fontsize=24,fontweight='extra bold',rotation=270)
		plt.xlabel("$R_v\,[h^{-1}\mathrm{Mpc}]$",fontsize=24,fontweight='extra bold')
		plt.ylabel(r'$n_\mathrm{min}/\overline{n}$',fontsize=24,fontweight='extra bold')
		plt.tick_params(axis='both', labelsize=16)
			
	plt.tight_layout(w_pad=2)

	#save the figure
	fig_filename = figdir+'VIDE-Delta_compensation.pdf'
	plt.savefig(fig_filename,bbox_inches='tight')

def navg_Rv_hist2d_multiple(vPrefix,xmax=115,nbins=40):

	#the samples for which to plot the void histograms
	sampleHandles = ("MDR1_DM_Main2","MDR1_Main2","MDR1_Main3","MDR1_CMASS")

	#set up the figure with subplots
	fig,axes = plt.subplots(2,2,figsize=(12,8))

	fPrefix = ''
	for name in sampleHandles:	

		ind = sampleHandles.index(name)
		fPrefix += name.replace('MDR1_','')+'-'

		#load the void and sample info
		filename = structdir+name+'/'+vPrefix+'_info.txt'
		if not os.access(filename, os.F_OK):
			print "%s not found for %s, check the file paths" %(vPrefix, sampleHandles[0])
			exit(-1)
		VoidsInfo = np.loadtxt(filename,skiprows=2)
		sinfoFile = structdir+name+'/'+'sample_info.dat'
		parms = imp.load_source("name",sinfoFile)
		#meanNNsep = parms.tracerDens**(-1.0/3)
		#for Main2 ParisVoids, the largest voids are ~700 Mpc/h(!!), so add a fudge to make plots look nice
		if name=="MDR1_Main2" and vPrefix=="ParisVoids":
			VoidsInfo = VoidsInfo[test_bin(VoidsInfo[:,6],0,150)] 

		#make the histograms: 7th column is radius, 5th column is minimum tracer density
		H, xedges, yedges = np.histogram2d(VoidsInfo[:,6],VoidsInfo[:,5],bins=nbins)
		#mask the bins which have no voids, for clarity in plotting
		masked_hist = np.ma.masked_where(H==0,H)
		if ind==0: max_scale = np.max(H) 

		#plot the histogram
		ax = axes.flat[ind]
		im = ax.pcolormesh(xedges,yedges,masked_hist.transpose(),cmap='YlOrRd',norm=LogNorm(),vmax=max_scale)

		#set standard x-y extents for all subplots
		ax.set_xlim([0,xmax])
		ax.set_ylim([0,5])

		ax.set_xlabel("$R_v\,[h^{-1}\mathrm{Mpc}]$",fontsize=24,fontweight='extra bold')
		ax.set_ylabel(r'$n_\mathrm{avg}/\overline{n}$',fontsize=24,fontweight='extra bold')
	 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
	 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
		ax.tick_params(axis='both', labelsize=16)
		
	
	plt.tight_layout(w_pad=2)

	#add the colorbar
	cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
	cbar = plt.colorbar(im,cax=cax,**kw)
	cbar.solids.set_edgecolor("face")
	cbar.ax.get_yaxis().labelpad=20
	cbar.set_label("$\mathrm{counts}$",fontsize=24,fontweight='extra bold',rotation=270)

	#save the figure
	fig_filename = figdir+fPrefix+vPrefix+'_navg-rad-hist.pdf'
	plt.savefig(fig_filename,bbox_inches='tight')

def nmin_Rv_Phi2d(sampleHandle,vPrefix,nbins=60):

	filename = structdir+sampleHandle+'/'+vPrefix+'_metrics.txt'
	VoidsInfo = np.loadtxt(filename,skiprows=1)

	plt.figure(figsize=(10,8))

	#bin the data and calculate mean and error of Phi in each bin
	xbins = np.linspace(min(VoidsInfo[:,1]),max(VoidsInfo[:,1]),nbins)
	ybins = np.linspace(0,1,nbins)
	xedges, yedges, binned_mean, binned_err = binner_2d(VoidsInfo[:,1],VoidsInfo[:,2],VoidsInfo[:,4]-1,[xbins,ybins])

	#plot the mean Phi values
	max_scale = np.max(np.abs(binned_mean))
	#set the colormap to display the whole range efficiently		
	vmax = np.max(binned_mean)
	vmin = np.abs(np.min(binned_mean))
	midpoint = 1 - vmax/(vmax+vmin)
	orig_cmap = mpl.cm.RdYlGn_r
	shifted_cmap = shiftedColorMap(orig_cmap, midpoint=midpoint, name='shifted')

	#plot the histogram
	im = plt.pcolormesh(xedges, yedges, binned_mean.transpose(),cmap=shifted_cmap,linewidth=0)
#	plt.pcolormesh(xedges, yedges, binned_mean.transpose(),cmap='RdYlGn_r',vmin=-max_scale,vmax=max_scale)
#	plt.pcolormesh(xedges, yedges, binned_mean.transpose(),cmap='RdYlGn_r',vmin=-1,vmax=1)
	cbar = plt.colorbar()

	#make the plot prettier
 	ax = plt.gca()
 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)

	#put the axes labels in
	plt.xlabel("$R_v\,[h^{-1}\mathrm{Mpc}]$",fontsize=24,fontweight='extra bold')
	plt.ylabel(r'$n_\mathrm{min}/\overline{n}$',fontsize=24,fontweight='extra bold')
	cbar.solids.set_edgecolor("face")
#	cbar.set_label(r"$10^5\Phi_\mathrm{c}$",fontsize=24,fontweight='extra bold',rotation=270)
	cbar.set_label(r"$\delta_\mathrm{min}$",fontsize=24,fontweight='extra bold',rotation=270)
	cbar.ax.get_yaxis().labelpad=25
	plt.tight_layout()

	#save the figure
	fig_filename = figdir+'/'+sampleHandle+'_'+vPrefix+'_nmin-rad-Phi'+'.pdf'
	#plt.savefig(fig_filename,bbox_inches='tight')
	#plt.savefig(figdir+'/greenbanana2.pdf',bbox_inches='tight')

def contamination(sampleHandle,vPrefix,usePoisson=False,nbins=50):

	filename = structdir+sampleHandle+'/'+vPrefix+'_metrics.txt'
	catalogue = np.loadtxt(filename,skiprows=1)
	poisson_voids = catalogue[catalogue[:,4]>1,:]
	sinfoFile = structdir+sampleHandle+'/'+'sample_info.dat'
	parms = imp.load_source("name",sinfoFile)
	meanNNsep = parms.tracerDens**(-1.0/3)

	plt.figure(figsize=(10,8))

	#bin the data and calculate contamination fraction in each bin
	xbins = np.linspace(min(catalogue[:,1]),max(catalogue[:,1]),nbins)
	ybins = np.linspace(0,1,nbins)
	fullHist,x,y = np.histogram2d(catalogue[:,1],catalogue[:,2],bins=[xbins,ybins])
	poissHist,x,y = np.histogram2d(poisson_voids[:,1],poisson_voids[:,2],bins=[xbins,ybins])
	poiss_frac = np.ma.masked_where(fullHist==0,1.0*poissHist/fullHist)

	#plot the histogram
	plt.pcolormesh(x, y, poiss_frac.transpose(),cmap='YlOrRd',linewidth=0)
	cbar = plt.colorbar()

	if usePoisson:
		PoissHandle = sampleHandle.replace('MDR1_','').replace('DM_','')
		contour_levels = np.array((0.05,0.01))	#2-sigma and 3-sigma equivalent contours	
		#build up the Poisson data contours
		if vPrefix == 'MinimalVoids':
			filename = structdir+'Poiss_'+PoissHandle+'/MinimalInterp.txt'
			Poisson = np.loadtxt(filename,skiprows=2)
			f = interp2d(Poisson[:,0],Poisson[:,1],Poisson[:,2])
			x = np.arange(0.5*meanNNsep,Rmax,2.5)
			y = np.arange(0,1,0.05)
			CS = plt.contour(x,y,f(x,y),contour_levels,colors='k')
		else:
			filename = structdir+'Poiss_'+PoissHandle+'/Poisson_'+PoissHandle+'.txt'
			Poisson = np.loadtxt(filename,skiprows=2)

			Pxmin = np.min(Poisson[:,6])		
			Pxmax = np.max(Poisson[:,6])		
			Pymin = np.min(Poisson[:,4])		
			Pymax = np.max(Poisson[:,4])
			values = np.vstack((Poisson[:,6],Poisson[:,4]))	
			k = gaussian_kde(values)
			xi, yi = np.mgrid[Pxmin:Pxmax:30j,Pymin:Pymax:30j]
			zi = k(np.vstack([xi.flatten(),yi.flatten()]))	
			CS = plt.contour(xi,yi,zi.reshape(xi.shape),contour_levels,colors='k')
		for c in CS.collections:
			c.set_dashes([(0, (2.0, 2.0))])

	#add a minimum size line
	x = np.linspace(0.6*meanNNsep,1.8*meanNNsep,25)
	plt.plot(x,(3/(4*np.pi))*(meanNNsep/x)**3,'k--')

	#make the plot prettier
 	ax = plt.gca()
 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)
	ax.set_ylim([0,1])
	ax.set_xlim([0,55])
	#add a little arrow to show the NN separation
	ax.annotate('',xy=(meanNNsep,0.1),xytext=(meanNNsep,0),arrowprops=dict(facecolor='black',width=0.5,headwidth=2.5))

	#put the axes labels in
	plt.xlabel("$R_v\,[h^{-1}\mathrm{Mpc}]$",fontsize=24,fontweight='extra bold')
	plt.ylabel(r'$n_\mathrm{min}/\overline{n}$',fontsize=24,fontweight='extra bold')
	cbar.solids.set_edgecolor("face")
	cbar.ax.tick_params(axis='y', labelsize=16)
	cbar.set_label("Contamination fraction",fontsize=22,rotation=270)
	cbar.ax.get_yaxis().labelpad=25
	plt.tight_layout()

	#save the figure
	fig_filename = figdir+'/'+sampleHandle.replace('MDR1_','')+vPrefix.replace('Voids','')+'_contamination.pdf'
	plt.savefig(fig_filename,bbox_inches='tight')

def navg_Phi(samplePaths,vPrefix,labels,quantiles=False,nbins=10,useDM=False):

	num_plots = len(samplePaths)
	colours = kelly_RdYlGn[[0,3,7]]
	ps = np.asarray(pointstyles)[[0,1,3]]
	ms = [8,9,8]
#	colours = np.asarray(kelly_colours)[[19,7,16]]
#	ps = np.asarray(pointstyles)[[1,0,3]]
#	ms = [9,8,8]

	plt.figure(figsize=(8,8))

	fPrefix = ''
	for name in samplePaths:
		
		ind = samplePaths.index(name)
	
		filename = name+vPrefix+'_metrics.txt'
		VoidsInfo = np.loadtxt(filename,skiprows=2)

		if quantiles:
			#get bins with equal numbers of voids
			divs = np.arange(nbins+1).astype(float)/nbins
			xedges = mquantiles(VoidsInfo[:,3],divs)
		else:
			H,xedges = np.histogram(VoidsInfo[:,3],nbins)
		if useDM:
			xedges, binned_mean, binned_err = binner(VoidsInfo[:,3],VoidsInfo[:,9],xedges)
			xedges, binned_x, binned_x_err = binner(VoidsInfo[:,3],VoidsInfo[:,3],xedges) 
		else:
			xedges, binned_mean, binned_err = binner(VoidsInfo[:,3],VoidsInfo[:,5],xedges)
			xedges, binned_x, binned_x_err = binner(VoidsInfo[:,3],VoidsInfo[:,3],xedges) 

		#plot the result
		ax = plt.subplot()
		if quantiles:
			ax.errorbar(binned_x,binned_mean,yerr=2*binned_err,xerr=2*binned_x_err,fmt=ps[ind],\
				color=colours[ind],markersize=ms[ind],markeredgecolor='none',elinewidth=2,label=labels[ind])
		else:
			ax.errorbar(binned_x[H>20],binned_mean[H>20],yerr=2*binned_err[H>20],xerr=2*binned_x_err[H>20],fmt=ps[ind],\
				color=colours[ind],markersize=ms[ind],markeredgecolor='none',elinewidth=2,label=labels[ind])
		ax.legend(loc='upper right',numpoints=1,prop={'size':16})

	plt.xlabel(r'$n_\mathrm{avg}/\overline{n}$',fontsize=24,fontweight='extra bold')
	if useDM:
		plt.ylabel(r"$\Delta(r=3R_v)$",fontsize=24,fontweight='extra bold')
		plt.ylim([-0.1,0.1])
		plt.xlim([0.7,1.5])
		plt.grid(True)
	else:
		plt.ylabel(r"$10^5\Phi_\mathrm{c}$",fontsize=24,fontweight='extra bold')
		plt.ylim([-1.5,2.5])
		plt.xlim([0.5,2.0])
	
	#make the plot prettier
 	ax = plt.gca()
 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)
	ax.axhline(0,linestyle=':',color='k')
	ax.axvline(1.0,linestyle=':',color='k')
	plt.tight_layout()

	#save the figure
	if useDM:
		fig_filename = figdir+'/'+fPrefix+'_'+vPrefix+'_navg-Delta.pdf'
	else:
		fig_filename = figdir+'/'+fPrefix+'_'+vPrefix+'_navg-Phi.pdf'
	plt.savefig(figdir+'navg-Phi_allz.png',bbox_inches='tight')
#	plt.savefig(figdir+'slide versions/navg-Phi.png',bbox_inches='tight')
	#fig_filename = figdir+'/PNGversions/'+fPrefix+'_'+vPrefix+'_navg-Phi.pdf'
	#plt.savefig(fig_filename)

def universal_Phi(samplePaths,vPrefix,labels,quantiles=False,nbins=10):

	colours = kelly_RdYlGn[[0,3,7]]
	ps = np.asarray(pointstyles)[[0,1,3]]
	ms = [8,9,8]

	plt.figure(figsize=(8,8))

	for name in samplePaths:
		
		ind = samplePaths.index(name)
	
		filename = name+vPrefix+'_metrics.txt'
		VoidsInfo = np.loadtxt(filename,skiprows=2)

		if quantiles:
			#get bins with equal numbers of voids
			divs = np.arange(nbins+1).astype(float)/nbins
			xedges = mquantiles(VoidsInfo[:,3],divs)
		else:
			H,xedges = np.histogram(VoidsInfo[:,3],nbins)
		xedges, binned_mean, binned_err = binner(VoidsInfo[:,3],VoidsInfo[:,5]/VoidsInfo[:,1]**0,xedges)
		xedges, binned_x, binned_x_err = binner(VoidsInfo[:,3],VoidsInfo[:,3],xedges) 

		#plot the result
		ax = plt.subplot()
		if quantiles:
			ax.errorbar(binned_x,binned_err*np.sqrt(len(VoidsInfo)/nbins),yerr=2*binned_err,xerr=2*binned_x_err,fmt=ps[ind],\
				color=colours[ind],markersize=ms[ind],markeredgecolor='none',elinewidth=2,label=labels[ind])
		else:
			ax.errorbar(binned_x[H>20],binned_mean[H>20],yerr=2*binned_err[H>20],xerr=2*binned_x_err[H>20],fmt=ps[ind],\
				color=colours[ind],markersize=ms[ind],markeredgecolor='none',elinewidth=2,label=labels[ind])
		ax.legend(loc='upper right',numpoints=1,prop={'size':16})

	plt.xlabel(r'$n_\mathrm{avg}/\overline{n}$',fontsize=24,fontweight='extra bold')
	plt.ylabel(r"$10^5\Phi_\mathrm{c}/R_v\;[h\,\mathrm{Mpc}^{-1}]$",fontsize=24,fontweight='extra bold')
#	plt.ylim([-0.1,0.1])
	plt.xlim([0.5,2.0])
	
	#make the plot prettier
 	ax = plt.gca()
 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)
	ax.axhline(0,linestyle=':',color='k')
	ax.axvline(1.0,linestyle=':',color='k')
	plt.tight_layout()

	#save the figure
#	plt.savefig(figdir+'universal-Phi_allz.png',bbox_inches='tight')
#	plt.savefig(figdir+'slide versions/navg-Phi.png',bbox_inches='tight')
	#fig_filename = figdir+'/PNGversions/'+fPrefix+'_'+vPrefix+'_navg-Phi.pdf'
	#plt.savefig(fig_filename)

def nmin_Phi(sampleHandles,vPrefix,nbins=10):

	num_plots = len(sampleHandles)
	colours = kelly_RdYlGn[0::(len(kelly_RdYlGn)+1)/num_plots]

	plt.figure(figsize=(11,7))

	fPrefix = ''
	for name in sampleHandles:
		
		ind = sampleHandles.index(name)
		fPrefix += name.replace('MDR1_','')+'-'
	
		filename = structdir+name+'/'+vPrefix+'_metrics.txt'
		VoidsInfo = np.loadtxt(filename,skiprows=2)

		#get bins with equal numbers of voids
		divs = np.arange(nbins+1).astype(float)/nbins
		xedges = mquantiles(VoidsInfo[:,2],divs)

		#the void radius is the 5th column, Phi is the 9th column
		xedges, binned_mean_Phi, binned_Phi_err = binner(VoidsInfo[:,2],VoidsInfo[:,5],xedges)
		xedges, binned_x, binned_x_err = binner(VoidsInfo[:,2],VoidsInfo[:,2],xedges) 

		#plot the result
		ax = plt.subplot()
		ax.errorbar(binned_x,binned_mean_Phi,yerr=binned_Phi_err,xerr=binned_x_err,fmt=pointstyles[ind],\
			color=colours[ind],markersize=8,markeredgecolor='none',label=name.replace('MDR1_',''))
		ax.legend(loc='upper right',numpoints=1,prop={'size':16})

	plt.xlabel(r'$n_\mathrm{min}/\overline{n}$',fontsize=24,fontweight='extra bold')
	plt.ylabel(r"$10^5\Phi_\mathrm{c}$",fontsize=24,fontweight='extra bold')

	#make the plot prettier
 	ax = plt.gca()
 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)
	plt.tight_layout()

	#save the figure
	fig_filename = figdir+'/'+fPrefix+'_'+vPrefix+'_nmin-Phi.pdf'
	#plt.savefig(fig_filename)
	#fig_filename = figdir+'/PNGversions/'+fPrefix+'_'+vPrefix+'_nmin-Phi.pdf'
	#plt.savefig(fig_filename)

	#plt.close()

def Rv_Phi(sampleHandles,vPrefix,nbins=10):

	num_plots = len(sampleHandles)
	colours = kelly_RdYlGn[0::(len(kelly_RdYlGn)+1)/num_plots]

	plt.figure(figsize=(11,7))

	fPrefix = ''
	for name in sampleHandles:
		
		ind = sampleHandles.index(name)
		fPrefix += name.replace('MDR1_','')+'-'
	
		filename = structdir+name+'/'+vPrefix+'_metrics.txt'
		VoidsInfo = np.loadtxt(filename,skiprows=2)

		#get bins with equal numbers of voids
		divs = np.arange(nbins+1).astype(float)/nbins
		xedges = mquantiles(VoidsInfo[:,1],divs)

		#the void radius is the 5th column, Phi is the 9th column
		xedges, binned_mean_Phi, binned_Phi_err = binner(VoidsInfo[:,1],VoidsInfo[:,5],xedges)
		xedges, binned_x, binned_x_err = binner(VoidsInfo[:,1],VoidsInfo[:,1],xedges) 

		#plot the result
		ax = plt.subplot()
		ax.errorbar(binned_x,binned_mean_Phi,yerr=binned_Phi_err,xerr=binned_x_err,fmt=pointstyles[ind],\
			color=colours[ind],markersize=8,markeredgecolor='none',label=name.replace('MDR1_',''))
		ax.legend(loc='upper right',numpoints=1,prop={'size':16})

	plt.xlabel(r'$R_v\,[h^{-1}\mathrm{Mpc}]$',fontsize=24,fontweight='extra bold')
	plt.ylabel(r"$10^5\Phi_\mathrm{c}$",fontsize=24,fontweight='extra bold')

	#make the plot prettier
 	ax = plt.gca()
 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)
	plt.tight_layout()

	#save the figure
	fig_filename = figdir+'/'+fPrefix+'_'+vPrefix+'_radius-Phi.pdf'
	#plt.savefig(fig_filename)
	#fig_filename = figdir+'/PNGversions/'+fPrefix+'_'+vPrefix+'_radius-Phi.pdf'
	#plt.savefig(fig_filename)

	#plt.close()

def Rv_nmin(sampleHandles,vPrefix,nbins=10):

	num_plots = len(sampleHandles)
	colours = kelly_RdYlGn[0::(len(kelly_RdYlGn)+1)/num_plots]

	fPrefix = ''
	for name in sampleHandles:
		
		ind = sampleHandles.index(name)
		fPrefix += name.replace('MDR1_','')+'-'
	
		filename = structdir+name+'/'+vPrefix+'_info.txt'
		VoidsInfo = np.loadtxt(filename,skiprows=2)

		#get bins with equal numbers of voids
		divs = np.arange(nbins+1).astype(float)/nbins
		xedges = mquantiles(VoidsInfo[:,6],divs)

		#the void radius is the 5th column, Phi is the 9th column
		xedges, binned_mean, binned_err = binner(VoidsInfo[:,6],VoidsInfo[:,4],xedges)
		xedges, binned_x, binned_x_err = binner(VoidsInfo[:,6],VoidsInfo[:,6],xedges) 

		#plot the result
		ax = plt.subplot()
		ax.errorbar(binned_x,binned_mean,yerr=binned_err,xerr=binned_x_err,fmt=pointstyles[ind],\
			color=colours[ind],markersize=8,markeredgecolor='none',label=name.replace('MDR1_',''))
		ax.legend(loc='upper right',numpoints=1,prop={'size':16})


	plt.xlabel(r'$R_v\,[h^{-1}\mathrm{Mpc}]$',fontsize=24,fontweight='extra bold')
	plt.ylabel(r"$n_\mathrm{min}/\overline{n}$",fontsize=24,fontweight='extra bold')

	#make the plot prettier
 	ax = plt.gca()
 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)
	plt.tight_layout()

	#save the figure
	fig_filename = figdir+'/'+fPrefix+'_'+vPrefix+'_radius-nmin.pdf'
	#plt.savefig(fig_filename)
	#fig_filename = figdir+'/PNGversions/'+fPrefix+'_'+vPrefix+'_radius-Phi.pdf'
	#plt.savefig(fig_filename)

	#plt.close()

def Rv_navg(sampleHandles,vPrefix,nbins=10):

	num_plots = len(sampleHandles)
	colours = kelly_RdYlGn[0::(len(kelly_RdYlGn)+1)/num_plots]

	fPrefix = ''
	for name in sampleHandles:
		
		ind = sampleHandles.index(name)
		fPrefix += name.replace('MDR1_','')+'-'
	
		filename = structdir+name+'/'+vPrefix+'_info.txt'
		VoidsInfo = np.loadtxt(filename,skiprows=2)

		#get bins with equal numbers of voids
		divs = np.arange(nbins+1).astype(float)/nbins
		xedges = mquantiles(VoidsInfo[:,6],divs)

		#the void radius is the 5th column, Phi is the 9th column
		xedges, binned_mean_Phi, binned_Phi_err = binner(VoidsInfo[:,6],VoidsInfo[:,5],xedges)
		xedges, binned_x, binned_x_err = binner(VoidsInfo[:,6],VoidsInfo[:,6],xedges) 

		#plot the result
		ax = plt.subplot()
		ax.errorbar(binned_x,binned_mean_Phi,yerr=binned_Phi_err,xerr=binned_x_err,fmt=pointstyles[ind],\
			color=colours[ind],markersize=8,label=name.replace('MDR1_',''))
		ax.legend(loc='upper right',numpoints=1,prop={'size':16})


	plt.xlabel(r'$R_v\,[h^{-1}\mathrm{Mpc}]$',fontsize=24,fontweight='extra bold')
	plt.ylabel(r"$n_\mathrm{avg}/\overline{n}$",fontsize=24,fontweight='extra bold')

	#make the plot prettier
 	ax = plt.gca()
 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)
	plt.tight_layout()

	#save the figure
	fig_filename = figdir+'/'+fPrefix+'_'+vPrefix+'_radius-navg.png'
	#plt.savefig(fig_filename)

	#plt.close()

def nmin_profiles(sHandle,useDM=False,diff=True):

	diffPath = 'differential/' if diff else 'cumulative/'
	if useDM:
		profileDir = structdir+sHandle+'/profiles/'+diffPath+'DM/res1024_'
	else:
		profileDir = structdir+sHandle+'/profiles/'+diffPath	

	if sHandle=='MDR1_Main1':
		bcfiles = ['StrictV_0.12D0.21_4.5R13.8','StrictV_0.21D0.30_4.5R13.8',\
			'StrictV_0.30D0.39_4.5R13.8','StrictV_0.39D0.49_4.5R13.8',\
			'StrictV_0.49D0.58_4.5R13.8']
		ccfiles = ['StrictCV_0.12D0.21_4.5R13.8','StrictCV_0.21D0.30_4.5R13.8',\
			'StrictCV_0.30D0.39_4.5R13.8','StrictCV_0.39D0.49_4.5R13.8',\
			'StrictCV_0.49D0.58_4.5R13.8']
		labels = [r'$0.12<n_\mathrm{min}/\overline{n}<0.21$',r'$0.21<n_\mathrm{min}/\overline{n}<0.30$',\
			r'$0.30<n_\mathrm{min}/\overline{n}<0.39$',r'$0.39<n_\mathrm{min}/\overline{n}<0.49$',\
			r'$0.49<n_\mathrm{min}/\overline{n}<0.58$']
	else:
		bcfiles = ['VIDEbV_0.13D0.20_15.1R20.4','VIDEbV_0.20D0.26_15.1R20.4',\
			'VIDEbV_0.26D0.32_15.1R20.4','VIDEbV_0.32D0.38_15.1R20.4',\
			'VIDEbV_0.38D0.44_15.1R20.4']
		ccfiles = ['VIDEV_0.13D0.20_15.1R20.4','VIDEV_0.20D0.26_15.1R20.4',\
			'VIDEV_0.26D0.32_15.1R20.4','VIDEV_0.32D0.38_15.1R20.4',\
			'VIDEV_0.38D0.44_15.1R20.4']
		labels = [r'$0.13<n_\mathrm{min}/\overline{n}<0.20$',r'$0.20<n_\mathrm{min}/\overline{n}<0.26$',\
			r'$0.26<n_\mathrm{min}/\overline{n}<0.32$',r'$0.32<n_\mathrm{min}/\overline{n}<0.38$',\
			r'$0.38<n_\mathrm{min}/\overline{n}<0.44$']
#		bcfiles = ['StrictV_0.20D0.30_8.0R12.0','StrictV_0.30D0.40_8.0R12.0',\
#			'StrictV_0.40D0.50_8.0R12.0','StrictV_0.50D0.60_8.0R12.0',\
#			'StrictV_0.60D0.70_8.0R12.0']
#		ccfiles = ['StrictCV_0.20D0.30_8.0R12.0','StrictCV_0.30D0.40_8.0R12.0',\
#			'StrictCV_0.40D0.50_8.0R12.0','StrictCV_0.50D0.60_8.0R12.0',\
#			'StrictCV_0.60D0.70_8.0R12.0']
#		labels = [r'$0.2<n_\mathrm{min}/\overline{n}<0.3$',r'$0.3<n_\mathrm{min}/\overline{n}<0.4$',\
#			r'$0.4<n_\mathrm{min}/\overline{n}<0.5$',r'$0.5<n_\mathrm{min}/\overline{n}<0.6$',\
#			r'$0.6<n_\mathrm{min}/\overline{n}<0.7$']

	num_plots = len(bcfiles) 
	colours = kelly_RdYlGn[0::(len(kelly_RdYlGn)+1)/num_plots]

	plt.figure(figsize=(10,8))

#	plt.subplot(1,2,1)
	maxfev=10000
	x = np.linspace(0,3)
	for filename in bcfiles:
		ind = bcfiles.index(filename)
		data = np.loadtxt(profileDir+filename,skiprows=2)
		if useDM:
			plt.errorbar(data[:,0],data[:,1],yerr=data[:,2],color=colours[len(bcfiles)-1-ind],fmt=':'+pointstyles[len(bcfiles)-1-ind],\
				markersize=8,elinewidth=2,markeredgecolor='none',label=labels[ind])
			popt, pcov = curve_fit(HSW_profile,data[:,0],data[:,1],sigma=data[:,2],p0=[-1,1,2,4],absolute_sigma=True,maxfev=maxfev)
			plt.plot(x,HSW_profile(x,popt[0],popt[1],popt[2],popt[3]),color=colours[len(bcfiles)-1-ind],linewidth=1.5)
		else:
			plt.errorbar(data[:,0],data[:,1],yerr=[data[:,3],data[:,2]],color=colours[len(bcfiles)-1-ind],\
				fmt=':'+pointstyles[len(bcfiles)-1-ind],markersize=8,elinewidth=2,markeredgecolor='none',label=labels[ind])
	plt.legend(loc='lower right',numpoints=1,prop={'size':16})
	if 'DM' in sHandle:
		ymax = 2.0
	else:
		ymax = 3.0 if useDM else 3.5
	plt.ylim([0,ymax])
	plt.xlabel(r'$r/R_v$',fontsize=24,fontweight='extra bold')
	if useDM:
		if diff:
			plt.ylabel(r'$\rho(r)/\overline{\rho}$',fontsize=24,fontweight='extra bold')
		else:
			plt.ylabel(r'$1+\Delta(r)$',fontsize=24,fontweight='extra bold')
	else:
		if diff:
			plt.ylabel(r'$n(r)/\overline{n}$',fontsize=24,fontweight='extra bold')
		else:
			plt.ylabel(r'$1+\Delta_n(r)$',fontsize=24,fontweight='extra bold')

	#make the plot prettier
 	ax = plt.gca()
 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)

#	plt.subplot(1,2,2)
#	for filename in ccfiles:
#		ind = ccfiles.index(filename)
#		data = np.loadtxt(profileDir+filename,skiprows=2)
#		if useDM:
#			plt.errorbar(data[:,0],data[:,1],yerr=data[:,2],color=colours[len(bcfiles)-1-ind],fmt=':'+pointstyles[len(bcfiles)-1-ind],\
#				markersize=8,elinewidth=2,markeredgecolor='none',label=labels[ind])
#			popt, pcov = curve_fit(HSW_profile,data[:,0],data[:,1],sigma=data[:,2],p0=[-1,1,2,4],absolute_sigma=True,maxfev=maxfev)
#			plt.plot(x,HSW_profile(x,popt[0],popt[1],popt[2],popt[3]),color=colours[len(bcfiles)-1-ind],linewidth=1.5)
#		else:
#			plt.errorbar(data[:,0],data[:,1],yerr=[data[:,3],data[:,2]],color=colours[len(bcfiles)-1-ind],\
#				fmt=':'+pointstyles[len(bcfiles)-1-ind],markersize=8,elinewidth=2,markeredgecolor='none',label=labels[ind])
#	plt.legend(loc='lower right',numpoints=1,prop={'size':16})
#	plt.ylim([0,ymax])
#	plt.xlabel(r'$r/R_v$',fontsize=24,fontweight='extra bold')
#	if useDM:
##		if diff:
#			plt.ylabel(r'$\rho(r)/\overline{\rho}$',fontsize=24,fontweight='extra bold')
#		else:
#			plt.ylabel(r'$1+\Delta(r)$',fontsize=24,fontweight='extra bold')
#	else:
#		if diff:
#			plt.ylabel(r'$n(r)/\overline{n}$',fontsize=24,fontweight='extra bold')
#		else:
#			plt.ylabel(r'$1+\Delta_n(r)$',fontsize=24,fontweight='extra bold')
#
#	#make the plot prettier
# 	ax = plt.gca()
#	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
# 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
#	ax.tick_params(axis='both', labelsize=16)
#	plt.tight_layout()

	#save the figure
	DMstring = '_DM' if useDM else ''
	diffString = '_diff' if diff else '_cumul'
	#fig_filename = figdir+'/'+sHandle.replace('MDR1_','')+'_nmin-variation'+diffString+DMstring+'.pdf'
	fig_filename = figdir+'/rho_nmin_profiles_'+diffString+'.pdf'
	plt.savefig(fig_filename,bbox_inches='tight')
	#fig_filename = figdir+'/'+sHandle.replace('MDR1_','')+'_nmin-variation'+diffString+DMstring+'.png'
	#plt.savefig(fig_filename)

def RQ_profiles_bothc(sHandle,useDM=False,diff=True):

	diffPath = 'differential/' if diff else 'cumulative/'
	if useDM:
		profileDir = structdir+sHandle+'/old/profiles/'+diffPath+'DM/res1024_'
	else:
		profileDir = structdir+sHandle+'/old/profiles/'+diffPath	

	if sHandle == 'MDR1_Main1':
		filelist = glob.glob(profileDir+'StrictV_RQ_*')
		bcfiles = np.asarray(filelist)[np.argsort(filelist)]
		bcfiles = bcfiles[[0,1,4,7,8]]
		filelist = glob.glob(profileDir+'StrictCV_RQ_*')
		ccfiles = np.asarray(filelist)[np.argsort(filelist)]
		ccfiles = ccfiles[[0,1,4,7,8]]
		labels = [r'$4.5<R_v<11.4$',r'$11.4<R_v<13.6$',\
			r'$17.4<R_v<19.4$',r'$25.4<R_v<33.8$',\
			r'$33.8<R_v<97.7$']
	elif sHandle == 'MDR1_DM_Main1':
		filelist = glob.glob(profileDir+'StrictV_RQ_*')
		bcfiles = np.asarray(filelist)[np.argsort(filelist)]
		filelist = glob.glob(profileDir+'StrictCV_RQ_*')
		ccfiles = np.asarray(filelist)[np.argsort(filelist)]
		labels = [r'$4.5<R_v<11.5$',r'$11.5<R_v<13.5$',r'$13.5<R_v<15.3$',\
			r'$15.3<R_v<17.0$',r'$17.0<R_v<18.8$',r'$18.8<R_v<20.7$',\
			r'$20.7<R_v<23.1$',r'$23.1<R_v<26.5$',r'$26.5<R_v<57.5$']

	#get some info on the sample
	sinfoFile = structdir+sHandle+'/'+'sample_info.dat'
	parms = imp.load_source("name",sinfoFile)
	meanNNsep = parms.tracerDens**(-1.0/3)

	num_plots = len(ccfiles) 
	colours = kelly_RdYlGn[0::(len(kelly_RdYlGn)+1)/num_plots]
	HSW_params = np.zeros((len(filelist),8))
	maxfev=10000

	catalogue = np.loadtxt(structdir+sHandle+"/old/StrictVoids_info.txt",skiprows=2)
	bin_edges = mquantiles(catalogue[:,6],np.linspace(0,1,10))
	bin_means = np.zeros((len(bin_edges)-1,2))
	for i in range(len(bin_edges)-1):
		bin_means[i,0], bin_means[i,1] = bin_mean_val(bin_edges[i],bin_edges[i+1],catalogue[:,6],catalogue[:,6])

	plt.figure(figsize=(20,8))

	plt.subplot(1,2,1)
	for filename in bcfiles:
		ind = np.where(bcfiles==filename)[0][0]
		data = np.loadtxt(filename,skiprows=2)

		x = np.linspace(0.01,3)
		if useDM:
			fitdata = data
			plt.errorbar(data[:,0],data[:,1],yerr=data[:,2],color=colours[ind],fmt=':'+pointstyles[ind],\
				markersize=8,elinewidth=2,markeredgecolor='none',label=labels[ind])
		else:
			fitdata = data[data[:,0]*bin_means[ind,0]>(3.0/(4*np.pi))**(1.0/3)*meanNNsep,:]
			plt.errorbar(data[:,0],data[:,1],yerr=[data[:,3],data[:,2]],color=colours[ind],markerfacecolor='white',\
				fmt=':'+pointstyles[ind],markersize=8,elinewidth=2,markeredgecolor=colours[ind],markeredgewidth=1.3)
			plt.errorbar(fitdata[:,0],fitdata[:,1],yerr=[fitdata[:,3],fitdata[:,2]],color=colours[ind],\
				fmt=pointstyles[ind],markersize=8,elinewidth=2,markeredgecolor='none',label=labels[ind])
		popt, pcov = curve_fit(twop_HSW,fitdata[:,0],fitdata[:,1],sigma=fitdata[:,2],p0=[-1,1],absolute_sigma=True,maxfev=maxfev)
		plt.plot(x,twop_HSW(x,popt[0],popt[1]),color=colours[ind],linewidth=1.5)
		popt, pcov = curve_fit(HSW_profile,fitdata[:,0],fitdata[:,1],sigma=fitdata[:,2],p0=[-1,1,2,4],absolute_sigma=True,maxfev=maxfev)
		#plt.plot(x,HSW_profile(x,popt[0],popt[1],popt[2],popt[3]),color=colours[ind],linewidth=1.5)
		HSW_params[ind,0:4] = popt
		HSW_params[ind,4:8] = np.sqrt(np.diag(pcov))

	plt.legend(loc='lower right',numpoints=1,prop={'size':16})
	if 'DM' in sHandle:
		ymax = 1.8 if diff else 1.5
	else:
		ymax = 2.0 if useDM else 2.5
	plt.ylim([0,ymax])
	plt.xlabel(r'$r/R_v$',fontsize=24,fontweight='extra bold')
	if useDM:
		if diff:
			plt.ylabel(r'$\rho(r)/\overline{\rho}$',fontsize=24,fontweight='extra bold')
		else:
			plt.ylabel(r'$1+\Delta(r)$',fontsize=24,fontweight='extra bold')
	else:
		if diff:
			plt.ylabel(r'$n(r)/\overline{n}$',fontsize=24,fontweight='extra bold')
		else:
			plt.ylabel(r'$1+\Delta_n(r)$',fontsize=24,fontweight='extra bold')

	#make the plot prettier
 	ax = plt.gca()
 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)

	plt.subplot(1,2,2)
	for filename in ccfiles:
		ind = np.where(ccfiles==filename)[0][0]
		data = np.loadtxt(filename,skiprows=2)
		if useDM:
			plt.errorbar(data[:,0],data[:,1],yerr=data[:,2],color=colours[ind],fmt=':'+pointstyles[ind],\
				markersize=8,elinewidth=2,markeredgecolor='none',label=labels[ind])
		else:
			plt.errorbar(data[:,0],data[:,1],yerr=[data[:,3],data[:,2]],color=colours[ind],markeredgewidth=1.3,\
				fmt=':'+pointstyles[ind],markersize=8,elinewidth=2,markeredgecolor=colours[ind],label=labels[ind])
	plt.legend(loc='lower right',numpoints=1,prop={'size':16})
	plt.ylim([0,ymax])
	plt.xlabel(r'$r/R_v$',fontsize=24,fontweight='extra bold')
	if useDM:
		if diff:
			plt.ylabel(r'$\rho(r)/\overline{\rho}$',fontsize=24,fontweight='extra bold')
		else:
			plt.ylabel(r'$1+\Delta(r)$',fontsize=24,fontweight='extra bold')
	else:
		if diff:
			plt.ylabel(r'$n(r)/\overline{n}$',fontsize=24,fontweight='extra bold')
		else:
			plt.ylabel(r'$1+\Delta_n(r)$',fontsize=24,fontweight='extra bold')

	#make the plot prettier
 	ax = plt.gca()
 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)
	plt.tight_layout()

	#save the figure
	DMstring = '_DM' if useDM else ''
	diffString = '_diff' if diff else '_cumul'
	fig_filename = figdir+'/rho_RQ_profiles.pdf'
	plt.savefig(fig_filename,bbox_inches='tight')

	return HSW_params, bin_means
	
def RQ_profiles_onec(sHandle,vPrefix='StrictV',useDM=False,diff=True):

	diffPath = 'differential/' if diff else 'cumulative/'
	if useDM:
		profileDir = structdir+sHandle+'/old/profiles/'+diffPath+'DM/res1024_'
	else:
		profileDir = structdir+sHandle+'/old/profiles/'+diffPath	

	if sHandle == 'MDR1_Main1':
		filelist = glob.glob(profileDir+vPrefix+'_RQ_*')
		filenames = np.asarray(filelist)[np.argsort(filelist)]
		filenames = filenames[[0,1,4,7,8]]
		labels = [r'$4.5<R_v<11.4$',r'$11.4<R_v<13.6$',\
			r'$17.4<R_v<19.4$',r'$25.4<R_v<33.8$',\
			r'$33.8<R_v<97.7$']
	elif sHandle == 'MDR1_DM_Main1':
		filelist = glob.glob(profileDir+vPrefix+'_RQ_*')
		filenames = np.asarray(filelist)[np.argsort(filelist)]
		labels = [r'$4.5<R_v<11.5$',r'$11.5<R_v<13.5$',r'$13.5<R_v<15.3$',\
			r'$15.3<R_v<17.0$',r'$17.0<R_v<18.8$',r'$18.8<R_v<20.7$',\
			r'$20.7<R_v<23.1$',r'$23.1<R_v<26.5$',r'$26.5<R_v<57.5$']
	elif sHandle == 'MDR1_Main2':
		filelist = glob.glob(profileDir+vPrefix+'_RQ_*')
		filenames = np.asarray(filelist)[np.argsort(filelist)]
		labels = [r'$7.0<R_v<16.0$',r'$16.0<R_v<19.1$',r'$19.1<R_v<21.5$',\
			r'$21.5<R_v<24.1$',r'$24.1<R_v<26.7$',r'$26.7<R_v<29.7$',\
			r'$29.7<R_v<33.8$',r'$33.8<R_v<42.1$',r'$42.1<R_v<118.9$']

	num_plots = len(filenames) 
	colours = kelly_RdYlGn[0::(len(kelly_RdYlGn)+1)/num_plots]
	plt.figure(figsize=(10,8))
	for name in filenames:
		ind = np.where(filenames==name)[0][0]
		data = np.loadtxt(name,skiprows=2)
		if useDM:
			plt.errorbar(data[:,0],data[:,1],yerr=data[:,2],color=colours[ind],fmt=':'+pointstyles[ind],\
				markersize=8,elinewidth=2,markeredgecolor='none',label=labels[ind])
		else:
			plt.errorbar(data[:,0],data[:,1],yerr=[data[:,3],data[:,2]],color=colours[ind],\
				fmt=':'+pointstyles[ind],markersize=8,elinewidth=2,markeredgecolor='none',label=labels[ind])
	plt.legend(loc='lower right',numpoints=1,prop={'size':14})
	if 'DM' in sHandle:
		ymax = 1.8 if diff else 1.5
	else:
		ymax = 1.5 if useDM else 2.5
	plt.ylim([0,ymax])
	plt.xlabel(r'$r/R_v$',fontsize=24,fontweight='extra bold')
	if useDM:
		if diff:
			plt.ylabel(r'$\rho(r)/\overline{\rho}$',fontsize=24,fontweight='extra bold')
		else:
			plt.ylabel(r'$1+\Delta(r)$',fontsize=24,fontweight='extra bold')
	else:
		if diff:
			plt.ylabel(r'$n(r)/\overline{n}$',fontsize=24,fontweight='extra bold')
		else:
			plt.ylabel(r'$1+\Delta_n(r)$',fontsize=24,fontweight='extra bold')

	#make the plot prettier
 	ax = plt.gca()
 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)

	#save the figure
	DMstring = '_DM' if useDM else ''
	diffString = '_diff' if diff else '_cumul'
	fig_filename = figdir+'/'+sHandle.replace('MDR1_','')+'_'+vPrefix+'_radius-variation'+diffString+DMstring+'.pdf'
	#plt.savefig(fig_filename)
	fig_filename = figdir+'/slide versions/radius-variation.jpg'
	plt.savefig(fig_filename)

def AQ_profiles(sHandle,useDM=False):

	if useDM:
		profileDir = structdir+sHandle+'/profiles/differential/DM/res1024_'
	else:
		profileDir = structdir+sHandle+'/profiles/differential/'	

	if sHandle == 'MDR1_Main1':
		bcfiles = ['StrictV_AQ_1.72A1.86','StrictV_AQ_1.59A1.72','StrictV_AQ_1.46A1.59',\
			'StrictV_AQ_1.32A1.46','StrictV_AQ_1.19A1.32','StrictV_AQ_1.06A1.19',\
			'StrictV_AQ_0.92A1.06','StrictV_AQ_0.79A0.92','StrictV_AQ_0.65A0.79']
		ccfiles = ['StrictCV_AQ_1.72A1.86','StrictCV_AQ_1.59A1.72','StrictCV_AQ_1.46A1.59',\
			'StrictCV_AQ_1.32A1.46','StrictCV_AQ_1.19A1.32','StrictCV_AQ_1.06A1.19',\
			'StrictCV_AQ_0.92A1.06','StrictCV_AQ_0.79A0.92','StrictCV_AQ_0.65A0.79']
		labels = [r'$1.72<n_\mathrm{avg}/\overline{n}<1.86$',r'$1.59<n_\mathrm{avg}/\overline{n}<1.72$',\
			r'$1.46<n_\mathrm{avg}/\overline{n}<1.59$',r'$1.32<n_\mathrm{avg}/\overline{n}<1.46$',\
			r'$1.19<n_\mathrm{avg}/\overline{n}<1.32$',r'$1.06<n_\mathrm{avg}/\overline{n}<1.19$',\
			r'$0.92<n_\mathrm{avg}/\overline{n}<1.06$',r'$0.79<n_\mathrm{avg}/\overline{n}<0.92$',\
			r'$0.65<n_\mathrm{avg}/\overline{n}<0.79$']
	elif sHandle == 'MDR1_DM_Main1':
		bcfiles = ['StrictV_AQ_1.54A1.65','StrictV_AQ_1.43A1.54','StrictV_AQ_1.32A1.43',\
			'StrictV_AQ_1.21A1.32','StrictV_AQ_1.09A1.21','StrictV_AQ_0.98A1.09',\
			'StrictV_AQ_0.87A0.98','StrictV_AQ_0.76A0.87','StrictV_AQ_0.65A0.76']
		ccfiles = ['StrictCV_AQ_1.54A1.65','StrictCV_AQ_1.43A1.54','StrictCV_AQ_1.32A1.43',\
			'StrictCV_AQ_1.21A1.32','StrictCV_AQ_1.09A1.21','StrictCV_AQ_0.98A1.09',\
			'StrictCV_AQ_0.87A0.98','StrictCV_AQ_0.76A0.87','StrictCV_AQ_0.65A0.76']
		labels = [r'$1.54<n_\mathrm{avg}/\overline{n}<1.65$',r'$1.43<n_\mathrm{avg}/\overline{n}<1.54$',\
			r'$1.32<n_\mathrm{avg}/\overline{n}<1.43$',r'$1.21<n_\mathrm{avg}/\overline{n}<1.32$',\
			r'$1.09<n_\mathrm{avg}/\overline{n}<1.21$',r'$0.98<n_\mathrm{avg}/\overline{n}<1.09$',\
			r'$0.87<n_\mathrm{avg}/\overline{n}<0.98$',r'$0.76<n_\mathrm{avg}/\overline{n}<0.87$',\
			r'$0.65<n_\mathrm{avg}/\overline{n}<0.76$']

	num_plots = len(bcfiles) 
	colours = kelly_RdYlGn[0::(len(kelly_RdYlGn)+1)/num_plots]

	plt.figure(figsize=(18,8))

	plt.subplot(1,2,1)
	for filename in bcfiles:
		ind = bcfiles.index(filename)
		data = np.loadtxt(profileDir+filename,skiprows=2)
		if useDM:
			plt.errorbar(data[:,0],data[:,1],yerr=data[:,2],color=colours[ind],fmt=':'+pointstyles[ind],\
				markersize=8,elinewidth=2,markeredgecolor='none',label=labels[ind])
		else:
			plt.errorbar(data[:,0],data[:,1],yerr=[data[:,3],data[:,2]],color=colours[ind],\
				fmt=':'+pointstyles[ind],markersize=8,elinewidth=2,markeredgecolor='none',label=labels[ind])
	plt.legend(loc='upper right',numpoints=1,prop={'size':16})
	ymax = 3.0 if useDM else 3.5
	plt.ylim([0,ymax])
	plt.xlabel(r'$r/R_v$',fontsize=24,fontweight='extra bold')
	if useDM:
		plt.ylabel(r'$\rho(r)/\overline{\rho}$',fontsize=24,fontweight='extra bold')
	else:
		plt.ylabel(r'$n(r)/\overline{n}$',fontsize=24,fontweight='extra bold')

	#make the plot prettier
 	ax = plt.gca()
 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)

	plt.subplot(1,2,2)
	for filename in ccfiles:
		ind = ccfiles.index(filename)
		data = np.loadtxt(profileDir+filename,skiprows=2)
		if useDM:
			plt.errorbar(data[:,0],data[:,1],yerr=data[:,2],color=colours[ind],fmt=':'+pointstyles[ind],\
				markersize=8,markeredgecolor='none',label=labels[ind])
		else:
			plt.errorbar(data[:,0],data[:,1],yerr=[data[:,3],data[:,2]],color=colours[ind],\
				fmt=':'+pointstyles[ind],markersize=8,markeredgecolor='none',label=labels[ind])
	plt.legend(loc='upper right',numpoints=1,prop={'size':16})
	plt.ylim([0,ymax])
	plt.xlabel(r'$r/R_v$',fontsize=24,fontweight='extra bold')
	if useDM:
		plt.ylabel(r'$\rho(r)/\overline{\rho}$',fontsize=24,fontweight='extra bold')
	else:
		plt.ylabel(r'$n(r)/\overline{n}$',fontsize=24,fontweight='extra bold')

	#make the plot prettier
 	ax = plt.gca()
 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)

	#save the figure
	DMstring = '_DM' if useDM else ''
	fig_filename = figdir+'/'+sHandle.replace('MDR1_','')+'_navg-variation'+DMstring+'.pdf'
	plt.savefig(fig_filename)
	#fig_filename = figdir+'/'+sHandle.replace('MDR1_','')+'_navg-variation'+DMstring+'.png'
	#plt.savefig(fig_filename)

def avg_dens():

	colours = kelly_RdYlGn[[0,3,7]]

	filenames = glob.glob(structdir+'MDR1_DM_Main1/profiles/cumulative/DM/res1024_StrictCV_AQ*')
	filenames = np.asarray(filenames)[np.argsort(filenames)]
	densData = np.empty((len(filenames),4))
	for name in filenames:
		ind = np.where(filenames==name)[0][0]
		F = open(name,'r')
		F.readline()
		line = F.readline()
		densData[ind,0] = float(line.split()[3][:4])
		densData[ind,3] = float(line.split()[3][5:9])
		F.close()
		data = np.loadtxt(name,skiprows=2)
		densData[ind,1] = data[data[:,0]==0.95,1]
		densData[ind,2] = data[data[:,0]==0.95,2]
	DMMain1 = densData
	filenames = glob.glob(structdir+'MDR1_DM_Main2/profiles/cumulative/DM/res1024_StrictCV_AQ*')
	filenames = np.asarray(filenames)[np.argsort(filenames)]
	densData = np.empty((len(filenames),4))
	for name in filenames:
		ind = np.where(filenames==name)[0][0]
		F = open(name,'r')
		F.readline()
		line = F.readline()
		densData[ind,0] = float(line.split()[3][:4])
		densData[ind,3] = float(line.split()[3][5:9])
		F.close()
		data = np.loadtxt(name,skiprows=2)
		densData[ind,1] = data[data[:,0]==0.95,1]
		densData[ind,2] = data[data[:,0]==0.95,2]
	DMMain2 = densData
	filenames = glob.glob(structdir+'MDR1_DM_LOWZ/profiles/cumulative/DM/res1024_StrictCV_AQ*')
	filenames = np.asarray(filenames)[np.argsort(filenames)]
	densData = np.empty((len(filenames),4))
	for name in filenames:
		ind = np.where(filenames==name)[0][0]
		F = open(name,'r')
		F.readline()
		line = F.readline()
		densData[ind,0] = float(line.split()[3][:4])
		densData[ind,3] = float(line.split()[3][5:9])
		F.close()
		data = np.loadtxt(name,skiprows=2)
		densData[ind,1] = data[data[:,0]==0.95,1]
		densData[ind,2] = data[data[:,0]==0.95,2]
	DMLOWZ = densData
	filenames = glob.glob(structdir+'MDR1_Main1/profiles/cumulative/DM/res1024_StrictCV_AQ*')
	filenames = np.asarray(filenames)[np.argsort(filenames)]
	densData = np.empty((len(filenames),4))
	for name in filenames:
		ind = np.where(filenames==name)[0][0]
		F = open(name,'r')
		F.readline()
		line = F.readline()
		densData[ind,0] = float(line.split()[3][:4])
		densData[ind,3] = float(line.split()[3][5:9])
		F.close()
		data = np.loadtxt(name,skiprows=2)
		densData[ind,1] = data[data[:,0]==0.95,1]
		densData[ind,2] = data[data[:,0]==0.95,2]
	Main1 = densData
	filenames = glob.glob(structdir+'MDR1_Main2/profiles/cumulative/DM/res1024_StrictCV_AQ*')
	filenames = np.asarray(filenames)[np.argsort(filenames)]
	densData = np.empty((len(filenames),4))
	for name in filenames:
		ind = np.where(filenames==name)[0][0]
		F = open(name,'r')
		F.readline()
		line = F.readline()
		densData[ind,0] = float(line.split()[3][:4])
		densData[ind,3] = float(line.split()[3][5:9])
		F.close()
		data = np.loadtxt(name,skiprows=2)
		densData[ind,1] = data[data[:,0]==0.95,1]
		densData[ind,2] = data[data[:,0]==0.95,2]
	Main2 = densData
	filenames = glob.glob(structdir+'MDR1_LOWZ/profiles/cumulative/DM/res1024_StrictCV_AQ*')
	filenames = np.asarray(filenames)[np.argsort(filenames)]
	densData = np.empty((len(filenames),4))
	for name in filenames:
		ind = np.where(filenames==name)[0][0]
		F = open(name,'r')
		F.readline()
		line = F.readline()
		densData[ind,0] = float(line.split()[3][:4])
		densData[ind,3] = float(line.split()[3][5:9])
		F.close()
		data = np.loadtxt(name,skiprows=2)
		densData[ind,1] = data[data[:,0]==0.95,1]
		densData[ind,2] = data[data[:,0]==0.95,2]
	LOWZ = densData

	plt.figure(figsize=(20,8))
	plt.subplot(1,2,1)
	plt.errorbar(DMMain1[:,0],DMMain1[:,1],yerr=DMMain1[:,2],color=colours[0],fmt=pointstyles[0],markersize=8,\
		markeredgecolor='none',elinewidth=2,label='DM Main1')
	plt.errorbar(DMMain2[:,0],DMMain2[:,1],yerr=DMMain2[:,2],color=colours[1],fmt=pointstyles[1],markersize=9,\
		markeredgecolor='none',elinewidth=2,label='DM Main2')
	plt.errorbar(DMLOWZ[:,0],DMLOWZ[:,1],yerr=DMLOWZ[:,2],color=colours[2],fmt=pointstyles[2],markersize=9,\
		markeredgecolor='none',elinewidth=2,label='DM LOWZ')
	plt.legend(loc='lower right',numpoints=1,prop={'size':16})
	plt.xlim([0.5,2.5])
	plt.ylim([0.5,1.1])
	plt.xlabel(r'$n_\mathrm{avg}/\overline{n}$',fontsize=24,fontweight='extra bold')
	plt.ylabel(r'$1+\Delta(R_v)$',fontsize=24,fontweight='extra bold')
	ax = plt.gca()
 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)

	plt.subplot(1,2,2)
	plt.errorbar(Main1[:,0],Main1[:,1],yerr=Main1[:,2],color=colours[0],fmt=pointstyles[0],markersize=8,\
		markeredgecolor='none',elinewidth=2,label='Main1')
	plt.errorbar(Main2[:,0],Main2[:,1],yerr=Main2[:,2],color=colours[1],fmt=pointstyles[1],markersize=9,\
		markeredgecolor='none',elinewidth=2,label='Main2')
	plt.errorbar(LOWZ[:,0],LOWZ[:,1],yerr=LOWZ[:,2],color=colours[2],fmt=pointstyles[2],markersize=9,\
		markeredgecolor='none',elinewidth=2,label='LOWZ')
	plt.legend(loc='lower right',numpoints=1,prop={'size':16})
	plt.xlim([0.5,2.5])
	plt.ylim([0.5,1.1])
	plt.xlabel(r'$n_\mathrm{avg}/\overline{n}$',fontsize=24,fontweight='extra bold')
	plt.ylabel(r'$1+\Delta(R_v)$',fontsize=24,fontweight='extra bold')
	ax = plt.gca()
 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)

def profile_variation():

	colours = kelly_RdYlGn[[0,3,7]]
	Main1 = np.loadtxt(structdir+'MDR1_Main2/profiles/cumulative/DM/res1024_StrictCV_35R40',skiprows=2)
	Main2 = np.loadtxt(structdir+'MDR1_DM_LOWZ/profiles/cumulative/DM/res1024_StrictCV_35R40',skiprows=2)
	LOWZ = np.loadtxt(structdir+'MDR1_LOWZ/profiles/cumulative/DM/res1024_StrictCV_35R40',skiprows=2)

	plt.figure(figsize=(10,8))
	plt.errorbar(Main1[:,0],Main1[:,1],yerr=Main1[:,2],color=colours[0],fmt=':'+pointstyles[0],markersize=8,\
		markeredgecolor='none',elinewidth=2,label='Main1')
	plt.errorbar(Main2[:,0],Main2[:,1],yerr=Main2[:,2],color=colours[1],fmt=':'+pointstyles[1],markersize=9,\
		markeredgecolor='none',elinewidth=2,label='Main2')
	plt.errorbar(LOWZ[:,0],LOWZ[:,1],yerr=LOWZ[:,2],color=colours[2],fmt=':'+pointstyles[2],markersize=8,\
		markeredgecolor='none',elinewidth=2,label='LOWZ')

	plt.legend(loc='lower right',numpoints=1,prop={'size':16})
	plt.xlabel(r'$r/R_v$',fontsize=24,fontweight='extra bold')
	plt.ylabel(r'$1+\Delta(r)$',fontsize=24,fontweight='extra bold')
	
	ax = plt.gca()
 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)

def numdens_scatter():
	
	filename = structdir+'MDR1_DM_Main1/StrictVoids_info.txt'
	data = np.loadtxt(filename,skiprows=2)

	plt.figure(figsize=(11,7))

	#ZOBOV-measured min density is column 5, naive central estimate is column 8
	plt.scatter(data[:,4],data[:,7],s=10,c=kelly_RdYlGn[7],edgecolors='none')
	plt.xlim([0,1])
	plt.ylim([-0.1,5])

	#make the plot prettier
 	ax = plt.gca()
 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)

	plt.xlabel(r'$n_\mathrm{min}/\overline{n}$',fontsize=24,fontweight='extra bold')
	plt.ylabel(r'$n(r<0.25R_v)/\overline{n}$',fontsize=24,fontweight='extra bold')

	fig_filename = figdir+'/DM_Main1_scatter.pdf'
	plt.savefig(fig_filename)

def navg_compensation(vPrefix='MinimalVoids',useRoot=False,quantiles=True,nbins=10):

	colours = kelly_RdYlGn[[0,3,7]]
	plt.figure(figsize=(20,8))

	DMMain1 = np.loadtxt(structdir+'MDR1/MDR1_DM_Main1/'+vPrefix+'_metrics.txt',skiprows=2)
	DMMain2 = np.loadtxt(structdir+'MDR1/MDR1_DM_Main2/'+vPrefix+'_metrics.txt',skiprows=2)
	DMLOWZ = np.loadtxt(structdir+'MDR1/MDR1_DM_LOWZ/'+vPrefix+'_metrics.txt',skiprows=2)
	Main1 = np.loadtxt(structdir+'MDR1/MDR1_Main1/'+vPrefix+'_metrics.txt',skiprows=2)
	Main2 = np.loadtxt(structdir+'MDR1/MDR1_Main2/'+vPrefix+'_metrics.txt',skiprows=2)
	LOWZ = np.loadtxt(structdir+'MDR1/MDR1_LOWZ/'+vPrefix+'_metrics.txt',skiprows=2)

	if useRoot: #only consider root-level voids
		rootMain1 = np.loadtxt(structdir+'MDR1/MDR1_Main1/'+vPrefix+'_rootIDs.txt',skiprows=1)
		rootMain2 = np.loadtxt(structdir+'MDR1/MDR1_Main2/'+vPrefix+'_rootIDs.txt',skiprows=1)
		rootLOWZ = np.loadtxt(structdir+'MDR1/MDR1_LOWZ/'+vPrefix+'_rootIDs.txt',skiprows=1)
		rootDMMain1 = np.loadtxt(structdir+'MDR1/MDR1_DM_Main1/'+vPrefix+'_rootIDs.txt',skiprows=1)
		rootDMMain2 = np.loadtxt(structdir+'MDR1/MDR1_DM_Main2/'+vPrefix+'_rootIDs.txt',skiprows=1)
		rootDMLOWZ = np.loadtxt(structdir+'MDR1/MDR1_DM_LOWZ/'+vPrefix+'_rootIDs.txt',skiprows=1)

		Main1 = Main1[np.in1d(Main1[:,0],rootMain1)]
		Main2 = Main2[np.in1d(Main2[:,0],rootMain2)]
		LOWZ = LOWZ[np.in1d(LOWZ[:,0],rootLOWZ)]
		DMMain1 = DMMain1[np.in1d(DMMain1[:,0],rootDMMain1)]
		DMMain2 = DMMain2[np.in1d(DMMain2[:,0],rootDMMain2)]
		DMLOWZ = DMLOWZ[np.in1d(DMLOWZ[:,0],rootDMLOWZ)]

	ind = 7

	if quantiles:
		#get bins with equal numbers of voids
		divs = np.arange(nbins+1).astype(float)/nbins
		M1xedges = mquantiles(Main1[:,ind],divs)
		M1xedges, binned_M1x, binned_M1x_err = binner(Main1[:,ind],Main1[:,ind],M1xedges) 
		M2xedges = mquantiles(Main2[:,ind],divs)
		M2xedges, binned_M2x, binned_M2x_err = binner(Main2[:,ind],Main2[:,ind],M2xedges) 
		LZxedges = mquantiles(LOWZ[:,ind],divs)
		LZxedges, binned_LZx, binned_LZx_err = binner(LOWZ[:,ind],LOWZ[:,ind],LZxedges) 
		DM1xedges = mquantiles(DMMain1[:,ind],divs)
		DM1xedges, binned_DM1x, binned_DM1x_err = binner(DMMain1[:,ind],DMMain1[:,ind],DM1xedges) 
		DM2xedges = mquantiles(DMMain2[:,ind],divs)
		DM2xedges, binned_DM2x, binned_DM2x_err = binner(DMMain2[:,ind],DMMain2[:,ind],DM2xedges) 
		DLZxedges = mquantiles(DMLOWZ[:,ind],divs)
		DLZxedges, binned_DLZx, binned_DLZx_err = binner(DMLOWZ[:,ind],DMLOWZ[:,ind],DLZxedges) 

		xedges, binned_M1mean, binned_M1err = binner(Main1[:,ind],Main1[:,9],M1xedges)
		xedges, binned_M2mean, binned_M2err = binner(Main2[:,ind],Main2[:,9],M2xedges)
		xedges, binned_LZmean, binned_LZerr = binner(LOWZ[:,ind],LOWZ[:,9],LZxedges)
		xedges, binned_DM1mean, binned_DM1err = binner(DMMain1[:,ind],DMMain1[:,9],DM1xedges)
		xedges, binned_DM2mean, binned_DM2err = binner(DMMain2[:,ind],DMMain2[:,9],DM2xedges)
		xedges, binned_DLZmean, binned_DLZerr = binner(DMLOWZ[:,ind],DMLOWZ[:,9],DLZxedges)

		M1H, xedges = np.histogram(Main1[:,ind],bins=M1xedges) 		
		M2H, xedges = np.histogram(Main2[:,ind],bins=M2xedges) 		
		LZH, xedges = np.histogram(LOWZ[:,ind],bins=LZxedges) 		
		DM1H, xedges = np.histogram(DMMain1[:,ind],bins=DM1xedges) 		
		DM2H, xedges = np.histogram(DMMain2[:,ind],bins=DM2xedges) 		
		DLZH, xedges = np.histogram(DMLOWZ[:,ind],bins=DLZxedges) 		
	else:
		M1H, xedges = np.histogram(Main1[:,ind],bins=nbins) 		
		M2H, xedges = np.histogram(Main2[:,ind],bins=nbins) 		
		LZH, xedges = np.histogram(LOWZ[:,ind],bins=nbins) 		
		DM1H, xedges = np.histogram(DMMain1[:,ind],bins=nbins) 		
		DM2H, xedges = np.histogram(DMMain2[:,ind],bins=nbins) 		
		DLZH, xedges = np.histogram(DMLOWZ[:,ind],bins=nbins) 		

		M1xedges, binned_M1x, binned_M1x_err = binner(Main1[:,ind],Main1[:,ind],nbins) 
		M2xedges, binned_M2x, binned_M2x_err = binner(Main2[:,ind],Main2[:,ind],nbins) 
		LZxedges, binned_LZx, binned_LZx_err = binner(LOWZ[:,ind],LOWZ[:,ind],nbins) 
		DM1xedges, binned_DM1x, binned_DM1x_err = binner(DMMain1[:,ind],DMMain1[:,ind],nbins) 
		DM2xedges, binned_DM2x, binned_DM2x_err = binner(DMMain2[:,ind],DMMain2[:,ind],nbins) 
		DLZxedges, binned_DLZx, binned_DLZx_err = binner(DMLOWZ[:,ind],DMLOWZ[:,ind],nbins) 

		xedges, binned_M1mean, binned_M1err = binner(Main1[:,ind],Main1[:,9],M1xedges)
		xedges, binned_M2mean, binned_M2err = binner(Main2[:,ind],Main2[:,9],M2xedges)
		xedges, binned_LZmean, binned_LZerr = binner(LOWZ[:,ind],LOWZ[:,9],LZxedges)
		xedges, binned_DM1mean, binned_DM1err = binner(DMMain1[:,ind],DMMain1[:,9],DM1xedges)
		xedges, binned_DM2mean, binned_DM2err = binner(DMMain2[:,ind],DMMain2[:,9],DM2xedges)
		xedges, binned_DLZmean, binned_DLZerr = binner(DMLOWZ[:,ind],DMLOWZ[:,9],DLZxedges)

	#fit st lines to data
	M1slope, M1intercept, M1r, M1p, M1se = linregress(Main1[:,ind],Main1[:,9]) 
	M2slope, M2intercept, M2r, M2p, M2se = linregress(Main2[:,ind],Main2[:,9]) 
	LZslope, LZintercept, LZr, LZp, LZse = linregress(LOWZ[:,ind],LOWZ[:,9]) 
	DM1slope, DM1intercept, DM1r, DM1p, DM1se = linregress(DMMain1[:,ind],DMMain1[:,9]) 
	DM2slope, DM2intercept, DM2r, DM2p, DM2se = linregress(DMMain2[:,ind],DMMain2[:,9]) 
	DLZslope, DLZintercept, DLZr, DLZp, DLZse = linregress(DMLOWZ[:,ind],DMLOWZ[:,9]) 

	print M1r**2, M2r**2, LZr**2, DM1r**2, DM2r**2, DLZr**2

	plt.figure(figsize=(15,8))
	x = np.linspace(0.6,1.4)

	ax = plt.subplot(1,2,1)
	ax.plot(x,DM1intercept+DM1slope*x,color=colours[0],linestyle='--',linewidth=1.5)
	ax.plot(x,DM2intercept+DM2slope*x,color=colours[1],linestyle='--',linewidth=1.5)
	ax.plot(x,DLZintercept+DLZslope*x,color=colours[2],linestyle='--',linewidth=1.5)
	ax.errorbar(binned_DM1x[DM1H>20],binned_DM1mean[DM1H>20],yerr=binned_DM1err[DM1H>20],fmt=pointstyles[0],markersize=8,\
			elinewidth=1.5,markeredgecolor='none',color=colours[0],label='DM Main1')
	ax.errorbar(binned_DM2x[DM2H>20],binned_DM2mean[DM2H>20],yerr=binned_DM2err[DM2H>20],fmt=pointstyles[1],markersize=9,\
			elinewidth=1.5,markeredgecolor='none',color=colours[1],label='DM Main2')
	ax.errorbar(binned_DLZx[DLZH>20],binned_DLZmean[DLZH>20],yerr=binned_DLZerr[DLZH>20],fmt=pointstyles[3],markersize=8,\
			elinewidth=1.5,markeredgecolor='none',color=colours[2],label='DM LOWZ')
	ax.legend(loc='lower right',numpoints=1,prop={'size':16})
#	ax.set_xlim([0.6,1.4])
#	ax.set_ylim([-0.15,0.15])
	ax.set_xlabel(r'$n_\mathrm{avg}/\overline{n}$',fontsize=24,fontweight='extra bold')
	ax.set_ylabel(r"$\Delta(r=3R_v)$",fontsize=24,fontweight='extra bold')
	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)
	ax.axhline(0,linestyle=':',color='k')
	ax.axvline(1,linestyle=':',color='k')

	ax = plt.subplot(1,2,2)
	ax.plot(x,M1intercept+M1slope*x,color=colours[0],linestyle='--',linewidth=1.5)
	ax.plot(x,M2intercept+M2slope*x,color=colours[1],linestyle='--',linewidth=1.5)
	ax.plot(x,LZintercept+LZslope*x,color=colours[2],linestyle='--',linewidth=1.5)
	ax.errorbar(binned_M1x[M1H>20],binned_M1mean[M1H>20],yerr=binned_M1err[M1H>20],fmt=pointstyles[0],markersize=8,\
			elinewidth=1.5,markeredgecolor='none',color=colours[0],label='Main1')
	ax.errorbar(binned_M2x[M2H>20],binned_M2mean[M2H>20],yerr=binned_M2err[M2H>20],fmt=pointstyles[1],markersize=9,\
			elinewidth=1.5,markeredgecolor='none',color=colours[1],label='Main2')
	ax.errorbar(binned_LZx[LZH>20],binned_LZmean[LZH>20],yerr=binned_LZerr[LZH>20],fmt=pointstyles[3],markersize=8,\
			elinewidth=1.5,markeredgecolor='none',color=colours[2],label='LOWZ')
	ax.legend(loc='lower right',numpoints=1,prop={'size':16})
#	ax.set_xlim([0.6,1.4])
#	ax.set_ylim([-0.15,0.15])
	ax.set_xlabel(r'$n_\mathrm{avg}/\overline{n}$',fontsize=24,fontweight='extra bold')
	ax.set_ylabel(r"$\Delta(r=3R_v)$",fontsize=24,fontweight='extra bold')
	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)
	ax.axhline(0,linestyle=':',color='k')
	ax.axvline(1,linestyle=':',color='k')
	plt.tight_layout(w_pad=2)

	#plt.savefig(figdir+'navg_Delta_compensation.pdf',bbox_inches='tight')

	plt.figure(figsize=(10,8))
	M1res = M1intercept + M1slope*Main1[:,ind] - Main1[:,9]
	plt.scatter(Main1[:,ind],M1res,color=colours[0])
	M2res = M2intercept + M2slope*Main2[:,ind] - Main2[:,9]
	plt.scatter(Main2[:,ind],M2res,color=colours[1])
	LZres = LZintercept + LZslope*LOWZ[:,ind] - LOWZ[:,9]
	plt.scatter(LOWZ[:,ind],LZres,color=colours[2])

def Rv_compensation(vPrefix='IsolatedVoids',useRoot=False,quantiles=True,nbins=10):

	colours = kelly_RdYlGn[[0,3,7]]
	plt.figure(figsize=(20,8))

	DMMain1 = np.loadtxt(structdir+'MDR1_DM_Main1/'+vPrefix+'_metrics.txt',skiprows=2)
	DMMain2 = np.loadtxt(structdir+'MDR1_DM_Main2/'+vPrefix+'_metrics.txt',skiprows=2)
	DMLOWZ = np.loadtxt(structdir+'MDR1_DM_LOWZ/'+vPrefix+'_metrics.txt',skiprows=2)
	Main1 = np.loadtxt(structdir+'MDR1_Main1/'+vPrefix+'_metrics.txt',skiprows=2)
	Main2 = np.loadtxt(structdir+'MDR1_Main2/'+vPrefix+'_metrics.txt',skiprows=2)
	LOWZ = np.loadtxt(structdir+'MDR1_LOWZ/'+vPrefix+'_metrics.txt',skiprows=2)

	if useRoot: #only consider root-level voids
		rootMain1 = np.loadtxt(structdir+'MDR1_Main1/'+vPrefix+'_rootIDs.txt',skiprows=1)
		rootMain2 = np.loadtxt(structdir+'MDR1_Main2/'+vPrefix+'_rootIDs.txt',skiprows=1)
		rootLOWZ = np.loadtxt(structdir+'MDR1_LOWZ/'+vPrefix+'_rootIDs.txt',skiprows=1)
		rootDMMain1 = np.loadtxt(structdir+'MDR1_DM_Main1/'+vPrefix+'_rootIDs.txt',skiprows=1)
		rootDMMain2 = np.loadtxt(structdir+'MDR1_DM_Main2/'+vPrefix+'_rootIDs.txt',skiprows=1)
		rootDMLOWZ = np.loadtxt(structdir+'MDR1_DM_LOWZ/'+vPrefix+'_rootIDs.txt',skiprows=1)

		Main1 = Main1[np.in1d(Main1[:,0],rootMain1)]
		Main2 = Main2[np.in1d(Main2[:,0],rootMain2)]
		LOWZ = LOWZ[np.in1d(LOWZ[:,0],rootLOWZ)]
		DMMain1 = DMMain1[np.in1d(DMMain1[:,0],rootDMMain1)]
		DMMain2 = DMMain2[np.in1d(DMMain2[:,0],rootDMMain2)]
		DMLOWZ = DMLOWZ[np.in1d(DMLOWZ[:,0],rootDMLOWZ)]

	ind = 1

	if quantiles:
		#get bins with equal numbers of voids
		divs = np.arange(nbins+1).astype(float)/nbins
		M1xedges = mquantiles(Main1[:,ind],divs)
		M1xedges, binned_M1x, binned_M1x_err = binner(Main1[:,ind],Main1[:,ind],M1xedges) 
		M2xedges = mquantiles(Main2[:,ind],divs)
		M2xedges, binned_M2x, binned_M2x_err = binner(Main2[:,ind],Main2[:,ind],M2xedges) 
		LZxedges = mquantiles(LOWZ[:,ind],divs)
		LZxedges, binned_LZx, binned_LZx_err = binner(LOWZ[:,ind],LOWZ[:,ind],LZxedges) 
		DM1xedges = mquantiles(DMMain1[:,ind],divs)
		DM1xedges, binned_DM1x, binned_DM1x_err = binner(DMMain1[:,ind],DMMain1[:,ind],DM1xedges) 
		DM2xedges = mquantiles(DMMain2[:,ind],divs)
		DM2xedges, binned_DM2x, binned_DM2x_err = binner(DMMain2[:,ind],DMMain2[:,ind],DM2xedges) 
		DLZxedges = mquantiles(DMLOWZ[:,ind],divs)
		DLZxedges, binned_DLZx, binned_DLZx_err = binner(DMLOWZ[:,ind],DMLOWZ[:,ind],DLZxedges) 

		xedges, binned_M1mean, binned_M1err = binner(Main1[:,ind],Main1[:,9],M1xedges)
		xedges, binned_M2mean, binned_M2err = binner(Main2[:,ind],Main2[:,9],M2xedges)
		xedges, binned_LZmean, binned_LZerr = binner(LOWZ[:,ind],LOWZ[:,9],LZxedges)
		xedges, binned_DM1mean, binned_DM1err = binner(DMMain1[:,ind],DMMain1[:,9],DM1xedges)
		xedges, binned_DM2mean, binned_DM2err = binner(DMMain2[:,ind],DMMain2[:,9],DM2xedges)
		xedges, binned_DLZmean, binned_DLZerr = binner(DMLOWZ[:,ind],DMLOWZ[:,9],DLZxedges)

		M1H, xedges = np.histogram(Main1[:,ind],bins=M1xedges) 		
		M2H, xedges = np.histogram(Main2[:,ind],bins=M2xedges) 		
		LZH, xedges = np.histogram(LOWZ[:,ind],bins=LZxedges) 		
		DM1H, xedges = np.histogram(DMMain1[:,ind],bins=DM1xedges) 		
		DM2H, xedges = np.histogram(DMMain2[:,ind],bins=DM2xedges) 		
		DLZH, xedges = np.histogram(DMLOWZ[:,ind],bins=DLZxedges) 		
	else:
		M1H, xedges = np.histogram(Main1[:,ind],bins=nbins) 		
		M2H, xedges = np.histogram(Main2[:,ind],bins=nbins) 		
		LZH, xedges = np.histogram(LOWZ[:,ind],bins=nbins) 		
		DM1H, xedges = np.histogram(DMMain1[:,ind],bins=nbins) 		
		DM2H, xedges = np.histogram(DMMain2[:,ind],bins=nbins) 		
		DLZH, xedges = np.histogram(DMLOWZ[:,ind],bins=nbins) 		

		M1xedges, binned_M1x, binned_M1x_err = binner(Main1[:,ind],Main1[:,ind],nbins) 
		M2xedges, binned_M2x, binned_M2x_err = binner(Main2[:,ind],Main2[:,ind],nbins) 
		LZxedges, binned_LZx, binned_LZx_err = binner(LOWZ[:,ind],LOWZ[:,ind],nbins) 
		DM1xedges, binned_DM1x, binned_DM1x_err = binner(DMMain1[:,ind],DMMain1[:,ind],nbins) 
		DM2xedges, binned_DM2x, binned_DM2x_err = binner(DMMain2[:,ind],DMMain2[:,ind],nbins) 
		DLZxedges, binned_DLZx, binned_DLZx_err = binner(DMLOWZ[:,ind],DMLOWZ[:,ind],nbins) 

		xedges, binned_M1mean, binned_M1err = binner(Main1[:,ind],Main1[:,9],M1xedges)
		xedges, binned_M2mean, binned_M2err = binner(Main2[:,ind],Main2[:,9],M2xedges)
		xedges, binned_LZmean, binned_LZerr = binner(LOWZ[:,ind],LOWZ[:,9],LZxedges)
		xedges, binned_DM1mean, binned_DM1err = binner(DMMain1[:,ind],DMMain1[:,9],DM1xedges)
		xedges, binned_DM2mean, binned_DM2err = binner(DMMain2[:,ind],DMMain2[:,9],DM2xedges)
		xedges, binned_DLZmean, binned_DLZerr = binner(DMLOWZ[:,ind],DMLOWZ[:,9],DLZxedges)

	plt.figure(figsize=(18,8))
	ax = plt.subplot(1,2,1)
	ax.errorbar(binned_DM1x[DM1H>20],binned_DM1mean[DM1H>20],yerr=binned_DM1err[DM1H>20],fmt=pointstyles[0],markersize=8,\
			elinewidth=1.5,markeredgecolor='none',color=colours[0],label='DM Main1')
	ax.errorbar(binned_M1x[M1H>10],binned_M1mean[M1H>10],yerr=binned_M1err[M1H>10],fmt=pointstyles[3],markersize=8,\
			elinewidth=1.5,markeredgecolor='none',color=colours[2],label='Main1')
#	ax.errorbar(binned_DM2x[DM2H>20],binned_DM2mean[DM2H>20],yerr=binned_DM2err[DM2H>20],fmt=pointstyles[1],markersize=9,\
#			elinewidth=1.5,markeredgecolor='none',color=colours[1],label='DM Main2')
#	ax.errorbar(binned_DLZx[DLZH>20],binned_DLZmean[DLZH>20],yerr=binned_DLZerr[DLZH>20],fmt=pointstyles[3],markersize=8,\
#			elinewidth=1.5,markeredgecolor='none',color=colours[2],label='DM LOWZ')
	ax.legend(loc='upper right',numpoints=1,prop={'size':16})
	ax.set_xlim([5,45])
	ax.set_ylim([-0.1,0.3])
	ax.set_xlabel(r'$R_v\;[h^{-1}\mathrm{Mpc}]$',fontsize=24,fontweight='extra bold')
#	ax.set_xlabel(r'$n_\mathrm{min}/\overline{n}$',fontsize=24,fontweight='extra bold')
	ax.set_ylabel(r"$\Delta(r=3R_v)$",fontsize=24,fontweight='extra bold')
	ax.set_xticks([5,10,15,20,25,30,35,40,45])
	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)
	ax.axhline(0,linestyle=':',color='k')

	ax = plt.subplot(1,2,2)
	ax.errorbar(binned_M1x[M1H>10],binned_M1mean[M1H>10],yerr=binned_M1err[M1H>10],fmt=pointstyles[0],markersize=8,\
			elinewidth=1.5,markeredgecolor='none',color=colours[0],label='Main1')
	ax.errorbar(binned_M2x[M2H>10],binned_M2mean[M2H>10],yerr=binned_M2err[M2H>10],fmt=pointstyles[1],markersize=9,\
			elinewidth=1.5,markeredgecolor='none',color=colours[1],label='Main2')
	ax.errorbar(binned_LZx[LZH>10],binned_LZmean[LZH>10],yerr=binned_LZerr[LZH>10],fmt=pointstyles[3],markersize=8,\
			elinewidth=1.5,markeredgecolor='none',color=colours[2],label='LOWZ')
	ax.legend(loc='upper right',numpoints=1,prop={'size':16})
	ax.set_xlim([5,75])
	ax.set_ylim([-0.1,0.3])
	ax.set_xlabel(r'$R_v\;[h^{-1}\mathrm{Mpc}]$',fontsize=24,fontweight='extra bold')
#	ax.set_xlabel(r'$n_\mathrm{min}/\overline{n}$',fontsize=24,fontweight='extra bold')
	ax.set_ylabel(r"$\Delta(r=3R_v)$",fontsize=24,fontweight='extra bold')
	ax.set_xticks([10,20,30,40,50,60,70])
	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)
	ax.axhline(0,linestyle=':',color='k')
	plt.tight_layout(w_pad=2)

	print binned_M1mean, binned_M1err
	print binned_M2mean, binned_M2err
	print binned_LZmean, binned_LZerr
	#plt.savefig(figdir+'Rv_Delta_compensation.pdf',bbox_inches='tight')

def biases(quantiles=False,nbins=10,vPrefix='IsolatedVoids',useRoot=False):

	colours = kelly_RdYlGn[[0,3,7]]
	plt.figure(figsize=(10,8))

	#load the void data
	Main1 = np.loadtxt(structdir+'MDR1_Main1/'+vPrefix+'_metrics.txt',skiprows=1)
	Main2 = np.loadtxt(structdir+'MDR1_Main2/'+vPrefix+'_metrics.txt',skiprows=1)
	LOWZ = np.loadtxt(structdir+'MDR1_LOWZ/'+vPrefix+'_metrics.txt',skiprows=1)
	DMMain1 = np.loadtxt(structdir+'MDR1_DM_Main1/'+vPrefix+'_metrics.txt',skiprows=1)
	DMMain2 = np.loadtxt(structdir+'MDR1_DM_Main2/'+vPrefix+'_metrics.txt',skiprows=1)
	DMLOWZ = np.loadtxt(structdir+'MDR1_DM_LOWZ/'+vPrefix+'_metrics.txt',skiprows=1)

	if useRoot: #only consider root-level voids
		rootMain1 = np.loadtxt(structdir+'MDR1_Main1/'+vPrefix+'_rootIDs.txt',skiprows=1)
		rootMain2 = np.loadtxt(structdir+'MDR1_Main2/'+vPrefix+'_rootIDs.txt',skiprows=1)
		rootLOWZ = np.loadtxt(structdir+'MDR1_LOWZ/'+vPrefix+'_rootIDs.txt',skiprows=1)
		rootDMMain1 = np.loadtxt(structdir+'MDR1_DM_Main1/'+vPrefix+'_rootIDs.txt',skiprows=1)
		rootDMMain2 = np.loadtxt(structdir+'MDR1_DM_Main2/'+vPrefix+'_rootIDs.txt',skiprows=1)
		rootDMLOWZ = np.loadtxt(structdir+'MDR1_DM_LOWZ/'+vPrefix+'_rootIDs.txt',skiprows=1)

		Main1 = Main1[np.in1d(Main1[:,0],rootMain1)]
		Main2 = Main2[np.in1d(Main2[:,0],rootMain2)]
		LOWZ = LOWZ[np.in1d(LOWZ[:,0],rootLOWZ)]
		DMMain1 = DMMain1[np.in1d(DMMain1[:,0],rootDMMain1)]
		DMMain2 = DMMain2[np.in1d(DMMain2[:,0],rootDMMain2)]
		DMLOWZ = DMLOWZ[np.in1d(DMLOWZ[:,0],rootDMLOWZ)]

	#fit st lines to data
	M1slope, M1intercept, M1r, M1p, M1se = linregress(Main1[:,2]-1,Main1[:,4]-1) 
	M2slope, M2intercept, M2r, M2p, M2se = linregress(Main2[:,2]-1,Main2[:,4]-1) 
	LZslope, LZintercept, LZr, LZp, LZse = linregress(LOWZ[:,2]-1,LOWZ[:,4]-1) 
	DM1slope, DM1intercept, DM1r, DM1p, DM1se = linregress(DMMain1[:,2]-1,DMMain1[:,4]-1) 
	DM2slope, DM2intercept, DM2r, DM2p, DM2se = linregress(DMMain2[:,2]-1,DMMain2[:,4]-1) 
	DLZslope, DLZintercept, DLZr, DLZp, DLZse = linregress(DMLOWZ[:,2]-1,DMLOWZ[:,4]-1) 

	print M1r**2, M2r**2, LZr**2, DM1r**2, DM2r**2, DLZr**2

	if quantiles:
		#get bins with equal numbers of voids
		divs = np.arange(nbins+1).astype(float)/nbins
		M1xedges = mquantiles(Main1[:,2],divs)
		M1xedges, binned_M1x, binned_M1x_err = binner(Main1[:,2],Main1[:,2],M1xedges) 
		M2xedges = mquantiles(Main2[:,2],divs)
		M2xedges, binned_M2x, binned_M2x_err = binner(Main2[:,2],Main2[:,2],M2xedges) 
		LZxedges = mquantiles(LOWZ[:,2],divs)
		LZxedges, binned_LZx, binned_LZx_err = binner(LOWZ[:,2],LOWZ[:,2],LZxedges) 
		xedges, binned_M1mean, binned_M1err = binner(Main1[:,2],Main1[:,4],M1xedges)
		xedges, binned_M2mean, binned_M2err = binner(Main2[:,2],Main2[:,4],M2xedges)
		xedges, binned_LZmean, binned_LZerr = binner(LOWZ[:,2],LOWZ[:,4],LZxedges)

		DM1xedges = mquantiles(DMMain1[:,2],divs)
		DM1xedges, binned_DM1x, binned_DM1x_err = binner(DMMain1[:,2],DMMain1[:,2],DM1xedges) 
		DM2xedges = mquantiles(DMMain2[:,2],divs)
		DM2xedges, binned_DM2x, binned_DM2x_err = binner(DMMain2[:,2],DMMain2[:,2],DM2xedges) 
		DLZxedges = mquantiles(DMLOWZ[:,2],divs)
		DLZxedges, binned_DLZx, binned_DLZx_err = binner(DMLOWZ[:,2],DMLOWZ[:,2],DLZxedges) 

		M1H, xedges = np.histogram(Main1[:,2],bins=M1xedges) 		
		M2H, xedges = np.histogram(Main2[:,2],bins=M2xedges) 		
		LZH, xedges = np.histogram(LOWZ[:,2],bins=LZxedges) 		
		DM1H, xedges = np.histogram(DMMain1[:,2],bins=DM1xedges) 		
		DM2H, xedges = np.histogram(DMMain2[:,2],bins=DM2xedges) 		
		DLZH, xedges = np.histogram(DMLOWZ[:,2],bins=DLZxedges) 		

		xedges, binned_DM1mean, binned_DM1err = binner(DMMain1[:,2],DMMain1[:,4],DM1xedges)
		xedges, binned_DM2mean, binned_DM2err = binner(DMMain2[:,2],DMMain2[:,4],DM2xedges)
		xedges, binned_DLZmean, binned_DLZerr = binner(DMLOWZ[:,2],DMLOWZ[:,4],DLZxedges)
	else:
		M1H, xedges = np.histogram(Main1[:,2],bins=nbins) 		
		M2H, xedges = np.histogram(Main2[:,2],bins=nbins) 		
		LZH, xedges = np.histogram(LOWZ[:,2],bins=nbins) 		
		DM1H, xedges = np.histogram(DMMain1[:,2],bins=nbins) 		
		DM2H, xedges = np.histogram(DMMain2[:,2],bins=nbins) 		
		DLZH, xedges = np.histogram(DMLOWZ[:,2],bins=nbins) 		

		M1xedges, binned_M1x, binned_M1x_err = binner(Main1[:,2],Main1[:,2],nbins) 
		M2xedges, binned_M2x, binned_M2x_err = binner(Main2[:,2],Main2[:,2],nbins) 
		LZxedges, binned_LZx, binned_LZx_err = binner(LOWZ[:,2],LOWZ[:,2],nbins) 
		DM1xedges, binned_DM1x, binned_DM1x_err = binner(DMMain1[:,2],DMMain1[:,2],nbins) 
		DM2xedges, binned_DM2x, binned_DM2x_err = binner(DMMain2[:,2],DMMain2[:,2],nbins) 
		DLZxedges, binned_DLZx, binned_DLZx_err = binner(DMLOWZ[:,2],DMLOWZ[:,2],nbins) 

		xedges, binned_M1mean, binned_M1err = binner(Main1[:,2],Main1[:,4],M1xedges)
		xedges, binned_M2mean, binned_M2err = binner(Main2[:,2],Main2[:,4],M2xedges)
		xedges, binned_LZmean, binned_LZerr = binner(LOWZ[:,2],LOWZ[:,4],LZxedges)
		xedges, binned_DM1mean, binned_DM1err = binner(DMMain1[:,2],DMMain1[:,4],DM1xedges)
		xedges, binned_DM2mean, binned_DM2err = binner(DMMain2[:,2],DMMain2[:,4],DM2xedges)
		xedges, binned_DLZmean, binned_DLZerr = binner(DMLOWZ[:,2],DMLOWZ[:,4],DLZxedges)

	plt.figure(figsize=(22,8))
	ax = plt.subplot(1,3,1)
	x = np.linspace(-0.95,-0.15)
	ax.errorbar(binned_DM1x[DM1H>20]-1,binned_DM1mean[DM1H>20]-1,yerr=binned_DM1err[DM1H>20],fmt=pointstyles[0],markersize=8,\
			elinewidth=1.5,markeredgecolor='none',color=colours[0],label='DM voids')
	ax.errorbar(binned_M1x[M1H>20]-1,(binned_M1mean[M1H>20]-1),yerr=binned_M1err[M1H>20],fmt=pointstyles[3],markersize=8,\
			elinewidth=1.5,markeredgecolor='none',color=colours[2],label='HOD voids')
	ax.plot(x,DM1intercept+DM1slope*x,color=colours[0],linestyle='--',linewidth=1.5)
	ax.plot(x,M1intercept+M1slope*x,color=colours[2],linestyle=':',linewidth=1.5)
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(loc='upper left',numpoints=1,prop={'size':16})
	plt.title('Main1',fontsize=24)
	plt.xlim([-1,-0.1])
	plt.ylim([-0.9,0])
	plt.xlabel(r'$\delta_{n\mathrm{,min}}^\mathrm{VTFE}$',fontsize=24,fontweight='extra bold')
	plt.ylabel(r"$\delta_\mathrm{min}$",fontsize=24,fontweight='extra bold')
	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)

	ax = plt.subplot(1,3,2)
	ax.plot(x,DM2intercept+DM2slope*x,color=colours[0],linestyle='--',linewidth=1)
	ax.plot(x,M2intercept+M2slope*x,color=colours[2],linestyle=':',linewidth=1.5)
	ax.errorbar(binned_DM2x[DM2H>20]-1,binned_DM2mean[DM2H>20]-1,yerr=binned_DM2err[DM2H>20],fmt=pointstyles[0],markersize=8,\
			elinewidth=1.5,markeredgecolor='none',color=colours[0],label='DM voids')
	ax.errorbar(binned_M2x[M2H>20]-1,(binned_M2mean[M2H>20]-1),yerr=binned_M2err[M2H>20],fmt=pointstyles[3],markersize=8,\
			elinewidth=1.5,markeredgecolor='none',color=colours[2],label='HOD voids')
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(loc='upper left',numpoints=1,prop={'size':16})
	plt.title('Main2',fontsize=24)
	plt.xlim([-1,-0.1])
	plt.ylim([-0.9,0])
	plt.xlabel(r'$\delta_{n\mathrm{,min}}^\mathrm{VTFE}$',fontsize=24,fontweight='extra bold')
	plt.ylabel(r"$\delta_\mathrm{min}$",fontsize=24,fontweight='extra bold')
	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)

	ax = plt.subplot(1,3,3)
	ax.plot(x,DLZintercept+DLZslope*x,color=colours[0],linestyle='--',linewidth=1)
	ax.plot(x,LZintercept+LZslope*x,color=colours[2],linestyle=':',linewidth=1.5)
	ax.errorbar(binned_DLZx[DLZH>20]-1,binned_DLZmean[DLZH>20]-1,yerr=binned_DLZerr[DLZH>20],fmt=pointstyles[0],markersize=8,\
			elinewidth=1.5,markeredgecolor='none',color=colours[0],label='DM voids')
	ax.errorbar(binned_LZx[LZH>20]-1,(binned_LZmean[LZH>20]-1),yerr=binned_LZerr[LZH>20],fmt=pointstyles[3],markersize=8,\
			elinewidth=1.5,markeredgecolor='none',color=colours[2],label='HOD voids')
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(loc='upper left',numpoints=1,prop={'size':16})
	plt.title('LOWZ',fontsize=24)
	plt.xlim([-1,-0.1])
	plt.ylim([-0.9,0])
	plt.xlabel(r'$\delta_{n\mathrm{,min}}^\mathrm{VTFE}$',fontsize=24,fontweight='extra bold')
	plt.ylabel(r"$\delta_\mathrm{min}$",fontsize=24,fontweight='extra bold')
	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)
	
 	plt.tight_layout()
	#plt.savefig(figdir+'biases.pdf')

def central_densities(sHandle='',paper1=True,vType='Isolated',quantiles=True,nbins=10):
	
	if paper1:
		bfile = structdir+'MDR1_DM_Main1/old/StrictVoids_metrics.txt'
		cfile = structdir+'MDR1_DM_Main1/old/StrictCVoids_metrics.txt'
		baryCM = np.loadtxt(bfile,skiprows=1)
		circumCM = np.loadtxt(cfile,skiprows=1)

		bfile = structdir+'MDR1_DM_Dense/VIDEbVoids_metrics.txt'
		cfile = structdir+'MDR1_DM_Dense/VIDEVoids_metrics.txt'
		baryCD = np.loadtxt(bfile,skiprows=1)
		circumCD = np.loadtxt(cfile,skiprows=1)

		#bin up the n_min data
		if quantiles:
			divs = np.arange(nbins+1).astype(float)/nbins
			Mxedges = mquantiles(baryCM[:,2],divs)
			Mxedges, binned_nminM, binned_nminM_err = binner(baryCM[:,2],baryCM[:,2],Mxedges)
			Dxedges = mquantiles(baryCD[:,2],divs)
			Dxedges, binned_nminD, binned_nminD_err = binner(baryCD[:,2],baryCD[:,2],Dxedges)
		else:
			Mxedges = [0,0.18,0.22,0.26,0.3,0.35,0.4,0.45,0.5,0.6]
			Mxedges, binned_nminM, binned_nminM_err = binner(baryCM[:,2],baryCM[:,2],Mxedges)	
			Dxedges, binned_nminD, binned_nminD_err = binner(baryCD[:,2],baryCD[:,2],nbins)	

		#get bin mean DM densities
		xedges, mean_bDensM, bDensM_err = binner(baryCM[:,2],baryCM[:,4],Mxedges)
		xedges, mean_cDensM, cDensM_err = binner(baryCM[:,2],circumCM[:,4],Mxedges)
		xedges, mean_bDensD, bDensD_err = binner(baryCD[:,2],baryCD[:,4],Dxedges)
		xedges, mean_cDensD, cDensD_err = binner(baryCD[:,2],circumCD[:,4],Dxedges)

		#reference 45-degree line
		x = np.linspace(0.11,0.59)

		fig,axes = plt.subplots(2,sharex=True,sharey=False,figsize=(12,12))

		ax = axes.flat[0]
		ax.errorbar(binned_nminD,mean_bDensD,yerr=2*bDensD_err,xerr=2*binned_nminD_err,fmt=pointstyles[0],\
				color=kelly_RdYlGn[0],markersize=8,elinewidth=2,markeredgecolor='none',label='Dense, barycentre')
		ax.errorbar(binned_nminD,mean_cDensD,yerr=2*cDensD_err,xerr=2*binned_nminD_err,fmt=pointstyles[1],\
				color=kelly_RdYlGn[6],markersize=9,elinewidth=2,markeredgecolor='none',label='Dense, circumcentre')
		ax.plot(x,x,'k--',linewidth=1.5)
		ax.legend(loc='upper right',numpoints=1,prop={'size':16})
		ax.set_ylim([0.0,1.0])
		ax.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
	 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
		ax.tick_params(axis='y', labelsize=16)
		ax.set_ylabel(r"$\rho(r=0)/\overline{\rho}$",fontsize=24,fontweight='extra bold')
		
		ax = axes.flat[1]
		ax.errorbar(binned_nminM,mean_bDensM,yerr=2*bDensM_err,xerr=2*binned_nminM_err,fmt=pointstyles[0],\
				color=kelly_RdYlGn[0],markersize=8,elinewidth=2,markeredgecolor='none',label='Main, barycentre')
		ax.errorbar(binned_nminM,mean_cDensM,yerr=2*cDensM_err,xerr=2*binned_nminM_err,fmt=pointstyles[1],\
				color=kelly_RdYlGn[6],markersize=9,elinewidth=2,markeredgecolor='none',label='Main, circumcentre')
		ax.plot(x,x,'k--',linewidth=1.5)
		ax.legend(loc='upper right',numpoints=1,prop={'size':16})
	 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
		ax.set_ylim([0.0,1.0])
	 	ax.set_yticks([0.0,0.2,0.4,0.6,0.8])
		ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
		ax.tick_params(axis='both', labelsize=16)
		ax.set_ylabel(r"$\rho(r=0)/\overline{\rho}$",fontsize=24,fontweight='extra bold')
		ax.set_xlim([0.1,0.6])
	 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})

		fig.subplots_adjust(hspace=0)
		plt.xlabel(r'$n_\mathrm{min}/\overline{n}$',fontsize=24,fontweight='extra bold')

		#save the figure
		fig_filename = figdir+'/central_densities.pdf'
		fig_filename = figdir+'/slide versions/central_densities.jpg'
		plt.savefig(fig_filename,bbox_inches='tight')
	else:

		num_plots = len(sHandle)
		colours = kelly_RdYlGn[0::(len(kelly_RdYlGn)+1)/num_plots]
		plt.figure(figsize=(10,8))

		for name in sHandle:
		
			ind = sHandle.index(name)
	
			filename = structdir+name+'/'+vType+'Voids_centre_properties.txt'
			#load the void data
			VoidsInfo = np.loadtxt(filename,skiprows=1)

			if quantiles:
				#get bins with equal numbers of voids
				divs = np.arange(nbins+1).astype(float)/nbins
				xedges = mquantiles(VoidsInfo[:,2],divs)
				xedges, binned_x, binned_x_err = binner(VoidsInfo[:,2],VoidsInfo[:,2],xedges) 
			else:
				xedges, binned_x, binned_x_err = binner(VoidsInfo[:,2],VoidsInfo[:,2],nbins) 
			xedges, binned_mean, binned_err = binner(VoidsInfo[:,2],VoidsInfo[:,4],xedges)

			#plot the result
			ax = plt.subplot()
			ax.errorbar(binned_x[binned_x<0.7],binned_mean[binned_x<0.7],yerr=binned_err[binned_x<0.7],xerr=binned_x_err[binned_x<0.7],fmt=pointstyles[ind],\
				color=colours[ind],elinewidth=1.5,markersize=8,markeredgecolor='none',\
				label=name.replace('MDR1_','').replace('_',' '))
			ax.legend(loc='upper left',numpoints=1,prop={'size':16})

		plt.xlabel(r'$n_\mathrm{min}/\overline{n}$',fontsize=24,fontweight='extra bold')
		plt.ylabel(r"$\rho_\mathrm{c}/\overline{\rho}$",fontsize=24,fontweight='extra bold')
		plt.ylim([0,1])

		#make the plot prettier
	 	ax = plt.gca()
	 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
	 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
		ax.tick_params(axis='both', labelsize=16)
		plt.tight_layout()


def Rv_rho(sHandle='',paper1=True,vType='Isolated',quantiles=True,nbins=10):
	
	if paper1:
		bfile = structdir+'MDR1_DM_Main1/VIDEbVoids_metrics.txt'
		cfile = structdir+'MDR1_DM_Main1/VIDEVoids_metrics.txt'
		baryCM = np.loadtxt(bfile,skiprows=1)
		circumCM = np.loadtxt(cfile,skiprows=1)

		bfile = structdir+'MDR1_DM_Dense/VIDEbVoids_metrics.txt'
		cfile = structdir+'MDR1_DM_Dense/VIDEVoids_metrics.txt'
		baryCD = np.loadtxt(bfile,skiprows=1)
		circumCD = np.loadtxt(cfile,skiprows=1)

		#bin up the R_v data
		if quantiles:
			divs = np.arange(nbins+1).astype(float)/nbins
			Mxedges = mquantiles(baryCM[:,1],divs)
			Mxedges, binned_RvM, binned_RvM_err = binner(baryCM[:,1],baryCM[:,1],Mxedges)
			Dxedges = mquantiles(baryCD[:,1],divs)
			Dxedges, binned_RvD, binned_RvD_err = binner(baryCD[:,1],baryCD[:,1],Dxedges)
		else:
			Mxedges = [0,8,12,16,20,24,28,32,40,50,60]
			Mxedges, binned_RvM, binned_RvM_err = binner(baryCM[:,1],baryCM[:,1],Mxedges)	
			Dxedges = [0,6,8,10,13,16,20,25,30,40,60]
			Dxedges, binned_RvD, binned_RvD_err = binner(baryCD[:,1],baryCD[:,1],Dxedges)	

		#get bin mean DM densities
		xedges, mean_bDensM, bDensM_err = binner(baryCM[:,1],baryCM[:,4],Mxedges)
		xedges, mean_cDensM, cDensM_err = binner(baryCM[:,1],circumCM[:,4],Mxedges)
		xedges, mean_bDensD, bDensD_err = binner(baryCD[:,1],baryCD[:,4],Dxedges)
		xedges, mean_cDensD, cDensD_err = binner(baryCD[:,1],circumCD[:,4],Dxedges)

		fig,axes = plt.subplots(2,sharex=True,sharey=False,figsize=(12,14))

		ax = axes.flat[0]
		ax.errorbar(binned_RvD,mean_bDensD,yerr=2*bDensD_err,xerr=2*binned_RvD_err,fmt=pointstyles[0],\
				color=kelly_RdYlGn[0],markersize=8,elinewidth=2,markeredgecolor='none',label='Dense, barycentre')
		ax.errorbar(binned_RvD,mean_cDensD,yerr=2*cDensD_err,xerr=2*binned_RvD_err,fmt=pointstyles[1],\
				color=kelly_RdYlGn[6],markersize=9,elinewidth=2,markeredgecolor='none',label='Dense, circumcentre')
		ax.legend(loc='upper right',numpoints=1,prop={'size':16})
	 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
		ax.set_ylim([0.0,1.0])
		ax.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
	 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
		ax.tick_params(axis='y', labelsize=16)
		ax.set_ylabel(r"$\rho(r=0)/\overline{\rho}$",fontsize=24,fontweight='extra bold')
		
		ax = axes.flat[1]
		ax.errorbar(binned_RvM,mean_bDensM,yerr=2*bDensM_err,xerr=2*binned_RvM_err,fmt=pointstyles[0],\
				color=kelly_RdYlGn[0],markersize=8,elinewidth=2,markeredgecolor='none',label='Main, barycentre')
		ax.errorbar(binned_RvM,mean_cDensM,yerr=2*cDensM_err,xerr=2*binned_RvM_err,fmt=pointstyles[1],\
				color=kelly_RdYlGn[6],markersize=9,elinewidth=2,markeredgecolor='none',label='Main, circumcentre')
		ax.legend(loc='upper right',numpoints=1,prop={'size':16})
	 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
		ax.set_ylim([0.0,1.0])
	 	ax.set_yticks([0.0,0.2,0.4,0.6,0.8])
		ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
		ax.tick_params(axis='both', labelsize=16)
		ax.set_ylabel(r"$\rho(r=0)/\overline{\rho}$",fontsize=24,fontweight='extra bold')

		fig.subplots_adjust(hspace=0)
		plt.xlabel(r'$R_v\,[h^{-1}\mathrm{Mpc}]$',fontsize=24,fontweight='extra bold')

		#save the figure
		fig_filename = figdir+'/DMDens_vs_Rv.pdf'
		plt.savefig(fig_filename,bbox_inches='tight')
	else:
		num_plots = len(sHandle)
		colours = kelly_RdYlGn[0::(len(kelly_RdYlGn)+1)/num_plots]
		plt.figure(figsize=(10,8))

		for name in sHandle:
		
			ind = sHandle.index(name)
	
			filename = structdir+name+'/'+vType+'Voids_centre_properties.txt'
			#load the void data
			VoidsInfo = np.loadtxt(filename,skiprows=2)

			if quantiles:
				#get bins with equal numbers of voids
				divs = np.arange(nbins+1).astype(float)/nbins
				xedges = mquantiles(VoidsInfo[:,1],divs)
				xedges, binned_x, binned_x_err = binner(VoidsInfo[:,1],VoidsInfo[:,1],xedges) 
			else:
				xedges, binned_x, binned_x_err = binner(VoidsInfo[:,1],VoidsInfo[:,1],nbins) 
			#the void radius is the 5th column, Phi is the 9th column
			xedges, binned_mean, binned_err = binner(VoidsInfo[:,1],VoidsInfo[:,4],xedges)

			#plot the result
			ax = plt.subplot()
			ax.errorbar(binned_x,binned_mean,yerr=2*binned_err,xerr=2*binned_x_err,fmt=pointstyles[ind],\
				color=colours[ind],elinewidth=1.5,markersize=8,markeredgecolor='none',\
				label=name.replace('MDR1_','').replace('_',' '))
			ax.legend(loc='upper right',numpoints=1,prop={'size':16})

		plt.xlabel(r'$R_v\,[h^{-1}\mathrm{Mpc}]$',fontsize=24,fontweight='extra bold')
		plt.ylabel(r"$\rho_\mathrm{c}/\overline{\rho}$",fontsize=24,fontweight='extra bold')

		#make the plot prettier
	 	ax = plt.gca()
	 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
	 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
		ax.tick_params(axis='both', labelsize=16)
		plt.tight_layout()

		#save the figure
		#fig_filename = figdir+'/'+sHandle+'_'+vType+'_radius-rho.pdf'
		#plt.savefig(fig_filename)

			
def numdens_plot(paper1=True,useCumul=True,useRoot=False,nbins=31):

	if paper1:
		dim2 = np.loadtxt(structdir+'NumDensData/Binned_DM_Dense_VIDE.txt')
		Main1 = np.loadtxt(structdir+'NumDensData/Binned_DM_Main1_VIDE.txt')
		SvdWfit = np.loadtxt(structdir+'NumDensData/CorrectSvdWfit_Rgt25_Dense.txt')
		Expfit = np.loadtxt(structdir+'NumDensData/Expfit_Rgt25_Dense.txt')

		plt.figure(figsize=(8,8))

		plt.xscale('log')
		plt.yscale('log',nonposy='clip')
		plt.errorbar(dim2[:,0],dim2[:,1],yerr=dim2[:,2],fmt=pointstyles[0],color=kelly_RdYlGn[0],markersize=8,\
			markeredgecolor='none',elinewidth=2,label='Dense')
		plt.errorbar(Main1[:,0],Main1[:,1],yerr=Main1[:,2],fmt=pointstyles[1],color=kelly_RdYlGn[7],markersize=9,\
			markeredgecolor='none',elinewidth=2,label='Main')
		plt.plot(SvdWfit[:,0],SvdWfit[:,1],'--',color=kelly_RdYlGn[1],linewidth=2,label=r'SvdW, $\delta_v=-0.40$')
		plt.plot(Expfit[:,0],Expfit[:,1],color=kelly_RdYlGn[6],linewidth=1.5,label=r'$5.3\times10^{-4}\exp(-R_v^{0.60})$')

		plt.xlim([2,105])
		plt.ylim([1e-10,1e-3])
		plt.xlabel(r'$R_v\,[h^{-1}\mathrm{Mpc}]$',fontsize=24,fontweight='extra bold')
		plt.ylabel(r'$dn/dR_v$',fontsize=24,fontweight='extra bold')
		plt.legend(loc='upper right',numpoints=1,prop={'size':16})
		plt.xticks([2,5,10,20,50,100])
		plt.yticks([1e-9,1e-7,1e-5,1e-3])
	
		#make the plot prettier
	 	ax = plt.gca()
	 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
		ax.tick_params(axis='both', labelsize=16)
		plt.tight_layout()

		#save the figure
		fig_filename = figdir+'/fitted_num_densities.png'
		plt.savefig(fig_filename)
	else:
		#--------------------------------------#
		#--------------first figure------------#
		fig,axes = plt.subplots(1,3,sharex=False,sharey=True,figsize=(21,8))

		data1 = np.loadtxt(structdir+'MDR1_DM_Main1/VIDEVoids_info.txt',skiprows=2)
		data2 = np.loadtxt(structdir+'MDR1_Main1/VIDEVoids_info.txt',skiprows=2)

		if useRoot: #use only root-level voids
			rootIDs1 = np.loadtxt(structdir+'MDR1_DM_Main1/VIDEVoids_rootIDs.txt')
			rootIDs2 = np.loadtxt(structdir+'MDR1_Main1/VIDEVoids_rootIDs.txt')
			data1 = data1[np.in1d(data1[:,0],rootIDs1)]
			data2 = data2[np.in1d(data2[:,0],rootIDs2)]

		bins1 = np.logspace(np.log10(np.min(data1[:,6])),np.log10(np.max(data1[:,6])+0.1),nbins)
		bins2 = np.logspace(np.log10(np.min(data2[:,6])),np.log10(np.max(data2[:,6])+0.1),nbins)

		hist1, bins1 = np.histogram(data1[:,6], bins=bins1)
		hist2, bins2 = np.histogram(data2[:,6], bins=bins2)

		dNdR1 = np.empty((hist1.shape[0],3))
		dNdR2 = np.empty((hist2.shape[0],3))
		for i in range(hist1.shape[0]):
			dNdR1[i,0] = (bins1[i]+bins1[i+1])/2.0
			dNdR2[i,0] = (bins2[i]+bins2[i+1])/2.0
			if useCumul:
				dNdR1[i,1] = 1e-9*np.sum(hist1[i:])
				dNdR2[i,1] = 1e-9*np.sum(hist2[i:])
				dNdR1[i,2] = 1e-9*np.sqrt(np.sum(hist1[i:]))
				dNdR2[i,2] = 1e-9*np.sqrt(np.sum(hist2[i:]))
			else:
				dNdR1[i,1] = 1e-9*hist1[i]/(bins1[i+1]-bins1[i])
				dNdR2[i,1] = 1e-9*hist2[i]/(bins2[i+1]-bins2[i])
				dNdR1[i,2] = 1e-9*np.sqrt(hist1[i])/(bins1[i+1]-bins1[i])
				dNdR2[i,2] = 1e-9*np.sqrt(hist2[i])/(bins2[i+1]-bins2[i])

		ax = axes[0]
		ax.set_xscale('log')
		ax.set_yscale('log',nonposy='clip')
		ax.errorbar(dNdR1[:,0],dNdR1[:,1],yerr=dNdR1[:,2],fmt=pointstyles[0],color=kelly_RdYlGn[0],markeredgecolor='none',\
			markersize=8,elinewidth=2,label='DM only')
		ax.errorbar(dNdR2[:,0],dNdR2[:,1],yerr=dNdR2[:,2],fmt=pointstyles[3],color=kelly_RdYlGn[7],markeredgecolor='none',\
			markersize=8,elinewidth=2,label=r"HOD")
		
		ax.set_xlim([4,600])
		ax.set_xlabel(r'$R_v\,[h^{-1}\mathrm{Mpc}]$',fontsize=24,fontweight='extra bold')
		ax.set_xticks([5,10,50,100,500])
		if useCumul: 
			ax.set_ylim([1e-10,1e-4])
			ax.set_ylabel(r'$n(>R_v)\,[h^3\mathrm{Mpc}^{-3}]$',fontsize=24,fontweight='extra bold')
			ax.set_yticks([1e-9,1e-7,1e-5])
		else:
			ax.set_ylim([1e-12,1e-5])
			ax.set_ylabel(r'$dn/dR_v$',fontsize=24,fontweight='extra bold')
			ax.set_yticks([1e-11,1e-9,1e-7,1e-5,1e-3])
		ax.set_title('VIDE',fontsize=24)
		ax.legend(loc='upper right',numpoints=1,prop={'size':16})
	
		#make the plot prettier
	 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
		ax.tick_params(axis='both', labelsize=16)
		
		data1 = np.loadtxt(structdir+'MDR1_DM_Main1/IsolatedVoids_info.txt',skiprows=2)
		data2 = np.loadtxt(structdir+'MDR1_Main1/IsolatedVoids_info.txt',skiprows=2)

		if useRoot: #use only root-level voids
			rootIDs1 = np.loadtxt(structdir+'MDR1_DM_Main1/IsolatedVoids_rootIDs.txt')
			rootIDs2 = np.loadtxt(structdir+'MDR1_Main1/IsolatedVoids_rootIDs.txt')
			data1 = data1[np.in1d(data1[:,0],rootIDs1)]
			data2 = data2[np.in1d(data2[:,0],rootIDs2)]

		bins1 = np.logspace(np.log10(np.min(data1[:,6])),np.log10(np.max(data1[:,6])+0.1),nbins)
		bins2 = np.logspace(np.log10(np.min(data2[:,6])),np.log10(np.max(data2[:,6])+0.1),nbins)

		hist1, bins1 = np.histogram(data1[:,6], bins=bins1)
		hist2, bins2 = np.histogram(data2[:,6], bins=bins2)

		dNdR1 = np.empty((hist1.shape[0],3))
		dNdR2 = np.empty((hist2.shape[0],3))
		for i in range(hist1.shape[0]):
			dNdR1[i,0] = (bins1[i]+bins1[i+1])/2.0
			dNdR2[i,0] = (bins2[i]+bins2[i+1])/2.0
			if useCumul:
				dNdR1[i,1] = 1e-9*np.sum(hist1[i:])
				dNdR2[i,1] = 1e-9*np.sum(hist2[i:])
				dNdR1[i,2] = 1e-9*np.sqrt(np.sum(hist1[i:]))
				dNdR2[i,2] = 1e-9*np.sqrt(np.sum(hist2[i:]))
			else:
				dNdR1[i,1] = 1e-9*hist1[i]/(bins1[i+1]-bins1[i])
				dNdR2[i,1] = 1e-9*hist2[i]/(bins2[i+1]-bins2[i])
				dNdR1[i,2] = 1e-9*np.sqrt(hist1[i])/(bins1[i+1]-bins1[i])
				dNdR2[i,2] = 1e-9*np.sqrt(hist2[i])/(bins2[i+1]-bins2[i])

		ax = axes[1]
		ax.set_xscale('log')
		ax.set_yscale('log',nonposy='clip')
		ax.errorbar(dNdR1[:,0],dNdR1[:,1],yerr=dNdR1[:,2],fmt=pointstyles[0],color=kelly_RdYlGn[0],markeredgecolor='none',\
			markersize=8,elinewidth=2,label='DM only')
		ax.errorbar(dNdR2[:,0],dNdR2[:,1],yerr=dNdR2[:,2],fmt=pointstyles[3],color=kelly_RdYlGn[7],markeredgecolor='none',\
			markersize=8,elinewidth=2,label=r"HOD")
		
		ax.set_xlim([4,600])
		ax.set_xlabel(r'$R_v\,[h^{-1}\mathrm{Mpc}]$',fontsize=24,fontweight='extra bold')
		ax.set_xticks([5,10,50,100,500])
		ax.set_title('Isolated',fontsize=24)
		ax.legend(loc='upper right',numpoints=1,prop={'size':16})
	
		#make the plot prettier
	 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
		ax.tick_params(axis='both', labelsize=16)

		data1 = np.loadtxt(structdir+'MDR1_DM_Main1/MinimalVoids_info.txt',skiprows=2)
		data2 = np.loadtxt(structdir+'MDR1_Main1/MinimalVoids_info.txt',skiprows=2)

		if useRoot: #use only root-level voids
			rootIDs1 = np.loadtxt(structdir+'MDR1_DM_Main1/MinimalVoids_rootIDs.txt')
			rootIDs2 = np.loadtxt(structdir+'MDR1_Main1/MinimalVoids_rootIDs.txt')
			data1 = data1[np.in1d(data1[:,0],rootIDs1)]
			data2 = data2[np.in1d(data2[:,0],rootIDs2)]

		bins1 = np.logspace(np.log10(np.min(data1[:,6])),np.log10(np.max(data1[:,6])+0.1),nbins)
		bins2 = np.logspace(np.log10(np.min(data2[:,6])),np.log10(np.max(data2[:,6])+0.1),nbins)

		hist1, bins1 = np.histogram(data1[:,6], bins=bins1)
		hist2, bins2 = np.histogram(data2[:,6], bins=bins2)

		dNdR1 = np.empty((hist1.shape[0],3))
		dNdR2 = np.empty((hist2.shape[0],3))
		for i in range(hist1.shape[0]):
			dNdR1[i,0] = (bins1[i]+bins1[i+1])/2.0
			dNdR2[i,0] = (bins2[i]+bins2[i+1])/2.0
			if useCumul:
				dNdR1[i,1] = 1e-9*np.sum(hist1[i:])
				dNdR2[i,1] = 1e-9*np.sum(hist2[i:])
				dNdR1[i,2] = 1e-9*np.sqrt(np.sum(hist1[i:]))
				dNdR2[i,2] = 1e-9*np.sqrt(np.sum(hist2[i:]))
			else:
				dNdR1[i,1] = 1e-9*hist1[i]/(bins1[i+1]-bins1[i])
				dNdR2[i,1] = 1e-9*hist2[i]/(bins2[i+1]-bins2[i])
				dNdR1[i,2] = 1e-9*np.sqrt(hist1[i])/(bins1[i+1]-bins1[i])
				dNdR2[i,2] = 1e-9*np.sqrt(hist2[i])/(bins2[i+1]-bins2[i])

		ax = axes[2]
		ax.set_xscale('log')
		ax.set_yscale('log',nonposy='clip')
		ax.errorbar(dNdR1[:,0],dNdR1[:,1],yerr=dNdR1[:,2],fmt=pointstyles[0],color=kelly_RdYlGn[0],markeredgecolor='none',\
			markersize=8,elinewidth=2,label='DM only')
		ax.errorbar(dNdR2[:,0],dNdR2[:,1],yerr=dNdR2[:,2],fmt=pointstyles[3],color=kelly_RdYlGn[7],markeredgecolor='none',\
			markersize=8,elinewidth=2,label=r"HOD")
		
		ax.set_xlim([4,600])
		ax.set_xlabel(r'$R_v\,[h^{-1}\mathrm{Mpc}]$',fontsize=24,fontweight='extra bold')
		ax.set_xticks([5,10,50,100,500])
		ax.set_title('Minimal',fontsize=24)
		ax.legend(loc='upper right',numpoints=1,prop={'size':16})
	
		#make the plot prettier
	 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
		ax.tick_params(axis='both', labelsize=16)
		plt.tight_layout()

		#save the figure
		fig_filename = figdir+'/DM-HOD_num_densities.pdf'
		#plt.savefig(fig_filename,bbox_inches='tight')

		#--------------------------------------#
		#-------------second figure------------#
		fig,axes = plt.subplots(1,2,sharex=False,sharey=True,figsize=(14,8))

		data1 = np.loadtxt(structdir+'MDR1_Main1/IsolatedVoids_info.txt',skiprows=2)
		data2 = np.loadtxt(structdir+'MDR1_Main2/IsolatedVoids_info.txt',skiprows=2)
		data3 = np.loadtxt(structdir+'MDR1_LOWZ/IsolatedVoids_info.txt',skiprows=2)

		bins1 = np.logspace(np.log10(np.min(data1[:,6])),np.log10(np.max(data1[:,6])+0.1),nbins)
		bins2 = np.logspace(np.log10(np.min(data2[:,6])),np.log10(np.max(data2[:,6])+0.1),nbins)
		bins3 = np.logspace(np.log10(np.min(data3[:,6])),np.log10(np.max(data3[:,6])+0.1),nbins)

		hist1, bins1 = np.histogram(data1[:,6], bins=bins1)
		hist2, bins2 = np.histogram(data2[:,6], bins=bins2)
		hist3, bins3 = np.histogram(data3[:,6], bins=bins3)

		dNdR1 = np.empty((hist1.shape[0],3))
		dNdR2 = np.empty((hist2.shape[0],3))
		dNdR3 = np.empty((hist3.shape[0],3))
		for i in range(hist1.shape[0]):
			dNdR1[i,0] = (bins1[i]+bins1[i+1])/2.0
			dNdR2[i,0] = (bins2[i]+bins2[i+1])/2.0
			dNdR3[i,0] = (bins3[i]+bins3[i+1])/2.0
			if useCumul:
				dNdR1[i,1] = 1e-9*np.sum(hist1[i:])#/(bins1[i+1]-bins1[i])
				dNdR2[i,1] = 1e-9*np.sum(hist2[i:])#/(bins2[i+1]-bins2[i])
				dNdR3[i,1] = 1e-9*np.sum(hist3[i:])#/(bins2[i+1]-bins2[i])
				dNdR1[i,2] = 1e-9*np.sqrt(np.sum(hist1[i:]))#/(bins1[i+1]-bins1[i])
				dNdR2[i,2] = 1e-9*np.sqrt(np.sum(hist2[i:]))#/(bins2[i+1]-bins2[i])
				dNdR3[i,2] = 1e-9*np.sqrt(np.sum(hist3[i:]))#/(bins2[i+1]-bins2[i])
			else:
				dNdR1[i,1] = 1e-9*hist1[i]/(bins1[i+1]-bins1[i])
				dNdR2[i,1] = 1e-9*hist2[i]/(bins2[i+1]-bins2[i])
				dNdR3[i,1] = 1e-9*hist3[i]/(bins3[i+1]-bins3[i])
				dNdR1[i,2] = 1e-9*np.sqrt(hist1[i])/(bins1[i+1]-bins1[i])
				dNdR2[i,2] = 1e-9*np.sqrt(hist2[i])/(bins2[i+1]-bins2[i])
				dNdR2[i,2] = 1e-9*np.sqrt(hist3[i])/(bins3[i+1]-bins3[i])

		ax = axes[0]
		ax.set_xscale('log')
		ax.set_yscale('log',nonposy='clip')
		ax.errorbar(dNdR1[:,0],dNdR1[:,1],yerr=dNdR1[:,2],fmt=pointstyles[0],color=kelly_RdYlGn[0],markeredgecolor='none',\
			markersize=8,elinewidth=2,label='Main1')
		ax.errorbar(dNdR2[:,0],dNdR2[:,1],yerr=dNdR2[:,2],fmt=pointstyles[1],color=kelly_RdYlGn[3],markeredgecolor='none',\
			markersize=9,elinewidth=2,label=r"Main2")
		ax.errorbar(dNdR3[:,0],dNdR3[:,1],yerr=dNdR3[:,2],fmt=pointstyles[3],color=kelly_RdYlGn[7],markeredgecolor='none',\
			markersize=8,elinewidth=2,label=r"LOWZ")
		
		ax.set_xlim([4,220])
		ax.set_xlabel(r'$R_v\,[h^{-1}\mathrm{Mpc}]$',fontsize=24,fontweight='extra bold')
		ax.set_xticks([5,10,50,100])
		if useCumul: 
			ax.set_ylim([1e-10,1e-4])
			ax.set_ylabel(r'$n(>R_v)\,[h^3\mathrm{Mpc}^{-3}]$',fontsize=24,fontweight='extra bold')
			ax.set_yticks([1e-9,1e-7,1e-5])
		else:
			ax.set_ylim([1e-12,1e-5])
			ax.set_ylabel(r'$dn/dR_v$',fontsize=24,fontweight='extra bold')
			ax.set_yticks([1e-11,1e-9,1e-7,1e-5,1e-3])
		ax.legend(loc='upper right',numpoints=1,prop={'size':16})
	
		#make the plot prettier
	 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
		ax.tick_params(axis='both', labelsize=16)

		data1 = np.loadtxt(structdir+'MDR1_Main1/MinimalVoids_info.txt',skiprows=2)
		data2 = np.loadtxt(structdir+'MDR1_Main2/MinimalVoids_info.txt',skiprows=2)
		data3 = np.loadtxt(structdir+'MDR1_LOWZ/MinimalVoids_info.txt',skiprows=2)

		if useRoot: #use only root-level voids
			rootIDs1 = np.loadtxt(structdir+'MDR1_Main1/MinimalVoids_rootIDs.txt')
			rootIDs2 = np.loadtxt(structdir+'MDR1_Main2/MinimalVoids_rootIDs.txt')
			rootIDs3 = np.loadtxt(structdir+'MDR1_LOWZ/MinimalVoids_rootIDs.txt')
			data1 = data1[np.in1d(data1[:,0],rootIDs1)]
			data2 = data2[np.in1d(data2[:,0],rootIDs2)]
			data3 = data3[np.in1d(data3[:,0],rootIDs3)]

		bins1 = np.logspace(np.log10(np.min(data1[:,6])),np.log10(np.max(data1[:,6])+0.1),nbins)
		bins2 = np.logspace(np.log10(np.min(data2[:,6])),np.log10(np.max(data2[:,6])+0.1),nbins)
		bins3 = np.logspace(np.log10(np.min(data3[:,6])),np.log10(np.max(data3[:,6])+0.1),nbins)

		hist1, bins1 = np.histogram(data1[:,6], bins=bins1)
		hist2, bins2 = np.histogram(data2[:,6], bins=bins2)
		hist3, bins3 = np.histogram(data3[:,6], bins=bins3)

		dNdR1 = np.empty((hist1.shape[0],3))
		dNdR2 = np.empty((hist2.shape[0],3))
		dNdR3 = np.empty((hist3.shape[0],3))
		for i in range(hist1.shape[0]):
			dNdR1[i,0] = (bins1[i]+bins1[i+1])/2.0
			dNdR2[i,0] = (bins2[i]+bins2[i+1])/2.0
			dNdR3[i,0] = (bins3[i]+bins3[i+1])/2.0
			if useCumul:
				dNdR1[i,1] = 1e-9*np.sum(hist1[i:])#/(bins1[i+1]-bins1[i])
				dNdR2[i,1] = 1e-9*np.sum(hist2[i:])#/(bins2[i+1]-bins2[i])
				dNdR3[i,1] = 1e-9*np.sum(hist3[i:])#/(bins2[i+1]-bins2[i])
				dNdR1[i,2] = 1e-9*np.sqrt(np.sum(hist1[i:]))#/(bins1[i+1]-bins1[i])
				dNdR2[i,2] = 1e-9*np.sqrt(np.sum(hist2[i:]))#/(bins2[i+1]-bins2[i])
				dNdR3[i,2] = 1e-9*np.sqrt(np.sum(hist3[i:]))#/(bins2[i+1]-bins2[i])
			else:
				dNdR1[i,1] = 1e-9*hist1[i]/(bins1[i+1]-bins1[i])
				dNdR2[i,1] = 1e-9*hist2[i]/(bins2[i+1]-bins2[i])
				dNdR3[i,1] = 1e-9*hist3[i]/(bins3[i+1]-bins3[i])
				dNdR1[i,2] = 1e-9*np.sqrt(hist1[i])/(bins1[i+1]-bins1[i])
				dNdR2[i,2] = 1e-9*np.sqrt(hist2[i])/(bins2[i+1]-bins2[i])
				dNdR2[i,2] = 1e-9*np.sqrt(hist3[i])/(bins3[i+1]-bins3[i])

		ax = axes[1]
		ax.set_xscale('log')
		ax.set_yscale('log',nonposy='clip')
		ax.errorbar(dNdR1[:,0],dNdR1[:,1],yerr=dNdR1[:,2],fmt=pointstyles[0],color=kelly_RdYlGn[0],markeredgecolor='none',\
			markersize=8,elinewidth=2,label='Main1')
		ax.errorbar(dNdR2[:,0],dNdR2[:,1],yerr=dNdR2[:,2],fmt=pointstyles[1],color=kelly_RdYlGn[3],markeredgecolor='none',\
			markersize=9,elinewidth=2,label=r"Main2")
		ax.errorbar(dNdR3[:,0],dNdR3[:,1],yerr=dNdR3[:,2],fmt=pointstyles[3],color=kelly_RdYlGn[7],markeredgecolor='none',\
			markersize=8,elinewidth=2,label=r"LOWZ")
		
		ax.set_xlim([4,220])
		ax.set_xlabel(r'$R_v\,[h^{-1}\mathrm{Mpc}]$',fontsize=24,fontweight='extra bold')
		ax.set_xticks([5,10,50,100])
		ax.legend(loc='upper right',numpoints=1,prop={'size':16})
	
		#make the plot prettier
	 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
		ax.tick_params(axis='both', labelsize=16)
		plt.tight_layout()

		#save the figure
		fig_filename = figdir+'/HOD_num_densities.pdf'
		#plt.savefig(fig_filename,bbox_inches='tight')

def linear_fits(xcol=3,ycol=9,vPrefix='IsolatedVoids',usebinned=False,useDM=False,nbins=10,xmax=2):

	colours = kelly_RdYlGn[[0,3,7]]
	plt.figure(figsize=(20,8))

	if useDM:
		Main1 = np.loadtxt(structdir+'MDR1_DM_Main1/'+vPrefix+'_metrics.txt',skiprows=2)
		Main2 = np.loadtxt(structdir+'MDR1_DM_Main2/'+vPrefix+'_metrics.txt',skiprows=2)
		LOWZ = np.loadtxt(structdir+'MDR1_DM_LOWZ/'+vPrefix+'_metrics.txt',skiprows=2)
	else:
		Main1 = np.loadtxt(structdir+'MDR1_Main1/'+vPrefix+'_metrics.txt',skiprows=2)
		Main2 = np.loadtxt(structdir+'MDR1_Main2/'+vPrefix+'_metrics.txt',skiprows=2)
		LOWZ = np.loadtxt(structdir+'MDR1_LOWZ/'+vPrefix+'_metrics.txt',skiprows=2)

	divs = np.arange(nbins+1).astype(float)/nbins
	M1xedges = mquantiles(Main1[:,xcol],divs)
	M1xedges, binned_M1x, binned_M1x_err = binner(Main1[:,xcol],Main1[:,xcol],M1xedges) 
	xedges, binned_M1mean, binned_M1err = binner(Main1[:,xcol],Main1[:,ycol],M1xedges)
	M2xedges = mquantiles(Main2[:,xcol],divs)
	M2xedges, binned_M2x, binned_M2x_err = binner(Main2[:,xcol],Main2[:,xcol],M2xedges) 
	xedges, binned_M2mean, binned_M2err = binner(Main2[:,xcol],Main2[:,ycol],M2xedges)
	LZxedges = mquantiles(LOWZ[:,xcol],divs)
	LZxedges, binned_LZx, binned_LZx_err = binner(LOWZ[:,xcol],LOWZ[:,xcol],LZxedges) 
	xedges, binned_LZmean, binned_LZerr = binner(LOWZ[:,xcol],LOWZ[:,ycol],LZxedges)
	
	if usebinned:
		M1p, M1cov = curve_fit(linf,binned_M1x,binned_M1mean,sigma=binned_M1err,absolute_sigma=True) 
		M2p, M2cov = curve_fit(linf,binned_M2x,binned_M2mean,sigma=binned_M2err,absolute_sigma=True) 
		LZp, LZcov = curve_fit(linf,binned_LZx,binned_LZmean,sigma=binned_LZerr,absolute_sigma=True)
	
		M1intercept, M1slope = M1p[0], M1p[1]
		M2intercept, M2slope = M2p[0], M2p[1]
		LZintercept, LZslope = LZp[0], LZp[1]
		M1res = M1intercept + M1slope*Main1[:,xcol] - Main1[:,ycol]
		M2res = M2intercept + M2slope*Main2[:,xcol] - Main2[:,ycol]
		LZres = LZintercept + LZslope*LOWZ[:,xcol] - LOWZ[:,ycol]
		M1r = np.sqrt(1 - np.sum(M1res**2)/np.sum((Main1[:,ycol]-np.mean(Main1[:,ycol]))**2))
		M2r = np.sqrt(1 - np.sum(M2res**2)/np.sum((Main2[:,ycol]-np.mean(Main2[:,ycol]))**2))
		LZr = np.sqrt(1 - np.sum(LZres**2)/np.sum((LOWZ[:,ycol]-np.mean(LOWZ[:,ycol]))**2))
	else:
		M1slope, M1intercept, M1r, M1p, M1se = linregress(Main1[:,xcol],Main1[:,ycol]) 
		M2slope, M2intercept, M2r, M2p, M2se = linregress(Main2[:,xcol],Main2[:,ycol]) 
		LZslope, LZintercept, LZr, LZp, LZse = linregress(LOWZ[:,xcol],LOWZ[:,ycol])

		M1res = M1intercept + M1slope*Main1[:,xcol] - Main1[:,ycol]
		M2res = M2intercept + M2slope*Main2[:,xcol] - Main2[:,ycol]
		LZres = LZintercept + LZslope*LOWZ[:,xcol] - LOWZ[:,ycol]

	ax = plt.subplot(1,2,1)
	x = np.linspace(min(Main1[:,xcol]),xmax)
	ax.plot(x,M1intercept+M1slope*x,color=colours[0],linestyle='--',linewidth=1.5)
	x = np.linspace(min(Main2[:,xcol]),xmax)
	ax.plot(x,M2intercept+M2slope*x,color=colours[1],linestyle='--',linewidth=1.5)
	x = np.linspace(min(LOWZ[:,xcol]),xmax)
	ax.plot(x,LZintercept+LZslope*x,color=colours[2],linestyle='--',linewidth=1.5)
	ax.errorbar(binned_M1x,binned_M1mean,yerr=binned_M1err,fmt=pointstyles[0],markersize=8,\
			elinewidth=1.5,markeredgecolor='none',color=colours[0],label='$m=%0.2f,\,c=%0.2f,\,R^2=%0.2f$'%(M1slope,M1intercept,M1r**2))
	ax.errorbar(binned_M2x,binned_M2mean,yerr=binned_M2err,fmt=pointstyles[1],markersize=9,\
			elinewidth=1.5,markeredgecolor='none',color=colours[1],label='$m=%0.2f,\,c=%0.2f,\,R^2=%0.2f$'%(M2slope,M2intercept,M2r**2))
	ax.errorbar(binned_LZx,binned_LZmean,yerr=binned_LZerr,fmt=pointstyles[3],markersize=8,\
			elinewidth=1.5,markeredgecolor='none',color=colours[2],label='$m=%0.2f,\,c=%0.2f,\,R^2=%0.2f$'%(LZslope,LZintercept,LZr**2))
	ax.legend(loc='lower right',numpoints=1,prop={'size':16})

	print M1p, M2p, LZp

	ax = plt.subplot(1,2,2)
	ax.scatter(Main1[:,xcol],M1res,color=colours[0])
	ax.scatter(Main2[:,xcol],M2res,color=colours[1])
	ax.scatter(LOWZ[:,xcol],LZres,color=colours[2])

def profile_fits(vPrefix='Isolated',stackType='_RQ',diff=True,useDM=True):

	samples = ['MDR1_Main1','MDR1_Main2','MDR1_LOWZ']
	diff_str = 'differential/' if diff else 'cumulative/'
	DM_str = 'DM/res1024_' if useDM else ''
	maxfev=100000
	HSW_params,NH_params,new_params = np.zeros((27,8)),np.zeros((27,12)),np.zeros((27,10))
	bin_means = np.zeros((27,3))
	labels = np.asarray(np.zeros((9,1)),dtype=str)

	plt.figure(figsize=(24,8))
	for name in samples:
		s_ind = samples.index(name)
		filelist = glob.glob(structdir+name+'/profiles/'+diff_str+DM_str+vPrefix+'V'+stackType+'*')
		filelist = np.asarray(filelist)[np.argsort(filelist)]
		metrics = np.loadtxt(structdir+name+'/'+vPrefix+'Voids_metrics.txt',skiprows=1)
		if stackType=='_RQ':
			bin_edges = mquantiles(metrics[:,1],np.linspace(0,1,10))
			for i in range(len(bin_edges)-1):
				bin_means[9*s_ind+i,0], junk = bin_mean_val(bin_edges[i],bin_edges[i+1],metrics[:,1],metrics[:,1])
				bin_means[9*s_ind+i,1], junk = bin_mean_val(bin_edges[i],bin_edges[i+1],metrics[:,1],metrics[:,2])
				bin_means[9*s_ind+i,2], junk = bin_mean_val(bin_edges[i],bin_edges[i+1],metrics[:,1],metrics[:,3])
				labels[i] = r'$\overline{R}_v=%0.1f\,h^{-1}$Mpc' %bin_means[9*s_ind+i,0]
		elif stackType=='_AQ':
			bin_edges = mquantiles(metrics[:,3],np.linspace(0,1,10))
			for i in range(len(bin_edges)-1):
				bin_means[9*s_ind+i,0], junk = bin_mean_val(bin_edges[i],bin_edges[i+1],metrics[:,3],metrics[:,1])
				bin_means[9*s_ind+i,1], junk = bin_mean_val(bin_edges[i],bin_edges[i+1],metrics[:,3],metrics[:,2])
				bin_means[9*s_ind+i,2], junk = bin_mean_val(bin_edges[i],bin_edges[i+1],metrics[:,3],metrics[:,3])
				labels[i] = r"$%0.2f\leq n_\mathrm{avg}<%0.2f$" %(bin_edges[i],bin_edges[i+1])
			filelist = filelist[::-1]
			labels = labels[::-1]
		elif stackType=='_DQ':
			bin_edges = mquantiles(metrics[:,2],np.linspace(0,1,10))
			for i in range(len(bin_edges)-1):
				bin_means[9*s_ind+i,0], junk = bin_mean_val(bin_edges[i],bin_edges[i+1],metrics[:,2],metrics[:,1])
				bin_means[9*s_ind+i,1], junk = bin_mean_val(bin_edges[i],bin_edges[i+1],metrics[:,2],metrics[:,2])
				bin_means[9*s_ind+i,2], junk = bin_mean_val(bin_edges[i],bin_edges[i+1],metrics[:,2],metrics[:,3])
				labels[i] = '$%0.2f\leq n_\mathrm{min}<%0.2f$' %(bin_edges[i],bin_edges[i+1])
			filelist = filelist[::-1]
			labels = labels[::-1]
		elif stackType=='_lR':
			H,bin_edges = np.histogram(np.log10(metrics[:,1]),bins=9)
			for i in range(H.shape[0]):
				bin_means[9*s_ind+i,0], junk = bin_mean_val(10**bin_edges[i],10**bin_edges[i+1],metrics[:,1],metrics[:,1])
				bin_means[9*s_ind+i,1], junk = bin_mean_val(10**bin_edges[i],10**bin_edges[i+1],metrics[:,1],metrics[:,2])
				bin_means[9*s_ind+i,2], junk = bin_mean_val(10**bin_edges[i],10**bin_edges[i+1],metrics[:,1],metrics[:,3])
				labels[i] = '$%0.1f\leq R_v<%0.1f$' %(10**bin_edges[i],10**bin_edges[i+1])

		ax = plt.subplot(1,3,s_ind+1)
		for filename in filelist:
			f_ind = np.where(filelist==filename)[0][0]
			x = np.linspace(0.01,3)
			data = np.loadtxt(filename,skiprows=2)
			
			popt, pcov = curve_fit(HSW_profile,data[:,0],data[:,1],sigma=data[:,2],p0=[-1,1,2,4],absolute_sigma=True,maxfev=maxfev)
			HSW_params[s_ind*9+f_ind,0:4] = popt
			HSW_params[s_ind*9+f_ind,4:8] = np.sqrt(np.diag(pcov))

			popt, pcov = curve_fit(NH_profile,data[:,0],data[:,1],sigma=data[:,2],p0=[-1,1,2,4],absolute_sigma=True,maxfev=maxfev)
#			NH_params[s_ind*9+f_ind,0:4] = popt
#			NH_params[s_ind*9+f_ind,4:8] = np.sqrt(np.diag(pcov))

			popt, pcov = curve_fit(pol_profile,data[:,0],data[:,1],sigma=data[:,2],p0=[1,1,1,1,1,1],absolute_sigma=True,maxfev=maxfev)
			NH_params[s_ind*9+f_ind,0:6] = popt
			NH_params[s_ind*9+f_ind,6:12] = np.sqrt(np.diag(pcov))

			popt, pcov = curve_fit(new_profile,data[:,0],data[:,1],sigma=data[:,2],p0=[-1,1,1,2,4],absolute_sigma=True,maxfev=maxfev)
			new_params[s_ind*9+f_ind,0:5] = popt
			new_params[s_ind*9+f_ind,5:10] = np.sqrt(np.diag(pcov))
			
			if f_ind%2==0:
				ax.errorbar(data[:,0],data[:,1],yerr=data[:,2],color=kelly_RdYlGn[f_ind],fmt=pointstyles[f_ind],\
					elinewidth=1.5,markersize=8,markeredgecolor='none',label=labels[f_ind,0])
#				ax.plot(x,new_profile(x,new_params[s_ind*9+f_ind,0],new_params[s_ind*9+f_ind,1],new_params[s_ind*9+f_ind,2],new_params[s_ind*9+f_ind,3],\
#					new_params[s_ind*9+f_ind,4]),color=kelly_RdYlGn[f_ind],linewidth=1.5)
				ax.plot(x,HSW_profile(x,HSW_params[s_ind*9+f_ind,0],HSW_params[s_ind*9+f_ind,1],HSW_params[s_ind*9+f_ind,2],HSW_params[s_ind*9+f_ind,3]),\
					color=kelly_RdYlGn[f_ind],linestyle='--',linewidth=1.5)
				ax.plot(x,pol_profile(x,NH_params[s_ind*9+f_ind,0],NH_params[s_ind*9+f_ind,1],NH_params[s_ind*9+f_ind,2],NH_params[s_ind*9+f_ind,3],\
					NH_params[s_ind*9+f_ind,4],NH_params[s_ind*9+f_ind,5]),color=kelly_RdYlGn[f_ind],linestyle=':',linewidth=1.5)
			ax.set_xlabel(r'$r/R_v$',fontsize=24,fontweight='extra bold')
			if useDM:
				if diff:
					ax.set_ylabel(r'$\rho(r)/\overline{\rho}$',fontsize=24,fontweight='extra bold')
				else:
					ax.set_ylabel(r'$1+\Delta(r)$',fontsize=24,fontweight='extra bold')
			else:
				if diff:
					ax.set_ylabel(r'$n(r)/\overline{n}$',fontsize=24,fontweight='extra bold')
				else:
					ax.set_ylabel(r'$1+\Delta_n(r)$',fontsize=24,fontweight='extra bold')

	 	ax.set_ylim([0,1.6])
		ax.set_title(name.replace('MDR1_',''),fontsize=20)
		ax.legend(loc='lower right',numpoints=1,prop={'size':12,'family':'serif'})
		ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
		ax.tick_params(axis='both', labelsize=16)

	plt.tight_layout()
	return HSW_params, NH_params, new_params, bin_means		

def split_navg(sHandle='MDR1_Main1',vPrefix='Isolated'):

	data = np.loadtxt(structdir+sHandle+'/'+vPrefix+'Voids_metrics.txt',skiprows=1)
	navg_bins = mquantiles(data[:,3],np.linspace(0,1,10))

	fig,axes = plt.subplots(3,3,sharex=False,sharey=False,figsize=(14,8))
	for i in range(9):
		select = data[np.logical_and(data[:,3]>=navg_bins[i],data[:,3]<navg_bins[i+1]),:]
		x = np.linspace(np.min(select[:,1]),np.max(select[:,1]))
		popt, pcov = curve_fit(linf,select[:,1],select[:,9])
		R_bins = mquantiles(select[:,1],np.linspace(0,1,10))
		R_bins, binned_DM, binned_DM_err = binner(select[:,1],select[:,9],R_bins) 
		R_bins, binned_R, binned_R_err = binner(select[:,1],select[:,1],R_bins) 

		ax = axes.flat[i]
		ax.errorbar(binned_R,binned_DM,yerr=binned_DM_err,xerr=binned_R_err,color=kelly_RdYlGn[0],fmt=pointstyles[0],\
			markersize=8,elinewidth=1.5,markeredgecolor='none')
		ax.plot(x,linf(x,popt[0],popt[1]),color=kelly_RdYlGn[0],linestyle='--',linewidth=1.5)
		ax.set_xlabel(r'$R_v\;[h^{-1}\mathrm{Mpc}]$',fontsize=16,fontweight='extra bold')
		ax.set_ylabel(r'$\Delta(r=3R_v)$',fontsize=16,fontweight='extra bold')
		ax.set_title(r'$%0.2f\leq n_\mathrm{avg}<%0.2f$' %(navg_bins[i],navg_bins[i+1]),fontsize=16,fontweight='extra bold')
	plt.tight_layout()

	fig,axes = plt.subplots(3,3,sharex=False,sharey=False,figsize=(14,8))
	for i in range(9):
		select = data[np.logical_and(data[:,3]>=navg_bins[i],data[:,3]<navg_bins[i+1]),:]
		x = np.linspace(np.min(select[:,2]),np.max(select[:,2]))
		popt, pcov = curve_fit(linf,select[:,2],select[:,9])
		n_bins = mquantiles(select[:,2],np.linspace(0,1,10))
		n_bins, binned_DM, binned_DM_err = binner(select[:,2],select[:,9],n_bins) 
		n_bins, binned_n, binned_n_err = binner(select[:,2],select[:,2],n_bins) 

		ax = axes.flat[i]
		ax.errorbar(binned_n,binned_DM,yerr=binned_DM_err,xerr=binned_n_err,color=kelly_RdYlGn[0],fmt=pointstyles[0],\
			markersize=8,elinewidth=1.5,markeredgecolor='none')
		ax.plot(x,linf(x,popt[0],popt[1]),color=kelly_RdYlGn[0],linestyle='--',linewidth=1.5)
		ax.set_xlabel(r'$n_\mathrm{min}/\overline{n}$',fontsize=16,fontweight='extra bold')
		ax.set_ylabel(r'$\Delta(r=3R_v)$',fontsize=16,fontweight='extra bold')
		ax.set_title(r'$%0.2f\leq n_\mathrm{avg}<%0.2f$' %(navg_bins[i],navg_bins[i+1]),fontsize=16,fontweight='extra bold')
	plt.tight_layout()

def hsw_fitparams():

	#get some info on the sample
	sinfoFile = structdir+'MDR1_DM_Main1/sample_info.dat'
	parms = imp.load_source("name",sinfoFile)
	meanNNsep = parms.tracerDens**(-1.0/3)

	filelist = glob.glob(structdir+'MDR1_DM_Main1/profiles/differential/VIDEbV_newSQ_*')
	nfiles = np.asarray(filelist)[np.argsort(filelist)]
	filelist = glob.glob(structdir+'MDR1_DM_Main1/profiles/differential/DM/res1024_VIDEbV_newSQ_*')
	dmfiles = np.asarray(filelist)[np.argsort(filelist)]

	catalogue = np.loadtxt(structdir+'MDR1_DM_Main1/VIDEbVoids_info.txt',skiprows=2)
	#bin_edges = mquantiles(catalogue[:,6],np.linspace(0,1,16))
	#bin_edges = 10**np.linspace(np.log10(min(catalogue[:,6])),np.log10(max(catalogue[:,6])),16)[3:]
	bin_edges = [0,10.5,12.4,14.1,17.5,20.7,24.5,29.1,34.5,40.9,48.5,58]
	bin_means = np.zeros((len(bin_edges)-1,2))
	for i in range(len(bin_edges)-1):
		bin_means[i,0], bin_means[i,1] = bin_mean_val(bin_edges[i],bin_edges[i+1],catalogue[:,6],catalogue[:,6])
	#bin_means = bin_means[:-1]

	nHSW, dmHSW = np.zeros((len(nfiles),8)),np.zeros((len(dmfiles),8))
	maxfev=100000
	print bin_means.shape
	print len(nfiles)
	for filename in nfiles:
		ind = np.where(nfiles==filename)[0][0]
		data = np.loadtxt(filename,skiprows=2)
		data = data[data[:,0]*bin_means[ind,0]>(3.0/(4*np.pi))**(1.0/3)*meanNNsep,:]
		popt, pcov = curve_fit(HSW_profile,data[:,0],data[:,1],sigma=data[:,2],p0=[-1,1,2,4],absolute_sigma=True,maxfev=maxfev)
		nHSW[ind,0:4] = popt
		nHSW[ind,4:8] = np.sqrt(np.diag(pcov))
	for filename in dmfiles:
		ind = np.where(dmfiles==filename)[0][0]
		data = np.loadtxt(filename,skiprows=2)
		popt, pcov = curve_fit(HSW_profile,data[:,0],data[:,1],sigma=data[:,2],p0=[-1,1,2,4],absolute_sigma=True,maxfev=maxfev)
		dmHSW[ind,0:4] = popt
		dmHSW[ind,4:8] = np.sqrt(np.diag(pcov))
	
	fig,axes = plt.subplots(2,sharex=True,sharey=False,figsize=(12,12))

	ax = axes.flat[0]
	ax.errorbar(bin_means[:,0],nHSW[:,0],yerr=2*nHSW[:,4],xerr=2*bin_means[:,1],fmt=pointstyles[0],\
		color=kelly_RdYlGn[0],markersize=8,elinewidth=2,markeredgecolor='none',label=r'$n(r)$ fit')
	ax.errorbar(bin_means[:,0],dmHSW[:,0],yerr=2*dmHSW[:,4],xerr=2*bin_means[:,1],fmt=pointstyles[3],\
		color=kelly_RdYlGn[6],markersize=7,elinewidth=2,markeredgecolor='none',label=r'$\rho(r)$ fit')
	ax.legend(loc='upper right',numpoints=1,prop={'size':16})
#	ax.set_ylim([-1.0,-0.3])
#	ax.set_yticks([-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3])
	ax.set_ylim([-1.05,-0.3])
	ax.set_yticks([-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3])
	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='y', labelsize=16)
	ax.set_ylabel(r"$\delta_c$",fontsize=24,fontweight='extra bold')
	
	ax = axes.flat[1]
	ax.errorbar(bin_means[:,0],nHSW[:,1],yerr=2*nHSW[:,5],xerr=2*bin_means[:,1],fmt=pointstyles[0],\
			color=kelly_RdYlGn[0],markersize=8,elinewidth=2,markeredgecolor='none')
	ax.errorbar(bin_means[:,0],dmHSW[:,1],yerr=2*dmHSW[:,5],xerr=2*bin_means[:,1],fmt=pointstyles[3],\
			color=kelly_RdYlGn[6],markersize=7,elinewidth=2,markeredgecolor='none')
	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
#	ax.set_ylim([0.6,1.0])
#	ax.set_yticks([0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95])
	ax.set_ylim([0.6,1.5])
#	ax.set_yticks([0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95])
	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)
	ax.set_ylabel(r"$r_s/R_v$",fontsize=24,fontweight='extra bold')
	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})

	fig.subplots_adjust(hspace=0)
	plt.xlabel(r'$R_v\,[h^{-1}\mathrm{Mpc}]$',fontsize=24,fontweight='extra bold')

	#save the figure
	fig_filename = figdir+'/Main_HSW1.pdf'
	#plt.savefig(fig_filename,bbox_inches='tight')

	fig,axes = plt.subplots(2,sharex=True,sharey=False,figsize=(12,12))
	xa = np.linspace(0.57,0.98)
	xb1 = np.linspace(0.57,0.91)
	xb2 = np.linspace(0.91,0.98)

	ax = axes.flat[0]
	ax.errorbar(nHSW[:,1],nHSW[:,2],yerr=2*nHSW[:,6],xerr=2*nHSW[:,5],fmt=pointstyles[0],\
			color=kelly_RdYlGn[0],markersize=8,elinewidth=2,markeredgecolor='none',label=r'$n(r)$ fit')
	ax.errorbar(dmHSW[:,1],dmHSW[:,2],yerr=2*dmHSW[:,6],xerr=2*dmHSW[:,5],fmt=pointstyles[3],\
			color=kelly_RdYlGn[6],markersize=7,elinewidth=2,markeredgecolor='none',label=r'$\rho(r)$ fit')
	ax.plot(xa,-2.0*xa+4,'k--',linewidth=1.5)
	ax.legend(loc='upper right',numpoints=1,prop={'size':16})
	ax.set_ylim([1.0,4.4])
	ax.set_yticks([1.0,1.5,2.0,2.5,3.0,3.5,4.0])
	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='y', labelsize=16)
	ax.set_ylabel(r"$\alpha$",fontsize=24,fontweight='extra bold')
	
	ax = axes.flat[1]
	ax.errorbar(nHSW[:,1],nHSW[:,3],yerr=2*nHSW[:,7],xerr=2*nHSW[:,5],fmt=pointstyles[0],\
			color=kelly_RdYlGn[0],markersize=8,elinewidth=2,markeredgecolor='none',label=r'$n(r)$ fit')
	ax.errorbar(dmHSW[:,1],dmHSW[:,3],yerr=2*dmHSW[:,7],xerr=2*dmHSW[:,5],fmt=pointstyles[3],\
			color=kelly_RdYlGn[6],markersize=7,elinewidth=2,markeredgecolor='none',label=r'$\rho(r)$ fit')
	ax.plot(xb1,17.5*xb1-6.5,'k--',linewidth=1.5)
	ax.plot(xb2,-9.8*xb2+18.4,'k--',linewidth=1.5)
	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
	ax.set_ylim([4.0,11.5])
	ax.set_yticks([4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0])
	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)
	ax.set_ylabel(r"$\beta$",fontsize=24,fontweight='extra bold')
	ax.set_xlim([0.57,1.0])
	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})

	fig.subplots_adjust(hspace=0)
	plt.xlabel(r'$r_s/R_v$',fontsize=24,fontweight='extra bold')

	#save the figure
	fig_filename = figdir+'/Main_HSW2.pdf'
	#plt.savefig(fig_filename,bbox_inches='tight')
	return nHSW,dmHSW,bin_means

def Dense_profiles(stackType='SQ'):

	#get some info on the sample
	sinfoFile = structdir+'MDR1_DM_Dense/sample_info.dat'
	parms = imp.load_source("name",sinfoFile)
	meanNNsep = parms.tracerDens**(-1.0/3)

	filelist = glob.glob('/home/seshadri/Workspace/structures/MDR1_DM_Dense/profiles/differential/DM/res1024_VIDEbV_'+stackType+'_*')
	dmfiles = np.asarray(filelist)[np.argsort(filelist)]
	filelist = glob.glob('/home/seshadri/Workspace/structures/MDR1_DM_Dense/profiles/differential/VIDEbV_'+stackType+'_*')
	nfiles = np.asarray(filelist)[np.argsort(filelist)]
	metrics = np.loadtxt('/home/seshadri/Workspace/structures/MDR1_DM_Dense/VIDEbVoids_metrics.txt',skiprows=1)
	if stackType=='SQ': 
		bin_edges = [0,4,6,8,9,10,11,12,14,16,20,25,30,35,100]
	elif stackType=='RQ':
		bin_edges = mquantiles(metrics[:,1],np.linspace(0,1,15))
	maxfev = 100000
	x = np.linspace(0.01,3)
	nHSW, dmHSW = np.zeros((len(nfiles),8)),np.zeros((len(nfiles),8))
	bin_means = np.zeros((len(filelist),3))
	labels = np.asarray(np.zeros((len(filelist),1)),dtype=str)
	for i in range(len(bin_edges)-1):
		bin_means[i,0], junk = bin_mean_val(bin_edges[i],bin_edges[i+1],metrics[:,1],metrics[:,1])
		bin_means[i,1], junk = bin_mean_val(bin_edges[i],bin_edges[i+1],metrics[:,1],metrics[:,2])
		bin_means[i,2], junk = bin_mean_val(bin_edges[i],bin_edges[i+1],metrics[:,1],metrics[:,3])
		labels[i] = '$%0.1f\leq R_v<%0.1f$' %(bin_edges[i],bin_edges[i+1])


	plt.figure(figsize=(14,8))
	ax = plt.subplot(1,2,1)
	for filename in nfiles:
		f_ind = np.where(nfiles==filename)[0][0]
		data = np.loadtxt(filename,skiprows=2)
		data = data[data[:,0]*bin_means[f_ind,0]>(3.0/(4*np.pi))**(1.0/3)*meanNNsep,:]
		popt, pcov = curve_fit(HSW_profile,data[:,0],data[:,1],sigma=data[:,2],p0=[-1,1,2,4],absolute_sigma=True,maxfev=maxfev)
		nHSW[f_ind,0:4] = popt
		nHSW[f_ind,4:8] = np.sqrt(np.diag(pcov))
		ax.errorbar(data[:,0],data[:,1],yerr=data[:,2],color=kelly_RdYlGn[f_ind%9],fmt=pointstyles[f_ind%9],\
			elinewidth=1.5,markersize=8,markeredgecolor='none',label=labels[f_ind,0])
		ax.plot(x,HSW_profile(x,popt[0],popt[1],popt[2],popt[3]),color=kelly_RdYlGn[f_ind%9],linestyle='--',linewidth=1.5)
	ax.set_xlabel(r'$r/R_v$',fontsize=24,fontweight='extra bold')
	ax.set_ylabel(r'$n(r)/\overline{n}$',fontsize=24,fontweight='extra bold')
	ax.legend(loc='lower right',numpoints=1,prop={'size':12,'family':'serif'})
	ax.set_ylim([0,2])
	ax = plt.subplot(1,2,2)
	for filename in dmfiles:
		f_ind = np.where(dmfiles==filename)[0][0]
		data = np.loadtxt(filename,skiprows=2)
		popt, pcov = curve_fit(HSW_profile,data[:,0],data[:,1],sigma=data[:,2],p0=[-1,1,2,4],absolute_sigma=True,maxfev=maxfev)
		dmHSW[f_ind,0:4] = popt
		dmHSW[f_ind,4:8] = np.sqrt(np.diag(pcov))
		ax.errorbar(data[:,0],data[:,1],yerr=data[:,2],color=kelly_RdYlGn[f_ind%9],fmt=pointstyles[f_ind%9],\
			elinewidth=1.5,markersize=8,markeredgecolor='none',label=labels[f_ind,0])
		ax.plot(x,HSW_profile(x,popt[0],popt[1],popt[2],popt[3]),color=kelly_RdYlGn[f_ind%9],linestyle='--',linewidth=1.5)
	ax.set_xlabel(r'$r/R_v$',fontsize=24,fontweight='extra bold')
	ax.set_ylabel(r'$\rho(r)/\overline{\rho}$',fontsize=24,fontweight='extra bold')
	ax.legend(loc='lower right',numpoints=1,prop={'size':12,'family':'serif'})
	ax.set_ylim([0,2])

	fig,axes = plt.subplots(2,sharex=True,sharey=False,figsize=(12,12))

	ax = axes.flat[0]
	ax.errorbar(bin_means[1:,0],nHSW[1:,0],yerr=2*nHSW[1:,4],xerr=2*bin_means[1:,1],fmt=pointstyles[0],\
		color=kelly_RdYlGn[0],markersize=8,elinewidth=2,markeredgecolor='none',label=r'$n(r)$ fit')
	ax.errorbar(bin_means[1:,0],dmHSW[1:,0],yerr=2*dmHSW[1:,4],xerr=2*bin_means[1:,1],fmt=pointstyles[3],\
		color=kelly_RdYlGn[6],markersize=7,elinewidth=2,markeredgecolor='none',label=r'$\rho(r)$ fit')
	ax.legend(loc='upper right',numpoints=1,prop={'size':16})
	#ax.set_ylim([-1.0,-0.3])
	#ax.set_yticks([-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3])
	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='y', labelsize=16)
	ax.set_ylabel(r"$\delta_c$",fontsize=24,fontweight='extra bold')
	
	ax = axes.flat[1]
	ax.errorbar(bin_means[1:,0],nHSW[1:,1],yerr=2*nHSW[1:,5],xerr=2*bin_means[1:,1],fmt=pointstyles[0],\
			color=kelly_RdYlGn[0],markersize=8,elinewidth=2,markeredgecolor='none')
	ax.errorbar(bin_means[1:,0],dmHSW[1:,1],yerr=2*dmHSW[1:,5],xerr=2*bin_means[1:,1],fmt=pointstyles[3],\
			color=kelly_RdYlGn[6],markersize=7,elinewidth=2,markeredgecolor='none')
	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
	#ax.set_ylim([0.6,1.0])
	#ax.set_yticks([0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95])
	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)
	ax.set_ylabel(r"$r_s/R_v$",fontsize=24,fontweight='extra bold')
	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})

	fig.subplots_adjust(hspace=0)
	plt.xlabel(r'$R_v\,[h^{-1}\mathrm{Mpc}]$',fontsize=24,fontweight='extra bold')

	#save the figure
	fig_filename = figdir+'/Dense_HSW1.pdf'
	#plt.savefig(fig_filename,bbox_inches='tight')

	fig,axes = plt.subplots(2,sharex=True,sharey=False,figsize=(12,12))
	xa = np.linspace(0.57,1.38)
	xb1 = np.linspace(0.57,0.91)
	xb2 = np.linspace(0.91,1.38)

	ax = axes.flat[0]
	ax.errorbar(nHSW[1:,1],nHSW[1:,2],yerr=2*nHSW[1:,6],xerr=2*nHSW[1:,5],fmt=pointstyles[0],\
			color=kelly_RdYlGn[0],markersize=8,elinewidth=2,markeredgecolor='none',label=r'$n(r)$ fit')
	ax.errorbar(dmHSW[1:,1],dmHSW[1:,2],yerr=2*dmHSW[1:,6],xerr=2*dmHSW[1:,5],fmt=pointstyles[3],\
			color=kelly_RdYlGn[6],markersize=7,elinewidth=2,markeredgecolor='none',label=r'$\rho(r)$ fit')
	ax.plot(xa,-2.0*xa+4,'k--',linewidth=1.5)
	ax.legend(loc='upper right',numpoints=1,prop={'size':16})
	#ax.set_ylim([1.0,4.4])
	#ax.set_yticks([1.0,1.5,2.0,2.5,3.0,3.5,4.0])
	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='y', labelsize=16)
	ax.set_ylabel(r"$\alpha$",fontsize=24,fontweight='extra bold')
	
	ax = axes.flat[1]
	ax.errorbar(nHSW[1:,1],nHSW[1:,3],yerr=2*nHSW[1:,7],xerr=2*nHSW[1:,5],fmt=pointstyles[0],\
			color=kelly_RdYlGn[0],markersize=8,elinewidth=2,markeredgecolor='none',label=r'$n(r)$ fit')
	ax.errorbar(dmHSW[1:,1],dmHSW[1:,3],yerr=2*dmHSW[1:,7],xerr=2*dmHSW[1:,5],fmt=pointstyles[3],\
			color=kelly_RdYlGn[6],markersize=7,elinewidth=2,markeredgecolor='none',label=r'$\rho(r)$ fit')
	ax.plot(xb1,17.5*xb1-6.5,'k--',linewidth=1.5)
	ax.plot(xb2,-9.8*xb2+18.4,'k--',linewidth=1.5)
	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
	#ax.set_ylim([4.0,11.5])
	#ax.set_yticks([4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0])
	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)
	ax.set_ylabel(r"$\beta$",fontsize=24,fontweight='extra bold')
	#ax.set_xlim([0.57,1.0])
	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})

	fig.subplots_adjust(hspace=0)
	plt.xlabel(r'$r_s/R_v$',fontsize=24,fontweight='extra bold')

	#save the figure
	fig_filename = figdir+'/Dense_HSW2.pdf'
	#plt.savefig(fig_filename,bbox_inches='tight')
