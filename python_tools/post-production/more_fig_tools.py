import numpy as np
import os, glob
import imp
import subprocess
from corr import *
from periodic_kdtree import PeriodicCKDTree
from scipy.spatial import cKDTree
from scipy.optimize import brentq, curve_fit, fsolve
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.stats.mstats import mquantiles
from scipy.stats import linregress, gaussian_kde
from scipy.special import erf
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from postprocess import read_boundaries
from tools import comovr, binner, test_bin, bin_mean_val
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
import mpl_toolkits.axisartist.angle_helper as angle_helper
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import MaxNLocator
import healpy as hp
from matplotlib.projections.geo import GeoAxes
import statsmodels.api as sm
from fig_tools import shiftedColorMap

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
	'#232C16', #19 Dark Olive Green
]
kelly_RdYlGn = np.asarray(kelly_colours)[[7,16,14,0,12,2,18,4,15]]
kelly_RdYlBu = np.asarray(kelly_colours)[[11,9,3,0,12,2,18,4,15]]

pointstyles = np.array(['o','^','v','s','D','<','>','p','d','*','h','H','8'])

def linf(x,A,B):
	return A+B*x
	
def phi1(r,phi0,r0):
	return phi0*(1./(1+r**2./r0**2.))
		
def phi2(r,phi0,r0,r1):
	return phi0*(1.-r/r1)/(1+r**2./r0**2.)
		
def phi3(r,phi0,r0,alpha):
	return phi0/(1+(r/r0)**alpha)

def skew_normal(x,mu,sigma,alpha):
    normpdf = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(np.power((x-mu),2)/(2*np.power(sigma,2))))
    normcdf = (0.5*(1+erf((alpha*((x-mu)/sigma))/(np.sqrt(2)))))
    return 2*normpdf*normcdf

#    return (1+erf(alpha*(x-mu)/(np.sqrt(2)*sigma)))*np.exp(-(x-mu)**2.0/(2*sigma**2.0))/(np.sqrt(2*np.pi)*sigma)
				
def fit_skew_dist(values):
	maxfev=10000
	if len(values)>20:
		H,edges = np.histogram(values,bins=15,density=True)
#		H = H*(edges[1:]-edges[:-1])
		x = 0.5*(edges[1:]+edges[:-1])
		popt, pcov = curve_fit(skew_normal,x,H,p0=[0,np.std(H),0],maxfev=maxfev)
		errors = np.sqrt(np.diag(pcov))
		return popt, errors
	else:
		return [np.mean(values),np.std(values),0], [np.std(values)/np.sqrt(len(values)),0,0]
				
def prettify_plot(ax):
 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
 	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)

def stacking_AISW():
	
	aisw_vals = np.array([8.7, 12.16, 1.0, 6.0, 8.5, 1.64])
	aisw_errs = np.array([2.18, 5.08, 7.5, 3.75, 4.1, 0.53])
	Granett_Aisw = np.array([8.7,8.7*2,4.4])
	Granett_Aisw_errs = np.array([2.18,2.18*2,1.1])
	Planck_xcorr = np.array([1.0,2.1,1.37])
	Planck_xcorr_errs = np.array([0.25,0.84,0.56])
	labels = np.array(['Planck Collab. XIX 2014/\nGranett et al. 2008', 'Cai et al. 2014', \
	    'Hotchkiss et al. 2015', 'Cai et al. 2017', 'Kovacs et al. (DES) 2017', \
	    'Nadathur & Crittenden 2016'])
	y = 1+np.arange(len(aisw_vals))
	
	# figure 1: just Granett
	plt.figure(figsize=(7,4))
	(_,caps,_) = plt.errorbar(aisw_vals[0],y[0],xerr=aisw_errs[0],fmt='d',color='#3130ff',\
	                          elinewidth=1.5,markersize=8,markeredgecolor='none')
	for cap in caps: cap.set_markeredgewidth(2)
	plt.tick_params(axis='y',which='both',left='off',right='off',labelleft='off')
	plt.tick_params(labelsize=14)
	plt.axvline(1,c='k',ls='--')
	plt.xlabel(r'$A_\mathrm{ISW}$',fontsize=22)
	# plt.xlim([0,len(aisw_vals)])
	plt.gca().invert_yaxis()
	plt.savefig('/Users/seshadri/Workspace/figures/ISWproject/Aisw_1.png',bbox_inches='tight')
	
	# figure 2: Granett variations
	plt.figure(figsize=(7,4))
	yvals = np.array([y[0]-0.02,y[0]+0.02])
	(_,caps,barline) = plt.errorbar(Granett_Aisw[1:],yvals,xerr=Granett_Aisw_errs[1:],fmt='d',color='#3130ff',\
	                          elinewidth=1.5,markersize=8,markeredgecolor='none')
	for cap in caps: cap.set_markeredgewidth(2)
	barline[0].set_linestyle('--')
	(_,caps,_) = plt.errorbar(aisw_vals[0],y[0],xerr=aisw_errs[0],fmt='d',color='#3130ff',\
	                          elinewidth=1.5,markersize=8,markeredgecolor='none')
	for cap in caps: cap.set_markeredgewidth(2)
	plt.tick_params(axis='y',which='both',left='off',right='off',labelleft='off')
	plt.tick_params(labelsize=14)
	plt.axvline(1,c='k',ls='--')
	plt.xlabel(r'$A_\mathrm{ISW}$',fontsize=22)
	plt.ylim([0.7,1.3])
	plt.gca().invert_yaxis()
	plt.savefig('/Users/seshadri/Workspace/figures/ISWproject/Aisw_2.png',bbox_inches='tight')
	
	# figure 3: just Granett + Planck limits
	plt.figure(figsize=(7,4))
	(_,caps,_) = plt.errorbar(aisw_vals[0],y[0],xerr=aisw_errs[0],fmt='d',color='#3130ff',\
	                          elinewidth=1.5,markersize=8,markeredgecolor='none')
	for cap in caps: cap.set_markeredgewidth(2)
	yvals = np.linspace(0,2)
	xvals = np.ones_like(yvals)
	plt.fill_betweenx(yvals,(Planck_xcorr[0]-Planck_xcorr_errs[0])*xvals,(Planck_xcorr[0]+Planck_xcorr_errs[0])*xvals,\
	                  color='gray',alpha=0.3)
	plt.yticks([0,1],['',''])
	plt.tick_params(labelsize=14)
	plt.axvline(1,c='k',ls='--')
	plt.xlabel(r'$A_\mathrm{ISW}$',fontsize=22)
	plt.annotate('Planck combined x-corr.',xy=(1.3,0.4),rotation=270,fontsize=10)
	plt.gca().invert_yaxis()
	plt.savefig('/Users/seshadri/Workspace/figures/ISWproject/Aisw_3.png',bbox_inches='tight')
	
	# figure 4: Granett + other CTH stacking + Planck limits
	plt.figure(figsize=(7,8))
	(_,caps,_) = plt.errorbar(aisw_vals[1:-1],y[1:-1],xerr=aisw_errs[1:-1],fmt='o',color='k',elinewidth=1.5,
	    markersize=8)
	for cap in caps: cap.set_markeredgewidth(2)
	(_,caps,barline) = plt.errorbar(aisw_vals[0],y[0],xerr=aisw_errs[0],fmt='d',color='#3130ff',elinewidth=1.5,
	    markersize=7)
	for cap in caps: cap.set_markeredgewidth(2)
	yvals = np.linspace(0,6)
	xvals = np.ones_like(yvals)
	plt.fill_betweenx(yvals,(Planck_xcorr[0]-Planck_xcorr_errs[0])*xvals,(Planck_xcorr[0]+Planck_xcorr_errs[0])*xvals,\
	                  color='gray',alpha=0.3)
	plt.yticks(y[:-1],labels[:-1])#,rotation='vertical')
	plt.tick_params(labelsize=14)
	plt.axvline(1,c='k',ls='--')
	plt.xlabel(r'$A_\mathrm{ISW}$',fontsize=22)
	plt.ylim([0,len(aisw_vals)])
	plt.xlim([0,15])
	plt.gca().invert_yaxis()
	plt.savefig('/Users/seshadri/Workspace/figures/ISWproject/Aisw_4.png',bbox_inches='tight')
	
	# figure 5: all stacking + Planck limits
	plt.figure(figsize=(7,8))
	(_,caps,_) = plt.errorbar(aisw_vals[1:-1],y[1:-1],xerr=aisw_errs[1:-1],fmt='o',color='k',elinewidth=1.5,
	    markersize=8)
	for cap in caps: cap.set_markeredgewidth(2)
	(_,caps,_) = plt.errorbar(aisw_vals[0],y[0],xerr=aisw_errs[0],fmt='d',color='#3130ff',elinewidth=1.5,
	    markersize=7)
	for cap in caps: cap.set_markeredgewidth(2)
	(_,caps,_) = plt.errorbar(aisw_vals[-1],y[-1],xerr=aisw_errs[-1],fmt='s',color=kelly_RdYlGn[7],
	    markeredgecolor='none',elinewidth=1.5,markersize=7)
	for cap in caps: cap.set_markeredgewidth(2)	
	yvals = np.linspace(0,7)
	xvals = np.ones_like(yvals)
	plt.fill_betweenx(yvals,(Planck_xcorr[0]-Planck_xcorr_errs[0])*xvals,(Planck_xcorr[0]+Planck_xcorr_errs[0])*xvals,\
	                  color='gray',alpha=0.3)
	plt.yticks(y,labels)
	plt.tick_params(labelsize=14)
	plt.axvline(1,c='k',ls='--')
	plt.xlabel(r'$A_\mathrm{ISW}$',fontsize=22)
	plt.ylim([0,len(aisw_vals)+1])
	plt.xlim([0,15])
	plt.gca().invert_yaxis()
	plt.savefig('/Users/seshadri/Workspace/figures/ISWproject/Aisw_5.png',bbox_inches='tight')
	
    # figure 6: our stacking vs Planck x-corr
    plt.figure(figsize=(7,4))
    (_,caps,_) = plt.errorbar(Planck_xcorr[1:],[1,2],xerr=Planck_xcorr_errs[1:],fmt='o',color='k',elinewidth=1.5,
        markersize=7)
    for cap in caps: cap.set_markeredgewidth(2)
    (_,caps,_) = plt.errorbar(aisw_vals[-1],3,xerr=aisw_errs[-1],fmt='s',color=kelly_RdYlGn[7],
        markeredgecolor='none',elinewidth=1.5,markersize=7)
    for cap in caps: cap.set_markeredgewidth(2)
    yvals = np.linspace(0,7)
    xvals = np.ones_like(yvals)
    plt.fill_betweenx(yvals,(Planck_xcorr[0]-Planck_xcorr_errs[0])*xvals,(Planck_xcorr[0]+Planck_xcorr_errs[0])*xvals,\
                      color='gray',alpha=0.3)
    plt.tick_params(labelsize=14)
    plt.tick_params(axis='y',which='both',left='off',right='off',labelleft='off')
    plt.axvline(1,c='k',ls='--')
    plt.xlabel(r'$A_\mathrm{ISW}$',fontsize=22)
    plt.ylim([0,4])
    plt.xlim([0,3])
    plt.gca().invert_yaxis()
    plt.annotate('WMAP7 x SDSS LRGs',xy=(1.7,0.85),fontsize=10)
    plt.annotate('Giannantonio et al. 2012',xy=(1.6,1.2),fontsize=10)
    plt.annotate('Planck x CMASS/LOWZ',xy=(0.9,1.85),fontsize=10)
    plt.annotate('Planck XXI 2015',xy=(1.1,2.2),fontsize=10)
    plt.annotate('Planck combined (all tracers)',xy=(0.65,0.9),fontsize=10,rotation=90)
    plt.annotate('Planck+CMASS stacking',xy=(1.18,2.85),fontsize=10,color=kelly_RdYlGn[7])
    plt.annotate('Nadathur & Crittenden 2016',xy=(1.12,3.2),fontsize=10,color=kelly_RdYlGn[7])
    plt.savefig('/Users/seshadri/Workspace/figures/ISWproject/Aisw_6.png',bbox_inches='tight')
	
def setup_axes(fig, rect, theta, radius, rot=True):

    # PolarAxes.PolarTransform takes radian. However, we want our coordinate
    # system in degree
    if rot:
		shift = 90 - 0.5*(theta[1]+theta[0])
		tr = Affine2D().translate(shift,0).scale(np.pi/180., 1.) + PolarAxes.PolarTransform()
    else:
		tr = Affine2D().scale(np.pi/180., 1.) + PolarAxes.PolarTransform()

    # Find grid values appropriate for the coordinate (degree).
    # The argument is an approximate number of grids.
    Nticks = min(360./(theta[1]-theta[0]),10)
    grid_locator1 = angle_helper.LocatorD(Nticks)

    # And also use an appropriate formatter:
    tick_formatter1 = angle_helper.FormatterDMS()

    # set up number of ticks for the r-axis
    grid_locator2 = MaxNLocator(4)

    # the extremes are passed to the function
    grid_helper = floating_axes.GridHelperCurveLinear(tr,
                                extremes=(theta[0], theta[1], radius[0], radius[1]),
                                grid_locator1=grid_locator1,
                                grid_locator2=grid_locator2,
                                tick_formatter1=tick_formatter1,
                                tick_formatter2=None,
                                )

    ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    fig.add_subplot(ax1)

    # adjust axis
    # the axis artist lets you call axis with
    # "bottom", "top", "left", "right"
    ax1.axis["left"].set_axis_direction("bottom")
    ax1.axis["right"].set_axis_direction("top")

    ax1.axis["bottom"].toggle(ticklabels=False,ticks=False)	
    ax1.axis["top"].set_axis_direction("bottom")
    ax1.axis["top"].toggle(ticklabels=True, label=True)
    ax1.axis["top"].major_ticklabels.set_axis_direction("top")
    ax1.axis["top"].label.set_axis_direction("top")
    ax1.axis["top"].label.set_text(ur"RA [\u00b0]")
    ax1.axis["left"].label.set_text(r"$\chi$ [Mpc/h]")

    # create a parasite axes
    aux_ax = ax1.get_aux_axes(tr)

    aux_ax.patch = ax1.patch # for aux_ax to have a clip path as in ax
    ax1.patch.zorder=0.9 # but this has a side effect that the patch is
                         # drawn twice, and possibly over some other
                         # artists. So, we decrease the zorder a bit to
                         # prevent this.

    return ax1, aux_ax
		
def skyplot(R, RA, Dec, sample='CMASS', vType = 'Isolated',use_baryC=False, showBuff=False):

	galaxy_sky_file = os.getenv('HOME')+'/Workspace/Surveys/SDSS_DR11/'+sample+'/galaxy_DR11v1cut_'+sample+'_North.txt'
	void_dir = os.getenv('HOME')+'/Workspace/structures/SDSS_DR11/'+sample+'_North/'
	posn_file = void_dir+sample+'_North_pos.dat'
	boundary_file = void_dir+vType+'Voids_boundaries.dat'
	parms = imp.load_source("name",void_dir+'sample_info.dat')
	
	#load up the galaxy and void sky positions
	galaxy_sky = np.loadtxt(galaxy_sky_file,skiprows=2)
	if use_baryC:
		void_sky = np.loadtxt(void_dir+'barycentres/'+vType+'_baryC_Voids_skypos.txt',skiprows=1)
	else:
		void_sky = np.loadtxt(void_dir+vType+'Voids_skypos.txt',skiprows=1)
		void_info = np.loadtxt(void_dir+vType+'Voids_info.txt',skiprows=2)
	
	#put void centres into observer coordinates
	void_info[:,1:4] = void_info[:,1:4]-parms.boxLen/2.0

	if showBuff:	#load up the positions of the buffer mocks to display
		with open(posn_file,'r') as File:
			Npart = np.fromfile(File,dtype=np.int32,count=1)
			Buffers = np.empty([Npart,7])
			Buffers[:,0] = np.fromfile(File, dtype=np.float64,count=Npart)
			Buffers[:,1] = np.fromfile(File, dtype=np.float64,count=Npart)
			Buffers[:,2] = np.fromfile(File, dtype=np.float64,count=Npart)
			Buffers[:,3] = np.fromfile(File, dtype=np.float64,count=Npart)
			Buffers[:,4] = np.fromfile(File, dtype=np.float64,count=Npart)
			Buffers[:,5] = np.fromfile(File, dtype=np.float64,count=Npart)
			Buffers[:,6] = np.fromfile(File, dtype=np.float64,count=Npart)
		Buffers = Buffers[len(galaxy_sky):]
		Buffers[:,:3] = Buffers[:,:3] - parms.boxLen/2.0
		Buffers_r = np.linalg.norm(Buffers[:,:3],axis=1)
	#select galaxies within the slice
	zrange = np.asarray([brentq(lambda x: comovr(x,0.308) - rr, 0.0, 1.0) for rr in R])
	galaxy_sky = galaxy_sky[(galaxy_sky[:,2]>zrange[0]) & (galaxy_sky[:,2]<zrange[1])]
	if showBuff: Buffers = Buffers[(Buffers_r>R[0]) & (Buffers_r<R[1]) & (Buffers[:,3]>0)]
	
	plt.figure(figsize=(10,8))
	m = Basemap(projection="lcc",lon_0=np.mean(RA),lat_0=np.mean(Dec),llcrnrlon=RA[0],\
		llcrnrlat=Dec[0],urcrnrlon=RA[1],urcrnrlat=Dec[1])
	m.drawparallels(np.arange(-90,90,5),linewidth=0.4); 
	m.drawmeridians(np.arange(0,360,5),linewidth=0.4)
	x, y = m(galaxy_sky[:,0],galaxy_sky[:,1])
	m.scatter(x,y,edgecolor='none',color='b',s=1.5)
	if showBuff:
		x, y = m(Buffers[:,3],Buffers[:,4])
		m.scatter(x,y,edgecolor='none',color='r',s=3)

	ax = plt.gca()
	ax.set_yticklabels(ax.get_yticks(), {'family':'serif'})
#	plt.scatter(galaxy_sky[:,0],galaxy_sky[:,1],color='b',s=1)
#	plt.xlim(RA); plt.ylim(Dec)
#	if showBuff:
#		plt.scatter(Buffers[:,3],Buffers[:,4],color='r',s=2)		

	#load up the boundary data
	junk, void_boundaries = read_boundaries(boundary_file)
	void_boundaries = np.asarray(void_boundaries)
		
	R_centre = np.mean(R)	#central radial coordinate
	#select sample of voids near this slice
	centre_dist = np.linalg.norm(void_info[:,1:4],axis=1)
	likely_void_criterion = (centre_dist>R_centre-void_info[:,6]) & \
				(centre_dist<R_centre+void_info[:,6]) & (void_info[:,5]<1) & \
				(RA[0]<void_sky[:,1]) & (void_sky[:,1]<RA[1]) & \
				(Dec[0]<void_sky[:,2]) & (void_sky[:,2]<Dec[1])
	likely_voids = void_info[likely_void_criterion]
	void_boundaries = void_boundaries[likely_void_criterion]
	
#	plt.scatter(void_sky[likely_void_criterion,1],void_sky[likely_void_criterion,2],color='b',s=10)
	

	print "%d likely voids" %len(likely_voids)
	print likely_voids[:,0]
	for i in range(len(likely_voids)):
		off_filename = os.getenv('HOME')+'/Workspace/structures/DR11_void_surfaces/surfaces/' +\
					sample+'_North_IsolVoids_%d.off' %likely_voids[i,0]
		xyz_filename = os.getenv('HOME')+'/Workspace/structures/DR11_void_surfaces/pointdata/' +\
					sample+'_North_IsolVoids_%d.xyz' %likely_voids[i,0]
					
		if not os.access(off_filename,os.F_OK):
			#the OFF file doesn't yet exist, so create it
		
			#step 1: save the boundary data in the xyz file
			np.savetxt(xyz_filename,void_boundaries[i]-parms.boxLen/2.0,fmt='%0.6f')
			
			#step 2: call clean_points to estimate normals, orient normals and clean
			logFile = "./cleanpoints.out"
			log = open(logFile,'w')
			cmd = ["/Users/seshadri/CGAL-4.7/examples/DR11_void_surfaces/clean_points",xyz_filename]
			subprocess.call(cmd,stdout=log,stderr=log)
			log.close()		
			
			#step 3: call reconstruct_surface to get the boundary surface from points and normals
			logFile = "./reconstruct_surface.out"
			log = open(logFile,'w')
			cmd = ["/Users/seshadri/CGAL-4.7/examples/DR11_void_surfaces/reconstruct_surface", \
				xyz_filename,off_filename]
			subprocess.call(cmd,stdout=log,stderr=log)
			log.close()
			
#			#alt step 2: call void_alpha_shapes_3 to get the 3D alpha shape of this point set
#			logFile = "./void_alpha_shapes_3.out"
#			log = open(logFile,'w')
#			cmd = ["/Users/seshadri/CGAL-4.7/examples/DR11_void_surfaces/void_alpha_shapes_3",xyz_filename,off_filename]
#			subprocess.call(cmd,stdout=log,stderr=log)
#			log.close()		

		
		#load data from the .off file
		verts, faces = read_off(off_filename)
		
		#process data to find required intersection 
		pts = np.fromfunction(lambda i,j: verts[faces[i,j],:],faces.shape,dtype=int)
		line_ends_3d = np.zeros((faces.shape[0],2,3))
		line_mask = np.zeros(pts.shape[0])
		for i_face in range(pts.shape[0]):
			n_pts = 0
			p1,p2 = sphere_line_intersection(pts[i_face,0],pts[i_face,1],R_centre)
			if np.any(p1) and np.any(p2): 
				print 'both!'
				line_ends_3d[i_face][n_pts] = 0.5*(p1+p2)
				n_pts+=1
			elif np.any(p1):
				line_ends_3d[i_face][n_pts] = p1
				n_pts+=1
			elif np.any(p2):
				line_ends_3d[i_face][n_pts] = p2
				n_pts+=1
			p1,p2 = sphere_line_intersection(pts[i_face,1],pts[i_face,2],R_centre)
			if np.any(p1) and np.any(p2): 
				print 'both!'
				line_ends_3d[i_face][n_pts] = 0.5*(p1+p2)
				n_pts+=1
			elif np.any(p1):
				line_ends_3d[i_face][n_pts] = p1
				n_pts+=1
			elif np.any(p2):
				line_ends_3d[i_face][n_pts] = p2
				n_pts+=1
			p1,p2 = sphere_line_intersection(pts[i_face,2],pts[i_face,0],R_centre)
			if np.any(p1) and np.any(p2): 
				print 'both!'
				line_ends_3d[i_face][n_pts] = 0.5*(p1+p2)
				n_pts+=1
			elif np.any(p1):
				line_ends_3d[i_face][n_pts] = p1
				n_pts+=1
			elif np.any(p2):
				line_ends_3d[i_face][n_pts] = p2
				n_pts+=1
			if n_pts>0: line_mask[i_face]=1
		
		if np.any(line_mask):		#some voids may not intersect the surface at all
			line_ends_3d = line_ends_3d[line_mask>0]
			line_ends_ang = np.zeros((line_ends_3d.shape[0],2,2))
			line_ends_ang[:,:,1] = 90 - np.degrees(np.arccos(line_ends_3d[:,:,2]/R_centre))
			line_ends_ang[:,:,0] = np.degrees(np.arctan2(line_ends_3d[:,:,1],line_ends_3d[:,:,0]))
			line_ends_ang[line_ends_ang[:,:,0]<0,0] += 360
			
			#plot
#			lines = LineCollection(line_ends_ang,colors='k',linestyle='dashed',linewidth=1.5)
#			plt.gca().add_collection(lines)
			for i in range(len(line_ends_ang)):
				m.plot(line_ends_ang[i,:,0],line_ends_ang[i,:,1],latlon=True,color='k',linestyle='--')
			
	plt.savefig(figdir+'CMASS_North_sky_Isol.pdf',bbox_inches='tight')

def DecSlice_w_boundaries(R=[350,700],RA=[120,180],Dec=[12,14],vType='Isolated',sample='LOWZ',showBuff=True):
	
	galaxy_sky_file = os.getenv('HOME')+'/Workspace/Surveys/SDSS_DR11/'+sample+'/galaxy_DR11v1cut_'+sample+'_North.txt'
	void_dir = os.getenv('HOME')+'/Workspace/structures/SDSS_DR11/'+sample+'_North/'
	posn_file = void_dir+sample+'_North_pos.dat'
	parms = imp.load_source("name",void_dir+'sample_info.dat')
	
	#load up the galaxy sky positions
	galaxy_sky = np.loadtxt(galaxy_sky_file,skiprows=2)
	if showBuff:	#load up the positions of the buffer mocks to display
		with open(posn_file,'r') as File:
			Npart = np.fromfile(File,dtype=np.int32,count=1)
			Buffers = np.empty([Npart,7])
			Buffers[:,0] = np.fromfile(File, dtype=np.float64,count=Npart)
			Buffers[:,1] = np.fromfile(File, dtype=np.float64,count=Npart)
			Buffers[:,2] = np.fromfile(File, dtype=np.float64,count=Npart)
			Buffers[:,3] = np.fromfile(File, dtype=np.float64,count=Npart)
			Buffers[:,4] = np.fromfile(File, dtype=np.float64,count=Npart)
			Buffers[:,5] = np.fromfile(File, dtype=np.float64,count=Npart)
			Buffers[:,6] = np.fromfile(File, dtype=np.float64,count=Npart)
		Buffers = Buffers[len(galaxy_sky):]
		Buffers[:,:3] = Buffers[:,:3] - parms.boxLen/2.0
		Buffers_r = np.linalg.norm(Buffers[:,:3],axis=1)
	#select galaxies within the slice
	zrange = np.asarray([brentq(lambda x: comovr(x,0.308) - rr, 0.0, 1.0) for rr in R])
	galaxy_sky = galaxy_sky[(galaxy_sky[:,2]>zrange[0]) & (galaxy_sky[:,2]<zrange[1]) &\
				(galaxy_sky[:,0]>RA[0]) & (galaxy_sky[:,0]<RA[1]) \
				& (galaxy_sky[:,1]>Dec[0]) & (galaxy_sky[:,1]<Dec[1])]
	#get their r distances
	zrange = np.linspace(0.1,0.8,25)
	vfunc = np.vectorize(comovr)
	rvals = vfunc(zrange,0.308)
	rinterp = interp1d(zrange,rvals)
	galaxy_r = rinterp(galaxy_sky[:,2])

	fig = plt.figure(figsize=(10,10))
	ax1, aux_ax1 = setup_axes(fig, 111, theta=RA, radius=R)

	#plot the galaxies
	aux_ax1.scatter(galaxy_sky[:,0],galaxy_r,color='b',alpha=0.5,\
		marker='.',edgecolor='none',s=5)
	if showBuff: 	#add the buffer mocks
		Buffer_slice = (Buffers_r>R[0]) & (Buffers_r<R[1]) & (Buffers[:,3]>RA[0]) \
				& (Buffers[:,3]<RA[1]) & (Buffers[:,4]>Dec[0]) & (Buffers[:,4]<Dec[1])
		aux_ax1.scatter(Buffers[Buffer_slice,3],Buffers_r[Buffer_slice],color=kelly_RdYlGn[0],\
			marker='.',edgecolor='none',s=5)

	#load void information
	info = np.loadtxt(void_dir+vType+'Voids_info.txt',skiprows=2)
	sky = np.loadtxt(void_dir+vType+'Voids_skypos.txt',skiprows=1)
	void_r = rinterp(sky[:,3])
	dist_from_slice = np.abs(void_r*(np.deg2rad(sky[:,2])-np.deg2rad(np.mean(Dec))))
	voids_to_plot = (sky[:,1]>RA[0]) & (sky[:,1]<RA[1]) & (dist_from_slice<info[:,6])
	
	#plot the void boundaries
	coneTheta = 90 - np.mean(Dec)
	ids, boundaries = read_boundaries(void_dir+vType+'Voids_boundaries.dat')
	boundaries = np.asarray(boundaries)[voids_to_plot]
	ids = np.asarray(ids)[voids_to_plot]
	info = info[voids_to_plot]
	no_intersect=0
	print 'Plotting up to %d void boundaries' %len(ids)
	for i in range(boundaries.shape[0]):
		off_filename = os.getenv('HOME')+'/Workspace/structures/DR11_void_surfaces/surfaces/' +\
					sample+'_North_'+vType+'Voids_%d.off' %ids[i]
		xyz_filename = os.getenv('HOME')+'/Workspace/structures/DR11_void_surfaces/pointdata/' +\
					sample+'_North_'+vType+'Voids_%d.xyz' %ids[i]
					
		if not os.access(off_filename,os.F_OK):
			#the OFF file doesn't yet exist, so create it
		
			#step 1: save the boundary data in the xyz file
			np.savetxt(xyz_filename,boundaries[i],fmt='%0.6f')
			
			#step 2: call reconstruct_surface to get the boundary surface from points and normals
			logFile = "./reconstruct_surface.out"
			log = open(logFile,'w')
			cmd = ["/Users/seshadri/CGAL-4.7/examples/DR11_void_surfaces/reconstruct_surface", \
				xyz_filename,off_filename]
			subprocess.call(cmd,stdout=log,stderr=log)
			log.close()
				
		#load data from the .off file
		verts, faces = read_off(off_filename)
		verts -= parms.boxLen/2.0
		
		#process data to find required intersection 
		pts = np.fromfunction(lambda i,j: verts[faces[i,j],:],faces.shape,dtype=int)
		line_ends_3d = np.zeros((faces.shape[0],2,3))
		line_mask = np.zeros(pts.shape[0])
		for i_face in range(pts.shape[0]):
			n_pts = 0
			p1,p2 = cone_line_intersection(pts[i_face,0],pts[i_face,1],coneTheta)
			if np.any(p1) and np.any(p2): 
				print 'both!'
				line_ends_3d[i_face][n_pts] = 0.5*(p1+p2)
				n_pts+=1
			elif np.any(p1):
				line_ends_3d[i_face][n_pts] = p1
				n_pts+=1
			elif np.any(p2):
				line_ends_3d[i_face][n_pts] = p2
				n_pts+=1
			p1,p2 = cone_line_intersection(pts[i_face,1],pts[i_face,2],coneTheta)
			if np.any(p1) and np.any(p2): 
				print 'both!'
				line_ends_3d[i_face][n_pts] = 0.5*(p1+p2)
				n_pts+=1
			elif np.any(p1):
				line_ends_3d[i_face][n_pts] = p1
				n_pts+=1
			elif np.any(p2):
				line_ends_3d[i_face][n_pts] = p2
				n_pts+=1
			p1,p2 = cone_line_intersection(pts[i_face,2],pts[i_face,0],coneTheta)
			if np.any(p1) and np.any(p2): 
				print 'both!'
				line_ends_3d[i_face][n_pts] = 0.5*(p1+p2)
				n_pts+=1
			elif np.any(p1):
				line_ends_3d[i_face][n_pts] = p1
				n_pts+=1
			elif np.any(p2):
				line_ends_3d[i_face][n_pts] = p2
				n_pts+=1
			if n_pts>0: line_mask[i_face]=1
		
		if np.any(line_mask):		#some voids may not intersect the surface at all
			line_ends_3d = line_ends_3d[line_mask>0]

			pts = np.append(line_ends_3d[:,0,:],line_ends_3d[:,1,:],axis=0)
			pts_slice = np.zeros((pts.shape[0],2))
			pts_slice[:,1] = np.linalg.norm(pts,axis=1)
			pts_slice[:,0] = np.degrees(np.arctan2(pts[:,1],pts[:,0]))
			pts_slice[pts_slice[:,0]<0,0] += 360

			line_ends_slice = np.zeros((line_ends_3d.shape[0],2,2))
			line_ends_slice[:,:,1] = np.linalg.norm(line_ends_3d,axis=2)
			line_ends_slice[:,:,0] = np.degrees(np.arctan2(line_ends_3d[:,:,1],line_ends_3d[:,:,0]))
			line_ends_slice[line_ends_slice[:,:,0]<0,0] += 360
			
			#plot
			if info[i,5]<1:
				lines = LineCollection(line_ends_slice,colors=kelly_RdYlGn[7],linestyle='dashed',linewidth=0.8)
			else:
				lines = LineCollection(line_ends_slice,colors=kelly_colours[0],linestyle='dashed',linewidth=0.8)
			aux_ax1.add_collection(lines)
		else: no_intersect+=1

	aux_ax1.scatter(galaxy_sky[:,0],galaxy_r,color='b',alpha=0.5,\
		marker='.',edgecolor='none',s=10)
		
#	print "%d voids did not intersect the cone" %no_intersect
	plt.savefig(figdir+'DR11/figure.pdf',bbox_inches='tight')
			
		
def DecSlicePlot(R, RA, Dec, sample='CMASS', use_baryC=False, showBuff = False):
	
	galaxy_sky_file = os.getenv('HOME')+'/Workspace/Surveys/SDSS_DR11/'+sample+'/galaxy_DR11v1cut_'+sample+'_North.txt'
	void_dir = os.getenv('HOME')+'/Workspace/structures/SDSS_DR11/'+sample+'_North/'
	posn_file = void_dir+sample+'_North_pos.dat'
	parms = imp.load_source("name",void_dir+'sample_info.dat')
	
	#load up the galaxy and void sky positions
	galaxy_sky = np.loadtxt(galaxy_sky_file,skiprows=2)
	if use_baryC:
		IV_sky = np.loadtxt(void_dir+'barycentres/Isolated_baryC_Voids_skypos.txt',skiprows=1)
		MV_sky = np.loadtxt(void_dir+'barycentres/Minimal_baryC_Voids_skypos.txt',skiprows=1)
		IV_info = np.loadtxt(void_dir+'barycentres/Isolated_baryC_Voids_info.txt',skiprows=2)
		MV_info = np.loadtxt(void_dir+'barycentres/Minimal_baryC_Voids_info.txt',skiprows=2)
	else:
		IV_sky = np.loadtxt(void_dir+'IsolatedVoids_skypos.txt',skiprows=1)
		MV_sky = np.loadtxt(void_dir+'MinimalVoids_skypos.txt',skiprows=1)
		IV_info = np.loadtxt(void_dir+'IsolatedVoids_info.txt',skiprows=2)
		MV_info = np.loadtxt(void_dir+'MinimalVoids_info.txt',skiprows=2)

	IV_info[:,1:4] = IV_info[:,1:4] - parms.boxLen/2.0
	MV_info[:,1:4] = MV_info[:,1:4] - parms.boxLen/2.0
	IV_r = np.linalg.norm(IV_info[:,1:4],axis=1)
	MV_r = np.linalg.norm(MV_info[:,1:4],axis=1)

	if showBuff:	#load up the positions of the buffer mocks to display
		with open(posn_file,'r') as File:
			Npart = np.fromfile(File,dtype=np.int32,count=1)
			Buffers = np.empty([Npart,7])
			Buffers[:,0] = np.fromfile(File, dtype=np.float64,count=Npart)
			Buffers[:,1] = np.fromfile(File, dtype=np.float64,count=Npart)
			Buffers[:,2] = np.fromfile(File, dtype=np.float64,count=Npart)
			Buffers[:,3] = np.fromfile(File, dtype=np.float64,count=Npart)
			Buffers[:,4] = np.fromfile(File, dtype=np.float64,count=Npart)
			Buffers[:,5] = np.fromfile(File, dtype=np.float64,count=Npart)
			Buffers[:,6] = np.fromfile(File, dtype=np.float64,count=Npart)
		Buffers = Buffers[len(galaxy_sky):]
		Buffers[:,:3] = Buffers[:,:3] - parms.boxLen/2.0
		Buffers_r = np.linalg.norm(Buffers[:,:3],axis=1)

	#select galaxies within the slice
	zrange = np.asarray([brentq(lambda x: comovr(x,0.308) - rr, 0.0, 1.0) for rr in R])
	galaxy_sky = galaxy_sky[(galaxy_sky[:,2]>zrange[0]) & (galaxy_sky[:,2]<zrange[1]) &\
				(galaxy_sky[:,0]>RA[0]) & (galaxy_sky[:,0]<RA[1]) \
				& (galaxy_sky[:,1]>Dec[0]) & (galaxy_sky[:,1]<Dec[1])]
	#get their r distances
	zrange = np.linspace(0.1,0.8,25)
	vfunc = np.vectorize(comovr)
	rvals = vfunc(zrange,0.308)
	rinterp = interp1d(zrange,rvals)
	galaxy_r = rinterp(galaxy_sky[:,2])
		
	#slices to retain for plotting
	IV_slice = (IV_sky[:,1]>RA[0]) & (IV_sky[:,1]<RA[1]) \
				& (IV_sky[:,2]>Dec[0]) & (IV_sky[:,2]<Dec[1]) \
				& (IV_r>R[0]) & (IV_r<R[1]) #& (IV_info[:,5]<1)
	MV_slice = (MV_sky[:,1]>RA[0]) & (MV_sky[:,1]<RA[1]) \
				& (MV_sky[:,2]>Dec[0]) & (MV_sky[:,2]<Dec[1]) \
				& (MV_r>R[0]) & (MV_r<R[1]) 
				
	fig = plt.figure(figsize=(10,10))
	ax1, aux_ax1 = setup_axes(fig, 111, theta=RA, radius=R)
	aux_ax1.tick_params(labelsize=32)

	#plot the galaxies
	aux_ax1.scatter(galaxy_sky[:,0],galaxy_r,color='b',alpha=0.5,\
		marker='.',edgecolor='none',s=10)

	#select the buffer mocks to show
	if showBuff:
		Buffer_slice = (Buffers_r>R[0]) & (Buffers_r<R[1]) & (Buffers[:,3]>RA[0]) \
				& (Buffers[:,3]<RA[1]) & (Buffers[:,4]>Dec[0]) & (Buffers[:,4]<Dec[1])
		aux_ax1.scatter(Buffers[Buffer_slice,3],Buffers_r[Buffer_slice],color=kelly_RdYlGn[0],\
			marker='.',edgecolor='none',s=10)

	if use_baryC:
		aux_ax1.scatter(IV_sky[(IV_slice) & (IV_info[:,5]<1.05),1],IV_r[(IV_slice) & \
			(IV_info[:,5]<1.05)][:],color=kelly_colours[4],marker='s',s=10)
		aux_ax1.scatter(IV_sky[(IV_slice) & (IV_info[:,5]>1.05),1],IV_r[(IV_slice) & \
			(IV_info[:,5]>1.05)][:],color=kelly_colours[0],marker='s',s=10)
	else:
		aux_ax1.scatter(IV_sky[(IV_slice) & (IV_info[:,5]<1.05),1],IV_r[(IV_slice) & \
			(IV_info[:,5]<1.05)][:],color=kelly_colours[4],marker='o',s=10)
		aux_ax1.scatter(IV_sky[(IV_slice) & (IV_info[:,5]>1.05),1],IV_r[(IV_slice) & \
			(IV_info[:,5]>1.05)][:],color=kelly_colours[0],marker='o',s=10)

#	if use_baryC:
#		aux_ax1.scatter(MV_sky[(MV_slice) & (MV_info[:,5]<1.1),1],MV_r[(MV_slice) & \
#			(MV_info[:,5]<1.1)][:],color=kelly_colours[4],marker='s',s=10)
#		aux_ax1.scatter(MV_sky[(MV_slice) & (MV_info[:,5]>1.1),1],MV_r[(MV_slice) & \
#			(MV_info[:,5]>1.1)][:],color=kelly_colours[0],marker='s',s=10)
#	else:
#		aux_ax1.scatter(MV_sky[(MV_slice) & (MV_info[:,5]<1.1),1],MV_r[(MV_slice) & \
#			(MV_info[:,5]<1.1)][:],color=kelly_colours[4],marker='o',s=10)
#		aux_ax1.scatter(MV_sky[(MV_slice) & (MV_info[:,5]>1.1),1],MV_r[(MV_slice) & \
#			(MV_info[:,5]>1.1)][:],color=kelly_colours[0],marker='o',s=10)

	plt.savefig(figdir+'DR11/figure.pdf',bbox_inches='tight')

			
def read_off(filename):
	
	with open(filename,'r') as file:
		if 'OFF' != file.readline().strip():
			raise Exception('Not a valid OFF header')
		n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
		line = file.readline()
		while line=='\n' or line[0]=='#': line = file.readline() #skip the empty lines

		n_dim = len(line.strip().split(' '))	#find the dimensionality of the points
		verts = np.empty((n_verts,n_dim))

		for i_vert in range(n_verts):
			verts[i_vert] = np.asarray([float(s) for s in line.strip().split(' ')])
			line = file.readline()

		faces = []
		line = line.strip().split(' ')
		for i_face in range(n_faces):
			numV = int(line[0])
			line_length= len(line)
			faces.append([int(s) for s in line[line_length-numV:]])
			line = file.readline().strip().split(' ')
	return verts, np.asarray(faces)

def read_mesh(filename):

	with open(filename,'r') as file:
		if 'ply' not in file.readline().strip():
			raise Exception('Not a valid PLY header')
		for i in range(9):
			file.readline()
		n_verts = int(file.readline().strip().split(' ')[3])
		n_dim = 3	# only interested in 3D case		
		verts = np.empty((n_verts,n_dim))

		for i in range(3):
			file.readline()
		n_faces = int(file.readline().strip().split(' ')[3])
		
		# skip remaining empty/comment lines
		line = file.readline()
		while line=='\n' or line[0]=='#': line = file.readline()

		for i_vert in range(n_verts):
			verts[i_vert] = np.asarray([float(s) for s in line.strip().split(' ')[3:]])
			line = file.readline()

		faces = []
		line = line.strip().split(' ')
		for i_face in range(n_faces):
			faces.append([int(s)-1 for s in line[3:]])
			if i_face<10:
				print line[3:]
			line = file.readline().strip().split(' ')
	return verts, np.asarray(faces)
	
def cone_line_intersection(l1,l2,coneTheta,coneD=np.array([0,0,1]),coneV=np.array([0,0,0])):
	
	p1 = p2 = None
	
	#convert all input arrays into numpy matrices
	l1 = np.matrix(l1).transpose()
	l2 = np.matrix(l2).transpose()
	coneD = np.matrix(coneD).transpose()
	coneV = np.matrix(coneV).transpose()

	#determine the number of dimensions
	ndim = l1.shape[0]
	
	if np.linalg.norm(coneD)==0:
		raise Exception('what the hell are you playing at?')
	coneD = coneD/np.linalg.norm(coneD)	#to ensure it is normalised
		
	cos_sq = np.cos(coneTheta*np.pi/180.)**2
	U = l2-l1
	M = coneD*np.transpose(coneD) - cos_sq*np.identity(ndim)
	Delta = l1-coneV
	
	c2 = np.transpose(U)*M*U
	c1 = np.transpose(U)*M*Delta
	c0 = np.transpose(Delta)*M*Delta
	
	if c2==0:
		if c1==0:
			#either no intersection, or the line segment is ON the cone (infinite solutions)
			pass
		else:
			#single point of intersection for the line
			t = np.asscalar(-c0/c1)
			if 0<=t<=1: 
				#intersection is within the line segment
				test = np.dot(coneD.transpose(),Delta) + t * np.dot(coneD.transpose(),U) 
				if test>=0:
					#intersection in the +ve half cone					
					p1 = l1 + t * (l2-l1)
	else:
		delta = c1**2 - c0*c2
		if delta<0:
			#no solutions
			pass
		elif delta==0:
			#repeated root, line is tangent to the cone at one point
			t = np.asscalar(-c1/c2)
			if 0<=t<=1: 
				#tangent point is within the line segment
				test = np.dot(coneD.transpose(),Delta) + t * np.dot(coneD.transpose(),U) 
				if test>=0:
					#intersection in the +ve half cone					
					p1 = l1 + t * (l2-l1)
		elif delta>0:
			#two intersections
			t = np.asscalar((-c1 + np.sqrt(delta))/c2)
			if 0<=t<=1: 
				#tangent point is within the line segment
				test = np.dot(coneD.transpose(),Delta) + t * np.dot(coneD.transpose(),U) 
				if test>=0:
					#intersection in the +ve half cone					
					p1 = l1 + t * (l2-l1)
			t = np.asscalar((-c1 - np.sqrt(delta))/c2)
			if 0<=t<=1: 
				#tangent point is within the line segment
				test = np.dot(coneD.transpose(),Delta) + t * np.dot(coneD.transpose(),U) 
				if test>=0:
					#intersection in the +ve half cone					
					p2 = l1 + t * (l2-l1)
	#convert back to arrays
	if np.any(p1): p1 = np.asarray(p1.transpose())[0]
	if np.any(p2): p2 = np.asarray(p2.transpose())[0]
		
	return p1,p2
	
def sphere_line_intersection(l1, l2, r, sp=np.array([0,0,0])):

	# l1[0],l1[1],l1[2]  P1 coordinates (point of line)
	# l2[0],l2[1],l2[2]  P2 coordinates (point of line)
	# sp[0],sp[1],sp[2], r  P3 coordinates and radius (sphere)
	# x,y,z   intersection coordinates
	#
	# This function returns a pointer array which first index indicates
	# the number of intersection point, followed by coordinate pairs.

	p1 = p2 = None

	a = np.dot(l2-l1,l2-l1)
	b = 2.0*np.dot(l2-l1,l1-sp)
	c = np.dot(sp,sp) + np.dot(l1,l1) - 2.0*np.dot(sp,l1) - r*r

	i = b * b - 4.0 * a * c

	if i < 0.0:
		pass  # no intersections
	elif i == 0.0:
		# one intersection
		mu = -b / (2.0 * a)
		if 0<=mu<=1: p1 = l1 + mu * (l2-l1)

	elif i > 0.0:
		# first intersection
		mu = (-b + np.sqrt(i)) / (2.0 * a)
		if 0<=mu<=1: p1 = l1 + mu * (l2-l1)

		# second intersection
		mu = (-b - np.sqrt(i)) / (2.0 * a)
		if 0<=mu<=1: p2 = l1 + mu * (l2-l1)

	return p1, p2	
	
def survey_footprint(survey='CMASS'):

	class ThetaFormatterShiftPi(GeoAxes.ThetaFormatter):
	    """Shifts labelling by pi
	    Shifts labelling from -180,180 to 0-360"""
	    def __call__(self, x, pos=None):
	        if x != 0:
	            x *= -1
	        if x < 0:
	            x += 2*np.pi
	        return GeoAxes.ThetaFormatter.__call__(self, x, pos)

	fig = plt.figure(figsize=(12., 12./(3./2.)))
	# matplotlib is doing the mollweide projection
	ax = fig.add_subplot(111,projection='mollweide')
	
	cmap = plt.cm.RdYlBu_r
	vmin = 0.5; vmax = 1

	xsize = 2000
	ysize = xsize/2.
	theta = np.linspace(np.pi, 0, ysize)
	phi   = np.linspace(-np.pi, np.pi, xsize)
	longitude = np.radians(np.linspace(-180, 180, xsize))
	latitude = np.radians(np.linspace(-90, 90, ysize))
	
	survey_mask = hp.read_map('/Users/seshadri/Workspace/Surveys/SDSS_DR11/'+survey+\
		'/mask_DR11v1_'+survey+'_North_n128.fits',verbose=False)
	nside = hp.npix2nside(len(survey_mask))	
	#rotate the survey mask through 180 degrees in longitude
	mask_theta, mask_phi = hp.pix2ang(nside,np.nonzero(survey_mask))
	filled_vals = survey_mask[np.nonzero(survey_mask)]
	r = hp.rotator.Rotator(rot=(180,0,0))
	mask_theta, mask_phi = r(mask_theta,mask_phi)
	rotated_survey_mask = np.zeros_like(survey_mask)
	rotated_survey_mask[hp.ang2pix(nside,mask_theta,mask_phi)]=filled_vals
	m = hp.ma(rotated_survey_mask)
	m.mask = np.logical_not(rotated_survey_mask)
		    	
	# project the map to a rectangular matrix xsize x ysize
	PHI, THETA = np.meshgrid(phi, theta)
	grid_pix = hp.ang2pix(nside, THETA, PHI)
	grid_mask = m.mask[grid_pix]
	grid_map = np.ma.MaskedArray(m[grid_pix],grid_mask)
	
	# rasterized makes the map bitmap while the labels remain vectorial
	# flip longitude to the astro convention
	image = plt.pcolormesh(longitude[::-1], latitude, grid_map, vmin=vmin, vmax=vmax, rasterized=True, cmap=cmap)

#	boundary = find_boundary(survey_mask,0)
#	nside = hp.get_nside(boundary)	
#	#rotate through 180 degrees in longitude
#	b_theta, b_phi = hp.pix2ang(nside,np.nonzero(boundary))
#	filled_vals = boundary[np.nonzero(boundary)]
#	r = hp.rotator.Rotator(rot=(180,0,0))
#	b_theta, b_phi = r(b_theta,b_phi)
#	rotated_boundary = np.zeros_like(boundary)
#	rotated_boundary[hp.ang2pix(nside,b_theta,b_phi)]=filled_vals
#	m = hp.ma(rotated_boundary)
#	m.mask = np.logical_not(rotated_boundary)
#		    	
#	# project the map to a rectangular matrix xsize x ysize
#	PHI, THETA = np.meshgrid(phi, theta)
#	grid_pix = hp.ang2pix(nside, THETA, PHI)
#	grid_mask = m.mask[grid_pix]
#	grid_map = np.ma.MaskedArray(m[grid_pix],grid_mask)
#	image = plt.pcolormesh(longitude[::-1], latitude, grid_map, vmin=vmin, vmax=vmax, rasterized=True, cmap='afmhot_r')
	
	survey_mask = hp.read_map('/Users/seshadri/Workspace/Surveys/SDSS_DR11/'+survey+\
		'/mask_DR11v1_'+survey+'_South_n128.fits',verbose=False)
	nside = hp.npix2nside(len(survey_mask))	
	#rotate the survey mask through 180 degrees in longitude
	mask_theta, mask_phi = hp.pix2ang(nside,np.nonzero(survey_mask))
	filled_vals = survey_mask[np.nonzero(survey_mask)]
	r = hp.rotator.Rotator(rot=(180,0,0))
	mask_theta, mask_phi = r(mask_theta,mask_phi)
	rotated_survey_mask = np.zeros_like(survey_mask)
	rotated_survey_mask[hp.ang2pix(nside,mask_theta,mask_phi)]=filled_vals
	m = hp.ma(rotated_survey_mask)
	m.mask = np.logical_not(rotated_survey_mask)
		    	
	# project the map to a rectangular matrix xsize x ysize
	PHI, THETA = np.meshgrid(phi, theta)
	grid_pix = hp.ang2pix(nside, THETA, PHI)
	grid_mask = m.mask[grid_pix]
	grid_map = np.ma.MaskedArray(m[grid_pix],grid_mask)
	image = plt.pcolormesh(longitude[::-1], latitude, grid_map, vmin=vmin, vmax=vmax, rasterized=True, cmap=cmap)
	
#	# graticule
#	ax.set_longitude_grid(30)
#	ax.xaxis.set_major_formatter(ThetaFormatterShiftPi(30))
	
	# colorbar
	cb = fig.colorbar(image, orientation='horizontal', shrink=.6, pad=0.05, ticks=[vmin, vmax])
	cb.ax.xaxis.set_label_text('Completeness',fontsize=24)
	cb.ax.xaxis.labelpad = -8
	# workaround for issue with viewers, see colorbar docstring
	cb.solids.set_edgecolor("face")
	cb.ax.tick_params(labelsize=22)
	
	
	ax.tick_params(axis='x', labelsize=14)
	ax.tick_params(axis='y', labelsize=14)
	
	# remove tick labels
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	# remove grid
	ax.xaxis.set_ticks(np.radians(np.array([-180,-150,-120,-90,-60,-30,0,30,60,90,120,150,180])))
	ax.yaxis.set_ticks(np.radians(np.array([-75,-60,-45,-30,-15,0,15,30,45,60,75])))
	
	# remove white space around figure
	spacing = 0.01
	plt.subplots_adjust(bottom=spacing, top=1-spacing, left=spacing, right=1-spacing)
	
	plt.grid(True)
	
	plt.savefig(figdir+'DR11/'+survey+'_surveymask.pdf',bbox_inches='tight')

def selection_fns(sample='CMASS'):

	north = np.loadtxt(structdir+'SDSS_DR11/'+sample+'_North/'+sample+'_North_selfn.dat',skiprows=1)
	south = np.loadtxt(structdir+'SDSS_DR11/'+sample+'_South/'+sample+'_South_selfn.dat',skiprows=1)
	
	n_selfn = InterpolatedUnivariateSpline(north[:,0],north[:,2],k=1)
	s_selfn = InterpolatedUnivariateSpline(south[:,0],south[:,2],k=1)
	if sample=='CMASS': x = np.linspace(0.435,0.695,1000)
	else: x = np.linspace(0.155,0.425,1000)
	yn = savgol_filter(n_selfn(x),101,3)
	ys = savgol_filter(s_selfn(x),101,3)
	n_selfn = InterpolatedUnivariateSpline(x,yn,k=1)	
	s_selfn = InterpolatedUnivariateSpline(x,ys,k=1)	
	
	if sample=='CMASS': x = np.linspace(0.435,0.695)
	else: x = np.linspace(0.155,0.425)
	plt.figure(figsize=(12.,8.))
	plt.plot(north[1:-1,0],north[1:-1,2],'b',alpha=0.9,label='North',linewidth=2)
	plt.plot(south[1:-1,0],south[1:-1,2],kelly_colours[4],label='South',linewidth=2)
	plt.xlabel("$z$",fontsize=24,fontweight='extra bold')
	plt.ylabel(r'$\phi(z)$',fontsize=24,fontweight='extra bold')
	plt.xlim([north[0,0],north[-1,0]])
	ax = plt.gca(); prettify_plot(ax)
	plt.legend(loc='upper right',numpoints=1,prop={'size':16})
	
	plt.savefig(figdir+'DR11/'+sample+'_selfns.pdf',bbox_inches='tight')

def calc_ellipticities():
	
	samples = np.asarray(["CMASS_North","CMASS_South","LOWZ_North","LOWZ_South"])
	voidTypes = np.asarray(["Isolated","Minimal"])
	
	for sample in samples:
		for vType in voidTypes:
			print "%s, %s" %(sample,vType)
		
			parmsFile = structdir+'SDSS_DR11/'+sample+"/sample_info.dat"
			boundFile = structdir+'SDSS_DR11/'+sample+"/"+vType+"Voids_boundaries.dat"
			cInfoFile = structdir+'SDSS_DR11/'+sample+"/"+vType+"Voids_info.txt"
			bInfoFile = structdir+'SDSS_DR11/'+sample+"/barycentres/"+vType+"_baryC_Voids_info.txt"
			c_ellipFile = structdir+'SDSS_DR11/'+sample+"/"+vType+"Voids_ellipticities.txt"
			b_ellipFile = structdir+'SDSS_DR11/'+sample+"/barycentres/"+vType+"_baryC_Voids_ellipticities.txt"
			
			structIDs, BoundaryPosns = read_boundaries(boundFile)
			voidInfo_cc = np.loadtxt(cInfoFile,skiprows=2)
			voidInfo_bc = np.loadtxt(bInfoFile,skiprows=2)
			parms = imp.load_source("name",parmsFile)
			
			ellip_cc = np.zeros((structIDs.shape[0],4))
			ellip_bc = np.zeros((structIDs.shape[0],4))
			orient_cc = np.zeros((structIDs.shape[0],2))
			orient_bc = np.zeros((structIDs.shape[0],2))
			for i in range(len(structIDs)):
				boundaries_cc = BoundaryPosns[i] - voidInfo_cc[i,1:4]
				boundaries_bc = BoundaryPosns[i] - voidInfo_bc[i,1:4]
				los_cc = voidInfo_cc[i,1:4] - parms.boxLen/2.0
				los_bc = voidInfo_bc[i,1:4] - parms.boxLen/2.0
				
				M1_cc = np.zeros((3,3)); 
				M1_bc = np.zeros((3,3)); 
				
				M1_cc[0,0] = np.sum(boundaries_cc[:,1]**2 + boundaries_cc[:,2]**2)
				M1_cc[1,1] = np.sum(boundaries_cc[:,0]**2 + boundaries_cc[:,2]**2)
				M1_cc[2,2] = np.sum(boundaries_cc[:,0]**2 + boundaries_cc[:,1]**2)
				M1_cc[0,1] = M1_cc[1,0] = np.sum(boundaries_cc[:,0]*boundaries_cc[:,1])
				M1_cc[0,2] = M1_cc[2,0] = np.sum(boundaries_cc[:,0]*boundaries_cc[:,2])
				M1_cc[2,1] = M1_cc[1,2] = np.sum(boundaries_cc[:,2]*boundaries_cc[:,1])
				
				M1_bc[0,0] = np.sum(boundaries_bc[:,1]**2 + boundaries_bc[:,2]**2)
				M1_bc[1,1] = np.sum(boundaries_bc[:,0]**2 + boundaries_bc[:,2]**2)
				M1_bc[2,2] = np.sum(boundaries_bc[:,0]**2 + boundaries_bc[:,1]**2)
				M1_bc[0,1] = M1_bc[1,0] = np.sum(boundaries_bc[:,0]*boundaries_bc[:,1])
				M1_bc[0,2] = M1_bc[2,0] = np.sum(boundaries_bc[:,0]*boundaries_bc[:,2])
				M1_bc[2,1] = M1_bc[1,2] = np.sum(boundaries_bc[:,2]*boundaries_bc[:,1])
				
				w, v = np.linalg.eigh(M1_cc)
				ellip_cc[i,0] = np.sqrt(w[2]/w[0])
				ellip_cc[i,1] = np.sqrt(w[1]/w[0])
				ellip_cc[i,2] = 0.5*(w[2]-w[0])/np.sum(w)
				ellip_cc[i,3] = 0.5*(w[2]-2*w[1]+w[0])/np.sum(w)
				orient_cc[i,0] = np.rad2deg(np.arccos(np.dot(v[2],los_cc)/np.linalg.norm(los_cc)))
				orient_cc[i,1] = np.rad2deg(np.arccos(np.dot(v[1],los_cc)/np.linalg.norm(los_cc)))
				    
				w, v = np.linalg.eigh(M1_bc)
				ellip_bc[i,0] = np.sqrt(w[2]/w[0])
				ellip_bc[i,1] = np.sqrt(w[1]/w[0])
				ellip_bc[i,2] = 0.5*(w[2]-w[0])/np.sum(w)
				ellip_bc[i,3] = 0.5*(w[2]-2*w[1]+w[0])/np.sum(w)
				orient_bc[i,0] = np.rad2deg(np.arccos(np.dot(v[2],los_bc)/np.linalg.norm(los_bc)))
				orient_bc[i,1] = np.rad2deg(np.arccos(np.dot(v[1],los_bc)/np.linalg.norm(los_bc)))
			
			with open(c_ellipFile,'w') as F:
				F.write("VoidID a1/a3 a1/a2 theta1(deg) theta2(deg) e p\n")
				for i in range(len(structIDs)):
					F.write("%d %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f\n" %(structIDs[i],ellip_cc[i,0],ellip_cc[i,1],\
						orient_cc[i,0],orient_cc[i,1],ellip_cc[i,2],ellip_cc[i,3]))
			with open(b_ellipFile,'w') as F:
				F.write("VoidID a1/a3 a1/a2 theta1(deg) theta2(deg) e p\n")
				for i in range(len(structIDs)):
					F.write("%d %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f\n" %(structIDs[i],ellip_bc[i,0],ellip_bc[i,1],\
						orient_bc[i,0],orient_bc[i,1],ellip_bc[i,2],ellip_bc[i,3]))
	
def calc_ellip_BigMDPl():
	
	samples = np.asarray(["CMASS","LOWZ"])
	voidTypes = np.asarray(["Isolated","Minimal"])
	
	for sample in samples:
		for vType in voidTypes:
			print "%s, %s" %(sample,vType)
	
			parmsFile = structdir+'BigMDPl/'+sample+"/sample_info.dat"
			boundFile = structdir+'BigMDPl/'+sample+"/"+vType+"Voids_boundaries.dat"
			cInfoFile = structdir+'BigMDPl/'+sample+"/"+vType+"Voids_info.txt"
			c_ellipFile = structdir+'BigMDPl/'+sample+"/"+vType+"Voids_ellipticities.txt"
	
			structIDs, BoundaryPosns = read_boundaries(boundFile)
			voidInfo_cc = np.loadtxt(cInfoFile,skiprows=2)
			parms = imp.load_source("name",parmsFile)
	
			ellip_cc = np.zeros((structIDs.shape[0],4))
			for i in range(len(structIDs)):
				#account for PBC in the box
				delta = voidInfo_cc[i,1:4] - BoundaryPosns[i]
				Xleak = np.abs(delta[:,0])>parms.boxLen/2.
				Yleak = np.abs(delta[:,1])>parms.boxLen/2.
				Zleak = np.abs(delta[:,2])>parms.boxLen/2.
				BoundaryPosns[i][Xleak,0] += parms.boxLen*np.sign(delta[Xleak,0])
				BoundaryPosns[i][Yleak,1] += parms.boxLen*np.sign(delta[Yleak,1])
				BoundaryPosns[i][Zleak,2] += parms.boxLen*np.sign(delta[Zleak,2])

				boundaries_cc = BoundaryPosns[i] - voidInfo_cc[i,1:4]
	
				M1_cc = np.zeros((3,3)); 
	
				M1_cc[0,0] = np.sum(boundaries_cc[:,1]**2 + boundaries_cc[:,2]**2)
				M1_cc[1,1] = np.sum(boundaries_cc[:,0]**2 + boundaries_cc[:,2]**2)
				M1_cc[2,2] = np.sum(boundaries_cc[:,0]**2 + boundaries_cc[:,1]**2)
				M1_cc[0,1] = M1_cc[1,0] = np.sum(boundaries_cc[:,0]*boundaries_cc[:,1])
				M1_cc[0,2] = M1_cc[2,0] = np.sum(boundaries_cc[:,0]*boundaries_cc[:,2])
				M1_cc[2,1] = M1_cc[1,2] = np.sum(boundaries_cc[:,2]*boundaries_cc[:,1])
	
				w, v = np.linalg.eigh(M1_cc)
				ellip_cc[i,0] = np.sqrt(w[2]/w[0])
				ellip_cc[i,1] = np.sqrt(w[1]/w[0])
				ellip_cc[i,2] = 0.5*(w[2]-w[0])/np.sum(w)
				ellip_cc[i,3] = 0.5*(w[2]-2*w[1]+w[0])/np.sum(w)
	
			with open(c_ellipFile,'w') as F:
				F.write("VoidID a1/a3 a1/a2 e p\n")
				for i in range(len(structIDs)):
					F.write("%d %0.3f %0.3f %0.3f %0.3f\n" %(structIDs[i],ellip_cc[i,0],ellip_cc[i,1],ellip_cc[i,2],ellip_cc[i,3]))
					

def VoidPhiCCF(sample='CMASS',vType='Isolated',PhiFile='Phi1175_z052_g1_info.txt',numJack=4):
	
	boxLen = 2500.0 #Mpc/h
	Rmin = 3; 	Rmax = 200.0     #Mpc/h
	bins = 20
	
	D1D2_action='compute'; D1D2_name='junk/DD.txt'
	RR_action='compute'; RR_name='junk/RR.txt'
	D1R_action='compute'; D1R_name='junk/DR.txt'
	D2R_action='compute'; D2R_name='junk/DR.txt'
	
	voiddata = np.loadtxt(structdir+'BigMDPl/'+sample+'/'+vType+'Voids_info.txt',skiprows=2)
	Lambda = (voiddata[:,5]-1)*voiddata[:,6]**1.2 
	voiddata = voiddata[Lambda>10]
	Phidata = np.loadtxt(structdir+'BigMDPl/PhiVoids/'+PhiFile,skiprows=2)
	
	boxsplit = np.arange(numJack+1)*boxLen/numJack
	xi = np.zeros((bins,numJack**3))
	for i in range(numJack):
		for j in range(numJack):
			for k in range(numJack):
				index = i*numJack**2 + j*numJack + k
				voidbox = voiddata[(voiddata[:,1]>boxsplit[i]) & (voiddata[:,1]<boxsplit[i+1]) &\
						(voiddata[:,2]>boxsplit[j]) & (voiddata[:,2]<boxsplit[j+1]) &\
						(voiddata[:,3]>boxsplit[k]) & (voiddata[:,3]<boxsplit[k+1])]
				pos1 = voidbox[:,1:4]
				Phibox = Phidata[(Phidata[:,1]>boxsplit[i]) & (Phidata[:,1]<boxsplit[i+1]) &\
						(Phidata[:,2]>boxsplit[j]) & (Phidata[:,2]<boxsplit[j+1]) &\
						(Phidata[:,3]>boxsplit[k]) & (Phidata[:,3]<boxsplit[k+1])]
				pos2 = Phibox[:,1:4]
				points_r = 10000
				pos_r = np.random.random((points_r,3))*boxLen/numJack 	#generate a random subbox
				pos_r += np.array([i,j,k])*boxLen/numJack 	#put the centre in the right position
				
				r,xi_r = TPCCF(pos1,pos2,pos_r,boxLen,
				             D1D2_action,D1R_action,D2R_action,RR_action,
				             D1D2_name,D1R_name,D2R_name,RR_name,
				             bins,Rmin,Rmax,verbose=False)
				xi[:,index] = xi_r
	Jack_samples = np.zeros(xi.shape)
	for i in range(xi.shape[0]):
	    for j in range(xi.shape[1]):
	        Jack_samples[i,j] = np.mean(xi[i,:]) - xi[i,j]/(xi.shape[1])
	Jack_mean = np.mean(Jack_samples,axis=1)
	Jack_err = np.std(Jack_samples,axis=1)*np.sqrt(xi.shape[1])
	output = np.empty((bins,3))
	output[:,0] = r; output[:,1] = Jack_mean; output[:,2] = Jack_err
#	np.savetxt(structdir+'BigMDPl/PhiVoids/alt_xcorr/'+sample+vType+'_all_L<-20.txt',output,fmt='%0.6f',header='r xi(r) xi_err')
#	np.savetxt(structdir+'BigMDPl/PhiVoids/xcorr/'+sample+vType+'_all_L>10_ACF.txt',output,fmt='%0.6f',header='r xi(r) xi_err')
	np.savetxt(structdir+'BigMDPl/PhiVoids/xcorr/'+PhiFile.replace('_info.txt','_ACF.txt'),output,fmt='%0.6f',header='r xi(r) xi_err')

def plot_voidPhiCCF():
	
	fig,axes = plt.subplots(2,2,sharex=False,sharey=True,figsize=(18,14))
	verts = [21.7,28.1,40.2,44.1]
	colours = np.array(['#3130ff','#3366ff',wad_colours[4],kelly_colours[14],kelly_RdYlGn[3],kelly_RdYlGn[4],kelly_RdYlGn[7]])

	ax = axes.flat[0]
	ax.set_xscale('log')
	data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/CCFs/Main1_Isol_all.txt')
	ax.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=colours[3],alpha=0.7)
	ax.plot(data[:,0],data[:,1],c=colours[3],lw=2,label=r'$\mathrm{all\;voids}$')
	data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/CCFs/Main1_Isol_-infL-10.txt')
	ax.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=colours[0],alpha=0.7)
	ax.plot(data[:,0],data[:,1],c=colours[0],dashes=[8,3],lw=2,label=r'$\lambda_v<-10$')
	data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/CCFs/Main1_Isol_-10L0.txt')
	ax.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=colours[1],alpha=0.7)
	ax.plot(data[:,0],data[:,1],c=colours[1],dashes=[8,3,3,3],lw=2,label=r'$-10<\lambda_v<0$')
	data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/CCFs/Main1_Isol_0L10.txt')
	ax.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=colours[5],alpha=0.7)
	ax.plot(data[:,0],data[:,1],c=colours[5],dashes=[8,3,8,3,3,3],lw=2,label=r'$0<\lambda_v<10$')
	data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/CCFs/Main1_Isol_10Linf.txt')
	ax.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=colours[6],alpha=0.7)
	ax.plot(data[:,0],data[:,1],c=colours[6],dashes=[8,3,3,3,3,3],lw=2,label=r'$\lambda_v>10$')
	ax.tick_params(labelsize=14)
	ax.set_xlim([2.9,180]); ax.set_ylim([-1,12])
	ax.set_xticks([10,20,30,40,50,60,70,80,90,100],['10','','','','','','','','','100'])
	ax.set_xlabel(r'$r\;[h^{-1}\mathrm{Mpc}]$',fontsize=24)
	ax.set_ylabel(r'$\xi_{\mathrm{v}\Phi}(r)$',fontsize=24)
	ax.legend(loc='upper right',numpoints=1,prop={'size':22},borderpad=0.5)
	ax.axhline(0,c='k',ls=':')
	ax.axvline(verts[0],c='k',ls=':')
	ax.annotate(r'$\overline{R_v}=%0.1f$'%verts[0], xy=(10, 10),
	            xycoords='data',fontsize=22)
	ax.set_title('Main1',fontsize=22)

	ax = axes.flat[1]
	ax.set_xscale('log')
	data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/CCFs/Main2_Isol_all.txt')
	ax.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=colours[3],alpha=0.7)
	ax.plot(data[:,0],data[:,1],c=colours[3],lw=2,label=r'$\mathrm{all\;voids}$')
	data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/CCFs/Main2_Isol_-infL-10.txt')
	ax.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=colours[0],alpha=0.7)
	ax.plot(data[:,0],data[:,1],c=colours[0],dashes=[8,3],lw=2,label=r'$\lambda_v<-10$')
	data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/CCFs/Main2_Isol_-10L0.txt')
	ax.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=colours[1],alpha=0.7)
	ax.plot(data[:,0],data[:,1],c=colours[1],dashes=[8,3,3,3],lw=2,label=r'$-10<\lambda_v<0$')
	data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/CCFs/Main2_Isol_0L10.txt')
	ax.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=colours[5],alpha=0.7)
	ax.plot(data[:,0],data[:,1],c=colours[5],dashes=[8,3,8,3,3,3],lw=2,label=r'$0<\lambda_v<10$')
	data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/CCFs/Main2_Isol_10Linf.txt')
	ax.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=colours[6],alpha=0.7)
	ax.plot(data[:,0],data[:,1],c=colours[6],dashes=[8,3,3,3,3,3],lw=2,label=r'$\lambda_v>10$')
	ax.tick_params(labelsize=14)
	ax.set_xlim([2.9,180]); ax.set_ylim([-1,12])
	ax.set_xticks([10,20,30,40,50,60,70,80,90,100],['10','','','','','','','','','100'])
	ax.set_xlabel(r'$r\;[h^{-1}\mathrm{Mpc}]$',fontsize=24)
	ax.axhline(0,c='k',ls=':')
	ax.axvline(verts[1],c='k',ls=':')
	ax.set_title('Main2',fontsize=22)
	
	ax = axes.flat[2]
	ax.set_xscale('log')
	data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/CCFs/LOWZ_Isol_all.txt')
	ax.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=colours[3],alpha=0.7)
	ax.plot(data[:,0],data[:,1],c=colours[3],lw=2,label=r'$\mathrm{all\;voids}$')
	data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/CCFs/LOWZ_Isol_-infL-10.txt')
	ax.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=colours[0],alpha=0.7)
	ax.plot(data[:,0],data[:,1],c=colours[0],dashes=[8,3],lw=2,label=r'$\lambda_v<-10$')
	data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/CCFs/LOWZ_Isol_-10L0.txt')
	ax.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=colours[1],alpha=0.7)
	ax.plot(data[:,0],data[:,1],c=colours[1],dashes=[8,3,3,3],lw=2,label=r'$-10<\lambda_v<0$')
	data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/CCFs/LOWZ_Isol_0L10.txt')
	ax.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=colours[5],alpha=0.7)
	ax.plot(data[:,0],data[:,1],c=colours[5],dashes=[8,3,8,3,3,3],lw=2,label=r'$0<\lambda_v<10$')
	data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/CCFs/LOWZ_Isol_10Linf.txt')
	ax.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=colours[6],alpha=0.7)
	ax.plot(data[:,0],data[:,1],c=colours[6],dashes=[8,3,3,3,3,3],lw=2,label=r'$\lambda_v>10$')
	ax.tick_params(labelsize=14)
	ax.set_xlim([2.9,180]); ax.set_ylim([-1,12])
	ax.set_xticks([10,20,30,40,50,60,70,80,90,100],['10','','','','','','','','','100'])
	ax.set_xlabel(r'$r\;[h^{-1}\mathrm{Mpc}]$',fontsize=24)
	ax.set_ylabel(r'$\xi_{\mathrm{v}\Phi}(r)$',fontsize=24)
	ax.axhline(0,c='k',ls=':')
	ax.axvline(verts[2],c='k',ls=':')
	ax.set_title('LOWZ',fontsize=22)

	ax = axes.flat[3]
	ax.set_xscale('log')
	data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/CCFs/CMASS_Isol_all.txt')
	ax.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=colours[3],alpha=0.7)
	ax.plot(data[:,0],data[:,1],c=colours[3],lw=2,label=r'$\mathrm{all\;voids}$')
	data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/CCFs/CMASS_Isol_-infL-10.txt')
	ax.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=colours[0],alpha=0.7)
	ax.plot(data[:,0],data[:,1],c=colours[0],dashes=[8,3],lw=2,label=r'$\lambda_v<-10$')
	data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/CCFs/CMASS_Isol_-10L0.txt')
	ax.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=colours[1],alpha=0.7)
	ax.plot(data[:,0],data[:,1],c=colours[1],dashes=[8,3,3,3],lw=2,label=r'$-10<\lambda_v<0$')
	data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/CCFs/CMASS_Isol_0L10.txt')
	ax.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=colours[5],alpha=0.7)
	ax.plot(data[:,0],data[:,1],c=colours[5],dashes=[8,3,8,3,3,3],lw=2,label=r'$0<\lambda_v<10$')
	data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/CCFs/CMASS_Isol_10Linf.txt')
	ax.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=colours[6],alpha=0.7)
	ax.plot(data[:,0],data[:,1],c=colours[6],dashes=[8,3,3,3,3,3],lw=2,label=r'$\lambda_v>10$')
	ax.tick_params(labelsize=14)
	ax.set_xlim([2.9,180]); ax.set_ylim([-1,12])
	ax.set_xticks([10,20,30,40,50,60,70,80,90,100],['10','','','','','','','','','100'])
	ax.set_xlabel(r'$r\;[h^{-1}\mathrm{Mpc}]$',fontsize=24)
	ax.axhline(0,c='k',ls=':')
	ax.axvline(verts[3],c='k',ls=':')
	ax.set_title('CMASS',fontsize=22)
	
	plt.tight_layout(w_pad=2)
#	plt.savefig(figdir+'BigMDPl/VoidsPhiCCF_lambda.pdf',bbox_inches='tight')

def BigMDPl_Phi_profiles():
	
	samples=['Main1','Main2','LOWZ','CMASS']
	groups = ['-infARsc-30','-30ARsc-20','-20ARsc-10','-10ARsc0','0ARsc10','10ARsc20','20ARsc30']
	labels = ['$\lambda_v<-30$','$-30<\lambda_v<-20$','$-20<\lambda_v<-10$','$-10<\lambda_v<0$','$0<\lambda_v<10$','$10<\lambda_v<20$','$20<\lambda_v<30$']
#	colours = kelly_RdYlGn[[0,1,2,3,4,7,8]]
#	colours = np.array(['#23238e','#3130ff','#3366ff',wad_colours[4],kelly_RdYlGn[3],kelly_RdYlGn[4],kelly_RdYlGn[7]])
	colours = np.array(['#3130ff','#3366ff',wad_colours[4],kelly_colours[14],kelly_RdYlGn[3],kelly_RdYlGn[4],kelly_RdYlGn[7]])
	points = pointstyles[[0,3,1,2,4,9,7]]
	msizes = np.array([8,7,9,9,7,9,8])
	xlims = [360,360,360,360]
	val_lims = [-100,-30,-20,-10,0,10,20,30]
	fit_params = np.zeros((len(samples),4,2))
	fit_param_errors = np.zeros((len(samples),4,2))
	vals = np.zeros((len(samples),len(groups)))
	verts = [21.7,28.1,40.2,44.1]
	
	fig,axes = plt.subplots(2,2,sharex=False,sharey=True,figsize=(16,12))
	isubplot = 0
	for sample in samples:

		metrics = np.loadtxt(structdir+'BigMDPl/'+sample+'/IsolatedVoids_metrics.txt',skiprows=1)
		ax = axes.flat[isubplot]
		x = np.linspace(0,xlims[isubplot])
		ax.plot(x,0*x,'k:')
		igroup = 0
		for group in groups:		
			
			data = np.loadtxt(structdir+'BigMDPl/'+sample+'/profiles/differential/alt_Phi_res1175/IsolatedV_'\
					+group+'.txt')
			(_,caps,_) = ax.errorbar(data[:,0],data[:,1],yerr=data[:,2],fmt=points[igroup],markersize=\
					msizes[igroup],elinewidth=1.5,markeredgecolor='none',color=colours[igroup],label=labels[igroup])
			for cap in caps: cap.set_markeredgewidth(2)
			if igroup < 4:
				popt, pcov = curve_fit(phi1,data[:,0],data[:,1],sigma=data[:,2],absolute_sigma=True,p0=[3,100],maxfev=10000)
				ax.plot(x,phi1(x,*popt),color=colours[igroup],lw=2,ls='--')
				fit_params[isubplot,igroup] = popt
				fit_param_errors[isubplot,igroup] = np.sqrt(np.diag(pcov))
			else:
				popt, pcov = curve_fit(phi2,data[:,0],data[:,1],sigma=data[:,2],absolute_sigma=True,p0=[3,100,100],maxfev=10000)
				ax.plot(x,phi2(x,*popt),color=colours[igroup],lw=2,ls='--')				
			select = ((metrics[:,3]-1)*metrics[:,1]**1.2>val_lims[igroup])&((metrics[:,3]-1)*metrics[:,1]**1.2<val_lims[igroup+1])
			vals[isubplot,igroup] = np.mean((metrics[select,3]-1)*metrics[select,1]**1.2)
			igroup += 1

		ax.set_title(sample,fontsize=22)	
		ax.set_xlabel(r'$r\;[h^{-1}\mathrm{Mpc}]$',fontsize=24)
		ax.set_xlim([0,xlims[isubplot]])
		ax.tick_params(labelsize=14)
		ax.axvline(verts[isubplot],c='k',ls=':')
		if isubplot==0 or isubplot==2: ax.set_ylabel(r'$10^5\times\Phi(r)$',fontsize=24)
		if isubplot==0:
			ax.legend(loc='upper right',numpoints=1,prop={'size':16},borderpad=0.5)
		isubplot += 1
		
	plt.tight_layout(w_pad=1)
#	plt.subplots_adjust(wspace=0.1)
#	plt.savefig(figdir+'BigMDPl/figure.pdf',bbox_inches='tight')
#	return fit_params, fit_param_errors, vals

def BigMDPl_dens_profiles(useAlt=True,add_VTFE=False):
	

	lambda_bins = np.array([-1000,-30,-20,-10,0,10,20,30])
	colours = np.array(['#3130ff','#3366ff',wad_colours[4],kelly_colours[14],kelly_RdYlGn[3],kelly_RdYlGn[4],kelly_RdYlGn[7]])
	points = pointstyles[[0,3,1,2,4,9,7]]
	msizes = np.array([8,7,9,9,7,9,8])
	labels = ['$\lambda_v<-30$','$-30<\lambda_v<-20$','$-20<\lambda_v<-10$','$-10<\lambda_v<0$','$0<\lambda_v<10$','$10<\lambda_v<20$','$20<\lambda_v<30$']
	verts = [21.7,28.1,40.2,44.1]

	samples=['Main1','Main2','LOWZ','CMASS']
	if useAlt:
		altString = 'alt_'
	else:
		altString = ''
	fig,axes = plt.subplots(2,2,sharex=False,sharey=True,figsize=(16,13))
	bias = [1.29,1.4,2.0,2.0]

	isubplot = 0
	for sample in samples:
		ax = axes.flat[isubplot]
		ax.axhline(0,c='k',ls=':')
		profdata = np.loadtxt(structdir+'BigMDPl/'+sample+'/profiles/differential/'+altString+'DM_res2350/IsolatedV_all')
		metrics = np.loadtxt(structdir+'BigMDPl/'+sample+'/IsolatedVoids_metrics.txt',skiprows=1)
		
		igroup = 0
		for igroup in range(len(lambda_bins)-1):
			data = profdata[:,1:]
			select = ((metrics[:,3]-1)*metrics[:,1]**1.2>lambda_bins[igroup])&((metrics[:,3]-1)*metrics[:,1]**1.2<lambda_bins[igroup+1])
			data = data[:,select]-1
			#data[:,1] -= 1
			yerr = 5*np.std(data,axis=1)/np.sqrt(data.shape[1]-1)
			(_,caps,_) = ax.errorbar(profdata[:,0],np.mean(data,axis=1),yerr=yerr,fmt=':'+points[igroup],markersize=msizes[igroup],\
					elinewidth=1.5,markeredgecolor='none',color=colours[igroup],label=labels[igroup])
			for cap in caps: cap.set_markeredgewidth(2)
			igroup+=1
		if isubplot==0 or isubplot==2: 
			ax.set_ylabel(r'$\delta(r)$',fontsize=24,fontweight='extra bold')	
		if isubplot==0:	ax.legend(loc='lower right',numpoints=1,prop={'size':16},borderpad=0.5)
		ax.set_title(sample,fontsize=24)	
		if useAlt:
			ax.set_xlabel(r'$r\;[h^{-1}\mathrm{Mpc}]$',fontsize=24)
			ax.set_xlim([0,140])
		else:			
			ax.set_xlabel(r'$r/R_v$',fontsize=24)
		ax.axvline(verts[isubplot],c='k',ls=':')
		ax.set_ylim([-0.85,0.6])
		ax.tick_params(labelsize=14)
		ax.set_title(sample,fontsize=22)
		isubplot +=1
	
	plt.tight_layout(w_pad=1)
	plt.savefig(figdir+'BigMDPl/figure.pdf',bbox_inches='tight')
	
def find_lownu_voids(sample='CMASS',vType='Isolated',nbins=20,useQ=True):
	
	#the simulation data for calibration
	simdata = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/'+sample+'/'+vType+'Voids_metrics.txt',skiprows=1)
	rootIDs = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/'+sample+'/'+vType+'Voids_rootIDs.txt')
	simdata = simdata[np.in1d(simdata[:,0],rootIDs)]
	
	#the actual void data from the catalogue
	if sample=='CMASS' or sample=='LOWZ':
		realdataN = np.loadtxt('/Users/seshadri/Workspace/structures/SDSS_DR11/'+sample+'_North/'+vType+'Voids_info.txt',skiprows=2)
		realdataS = np.loadtxt('/Users/seshadri/Workspace/structures/SDSS_DR11/'+sample+'_South/'+vType+'Voids_info.txt',skiprows=2)
		realdata = np.vstack([realdataN,realdataS])
	else:
		realdata = np.loadtxt('/Users/seshadri/Workspace/structures/SDSS_DR7/'+sample+'/'+vType+'Voids_info.txt',skiprows=2)
	
	if sample=='CMASS':
		sigmaFile = '/Users/seshadri/Workspace/sigmaTH_z052_Planck.txt'
	elif sample=='LOWZ':
		sigmaFile = '/Users/seshadri/Workspace/sigmaTH_z032_Planck.txt'
	else:
		sigmaFile = '/Users/seshadri/Workspace/sigmaTH_z010_Planck.txt'
	sigmaTH = np.loadtxt(sigmaFile)
	sigmaTH = interp1d(sigmaTH[:,0],sigmaTH[:,1])
	
	#first the excursion of Delta(Rv)
	deltaR = (simdata[:,3]-1)*simdata[:,1]		#delta_avg * Rv
	xbins = mquantiles(deltaR,1.0*np.arange(nbins+1)/nbins) if useQ else nbins
	H, xedges = np.histogram(deltaR,bins=xbins)
	ymeans, ymean_errs = np.zeros(H.shape),np.zeros(H.shape)
	xmeans, xmean_errs = np.zeros(H.shape),np.zeros(H.shape)
	popt = np.zeros((H.shape[0],3))
	errors = np.zeros((H.shape[0],3))
	for i in range(H.shape[0]):
		bin_vals = simdata[(test_bin(deltaR,xedges[i],xedges[i+1])),7]
		xmeans[i] = np.mean(deltaR[test_bin(deltaR,xedges[i],xedges[i+1])])
		ymeans[i] = np.mean(bin_vals)
		xmean_errs[i] = np.std(deltaR[test_bin(deltaR,xedges[i],xedges[i+1])])/np.sqrt(len(bin_vals))
		ymean_errs[i] = np.std(bin_vals)/np.sqrt(len(bin_vals))
		popt[i], errors[i] = fit_skew_dist(bin_vals)
	upperlims = np.zeros(len(popt))
	for i in range(len(popt)):
		if errors[i,1]>0:
			upperlims[i] = fsolve(lambda x: quad(lambda y: skew_normal(y,popt[i,0],popt[i,1],popt[i,2]),-10,x)[0] - 0.954,-0.1)
		else:
			upperlims[i] = popt[i,0]+1.7*errors[i,0]
	linear_mean, cov = curve_fit(linf,xmeans,ymeans,sigma=ymean_errs,absolute_sigma=True,maxfev=1000)
	ul_slope, ul_intercept, r, p, se = linregress(xmeans,upperlims)
	estDeltaR = linear_mean[0] + linear_mean[1]*(realdata[:,5]-1)*realdata[:,6]
	nuR = estDeltaR/sigmaTH(realdata[:,6])
	nuR_twosigma = (ul_intercept + ul_slope*(realdata[:,5]-1)*realdata[:,6])/sigmaTH(realdata[:,6])
	plt.figure()
	plt.scatter(xmeans,upperlims)
	plt.scatter(xmeans,ymeans,c='r')
	
	#now the excursion of Delta(3Rv)
	delta = simdata[:,3]-1
	xbins = mquantiles(delta,1.0*np.arange(nbins+1)/nbins) if useQ else nbins
	H, xedges = np.histogram(delta,bins=xbins)
	ymeans, ymean_errs = np.zeros(H.shape),np.zeros(H.shape)
	xmeans, xmean_errs = np.zeros(H.shape),np.zeros(H.shape)
	upperlims = np.zeros(H.shape)
	for i in range(H.shape[0]):
		bin_vals = simdata[(test_bin(delta,xedges[i],xedges[i+1])),9]
		xmeans[i] = np.mean(delta[test_bin(delta,xedges[i],xedges[i+1])])
		ymeans[i] = np.mean(bin_vals)
		xmean_errs[i] = np.std(delta[test_bin(delta,xedges[i],xedges[i+1])])/np.sqrt(len(bin_vals))
		ymean_errs[i] = np.std(bin_vals)/np.sqrt(len(bin_vals))
		upperlims[i] = ymeans[i] + 1.7*np.std(bin_vals)
	linear_mean, cov = curve_fit(linf,xmeans,ymeans,sigma=ymean_errs,absolute_sigma=True,maxfev=1000)
	ul_slope, ul_intercept, r, p, se = linregress(xmeans,upperlims)
	estDelta3R = linear_mean[0] + linear_mean[1]*(realdata[:,5]-1)
	nu3R = estDelta3R/sigmaTH(3.*realdata[:,6])
	nu3R_twosigma = (ul_intercept + ul_slope*(realdata[:,5]-1))/sigmaTH(3*realdata[:,6])
	plt.figure()
	plt.scatter(xmeans,upperlims)
	plt.scatter(xmeans,ymeans,c='r')
	
	augmented_data = np.zeros((len(realdata),9))	
	augmented_data[:,0] = realdata[:,0]
	augmented_data[:,1] = realdata[:,4]
	augmented_data[:,2] = realdata[:,5]
	augmented_data[:,3] = realdata[:,6]
	augmented_data[:,4] = realdata[:,8]
	augmented_data[:,5] = nuR
	augmented_data[:,6] = nuR_twosigma
	augmented_data[:,7] = nu3R
	augmented_data[:,8] = nu3R_twosigma
	
	Rsupers = augmented_data[augmented_data[:,5]<-3]
	ThreeRsupers = augmented_data[augmented_data[:,7]<-3]
	
	if len(Rsupers)>0:
		print "%d voids are >3sigma excursions at Rv:" %len(Rsupers)
#		np.savetxt(structdir+sample+'_'+vType+'_Rsupervoids.txt',Rsupers[np.argsort(Rsupers[:,5])],\
#			header='VoidID MinDens WtdAvgDens R_eff(Mpc/h) EdgeFlag nuR nuR_max nu3R nu3R_max',fmt='%0.4f')		
	else:
		print "0 voids exceed 3sigma at Rv, writing 5 best instead"
		Rsupers = augmented_data[np.argsort(augmented_data[:,5])]
#		np.savetxt(structdir+sample+'_'+vType+'_Rsupervoids.txt',Rsupers[:5],\
#			header='VoidID MinDens WtdAvgDens R_eff(Mpc/h) EdgeFlag nuR nuR_max nu3R nu3R_max',fmt='%0.4f')		

		
	if len(ThreeRsupers)>0:
		print "%d voids are >3sigma excursions at 3Rv:" %len(ThreeRsupers)
#		np.savetxt(structdir+sample+'_'+vType+'_3Rsupervoids.txt',ThreeRsupers[np.argsort(ThreeRsupers[:,7])],\
#			header='VoidID MinDens WtdAvgDens R_eff(Mpc/h) EdgeFlag nuR nuR_max nu3R nu3R_max',fmt='%0.4f')		
	else:
		print "0 voids exceed 3sigma at 3Rv, writing 5 best instead"
		ThreeRsupers = augmented_data[np.argsort(augmented_data[:,7])]
#		np.savetxt(structdir+sample+'_'+vType+'_3Rsupervoids.txt',ThreeRsupers[:5],\
#			header='VoidID MinDens WtdAvgDens R_eff(Mpc/h) EdgeFlag nuR nuR_max nu3R nu3R_max',fmt='%0.4f')		
	
	plt.figure(figsize=(10,8))
	junk = plt.hist(nuR,bins=25,color=kelly_RdYlGn[0],normed=True,alpha=0.5,label=r'$\nu\,(R_v)$')
	junk = plt.hist(nu3R,bins=25,color=kelly_RdYlGn[7],normed=True,alpha=0.5,label=r'$\nu\,(3R_v)$')
	plt.legend(loc='upper right',numpoints=1,prop={'size':16},borderpad=0.5)
	ax = plt.gca(); prettify_plot(ax)
#	plt.savefig(figdir+'DR11/'+sample+'_'+vType+'_nu_dist.pdf',bbox_inches='tight')

def test_r(col2=4,nbins=10,useQ=True):
	
	cminfo = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/CMASS/IsolatedVoids_info.txt',skiprows=2)
	lzinfo = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/LOWZ/IsolatedVoids_info.txt',skiprows=2)
	cmmetrics = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/CMASS/IsolatedVoids_metrics.txt',skiprows=1)
	lzmetrics = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/LOWZ/IsolatedVoids_metrics.txt',skiprows=1)

	cmsigmaFile = '/Users/seshadri/Workspace/sigmaTH_z052_Planck.txt'
	lzsigmaFile = '/Users/seshadri/Workspace/sigmaTH_z032_Planck.txt'
	cmsigmaTH = np.loadtxt(cmsigmaFile)
	cmsigmaTH = interp1d(cmsigmaTH[:,0],cmsigmaTH[:,1])
	lzsigmaTH = np.loadtxt(lzsigmaFile)
	lzsigmaTH = interp1d(lzsigmaTH[:,0],lzsigmaTH[:,1])
	
	cminfo[:,4] -= 1; cminfo[:,5] -=1; cmmetrics[:,4] -= 1
	lzinfo[:,4] -= 1; lzinfo[:,5] -=1; lzmetrics[:,4] -= 1
	
	if col2<10:
		cmy = cmmetrics[:,col2]; lzy = lzmetrics[:,col2]
	else:
		cmy = cmmetrics[:,7]/cmsigmaTH(cminfo[:,6])
		lzy = lzmetrics[:,7]/lzsigmaTH(lzinfo[:,6])
		
	fig,axes = plt.subplots(1,4,sharex=False,sharey=True,figsize=(24,8))
	ax = axes.flat[0]
	divs = np.arange(nbins+1).astype(float)/nbins
	cmxedges = mquantiles(cminfo[:,4],divs)
	lzxedges = mquantiles(lzinfo[:,4],divs)
	cmxedges, binned_cmx, binned_cmxerr = binner(cminfo[:,4],cminfo[:,4],cmxedges)
	cmxedges, binned_cmy, binned_cmyerr = binner(cminfo[:,4],cmy,cmxedges)
	lzxedges, binned_lzx, binned_lzxerr = binner(lzinfo[:,4],lzinfo[:,4],lzxedges)
	lzxedges, binned_lzy, binned_lzyerr = binner(lzinfo[:,4],lzy,lzxedges)	
	(_,caps,_) = ax.errorbar(binned_lzx,binned_lzy,yerr=2*binned_lzyerr,xerr=2*binned_lzxerr,fmt=pointstyles[0],markersize=8,\
		elinewidth=1.5,markeredgecolor='none',color=kelly_RdYlGn[0],label='BigMD LOWZ mocks')
	for cap in caps: cap.set_markeredgewidth(2)
	(_,caps,_) = ax.errorbar(binned_cmx,binned_cmy,yerr=2*binned_cmyerr,xerr=2*binned_cmxerr,fmt=pointstyles[3],markersize=8,\
		elinewidth=1.5,markeredgecolor='none',color=kelly_RdYlGn[7],label='BigMD CMASS mocks')
	for cap in caps: cap.set_markeredgewidth(2)
	ax.set_xlabel(r'$\delta_{g,\mathrm{min}}$',fontsize=24,fontweight='extra bold')
	if col2==4:
		ax.set_ylabel(r'$\delta_\mathrm{min}$',fontsize=24,fontweight='extra bold')
	elif col2==7:
		ax.set_ylabel(r'$\Delta(R_v)$',fontsize=24,fontweight='extra bold')
	elif col2==9:
		ax.set_ylabel(r'$\Delta(3R_v)$',fontsize=24,fontweight='extra bold')
	elif col2>9:
		ax.set_ylabel(r'$\nu$',fontsize=24,fontweight='extra bold')
	ax.set_xticks([-0.9,-0.8,-0.7,-0.6,-0.5,-0.4])
	ax.set_xlim([-0.9,-0.35])
	ax.tick_params(axis='both', labelsize=16)
	ax.legend(loc='upper left',numpoints=1,prop={'size':16},borderpad=0.5)

	ax = axes.flat[1]
	divs = np.arange(nbins+1).astype(float)/nbins
	cmxedges = mquantiles(cminfo[:,5],divs)
	lzxedges = mquantiles(lzinfo[:,5],divs)
	cmxedges, binned_cmx, binned_cmxerr = binner(cminfo[:,5],cminfo[:,5],cmxedges)
	cmxedges, binned_cmy, binned_cmyerr = binner(cminfo[:,5],cmy,cmxedges)
	lzxedges, binned_lzx, binned_lzxerr = binner(lzinfo[:,5],lzinfo[:,5],lzxedges)
	lzxedges, binned_lzy, binned_lzyerr = binner(lzinfo[:,5],lzy,lzxedges)	
	(_,caps,_) = ax.errorbar(binned_lzx,binned_lzy,yerr=2*binned_lzyerr,xerr=2*binned_lzxerr,fmt=pointstyles[0],markersize=8,\
		elinewidth=1.5,markeredgecolor='none',color=kelly_RdYlGn[0],label='BigMD LOWZ mocks')
	for cap in caps: cap.set_markeredgewidth(2)
	(_,caps,_) = ax.errorbar(binned_cmx,binned_cmy,yerr=2*binned_cmyerr,xerr=2*binned_cmxerr,fmt=pointstyles[3],markersize=8,\
		elinewidth=1.5,markeredgecolor='none',color=kelly_RdYlGn[7],label='BigMD CMASS mocks')
	for cap in caps: cap.set_markeredgewidth(2)
	ax.set_xlabel(r'$\bar{\delta}_g$',fontsize=24,fontweight='extra bold')
	ax.tick_params(axis='both', labelsize=16)

	ax =axes.flat[2]
	divs = np.arange(nbins+1).astype(float)/nbins
	cmxedges = mquantiles(cminfo[:,6],divs)
	lzxedges = mquantiles(lzinfo[:,6],divs)
	cmxedges, binned_cmx, binned_cmxerr = binner(cminfo[:,6],cminfo[:,6],cmxedges)
	cmxedges, binned_cmy, binned_cmyerr = binner(cminfo[:,6],cmy,cmxedges)
	lzxedges, binned_lzx, binned_lzxerr = binner(lzinfo[:,6],lzinfo[:,6],lzxedges)
	lzxedges, binned_lzy, binned_lzyerr = binner(lzinfo[:,6],lzy,lzxedges)	
	(_,caps,_) = ax.errorbar(binned_lzx,binned_lzy,yerr=2*binned_lzyerr,xerr=2*binned_lzxerr,fmt=pointstyles[0],markersize=8,\
		elinewidth=1.5,markeredgecolor='none',color=kelly_RdYlGn[0],label='BigMD LOWZ mocks')
	for cap in caps: cap.set_markeredgewidth(2)
	(_,caps,_) = ax.errorbar(binned_cmx,binned_cmy,yerr=2*binned_cmyerr,xerr=2*binned_cmxerr,fmt=pointstyles[3],markersize=8,\
		elinewidth=1.5,markeredgecolor='none',color=kelly_RdYlGn[7],label='BigMD CMASS mocks')
	for cap in caps: cap.set_markeredgewidth(2)
	ax.set_xlabel(r'$R_v\;[h^{-1}\mathrm{Mpc}]$',fontsize=24,fontweight='extra bold')
	ax.tick_params(axis='both', labelsize=16)

	ax=axes.flat[3]	
	if useQ:
		divs = np.arange(nbins+1).astype(float)/nbins
		cmxedges = mquantiles(cminfo[:,9],divs)
		lzxedges = mquantiles(lzinfo[:,9],divs)
		cmxedges, binned_cmx, binned_cmxerr = binner(cminfo[:,9],cminfo[:,9],cmxedges)
		cmxedges, binned_cmy, binned_cmyerr = binner(cminfo[:,9],cmy,cmxedges)
		lzxedges, binned_lzx, binned_lzxerr = binner(lzinfo[:,9],lzinfo[:,9],lzxedges)
		lzxedges, binned_lzy, binned_lzyerr = binner(lzinfo[:,9],lzy,lzxedges)
	else:
		bins = [1.0,1.1,1.2,1.3,1.4,1.6,1.8,2.0,2.5]
		cmH, cmxedges = np.histogram(cminfo[cminfo[:,9]<2.4,9],bins)	
		cmxedges, binned_cmx, binned_cmxerr = binner(cminfo[:,9],cminfo[:,9],cmxedges)
		cmxedges, binned_cmy, binned_cmyerr = binner(cminfo[:,9],cmy,cmxedges)
		binned_cmx = binned_cmx[cmH>0]; binned_cmy = binned_cmy[cmH>0]
		binned_cmxerr = binned_cmxerr[cmH>0]; binned_cmyerr = binned_cmyerr[cmH>0]			
		lzH, lzxedges = np.histogram(lzinfo[:,9],bins)	
		lzxedges, binned_lzx, binned_lzxerr = binner(lzinfo[:,9],lzinfo[:,9],lzxedges)
		lzxedges, binned_lzy, binned_lzyerr = binner(lzinfo[:,9],lzy,lzxedges)
		binned_lzx = binned_lzx[lzH>0]; binned_lzy = binned_lzy[lzH>0]
		binned_lzxerr = binned_lzxerr[lzH>0]; binned_lzyerr = binned_lzyerr[lzH>0]		
	(_,caps,_) = ax.errorbar(binned_cmx+0.02,binned_cmy,yerr=2*binned_cmyerr,xerr=2*binned_cmxerr,fmt=pointstyles[3],markersize=8,\
		elinewidth=1.5,markeredgecolor='none',color=kelly_RdYlGn[7],label='CMASS')
	for cap in caps: cap.set_markeredgewidth(2)
	(_,caps,_) = ax.errorbar(binned_lzx,binned_lzy,yerr=2*binned_lzyerr,xerr=2*binned_lzxerr,fmt=pointstyles[0],markersize=8,\
		elinewidth=1.5,markeredgecolor='none',color=kelly_RdYlGn[0],label='LOWZ')
	for cap in caps: cap.set_markeredgewidth(2)
	ax.set_xlabel(r'$\mathrm{Density\;ratio,\;}r$',fontsize=24,fontweight='extra bold')
	ax.tick_params(axis='both', labelsize=16)	
	plt.subplots_adjust(wspace=0.1)
	
#	plt.savefig(figdir+'BigMDPl/significance_indicators.pdf',bbox_inches='tight')
#	plt.savefig(figdir+'BigMDPl/r_vs_deltag_significance.pdf',bbox_inches='tight')
	
def RSD_observables(directory='/Users/seshadri/Workspace/structures/BigMDPl/CMASS_RSD/',
	samples=['boxreal','boxredshift','realspace','redshiftspace','photoz002'],
	colY=11,colX=[5,6,4,7,8],nbins=15,structType='Voids'):
		
	if len(samples)==5:
		colours = np.array(['#3130ff',wad_colours[4],kelly_colours[14],kelly_RdYlGn[4],kelly_RdYlGn[7]])
	elif len(samples)==4:
		colours = np.array(['#3130ff',wad_colours[4],kelly_RdYlGn[3],kelly_RdYlGn[7]])
	points = pointstyles[[0,3,1,2,4,9,7]]
	msizes = np.array([8,7,9,9,7,9,8])
		
	fig,axes = plt.subplots(1,len(colX),sharex=False,sharey=True,figsize=(30,8))
	divs = np.arange(nbins+1).astype(float)/nbins
	for i in range(len(samples)):

		metrics = np.loadtxt(directory+samples[i]+'_'+structType+'_cat.txt')
		metrics = metrics[metrics[:,9]<=1]

		for j in range(len(colX)):
			ax = axes.flat[j]
			if colX[j]==8:
				xedges = [1.0,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.5,1.6,1.8,2.0,2.5]
			else:
				xedges = mquantiles(metrics[:,colX[j]],divs)
			xedges, x, xerr = binner(metrics[:,colX[j]],metrics[:,colX[j]],xedges)
			if i>1 and colY==11:
#				xedges, y, yerr = binner(metrics[:,colX[j]],metrics[:,colY]+0.381766,xedges)
				xedges, y, yerr = binner(metrics[:,colX[j]],metrics[:,colY],xedges)
			else:
				xedges, y, yerr = binner(metrics[:,colX[j]],metrics[:,colY],xedges)
			(_,caps,_) = ax.errorbar(x,y,yerr=yerr,fmt=points[i],markersize=msizes[i],\
				elinewidth=1.5,markeredgecolor='none',color=colours[i],label=samples[i])
			for cap in caps: cap.set_markeredgewidth(2)
			ax.tick_params(axis='both', labelsize=16)
			ax.axhline(0,linestyle=':',color='k')
				
	plt.subplots_adjust(wspace=0.1)


def test_observables(col2=5,nbins=12,useQ=True):
	
#	colours = kelly_RdYlGn[[0,1,3,7]]
	colours = np.array(['#3130ff',wad_colours[4],kelly_RdYlGn[3],kelly_RdYlGn[7]])

	cminfo = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/CMASS/IsolatedVoids_info.txt',skiprows=2)
	lzinfo = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/LOWZ/IsolatedVoids_info.txt',skiprows=2)
	m3info = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/Main3/IsolatedVoids_info.txt',skiprows=2)
	m2info = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/Main2/IsolatedVoids_info.txt',skiprows=2)
	m1info = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/Main1/IsolatedVoids_info.txt',skiprows=2)
	cmmetrics = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/CMASS/IsolatedVoids_metrics.txt',skiprows=1)
	lzmetrics = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/LOWZ/IsolatedVoids_metrics.txt',skiprows=1)
	m3metrics = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/Main3/IsolatedVoids_metrics.txt',skiprows=1)
	m2metrics = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/Main2/IsolatedVoids_metrics.txt',skiprows=1)
	m1metrics = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/Main1/IsolatedVoids_metrics.txt',skiprows=1)

	cmsigmaFile = '/Users/seshadri/Workspace/sigmaTH_z052_Planck.txt'
	lzsigmaFile = '/Users/seshadri/Workspace/sigmaTH_z032_Planck.txt'
	msigmaFile = '/Users/seshadri/Workspace/sigmaTH_z010_Planck.txt'

	cmsigmaTH = np.loadtxt(cmsigmaFile)
	cmsigmaTH = interp1d(cmsigmaTH[:,0],cmsigmaTH[:,1])
	lzsigmaTH = np.loadtxt(lzsigmaFile)
	lzsigmaTH = interp1d(lzsigmaTH[:,0],lzsigmaTH[:,1])
	msigmaTH = np.loadtxt(msigmaFile)
	msigmaTH = interp1d(msigmaTH[:,0],msigmaTH[:,1])
	
	cminfo[:,4] -= 1; cminfo[:,5] -=1; cmmetrics[:,4] -= 1
	lzinfo[:,4] -= 1; lzinfo[:,5] -=1; lzmetrics[:,4] -= 1
	m3info[:,4] -= 1; m3info[:,5] -=1; m3metrics[:,4] -= 1
	m2info[:,4] -= 1; m2info[:,5] -=1; m2metrics[:,4] -= 1
	m1info[:,4] -= 1; m1info[:,5] -=1; m1metrics[:,4] -= 1
	
	if col2<10:
		cmy = cmmetrics[:,col2]; lzy = lzmetrics[:,col2]; m3y = m3metrics[:,col2]; m2y = m2metrics[:,col2]; m1y = m1metrics[:,col2]
	else:
		cmy = cmmetrics[:,7]/cmsigmaTH(cminfo[:,6])
		lzy = lzmetrics[:,7]/lzsigmaTH(lzinfo[:,6])
		m3y = m3metrics[:,7]/msigmaTH(m2info[:,6])
		m2y = m2metrics[:,7]/msigmaTH(m2info[:,6])
		m1y = m1metrics[:,7]/msigmaTH(m1info[:,6])
		
	fig,axes = plt.subplots(1,4,sharex=False,sharey=True,figsize=(24,8))
	ax = axes.flat[0]
	divs = np.arange(nbins+1).astype(float)/nbins
	cmxedges = mquantiles(cminfo[:,4],divs)
	lzxedges = mquantiles(lzinfo[:,4],divs)
	m3xedges = mquantiles(m3info[:,4],divs)
	m2xedges = mquantiles(m2info[:,4],divs)
	m1xedges = mquantiles(m1info[:,4],divs)
	cmxedges, binned_cmx, binned_cmxerr = binner(cminfo[:,4],cminfo[:,4],cmxedges)
	cmxedges, binned_cmy, binned_cmyerr = binner(cminfo[:,4],cmy,cmxedges)
	lzxedges, binned_lzx, binned_lzxerr = binner(lzinfo[:,4],lzinfo[:,4],lzxedges)
	lzxedges, binned_lzy, binned_lzyerr = binner(lzinfo[:,4],lzy,lzxedges)	
	m3xedges, binned_m3x, binned_m3xerr = binner(m3info[:,4],m3info[:,4],m3xedges)
	m3xedges, binned_m3y, binned_m3yerr = binner(m3info[:,4],m3y,m3xedges)	
	m2xedges, binned_m2x, binned_m2xerr = binner(m2info[:,4],m2info[:,4],m2xedges)
	m2xedges, binned_m2y, binned_m2yerr = binner(m2info[:,4],m2y,m2xedges)	
	m1xedges, binned_m1x, binned_m1xerr = binner(m1info[:,4],m1info[:,4],m1xedges)
	m1xedges, binned_m1y, binned_m1yerr = binner(m1info[:,4],m1y,m1xedges)	
	(_,caps,_) = ax.errorbar(binned_m1x,binned_m1y,yerr=binned_m1yerr,fmt=pointstyles[0],markersize=8,\
		elinewidth=1.5,markeredgecolor='none',color=colours[0],label='Main1')
	for cap in caps: cap.set_markeredgewidth(2)
	(_,caps,_) = ax.errorbar(binned_m2x,binned_m2y,yerr=binned_m2yerr,fmt=pointstyles[1],markersize=9,\
		elinewidth=1.5,markeredgecolor='none',color=colours[1],label='Main2')
	for cap in caps: cap.set_markeredgewidth(2)
	(_,caps,_) = ax.errorbar(binned_lzx,binned_lzy,yerr=binned_lzyerr,fmt=pointstyles[4],markersize=8,\
		elinewidth=1.5,markeredgecolor='none',color=colours[2],label='LOWZ')
	for cap in caps: cap.set_markeredgewidth(2)
	(_,caps,_) = ax.errorbar(binned_cmx,binned_cmy,yerr=binned_cmyerr,fmt=pointstyles[3],markersize=7,\
		elinewidth=1.5,markeredgecolor='none',color=colours[3],label='CMASS')
	for cap in caps: cap.set_markeredgewidth(2)
	ax.set_xlabel(r'$\delta_{g,\mathrm{min}}$',fontsize=26,fontweight='extra bold')
	if col2==4:
		ax.set_ylabel(r'$\delta_\mathrm{min}$',fontsize=26,fontweight='extra bold')
		ax.legend(loc='upper left',numpoints=1,prop={'size':16},borderpad=0.5)
	elif col2==7:
		ax.set_ylabel(r'$\Delta(R_v)$',fontsize=26,fontweight='extra bold')
		ax.legend(loc='upper left',numpoints=1,prop={'size':16},borderpad=0.5)
	elif col2==9:
		ax.set_ylabel(r'$\Delta(3R_v)$',fontsize=26,fontweight='extra bold')
		ax.legend(loc='upper left',numpoints=1,prop={'size':16},borderpad=0.5)
		ax.set_ylim([-0.11,0.35])
	elif col2==5:
		ax.set_ylabel(r'$\Phi_0\times10^5$',fontsize=26,fontweight='extra bold')
		ax.legend(loc='upper right',numpoints=1,prop={'size':16},borderpad=0.5)
	elif col2>9:
		ax.set_ylabel(r'$\nu$',fontsize=26,fontweight='extra bold')
		ax.legend(loc='upper left',numpoints=1,prop={'size':16},borderpad=0.5)
	ax.set_xticks([-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4])
	ax.set_xlim([-1.,-0.38])
	ax.tick_params(axis='both', labelsize=16)
	ax.axhline(0,linestyle=':',color='k')

	ax = axes.flat[1]
	divs = np.arange(nbins+1).astype(float)/nbins
	cmxedges = mquantiles(cminfo[cminfo[:,5]<0.6,5],divs)
	lzxedges = mquantiles(lzinfo[lzinfo[:,5]<0.6,5],divs)
	m2xedges = mquantiles(m2info[m2info[:,5]<0.6,5],divs)
	m1xedges = mquantiles(m1info[m1info[:,5]<0.6,5],divs)
	cmxedges, binned_cmx, binned_cmxerr = binner(cminfo[:,5],cminfo[:,5],cmxedges)
	cmxedges, binned_cmy, binned_cmyerr = binner(cminfo[:,5],cmy,cmxedges)
	lzxedges, binned_lzx, binned_lzxerr = binner(lzinfo[:,5],lzinfo[:,5],lzxedges)
	lzxedges, binned_lzy, binned_lzyerr = binner(lzinfo[:,5],lzy,lzxedges)	
	m2xedges, binned_m2x, binned_m2xerr = binner(m2info[:,5],m2info[:,5],m2xedges)
	m2xedges, binned_m2y, binned_m2yerr = binner(m2info[:,5],m2y,m2xedges)	
	m1xedges, binned_m1x, binned_m1xerr = binner(m1info[:,5],m1info[:,5],m1xedges)
	m1xedges, binned_m1y, binned_m1yerr = binner(m1info[:,5],m1y,m1xedges)	
	(_,caps,_) = ax.errorbar(binned_m1x,binned_m1y,yerr=binned_m1yerr,fmt=pointstyles[0],markersize=8,\
		elinewidth=1.5,markeredgecolor='none',color=colours[0],label='Main1')
	for cap in caps: cap.set_markeredgewidth(2)
	(_,caps,_) = ax.errorbar(binned_m2x,binned_m2y,yerr=binned_m2yerr,fmt=pointstyles[1],markersize=9,\
		elinewidth=1.5,markeredgecolor='none',color=colours[1],label='Main2')
	for cap in caps: cap.set_markeredgewidth(2)
	(_,caps,_) = ax.errorbar(binned_lzx,binned_lzy,yerr=binned_lzyerr,fmt=pointstyles[4],markersize=8,\
		elinewidth=1.5,markeredgecolor='none',color=colours[2],label='LOWZ')
	for cap in caps: cap.set_markeredgewidth(2)
	(_,caps,_) = ax.errorbar(binned_cmx,binned_cmy,yerr=binned_cmyerr,fmt=pointstyles[3],markersize=7,\
		elinewidth=1.5,markeredgecolor='none',color=colours[3],label='CMASS')
	for cap in caps: cap.set_markeredgewidth(2)
	ax.set_xlabel(r'$\bar{\delta}_g$',fontsize=26,fontweight='extra bold')
	ax.set_xticks([-0.4,-0.2,0,0.2,0.4,0.6])
	ax.set_xlim([-0.4,0.6])
	ax.tick_params(axis='both', labelsize=16)
	ax.axhline(0,linestyle=':',color='k')
	ax.axvline(0,linestyle=':',color='k')

	ax = axes.flat[2]
	divs = np.arange(nbins+1).astype(float)/nbins
	cmxedges = mquantiles(cminfo[:,6],divs)
	lzxedges = mquantiles(lzinfo[:,6],divs)
	m2xedges = mquantiles(m2info[:,6],divs)
	m1xedges = mquantiles(m1info[:,6],divs)
	cmxedges, binned_cmx, binned_cmxerr = binner(cminfo[:,6],cminfo[:,6],cmxedges)
	cmxedges, binned_cmy, binned_cmyerr = binner(cminfo[:,6],cmy,cmxedges)
	lzxedges, binned_lzx, binned_lzxerr = binner(lzinfo[:,6],lzinfo[:,6],lzxedges)
	lzxedges, binned_lzy, binned_lzyerr = binner(lzinfo[:,6],lzy,lzxedges)	
	m2xedges, binned_m2x, binned_m2xerr = binner(m2info[:,6],m2info[:,6],m2xedges)
	m2xedges, binned_m2y, binned_m2yerr = binner(m2info[:,6],m2y,m2xedges)	
	m1xedges, binned_m1x, binned_m1xerr = binner(m1info[:,6],m1info[:,6],m1xedges)
	m1xedges, binned_m1y, binned_m1yerr = binner(m1info[:,6],m1y,m1xedges)	
	(_,caps,_) = ax.errorbar(binned_m1x,binned_m1y,yerr=binned_m1yerr,fmt=pointstyles[0],markersize=8,\
		elinewidth=1.5,markeredgecolor='none',color=colours[0],label='Main1')
	for cap in caps: cap.set_markeredgewidth(2)
	(_,caps,_) = ax.errorbar(binned_m2x,binned_m2y,yerr=binned_m2yerr,fmt=pointstyles[1],markersize=9,\
		elinewidth=1.5,markeredgecolor='none',color=colours[1],label='Main2')
	for cap in caps: cap.set_markeredgewidth(2)
	(_,caps,_) = ax.errorbar(binned_lzx,binned_lzy,yerr=binned_lzyerr,fmt=pointstyles[4],markersize=8,\
		elinewidth=1.5,markeredgecolor='none',color=colours[2],label='LOWZ')
	for cap in caps: cap.set_markeredgewidth(2)
	(_,caps,_) = ax.errorbar(binned_cmx,binned_cmy,yerr=binned_cmyerr,fmt=pointstyles[3],markersize=7,\
		elinewidth=1.5,markeredgecolor='none',color=colours[3],label='CMASS')
	for cap in caps: cap.set_markeredgewidth(2)
	ax.set_xlabel(r'$R_v\;[h^{-1}\mathrm{Mpc}]$',fontsize=26,fontweight='extra bold')
	ax.tick_params(axis='both', labelsize=16)
	ax.axhline(0,linestyle=':',color='k')

	ax=axes.flat[3]	
	if useQ:
		divs = np.arange(nbins+1).astype(float)/nbins
		cmxedges = mquantiles(cminfo[:,9],divs)
		lzxedges = mquantiles(lzinfo[:,9],divs)
		m2xedges = mquantiles(m2info[:,9],divs)
		m1xedges = mquantiles(m1info[:,9],divs)
		cmxedges, binned_cmx, binned_cmxerr = binner(cminfo[:,9],cminfo[:,9],cmxedges)
		cmxedges, binned_cmy, binned_cmyerr = binner(cminfo[:,9],cmy,cmxedges)
		lzxedges, binned_lzx, binned_lzxerr = binner(lzinfo[:,9],lzinfo[:,9],lzxedges)
		lzxedges, binned_lzy, binned_lzyerr = binner(lzinfo[:,9],lzy,lzxedges)
		m2xedges, binned_m2x, binned_m2xerr = binner(m2info[:,9],m2info[:,9],m2xedges)
		m2xedges, binned_m2y, binned_m2yerr = binner(m2info[:,9],m2y,m2xedges)
		m1xedges, binned_m1x, binned_m1xerr = binner(m1info[:,9],m1info[:,9],m1xedges)
		m1xedges, binned_m1y, binned_m1yerr = binner(m1info[:,9],m1y,m1xedges)
	else:
		bins = [1.0,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.5,1.6,1.8,2.0,2.5]
		cmH, cmxedges = np.histogram(cminfo[cminfo[:,9]<2.4,9],bins)	
		cmxedges, binned_cmx, binned_cmxerr = binner(cminfo[:,9],cminfo[:,9],cmxedges)
		cmxedges, binned_cmy, binned_cmyerr = binner(cminfo[:,9],cmy,cmxedges)
		binned_cmx = binned_cmx[cmH>0]; binned_cmy = binned_cmy[cmH>0]
		binned_cmxerr = binned_cmxerr[cmH>0]; binned_cmyerr = binned_cmyerr[cmH>0]			
		lzH, lzxedges = np.histogram(lzinfo[:,9],bins)	
		lzxedges, binned_lzx, binned_lzxerr = binner(lzinfo[:,9],lzinfo[:,9],lzxedges)
		lzxedges, binned_lzy, binned_lzyerr = binner(lzinfo[:,9],lzy,lzxedges)
		binned_lzx = binned_lzx[lzH>0]; binned_lzy = binned_lzy[lzH>0]
		binned_lzxerr = binned_lzxerr[lzH>0]; binned_lzyerr = binned_lzyerr[lzH>0]		
		m2H, m2xedges = np.histogram(m2info[:,9],bins)	
		m2xedges, binned_m2x, binned_m2xerr = binner(m2info[:,9],m2info[:,9],m2xedges)
		m2xedges, binned_m2y, binned_m2yerr = binner(m2info[:,9],m2y,m2xedges)
		binned_m2x = binned_m2x[m2H>0]; binned_m2y = binned_m2y[m2H>0]
		binned_m2xerr = binned_m2xerr[m2H>0]; binned_m2yerr = binned_m2yerr[m2H>0]		
		m1H, m1xedges = np.histogram(m1info[:,9],bins)	
		m1xedges, binned_m1x, binned_m1xerr = binner(m1info[:,9],m1info[:,9],m1xedges)
		m1xedges, binned_m1y, binned_m1yerr = binner(m1info[:,9],m1y,m1xedges)
		binned_m1x = binned_m1x[m1H>0]; binned_m1y = binned_m1y[m1H>0]
		binned_m1xerr = binned_m1xerr[m1H>0]; binned_m1yerr = binned_m1yerr[m1H>0]		
	(_,caps,_) = ax.errorbar(binned_m1x,binned_m1y,yerr=binned_m1yerr,fmt=pointstyles[0],markersize=8,\
		elinewidth=1.5,markeredgecolor='none',color=colours[0],label='Main1')
	for cap in caps: cap.set_markeredgewidth(2)
	(_,caps,_) = ax.errorbar(binned_m2x,binned_m2y,yerr=binned_m2yerr,fmt=pointstyles[1],markersize=9,\
		elinewidth=1.5,markeredgecolor='none',color=colours[1],label='Main2')
	for cap in caps: cap.set_markeredgewidth(2)
	(_,caps,_) = ax.errorbar(binned_lzx,binned_lzy,yerr=binned_lzyerr,fmt=pointstyles[4],markersize=8,\
		elinewidth=1.5,markeredgecolor='none',color=colours[2],label='LOWZ')
	for cap in caps: cap.set_markeredgewidth(2)
	(_,caps,_) = ax.errorbar(binned_cmx,binned_cmy,yerr=binned_cmyerr,fmt=pointstyles[3],markersize=7,\
		elinewidth=1.5,markeredgecolor='none',color=colours[3],label='CMASS')
	for cap in caps: cap.set_markeredgewidth(2)
	ax.set_xlabel(r'$\mathrm{Density\;ratio,\;}r$',fontsize=26,fontweight='extra bold')
	if not useQ: ax.set_xlim([1,2.4])
	ax.tick_params(axis='both', labelsize=16)	
	ax.axhline(0,linestyle=':',color='k')

	plt.subplots_adjust(wspace=0.1)

	plt.savefig(figdir+'BigMDPl/Phi_observables.pdf',bbox_inches='tight')
	
def Delta_navg_slope(cols=[3,9],samples=['Main1','Main2','LOWZ','CMASS'],nbins=15,vType='Isolated',scalePhi=False,useQ=False,useRoot=True):
	
	colours = kelly_RdYlGn[[0,1,4,7]]
	points = pointstyles[[0,1,3,4]]
	msizes = np.array([8,8,8,7])
	plt.figure(figsize=(8,10))
	ind = 0
	for sample in samples:
		metrics = np.loadtxt(structdir+'BigMDPl/'+sample+'/'+vType+'Voids_metrics.txt',skiprows=1)
		if useRoot:
			root = np.loadtxt(structdir+'BigMDPl/'+sample+'/'+vType+'Voids_rootIDs.txt',skiprows=1)
			metrics = metrics[np.in1d(metrics[:,0],root)]			
		#convert normalised densities to density contrasts
		metrics[:,2] -= 1
		metrics[:,3] -= 1
		metrics[:,4] -= 1
		if cols[0]==3:
			metrics = metrics[(metrics[:,3]<0.6)]
		if scalePhi: metrics[:,3] = metrics[:,3]*metrics[:,1]**1.2

		if useQ:
			divs = np.arange(nbins+1).astype(float)/nbins
			xedges = mquantiles(metrics[:,cols[0]],divs)
			xedges, binned_x, binned_xerr = binner(metrics[:,cols[0]],metrics[:,cols[0]],xedges)
			xedges, binned_y, binned_yerr = binner(metrics[:,cols[0]],metrics[:,cols[1]],xedges)
		else:
			H, xedges = np.histogram(metrics[:,cols[0]],nbins)	
			xedges, binned_x, binned_xerr = binner(metrics[:,cols[0]],metrics[:,cols[0]],xedges)
			xedges, binned_y, binned_yerr = binner(metrics[:,cols[0]],metrics[:,cols[1]],xedges)
			binned_x = binned_x[H>20]
			binned_y = binned_y[H>20]
			binned_xerr = binned_xerr[H>20]
			binned_yerr = binned_yerr[H>20]			
		
		slope, intercept, r, p, se = linregress(metrics[:,cols[0]],metrics[:,cols[1]]) 
		print "%s:\n\tslope: %0.4f\tintercept: %0.4f\trsq: %0.4f" %(sample,slope,intercept,r**2)
		
#		plt.scatter(metrics[:,cols[0]],metrics[:,cols[1]],s=1,color=colours[ind],edgecolor='none',alpha=0.5)
		(_,caps,_) = plt.errorbar(binned_x,binned_y,yerr=binned_yerr,xerr=binned_xerr,fmt=points[ind],markersize=msizes[ind],\
			elinewidth=1.5,markeredgecolor='none',color=colours[ind],label=sample)
		if cols==[3,9]:
			x = np.linspace(-0.59,0.59)
			plt.plot(x,slope*x+intercept,'--',color=colours[ind])
			
		for cap in caps: cap.set_markeredgewidth(2)
		ind += 1
		
	if cols==[3,9]:
#		plt.xlim([-0.6,0.6])
#		plt.ylim([-0.2,0.2])
		ax = plt.gca()
		ax.axhline(0,linestyle=':',color='k')
		ax.axvline(0,linestyle=':',color='k')
		plt.xlabel(r'$\bar\delta_v$',fontsize=24,fontweight='extra bold')
		plt.ylabel(r'$\Delta_{3R_v}$',fontsize=24,fontweight='extra bold')
		plt.legend(loc='upper left',numpoints=1,prop={'size':16},borderpad=0.5)
	elif cols==[3,5]:
#		plt.xlim([-0.4,0.6])
		plt.xlabel(r'$\bar\delta_v$',fontsize=24,fontweight='extra bold')
		if scalePhi:
#			plt.ylim([-0.1,0.1])
			plt.ylabel(r'$\Phi_c/R_v\,\,[10^{-5}h\mathrm{Mpc}^{-1}]$',fontsize=24,fontweight='extra bold')
		else:
			plt.ylim([-2,4])
			plt.ylabel(r'$\Phi_c\times10^5$',fontsize=24,fontweight='extra bold')			
		plt.legend(loc='upper right',numpoints=1,prop={'size':16},borderpad=0.5)
		
	ax = plt.gca(); prettify_plot(ax)
#	plt.savefig(figdir+'BigMDPl/figure.pdf',bbox_inches='tight')
	
#	elif cols[0]==2:
#		plt.xlim([-1.0,-0.3])
#		plt.xlabel(r'$\delta_{n,\mathrm{min}}$',fontsize=24,fontweight='extra bold')
#	elif cols[0]==1:
#		plt.xlim([0,150])
#		plt.xlabel(r'$R_v$',fontsize=24,fontweight='extra bold')
#		
#	if cols[1]==9:
#	elif cols[1]==5:
#		if scalePhi:
#			plt.ylim([-0.1,0.1])
#			plt.ylabel(r'$\Phi_c/R_v\,\,[10^{-5}h\mathrm{Mpc}^{-1}]$',fontsize=24,fontweight='extra bold')
#		else:
#			plt.ylim([-3,4])
#			plt.ylabel(r'$\Phi_c\times10^5$',fontsize=24,fontweight='extra bold')
#		plt.legend(loc='upper right',numpoints=1,prop={'size':16},borderpad=0.5)
		
#	plt.figure(figsize=(5,5))
#	res = intercept + slope * metrics[:,cols[0]] - metrics[:,cols[1]]
#	#plt.scatter(metrics[:,cols[0]],res,color=kelly_RdYlGn[0],edgecolor='none')
#	plt.hist2d(metrics[:,1],res,bins=30,cmap='viridis',norm=LogNorm())
#	bias = np.mean(res[metrics[:,1]>100])/(np.sqrt(len(res[metrics[:,1]>100]))*np.std(res[metrics[:,1]>100]))
#	print bias

def Phi_scaling(Rpower,samples=['Main1','Main2','LOWZ','CMASS'],nbins=15,vType='Isolated',useQ=True,useRoot=True):
	
#	colours = kelly_RdYlGn[[0,1,4,7]]
	colours = np.array(['#3130ff',wad_colours[4],kelly_RdYlGn[3],kelly_RdYlGn[7]])
	points = pointstyles[[0,1,4,3]]
	msizes = np.array([8,9,8,7])
	plt.figure(figsize=(10,8))
	ind = 0
	for sample in samples:
		metrics = np.loadtxt(structdir+'BigMDPl/'+sample+'/'+vType+'Voids_metrics.txt',skiprows=1)
		if useRoot:
			root = np.loadtxt(structdir+'BigMDPl/'+sample+'/'+vType+'Voids_rootIDs.txt',skiprows=1)
			metrics = metrics[np.in1d(metrics[:,0],root)]			
		#convert normalised densities to density contrasts
		metrics[:,3] -= 1
		metrics[:,1] = metrics[:,1]**Rpower
		#metrics = metrics[(metrics[:,3]<0.6)]

		if useQ:
			divs = np.arange(nbins+1).astype(float)/nbins
			xedges = mquantiles(metrics[:,3]*metrics[:,1],divs)
			xedges, binned_x, binned_xerr = binner(metrics[:,3]*metrics[:,1],metrics[:,3]*metrics[:,1],xedges)
			xedges, binned_y, binned_yerr = binner(metrics[:,3]*metrics[:,1],metrics[:,5],xedges)
		else:
			H, xedges = np.histogram(metrics[:,3]*metrics[:,1],nbins)	
			xedges, binned_x, binned_xerr = binner(metrics[:,3]*metrics[:,1],metrics[:,3]*metrics[:,1],xedges)
			xedges, binned_y, binned_yerr = binner(metrics[:,3]*metrics[:,1],metrics[:,5],xedges)
			binned_x = binned_x[H>20]
			binned_y = binned_y[H>20]
			binned_xerr = binned_xerr[H>20]
			binned_yerr = binned_yerr[H>20]			
		
		slope, intercept, r, p, se = linregress(metrics[:,3]*metrics[:,1],metrics[:,5]) 
		popt, pcov = curve_fit(linf,binned_x,binned_y,sigma=binned_yerr,absolute_sigma=True)
		print popt, np.sqrt(np.diag(pcov))
		print "%s:\n\tslope: %0.4f\tintercept: %0.4f\trsq: %0.4f" %(sample,slope,intercept,r**2)
		
		x = np.linspace(1.2*binned_x[0],1.1*binned_x[-1])
		plt.plot(x,linf(x,*popt),c=colours[ind],ls='--')
		(_,caps,_) = plt.errorbar(binned_x,binned_y,yerr=binned_yerr,fmt=points[ind],markersize=msizes[ind],\
			elinewidth=1.5,markeredgecolor='none',color=colours[ind],label=sample)			
		for cap in caps: cap.set_markeredgewidth(2)
		ind += 1
		
		plt.xlabel(r'$\lambda_v=\bar\delta_g \left(\frac{R_v}{h^{-1}\mathrm{Mpc}}\right)^{%0.1f}$'%Rpower,fontsize=26,fontweight='extra bold')
		plt.ylim([-2,4])
		plt.ylabel(r'$\Phi_0\times10^5$',fontsize=26,fontweight='extra bold')			
		plt.legend(loc='upper right',numpoints=1,prop={'size':18},borderpad=0.5)
		plt.axhline(0,c='k',ls=':')
		plt.axvline(0,c='k',ls=':')
		
	plt.tick_params(labelsize=18)
	#plt.savefig(figdir+'BigMDPl/Phi_scaling.pdf',bbox_inches='tight')
	
def BigMDPl_void_sizes():
	
	m1 = np.loadtxt(structdir+'BigMDPl/Main1/IsolatedVoids_metrics.txt',skiprows=1)
	m2 = np.loadtxt(structdir+'BigMDPl/Main2/IsolatedVoids_metrics.txt',skiprows=1)
	lz = np.loadtxt(structdir+'BigMDPl/LOWZ/IsolatedVoids_metrics.txt',skiprows=1)
	cm = np.loadtxt(structdir+'BigMDPl/CMASS/IsolatedVoids_metrics.txt',skiprows=1)

#	colours = kelly_RdYlGn[[0,1,3,7]]
	colours = np.array(['#3130ff',wad_colours[4],kelly_RdYlGn[3],kelly_RdYlGn[7]])
	
	m1_lam = (m1[:,3]-1)*m1[:,1]**1.2
	m2_lam = (m2[:,3]-1)*m2[:,1]**1.2
	lz_lam = (lz[:,3]-1)*lz[:,1]**1.2
	cm_lam = (cm[:,3]-1)*cm[:,1]**1.2
	
	plt.figure(figsize=(10,8))
	x = np.linspace(np.min(m1[:,1]),np.max(cm[:,1]))
	k = gaussian_kde(m1[:,1])
	plt.plot(x,k(x),c=colours[0],lw=1.5,label='Main1')
	k = gaussian_kde(m2[:,1])
	plt.plot(x,k(x),c=colours[1],lw=2,ls=(20,[6,6]),label='Main2')
	k = gaussian_kde(lz[:,1])
	plt.plot(x,k(x),c=colours[2],lw=2,ls=(20,[6,3,3,3]),label='LOWZ')
	k = gaussian_kde(cm[:,1])
	plt.plot(x,k(x),c=colours[3],lw=2,ls=(20,[6,2,2,2,2,2]),label='CMASS')
	plt.tick_params(labelsize=18)
	plt.yticks([])
	plt.xlim([4,110])
	plt.xticks([5,10,20,30,40,50,60,70,80,90,100])
	plt.legend(loc='upper right',numpoints=1,prop={'size':16},borderpad=0.5)
	plt.xlabel(r'$R_v\;[h^{-1}\mathrm{Mpc}]$',fontsize=26)
	plt.ylabel(r'$P\,(R_v)$',fontsize=26)
	
#	plt.subplot(1,2,2)
#	x = np.linspace(-70,90)
#	k = gaussian_kde(m1_lam,1)
#	plt.plot(x,k(x),c=colours[0],lw=2,label='Main1')
#	k = gaussian_kde(m2_lam,1)
#	plt.plot(x,k(x),c=colours[1],lw=2,ls=(20,[6,6]),label='Main2')
#	k = gaussian_kde(lz_lam,1)
#	plt.plot(x,k(x),c=kelly_RdYlGn[4],lw=2,ls=(20,[6,3,3,3]),label='LOWZ')
#	k = gaussian_kde(cm_lam,1)
#	plt.plot(x,k(x),c=kelly_RdYlGn[7],lw=2,ls=(20,[6,2,2,2,2,2]),label='CMASS')
#	plt.yticks([])
#	plt.xlim([-70,90])
#	plt.tick_params(labelsize=18)
#	plt.axvline(0,c='k',ls=':')
#	plt.xlabel(r'$\lambda_v = \bar\delta_g\left(\frac{R_v}{h^{-1}\mathrm{Mpc}}\right)^{1.2}$',fontsize=26)
#	plt.ylabel(r'$P\,(\lambda_v)$',fontsize=26)
#	plt.tight_layout(w_pad=2)
	plt.savefig(figdir+'BigMDPl/void_distributions.pdf',bbox_inches='tight')
	
		
def size_dist_w_inset(sample='LOWZ',vType='Isolated',sigma=2):
		
	def choose_label(argument):
		switcher = {
		1: '68% c.l.',
		2: '95% c.l.',
		3: '99.7% c.l.',
		}
		return switcher.get(argument, "nothing")
	band = '%dsigma/' %sigma
	xmax = 110 if vType=='Isolated' else 230
	plt.figure(figsize=(13,8))
	plt.yscale('log',nonposy='clip')
	plt.xscale('log')
	
	if sample=='LOWZ':
		MDR1data = np.loadtxt(structdir+'MDR1/MDR1_DM_LOWZ/distributions/MDR1_DM_LOWZ_'+vType+'_nR.txt')
		MDR1data = MDR1data[MDR1data[:,1]>0]	#drop bins which have no voids (wouldn't display anyway)
		plt.plot(MDR1data[:,0],MDR1data[:,1],'k--',color=kelly_RdYlGn[7],linewidth=2,label='naive subsampling')

	BigMDPldata = np.loadtxt(structdir+'BigMDPl/distributions/'+sample+'_'+vType+'_nR.txt')
	BigMDPldata = BigMDPldata[BigMDPldata[:,1]>0]	#drop bins which have no voids (wouldn't display anyway)
	plt.fill_between(BigMDPldata[:,0],BigMDPldata[:,1]-sigma*BigMDPldata[:,2],BigMDPldata[:,1]+sigma*BigMDPldata[:,2],\
		color=kelly_RdYlGn[0],alpha=0.7)
	plt.plot([],[],color=kelly_RdYlGn[0],alpha=0.7,linewidth=10,label='BigMD mocks '+choose_label(sigma))

	mockdata = np.loadtxt(structdir+'SDSS_DR11/mocks/'+band+sample+'_'+vType+'_nR.txt')
	mockdata = mockdata[mockdata[:,3]>0]	#drop the ones where the upper limit is also zero
	plt.fill_between(mockdata[:,0],mockdata[:,2],mockdata[:,3],color='b',alpha=0.5)
	plt.plot([],[],'b',alpha=0.5,linewidth=10,label='PATCHY mocks '+choose_label(sigma))

	dr11data = np.loadtxt(structdir+'SDSS_DR11/distributions/'+sample+'_'+vType+'_nR.txt')
	dr11data = dr11data[dr11data[:,1]>0]	#drop bins which have no voids (wouldn't display anyway)
	(_, caps, _) = plt.errorbar(dr11data[:,0],dr11data[:,1],yerr=dr11data[:,2],color='k',fmt=pointstyles[0],elinewidth=2,
		markersize=8,markeredgecolor='none',label='DR11 '+sample)
	for cap in caps: cap.set_markeredgewidth(2)
#	plt.scatter(dr11data[:,0],dr11data[:,1],color='k',marker=pointstyles[0],s=50,edgecolor='none',label='DR11 '+sample)
#				
	plt.xlim([8,xmax])
	plt.ylim([1e-10,3e-5])
# 	ax = plt.gca(); ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
	plt.tick_params(axis='both', labelsize=16)

	if vType=='Minimal': plt.legend(loc='upper right',numpoints=1,prop={'size':16},borderpad=0.5)
	plt.legend(loc='upper right',numpoints=1,prop={'size':16},borderpad=0.5)
	plt.xlabel(r'$R_v\,[h^{-1}\mathrm{Mpc}]$',fontsize=24,fontweight='extra bold')
	plt.ylabel(r'$n(>R_v)\,\,[h^3\mathrm{Mpc}^{-3}]$',fontsize=24,fontweight='extra bold')
	if vType=='Isolated':
		plt.xticks([10,20,50,100],[10,20,50,100])
	else:
		plt.xticks([10,20,50,100,200],[10,20,50,100,200])
		
	#----------------#
	#now add the inset
	plt.axes([0.22,0.22,0.32,0.32])
	
	if sample=='LOWZ':
		MDR1data = np.loadtxt(structdir+'MDR1/MDR1_DM_LOWZ/distributions/MDR1_DM_LOWZ_'+vType+'_dNdR.txt')
		MDR1data = MDR1data[MDR1data[:,1]>0]	#drop bins which have no voids (wouldn't display anyway)
		plt.plot(MDR1data[:,0],MDR1data[:,1],'k--',color=kelly_RdYlGn[7],linewidth=2)

	BigMDPldata = np.loadtxt(structdir+'BigMDPl/distributions/'+sample+'_'+vType+'_dNdR.txt')
	BigMDPldata = BigMDPldata[BigMDPldata[:,1]>0]	#drop bins which have no voids (wouldn't display anyway)
	plt.fill_between(BigMDPldata[:,0],BigMDPldata[:,1]-sigma*BigMDPldata[:,2],BigMDPldata[:,1]+sigma*BigMDPldata[:,2],
		color=kelly_RdYlGn[0],alpha=0.7)

	mockdata = np.loadtxt(structdir+'SDSS_DR11/mocks/'+band+sample+'_'+vType+'_dNdR.txt')
	mockdata = mockdata[mockdata[:,3]>0]	#drop the ones where the upper limit is also zero
	plt.fill_between(mockdata[:,0],mockdata[:,2],mockdata[:,3],color='b',alpha=0.5)

	dr11data = np.loadtxt(structdir+'SDSS_DR11/distributions/'+sample+'_'+vType+'_dNdR.txt')
	dr11data = dr11data[dr11data[:,1]>0]	#drop bins which have no voids (wouldn't display anyway)
	(_, caps, _) = plt.errorbar(dr11data[:,0],dr11data[:,1],yerr=dr11data[:,2],color='k',fmt=pointstyles[0],elinewidth=2,
		markersize=7,markeredgecolor='none',label='DR11 '+sample)
	for cap in caps: cap.set_markeredgewidth(2)

	plt.yscale('log',nonposy='clip')
	plt.xlim([8,60]); plt.ylim([5e-10,2e-7])
	plt.tick_params(axis='both', labelsize=12)
	plt.ylabel(r'$\mathcal{N}=dn/dR_v\,\,[h^4\mathrm{Mpc}^{-4}]$',fontsize=18,fontweight='extra bold')
	plt.xlabel(r'$R_v\,[h^{-1}\mathrm{Mpc}]$',fontsize=18,fontweight='extra bold')

	figname = figdir+'DR11/'+sample+'_'+vType+'_sizes_11x8.pdf'
	plt.savefig(figname,bbox_inches='tight')

def density_constraints(sigma=2,suffix='dNddelta'):
	
	def choose_label(argument):
		switcher = {
		1: '68% c.l.',
		2: '95% c.l.',
		3: '99.7% c.l.',
		}
		return switcher.get(argument, "nothing")
	band = '%dsigma/' %sigma

	fig = plt.figure(figsize=(12,8))
	lzmockdata = np.loadtxt(structdir+'SDSS_DR11/mocks/'+band+'LOWZ_Isolated_'+suffix+'.txt')
	lzmockdata = lzmockdata[lzmockdata[:,1]>0]
	lzmockdata[:,0] -= 1
	plt.fill_between(lzmockdata[:,0],lzmockdata[:,2]/lzmockdata[:,1],lzmockdata[:,3]/lzmockdata[:,1],color=kelly_RdYlGn[0],alpha=0.7)
	plt.plot([],[],color=kelly_RdYlGn[0],alpha=0.7,linewidth=10,label='LOWZ mocks '+choose_label(sigma))	

	cmmockdata = np.loadtxt(structdir+'SDSS_DR11/mocks/'+band+'CMASS_Isolated_'+suffix+'.txt')
	cmmockdata = cmmockdata[cmmockdata[:,1]>0]
	cmmockdata[:,0] -= 1
	plt.fill_between(cmmockdata[:,0],cmmockdata[:,2]/cmmockdata[:,1],cmmockdata[:,3]/cmmockdata[:,1],color='b',alpha=0.5)
	plt.plot([],[],color='b',alpha=0.5,linewidth=10,label='CMASS mocks '+choose_label(sigma))	

	plt.ylim([0,2]) 
	if 'delta' in suffix:
		plt.xlim([-1,0])
		x = np.linspace(-1,0)
		plt.plot(x,x/x,':k',linewidth=1.5)
	elif 'A' in suffix:
		plt.xlim([-0.35,1.5])
		x = np.linspace(-0.35,2)
		plt.plot(x,x/x,':k',linewidth=1.5)

	cm = np.loadtxt(structdir+'SDSS_DR11/distributions/CMASS_Isolated_'+suffix+'_fewbins.txt')
	lz = np.loadtxt(structdir+'SDSS_DR11/distributions/LOWZ_Isolated_'+suffix+'_fewbins.txt')
	cm = cm[cm[:,1]>0]; lz = lz[lz[:,1]>0]
	cm[:,0] -= 1; lz[:,0] -= 1
	cminterp = interp1d(cmmockdata[:,0],cmmockdata[:,1])
	lzinterp = interp1d(lzmockdata[:,0],lzmockdata[:,1])
	cmmock = cminterp(cm[:,0])
	lzmock = lzinterp(lz[:,0])	
	
#	(_, caps, _) = plt.errorbar(lz[:,0],lz[:,1]/lzmock,yerr=lz[:,2]/lzmock,fmt=pointstyles[0],color='k',
#		elinewidth=2,markersize=8,markeredgecolor='none',label='DR11 LOWZ')
#	for cap in caps: cap.set_markeredgewidth(2)
#	(_, caps, _) = plt.errorbar(cm[:,0],cm[:,1]/cmmock,yerr=cm[:,2]/cmmock,fmt=pointstyles[3],color='k',
#		elinewidth=2,markersize=8,markeredgecolor='none',label='DR11 CMASS')
#	for cap in caps: cap.set_markeredgewidth(2)
	
	plt.plot(lz[:,0],lz[:,1]/lzmock,'k-',lw=1.5,marker=pointstyles[0],markersize=8,label='DR11 LOWZ')
	plt.plot(cm[:,0],cm[:,1]/cmmock,'k--',lw=1.5,marker=pointstyles[3],markersize=7,label='DR11 CMASS')
	
	if suffix=='dNddelta':
		plt.legend(loc='upper center',numpoints=1,prop={'size':16},borderpad=0.5)
		plt.ylabel(r'$\mathcal{N}(\delta_{g,\mathrm{min}})/\mathcal{N}(\delta_{g,\mathrm{min}})|_{\Lambda\mathrm{CDM}}$',fontsize=22,fontweight='extra bold')
		plt.xlabel(r'$\delta_{g,\mathrm{min}}$',fontsize=24,fontweight='extra bold')
	elif suffix=='ndelta':
		plt.legend(loc='upper right',numpoints=1,prop={'size':16},borderpad=0.5)
		plt.ylabel(r'$n(<\delta_{g,\mathrm{min}})/n_{\Lambda\mathrm{CDM}}(<\delta_{g,\mathrm{min}})$',fontsize=22,fontweight='extra bold')
		plt.xlabel(r'$\delta_{g,\mathrm{min}}$',fontsize=24,fontweight='extra bold')
	elif suffix=='dNdA':
		plt.legend(loc='upper center',numpoints=1,prop={'size':16},borderpad=0.5)
		plt.ylabel(r'$\mathcal{N}(\bar{\delta}_g)/\mathcal{N}(\bar{\delta}_g)|_{\Lambda\mathrm{CDM}}$',fontsize=22,fontweight='extra bold')
		plt.xlabel(r'$\bar{\delta}_g$',fontsize=24,fontweight='extra bold')
	elif suffix=='nA':
		plt.legend(loc='upper right',numpoints=1,prop={'size':16},borderpad=0.5)
		plt.ylabel(r'$n(<\bar{\delta}_g)/n_{\Lambda\mathrm{CDM}}(<\bar{\delta}_g)$',fontsize=22,fontweight='extra bold')
		plt.xlabel(r'$\bar{\delta}_g$',fontsize=24,fontweight='extra bold')
		
	ax = plt.gca(); prettify_plot(ax)


	plt.savefig(figdir+'DR11/enhancements_Isolated_'+suffix+'_fewbins.pdf',bbox_inches='tight')
	plt.savefig('/Users/seshadri/Workspace/papers/DR11catalogue/enhancements_Isolated_'+suffix+'_fewbins.pdf',bbox_inches='tight')
	
	
def enhancement_limits(sigma=3,suffix='nR'):
	
	def choose_label(argument):
		switcher = {
		1: '68% c.l.',
		2: '95% c.l.',
		3: '99.7% c.l.',
		}
		return switcher.get(argument, "nothing")
	band = '%dsigma/' %sigma

	fig = plt.figure(figsize=(12,8))
	lzmockdata = np.loadtxt(structdir+'SDSS_DR11/mocks/'+band+'LOWZ_Isolated_'+suffix+'.txt')
	plt.fill_between(lzmockdata[:,0],lzmockdata[:,2]/lzmockdata[:,1],lzmockdata[:,3]/lzmockdata[:,1],color=kelly_RdYlGn[0],alpha=0.7)
	plt.plot([],[],color=kelly_RdYlGn[0],alpha=0.7,linewidth=10,label='LOWZ mocks '+choose_label(sigma))	
	cmmockdata = np.loadtxt(structdir+'SDSS_DR11/mocks/'+band+'CMASS_Isolated_'+suffix+'.txt')
	plt.fill_between(cmmockdata[:,0],cmmockdata[:,2]/cmmockdata[:,1],cmmockdata[:,3]/cmmockdata[:,1],color='b',alpha=0.5)
	plt.plot([],[],color='b',alpha=0.5,linewidth=10,label='CMASS mocks '+choose_label(sigma))	

	plt.ylim([0,2]); plt.xlim([8,110])
	plt.xscale('log')
	plt.xticks([10,20,50,100],[10,20,50,100])
	x = np.linspace(8,110)
	plt.plot(x,x/x,':k',linewidth=1.5)

	cm = np.loadtxt(structdir+'SDSS_DR11/distributions/CMASS_Isolated_'+suffix+'_fewbins.txt')
	lz = np.loadtxt(structdir+'SDSS_DR11/distributions/LOWZ_Isolated_'+suffix+'_fewbins.txt')
	cminterp = interp1d(cmmockdata[:,0],cmmockdata[:,1])
	lzinterp = interp1d(lzmockdata[:,0],lzmockdata[:,1])
	cmmock = cminterp(cm[:,0])
	lzmock = lzinterp(lz[:,0])	
	
	plt.plot(lz[:,0],lz[:,1]/lzmock,'k-',lw=1.5,marker=pointstyles[0],markersize=8,label='DR11 LOWZ')
	plt.plot(cm[:,0],cm[:,1]/cmmock,'k--',lw=1.5,marker=pointstyles[3],markersize=7,label='DR11 CMASS')

#	(_, caps, _) = plt.errorbar(lz[:,0],lz[:,1]/lzmock,yerr=lz[:,2]/lzmock,fmt=pointstyles[0],color='k',
#		elinewidth=2,markersize=8,markeredgecolor='none',label='DR11 LOWZ')
#	for cap in caps: cap.set_markeredgewidth(2)
#	(_, caps, _) = plt.errorbar(cm[:,0],cm[:,1]/cmmock,yerr=cm[:,2]/cmmock,fmt=pointstyles[3],color='k',
#		elinewidth=2,markersize=8,markeredgecolor='none',label='DR11 CMASS')
#	for cap in caps: cap.set_markeredgewidth(2)
		
	if suffix=='dNdR':
		plt.legend(loc='upper center',numpoints=1,prop={'size':16},borderpad=0.5)
		plt.ylabel(r'$\mathcal{N}(R_v)/\mathcal{N}(R_v)|_{\Lambda\mathrm{CDM}}$',fontsize=22,fontweight='extra bold')
	else:
		plt.legend(loc='upper left',numpoints=1,prop={'size':16},borderpad=0.5)
		plt.ylabel(r'$n(>R_v)/n_{\Lambda\mathrm{CDM}}(>R_v)$',fontsize=22,fontweight='extra bold')
	plt.xlabel(r'$R_v\,[h^{-1}\mathrm{Mpc}]$',fontsize=24,fontweight='extra bold')
	ax = plt.gca(); prettify_plot(ax)

	plt.savefig(figdir+'DR11/enhancements_Isolated_'+suffix+'.pdf',bbox_inches='tight')
	plt.savefig('/Users/seshadri/Workspace/papers/DR11catalogue/enhancements_Isolated_'+suffix+'_fewbins.pdf',bbox_inches='tight')

	
#	x1 = np.loadtxt(structdir+'SDSS_DR11/mocks/F4.txt')
#	x2 = np.loadtxt(structdir+'SDSS_DR11/mocks/F5.txt')
#	x3 = np.loadtxt(structdir+'SDSS_DR11/mocks/F6.txt')
#	plt.plot(x1[:,0],x1[:,1],color=kelly_RdYlGn[7],linewidth=2)
#	plt.plot(x2[:,0],x2[:,1],color=kelly_RdYlGn[6],linewidth=2)
#	plt.plot(x3[:,0],x3[:,1],color=kelly_RdYlGn[4],linewidth=2)
##	w1 = np.loadtxt(structdir+'SDSS_DR11/mocks/enhancementw0.6.txt')
##	w2 = np.loadtxt(structdir+'SDSS_DR11/mocks/enhancementw1.4.txt')
#	w1 = np.loadtxt(structdir+'SDSS_DR11/mocks/wa-0.2.txt')
#	w2 = np.loadtxt(structdir+'SDSS_DR11/mocks/wa+0.2.txt')
#	plt.plot(w1[:,0],w1[:,1],color=kelly_RdYlGn[0],linewidth=2)
#	plt.plot(w2[:,0],w2[:,1],color=kelly_RdYlGn[1],linewidth=2)
#		
#	ax = fig.add_axes([0.1,0.1,0.2,0.4])
#	ax.set_axis_off()
#	ax.text(0.6, 0.8, 'F4',
#        horizontalalignment='left',
#        verticalalignment='center',
#        fontsize=14, color=kelly_RdYlGn[7])
#	ax.text(0.6, 0.7, 'F5',
#        horizontalalignment='left',
#        verticalalignment='center',
#        fontsize=14, color=kelly_RdYlGn[6])
#	ax.text(0.6, 0.6, 'F6',
#        horizontalalignment='left',
#        verticalalignment='center',
#        fontsize=14, color=kelly_RdYlGn[5])
#	ax.text(0.6, 0.5, '$w_a=-0.2$',
#        horizontalalignment='left',
#        verticalalignment='center',
#        fontsize=15, color=kelly_RdYlGn[1])
#	ax.text(0.6, 0.4, '$w_a=+0.2$',
#        horizontalalignment='left',
#        verticalalignment='center',
#        fontsize=15, color=kelly_RdYlGn[0])
#
#	plt.savefig(figdir+'DR11/alt_enhancements_Isolated.pdf',bbox_inches='tight')


def deltaRhist(sample='LOWZ',vType='Isolated',nbins=40):
	
	north = np.loadtxt(structdir+'SDSS_DR11/'+sample+'_North/'+vType+'Voids_info.txt',skiprows=2)
	south = np.loadtxt(structdir+'SDSS_DR11/'+sample+'_South/'+vType+'Voids_info.txt',skiprows=2)
	data = np.vstack([north,south])
	sinfoFileN = structdir+'SDSS_DR11/'+sample+'_North/sample_info.dat'
	parmsN = imp.load_source("name",sinfoFileN)
	sinfoFileS = structdir+'SDSS_DR11/'+sample+'_South/sample_info.dat'
	parmsS = imp.load_source("name",sinfoFileS)
	meanNNsep = max(parmsN.tracerDens,parmsS.tracerDens)**(-1.0/3)

	
	hist, x, y = np.histogram2d(data[:,6],data[:,4]-1,bins=nbins)
	max_scale = np.max(hist)
	#mask the bins which have no voids, for clarity in plotting
	masked_hist = np.ma.masked_where(hist==0,hist)

	#now plot the histogram
	plt.figure(figsize=(12,8))
	plt.pcolormesh(x,y,masked_hist.transpose(),cmap='viridis',norm=LogNorm(),vmax=max_scale)
	cbar = plt.colorbar()
	tick_locs   = [1,10,20,50]
	cbar.locator     = mpl.ticker.FixedLocator(tick_locs)
	cbar.update_ticks()
	cbar.ax.set_yticklabels(['1','10','20','50'],fontsize=16)

#	x = np.linspace(0.6*meanNNsep,1.8*meanNNsep,25)
#	plt.plot(x,(3/(4*np.pi))*(meanNNsep/x)**3-1,'k--')

#	plt.pcolormesh(x,y,masked_hist.transpose(),cmap='viridis')
#	cbar = plt.colorbar()

	plt.tick_params(axis='both', labelsize=16)
	plt.xlabel("$R_v\,[h^{-1}\mathrm{Mpc}]$",fontsize=24,fontweight='extra bold')
	plt.ylabel(r'$\delta_{g,\mathrm{min}}$',fontsize=24,fontweight='extra bold')
	cbar.solids.set_edgecolor("face")
	cbar.ax.tick_params(labelsize=16) 
	cbar.ax.get_yaxis().labelpad=25
	cbar.set_label("$\mathrm{counts}$",fontsize=24,fontweight='extra bold',rotation=270)
	
	plt.savefig(figdir+'DR11/'+sample+'_'+vType+'_deltaRhist.pdf',bbox_inches='tight')

def ellipticity_figs(sample='LOWZ',vType='Isolated',useBC=True):
	
	if useBC:
		north = np.loadtxt(structdir+'SDSS_DR11/'+sample+'_North/barycentres/'+vType+'_baryC_Voids_ellipticities.txt',skiprows=1)
		south = np.loadtxt(structdir+'SDSS_DR11/'+sample+'_South/barycentres/'+vType+'_baryC_Voids_ellipticities.txt',skiprows=1)
		els = np.vstack([north,south])
		north = np.loadtxt(structdir+'SDSS_DR11/'+sample+'_North/barycentres/'+vType+'_baryC_Voids_info.txt',skiprows=2)
		south = np.loadtxt(structdir+'SDSS_DR11/'+sample+'_South/barycentres/'+vType+'_baryC_Voids_info.txt',skiprows=2)
		info = np.vstack([north,south])
	else:
		north = np.loadtxt(structdir+'SDSS_DR11/'+sample+'_North/'+vType+'Voids_ellipticities.txt',skiprows=1)
		south = np.loadtxt(structdir+'SDSS_DR11/'+sample+'_South/'+vType+'Voids_ellipticities.txt',skiprows=1)
		els = np.vstack([north,south])
		north = np.loadtxt(structdir+'SDSS_DR11/'+sample+'_North/'+vType+'Voids_info.txt',skiprows=2)
		south = np.loadtxt(structdir+'SDSS_DR11/'+sample+'_South/'+vType+'Voids_info.txt',skiprows=2)
		info = np.vstack([north,south])		

	north = np.loadtxt(structdir+'SDSS_DR11/'+sample+'_North/'+vType+'Voids_list.txt',skiprows=2)
	south = np.loadtxt(structdir+'SDSS_DR11/'+sample+'_South/'+vType+'Voids_list.txt',skiprows=2)
	list_info = np.vstack([north,south])		
		
#	fig = plt.figure(figsize=(10,6.67))
#	hist, x, y = np.histogram2d(els[:,5],els[:,6],bins=30)
#	masked_hist = np.ma.masked_where(hist==0,hist)
#	plt.pcolormesh(x,y,masked_hist.transpose(),cmap='viridis')#,norm=LogNorm())
#	x = np.linspace(0,0.25)
#	plt.plot(x,x,'--k'); plt.plot(x,-x,'--k') 
#	x = np.linspace(0,0.35)
#	plt.plot(x,0*x,'k')
#	plt.xlim([0,0.35]); plt.ylim([-0.25,0.25])
#	plt.tick_params(axis='both', labelsize=16)
#	plt.xlabel('ellipticity $e$',fontsize=22)
#	plt.ylabel('prolateness $p$',fontsize=22)
#	cbar = plt.colorbar()
#	cbar.ax.tick_params(labelsize=16) 
#	cbar.ax.get_yaxis().labelpad=25
#	cbar.set_label("$\mathrm{counts}$",fontsize=24,fontweight='extra bold',rotation=270)	
#	ax = fig.add_axes([0.6,0.3,0.2,0.4])
#	ax.set_axis_off()
#	ax.text(0.2, 0.6, 'oblate',
#        horizontalalignment='left',
#        verticalalignment='bottom',
#        fontsize=14, color='k')
#	ax.text(0.2, 0.4, 'prolate',
#        horizontalalignment='left',
#        verticalalignment='bottom',
#        fontsize=14, color='k')	
##	plt.savefig(figdir+'DR11/'+sample+'_'+vType+'_e-and-p.pdf',bbox_inches='tight')
#	
#	fig = plt.figure(figsize=(10,6.67))
#	junk = plt.hist(els[info[:,8]==0,5],bins=25,histtype='step',normed=True,color='b',linewidth=2)
#	junk = plt.hist(els[info[:,8]==1,5],bins=25,histtype='step',normed=True,linestyle='--',color=kelly_RdYlGn[0],linewidth=2)
#	plt.plot([],[],color='b',linewidth=2,label='central voids')
#	plt.plot([],[],color=kelly_RdYlGn[0],linewidth=2,linestyle='--',label='edge voids')
#	plt.tick_params(axis='both', labelsize=16)
#	plt.xlabel('ellipticity $e$',fontsize=22)
#	plt.ylabel('PDF',fontsize=22)
#	plt.legend(loc='upper right',numpoints=1,prop={'size':16},borderpad=0.5)
##	plt.savefig(figdir+'DR11/'+sample+'_'+vType+'_e-dist.pdf',bbox_inches='tight')
#	
#	fig = plt.figure(figsize=(10,6.67))
#	junk = plt.hist(els[info[:,8]==0,6],bins=25,histtype='step',normed=True,color='b',linewidth=2)
#	junk = plt.hist(els[info[:,8]==1,6],bins=25,histtype='step',normed=True,linestyle='--',color=kelly_RdYlGn[0],linewidth=2)
#	plt.plot([],[],color='b',linewidth=2,label='central voids')
#	plt.plot([],[],color=kelly_RdYlGn[0],linewidth=2,linestyle='--',label='edge voids')
#	plt.tick_params(axis='both', labelsize=16)
#	plt.xlabel('prolateness $p$',fontsize=22)
#	plt.ylabel('PDF',fontsize=22)
#	plt.legend(loc='upper right',numpoints=1,prop={'size':16},borderpad=0.5)
##	plt.savefig(figdir+'DR11/'+sample+'_'+vType+'_p-dist.pdf',bbox_inches='tight')
	
	plt.figure(figsize=(10,6.67))
	els = els[info[:,5]<1.1]
	info = info[info[:,5]<1.1]
	xedges = [0,0.8,0.9,1.0,1.1]
	h, xedges = np.histogram(info[:,5],bins=xedges)
	xedges, y, yerr = binner(info[:,5],els[:,1],xedges)
	xedges, x, xerr = binner(info[:,5],info[:,5],xedges)
	(_, caps, _) = plt.errorbar(x,y,yerr,xerr,fmt=pointstyles[0],color=kelly_RdYlBu[1],
		elinewidth=2,markersize=8,markeredgecolor='none')

def ISW_mf_plot(cmbmap='smica',galaxy_sample='CMASS'):

	def ampl(x,A):
		return A*x
	if galaxy_sample=='CMASS': 
		nbins = 8 
	elif galaxy_sample=='LOWZ': nbins=6
	output = np.zeros((2*nbins,3))

	void_data_temps = np.loadtxt('/Users/seshadri/Workspace/structures/ISW_analysis/CMASS_DR12Voids_MF_'+cmbmap+'_masked2.txt')
	clust_data_temps = np.loadtxt('/Users/seshadri/Workspace/structures/ISW_analysis/CMASS_DR12Clusters_MF_'+cmbmap+'_masked2.txt')
	rand_temps = np.loadtxt('/Users/seshadri/Workspace/structures/ISW_analysis/CMASS_DR12_Random2000_masked_MF_temps.txt')
	
	mock_means = np.mean(rand_temps.T[1:,:],axis=0)
	output[:nbins,0] = void_data_temps[:,3]
	output[:nbins,1] = void_data_temps[:,4]-mock_means[:nbins]
	output[nbins:,0] = clust_data_temps[:,3]
	output[nbins:,1] = clust_data_temps[:,4]-mock_means[nbins:]
	cmbvec = np.zeros((rand_temps.shape[1]-1,rand_temps.shape[0]))
	cmbvec = rand_temps.T[1:,:] - mock_means
	cmbQ = np.dot(cmbvec.T,cmbvec)/(cmbvec.shape[0]-1)
	X = np.transpose(np.mat(output[:,0]))
	Y = np.transpose(np.mat(output[:,1]))
	Aisw = np.array(np.linalg.inv(X.T*np.linalg.inv(cmbQ)*X)*(X.T*np.linalg.inv(cmbQ)*Y))[0,0]
	Aisw_err = np.array(np.sqrt(np.linalg.inv(X.T*np.linalg.inv(cmbQ)*X)))[0,0]
	D = np.sqrt(np.diag(cmbQ))
	cmbcorr = np.fromfunction(lambda i, j: cmbQ[i,j]/(D[i]*D[j]), (2*nbins, 2*nbins), dtype=int)
	
	print "A_ISW = %0.3f +/- %0.3f (%0.2f sigma)" %(Aisw,Aisw_err, Aisw/Aisw_err)	

	plt.figure(figsize=(20,8))
	plt.subplot(1,2,1)
	(_,caps,_) = plt.errorbar(void_data_temps[:,3],void_data_temps[:,4]-mock_means[:nbins],yerr=D[:nbins],\
			fmt='o',markersize=8,color='b',elinewidth=1.5,markeredgecolor='none')
	for cap in caps: cap.set_markeredgewidth(2)
	(_,caps,_) = plt.errorbar(clust_data_temps[:,3],clust_data_temps[:,4]-mock_means[nbins:],yerr=D[nbins:],\
			fmt='s',markersize=8,color='g',elinewidth=1.5,markeredgecolor='none')
	for cap in caps: cap.set_markeredgewidth(2)
	x = np.linspace(-6,5) 
	plt.plot(x,x,c='k',ls='--')
	plt.plot(x,Aisw*x,c='r')
	plt.axhline(0,c='k',ls=':')
	plt.axvline(0,c='k',ls=':')
	plt.xlim([-6,5])
	plt.tick_params(labelsize=16)
	plt.xlabel(r'$T_0^\mathrm{expected}\;[\mu\mathrm{K}]$',fontsize=24)
	plt.ylabel(r'$T_0^\mathrm{measured}\;[\mu\mathrm{K}]$',fontsize=24)
#	plt.savefig(figdir+'ISWproject/CMASS-DR12-mf-'+cmbmap+'.pdf',bbox_inches='tight')
	
#	void_data_temps = np.loadtxt('/Users/seshadri/Workspace/structures/ISW_analysis/CMASS_DR12Voids_MF_sevem.txt')
#	clust_data_temps = np.loadtxt('/Users/seshadri/Workspace/structures/ISW_analysis/CMASS_DR12Clusters_MF_sevem.txt')
#	mock_means = np.mean(rand_temps.T[1:,:],axis=0)
#	output[:nbins,0] = void_data_temps[:,3]
#	output[:nbins,1] = void_data_temps[:,4]-mock_means[:nbins]
#	output[nbins:,0] = clust_data_temps[:,3]
#	output[nbins:,1] = clust_data_temps[:,4]-mock_means[nbins:]
#	cmbvec = np.zeros((rand_temps.shape[1]-1,rand_temps.shape[0]))
#	cmbvec = rand_temps.T[1:,:] - mock_means
#	cmbQ = np.dot(cmbvec.T,cmbvec)/(cmbvec.shape[0]-1)
#	X = np.transpose(np.mat(output[:,0]))
#	Y = np.transpose(np.mat(output[:,1]))
#	Aisw = np.array(np.linalg.inv(X.T*np.linalg.inv(cmbQ)*X)*(X.T*np.linalg.inv(cmbQ)*Y))[0,0]
#	Aisw_err = np.array(np.sqrt(np.linalg.inv(X.T*np.linalg.inv(cmbQ)*X)))[0,0]
#	D = np.sqrt(np.diag(cmbQ))
#	(_,caps,_) = plt.errorbar(void_data_temps[:,3]+0.1,void_data_temps[:,4]-mock_means[:nbins],yerr=D[:nbins],\
#			fmt='o',markersize=8,color='b',elinewidth=1.5,markerfacecolor='none',markeredgecolor='b')
#	for cap in caps: cap.set_markeredgewidth(2)
#	(_,caps,_) = plt.errorbar(clust_data_temps[:,3]+0.1,clust_data_temps[:,4]-mock_means[nbins:],yerr=D[nbins:],\
#			fmt='s',markersize=8,color='g',elinewidth=1.5,markerfacecolor='none',markeredgecolor='g')
#	for cap in caps: cap.set_markeredgewidth(2)


#	plt.figure(figsize=(10,8))
	plt.subplot(1,2,2)
	cmap = ListedColormap(np.loadtxt("/Users/seshadri/Workspace/colormaps/Planck_Parchment_RGB.txt")/255.)
	pcol = plt.pcolormesh(cmbcorr[-1::-1,:],cmap=cmap,vmax=1,vmin=0)
	pcol.set_edgecolor("face")
	cbar = plt.colorbar()
	cbar.ax.tick_params(labelsize=16)
	plt.tick_params(axis='both',which='both',bottom='off',top='off',left='off',right='off',labelbottom='off',labelleft='off')
	plt.tight_layout(w_pad=10)
#	plt.savefig('/Users/seshadri/Workspace/papers/ISWdetection/fig3.pdf',bbox_inches='tight')
		
def ISW_cth_plot(cmbmap='smica',galaxy_sample='CMASS'):

	def ampl(x,A):
		return A*x
	if galaxy_sample=='CMASS': 
		nbins = 8 
	elif galaxy_sample=='LOWZ': nbins=6
	output = np.zeros((2*nbins,3))

	void_data_temps = np.loadtxt('/Users/seshadri/Workspace/structures/ISW_analysis/CMASS_DR12Voids_CTH2_'+cmbmap+'_masked.txt')
	clust_data_temps = np.loadtxt('/Users/seshadri/Workspace/structures/ISW_analysis/CMASS_DR12Clusters_CTH2_'+cmbmap+'_masked.txt')
	rand_temps = np.loadtxt('/Users/seshadri/Workspace/structures/ISW_analysis/CMASS_DR12_Random500_CTH_temps.txt')
	
	mock_means = np.mean(rand_temps.T[1:,:],axis=0)
	output[:nbins,0] = void_data_temps[:,1]
	output[:nbins,1] = void_data_temps[:,2]-mock_means[:nbins]
	output[nbins:,0] = clust_data_temps[:,1]
	output[nbins:,1] = clust_data_temps[:,2]-mock_means[nbins:]
	cmbvec = np.zeros((rand_temps.shape[1]-1,rand_temps.shape[0]))
	cmbvec = rand_temps.T[1:,:] - mock_means
	cmbQ = np.dot(cmbvec.T,cmbvec)/(cmbvec.shape[0]-1)
	X = np.transpose(np.mat(output[:,0]))
	Y = np.transpose(np.mat(output[:,1]))
	Aisw = np.array(np.linalg.inv(X.T*np.linalg.inv(cmbQ)*X)*(X.T*np.linalg.inv(cmbQ)*Y))[0,0]
	Aisw_err = np.array(np.sqrt(np.linalg.inv(X.T*np.linalg.inv(cmbQ)*X)))[0,0]
	D = np.sqrt(np.diag(cmbQ))
	cmbcorr = np.fromfunction(lambda i, j: cmbQ[i,j]/(D[i]*D[j]), (16, 16), dtype=int)
	
	print "A_ISW = %0.3f +/- %0.3f (%0.2f sigma)" %(Aisw,Aisw_err, Aisw/Aisw_err)	

	plt.figure(figsize=(20,8))
	plt.subplot(1,2,1)
	(_,caps,_) = plt.errorbar(void_data_temps[:,1],void_data_temps[:,2]-mock_means[:nbins],yerr=D[:nbins],\
			fmt='o',markersize=8,color='b',elinewidth=1.5,markeredgecolor='none')
	for cap in caps: cap.set_markeredgewidth(2)
	(_,caps,_) = plt.errorbar(clust_data_temps[:,1],clust_data_temps[:,2]-mock_means[nbins:],yerr=D[nbins:],\
			fmt='s',markersize=8,color='g',elinewidth=1.5,markeredgecolor='none')
	for cap in caps: cap.set_markeredgewidth(2)
	x = np.linspace(-1.5,1) 
	plt.plot(x,x,c='k',ls='--')
	plt.plot(x,Aisw*x,c='r')
	plt.axhline(0,c='k',ls=':')
	plt.axvline(0,c='k',ls=':')
	plt.xlim([-1.5,1])
	plt.tick_params(labelsize=16)
	plt.xlabel(r'$T_0^\mathrm{expected}\;[\mu\mathrm{K}]$',fontsize=24)
	plt.ylabel(r'$T_0^\mathrm{measured}\;[\mu\mathrm{K}]$',fontsize=24)
#	plt.savefig(figdir+'ISWproject/CMASS-DR12-mf-'+cmbmap+'.pdf',bbox_inches='tight')
	
#	void_data_temps = np.loadtxt('/Users/seshadri/Workspace/structures/ISW_analysis/CMASS_DR12Voids_MF_sevem.txt')
#	clust_data_temps = np.loadtxt('/Users/seshadri/Workspace/structures/ISW_analysis/CMASS_DR12Clusters_MF_sevem.txt')
#	mock_means = np.mean(rand_temps.T[1:,:],axis=0)
#	output[:nbins,0] = void_data_temps[:,3]
#	output[:nbins,1] = void_data_temps[:,4]-mock_means[:nbins]
#	output[nbins:,0] = clust_data_temps[:,3]
#	output[nbins:,1] = clust_data_temps[:,4]-mock_means[nbins:]
#	cmbvec = np.zeros((rand_temps.shape[1]-1,rand_temps.shape[0]))
#	cmbvec = rand_temps.T[1:,:] - mock_means
#	cmbQ = np.dot(cmbvec.T,cmbvec)/(cmbvec.shape[0]-1)
#	X = np.transpose(np.mat(output[:,0]))
#	Y = np.transpose(np.mat(output[:,1]))
#	Aisw = np.array(np.linalg.inv(X.T*np.linalg.inv(cmbQ)*X)*(X.T*np.linalg.inv(cmbQ)*Y))[0,0]
#	Aisw_err = np.array(np.sqrt(np.linalg.inv(X.T*np.linalg.inv(cmbQ)*X)))[0,0]
#	D = np.sqrt(np.diag(cmbQ))
#	(_,caps,_) = plt.errorbar(void_data_temps[:,3]+0.1,void_data_temps[:,4]-mock_means[:nbins],yerr=D[:nbins],\
#			fmt='o',markersize=8,color='b',elinewidth=1.5,markerfacecolor='none',markeredgecolor='b')
#	for cap in caps: cap.set_markeredgewidth(2)
#	(_,caps,_) = plt.errorbar(clust_data_temps[:,3]+0.1,clust_data_temps[:,4]-mock_means[nbins:],yerr=D[nbins:],\
#			fmt='s',markersize=8,color='g',elinewidth=1.5,markerfacecolor='none',markeredgecolor='g')
#	for cap in caps: cap.set_markeredgewidth(2)


	plt.subplot(1,2,2)
	cmap = ListedColormap(np.loadtxt("/Users/seshadri/Workspace/colormaps/Planck_Parchment_RGB.txt")/255.)
	pcol = plt.pcolormesh(cmbcorr[-1::-1,:],cmap=cmap,vmax=1,vmin=0)
	pcol.set_edgecolor("face")
	cbar = plt.colorbar()
	cbar.ax.tick_params(labelsize=16)
	plt.tick_params(axis='both',which='both',bottom='off',top='off',left='off',right='off',labelbottom='off',labelleft='off')
	plt.savefig(figdir+'ISWproject/CMASS-DR12-cth-data.pdf',bbox_inches='tight')

def mf_binned_plot(cmbmap='smica',galaxy_sample='CMASS'):

	def ampl(x,A):
		return A*x
	if galaxy_sample=='CMASS': 
		nbins = 8 
	elif galaxy_sample=='LOWZ': nbins=6
	output = np.zeros((2*nbins,3))
	bias = np.zeros((2*nbins,3))
	
	plt.figure(figsize=(18,8))
	plt.subplot(1,2,1)
	
	filenames = glob.glob('/Users/seshadri/Workspace/structures/QPM_DR12/mf_temp_data_smica/voids/CMASS_Voids_*')
	filenames = np.asarray(filenames)[np.argsort(filenames)]
	mock_void_temps = np.empty((len(filenames),nbins,6))
	data_temps = np.loadtxt('/Users/seshadri/Workspace/structures/QPM_DR12/mf_temp_data_'+cmbmap+'/voids/CMASS_DR12Voids_MF.txt')
	for i in range(len(filenames)):
	    mock_void_temps[i] = np.loadtxt(filenames[i])
	mock_void_mean = np.mean(mock_void_temps[:,:,4],axis=0)
	mock_void_errors = np.empty((mock_void_temps.shape[1],2))
	for i in range(mock_void_temps.shape[1]):
		mock_void_errors[i,0] = mock_void_mean[i] - mquantiles(mock_void_temps[:,i,4],0.1585)
		mock_void_errors[i,1] = mquantiles(mock_void_temps[:,i,4],0.8415) - mock_void_mean[i]	
	(_,caps,_) = plt.errorbar(data_temps[:,3],data_temps[:,4]-mock_void_mean,yerr=[mock_void_errors[:,0],mock_void_errors[:,1]],\
			fmt='o',markersize=8,color='b',elinewidth=1.5,markeredgecolor='none')
	for cap in caps: cap.set_markeredgewidth(2)
	output[:nbins,0] = data_temps[:,3]
	output[:nbins,1] = data_temps[:,4]-mock_void_mean
	output[:nbins,2] = 0.5*(mock_void_errors[:,0]+mock_void_errors[:,1])
	bias[:nbins,0] = np.mean(mock_void_temps[:,:,3],axis=0)
	bias[:nbins,1] = mock_void_mean
	bias[:nbins,2] = np.std(mock_void_temps[:,:,4],axis=0)/np.sqrt(len(filenames))
	
	filenames = glob.glob('/Users/seshadri/Workspace/structures/QPM_DR12/mf_temp_data_smica/clusters/CMASS_Clusters_*')
	filenames = np.asarray(filenames)[np.argsort(filenames)]
	mock_clust_temps = np.empty((len(filenames),nbins,6))
	data_temps = np.loadtxt('/Users/seshadri/Workspace/structures/QPM_DR12/mf_temp_data_'+cmbmap+'/clusters/CMASS_DR12Clusters_MF.txt')
	for i in range(len(filenames)):
	    mock_clust_temps[i] = np.loadtxt(filenames[i])
	mock_clust_mean = np.mean(mock_clust_temps[:,:,4],axis=0)
	mock_clust_errors = np.empty((mock_clust_temps.shape[1],2))
	for i in range(mock_clust_temps.shape[1]):
		mock_clust_errors[i,0] = mock_clust_mean[i] - mquantiles(mock_clust_temps[:,i,4],0.1585)
		mock_clust_errors[i,1] = mquantiles(mock_clust_temps[:,i,4],0.8415) - mock_clust_mean[i]	
	(_,caps,_) = plt.errorbar(data_temps[:,3],data_temps[:,4]-mock_clust_mean,yerr=[mock_clust_errors[:,0],mock_clust_errors[:,1]],\
			fmt='s',markersize=8,color='g',elinewidth=1.5,markeredgecolor='none')
	for cap in caps: cap.set_markeredgewidth(2)
	output[nbins:,0] = data_temps[:,3]
	output[nbins:,1] = data_temps[:,4]-mock_clust_mean
	output[nbins:,2] = 0.5*(mock_clust_errors[:,0]+mock_clust_errors[:,1])
	bias[nbins:,0] = np.mean(mock_clust_temps[:,:,3],axis=0)
	bias[nbins:,1] = mock_clust_mean
	bias[nbins:,2] = np.std(mock_clust_temps[:,:,4],axis=0)/np.sqrt(len(filenames))
	
	mockvec = np.zeros((len(filenames),2*nbins))
	mockvec[:,:nbins] = mock_void_temps[:,:,4] - mock_void_mean
	mockvec[:,nbins:] = mock_clust_temps[:,:,4] - mock_clust_mean
	mockQ = np.dot(mockvec.T,mockvec)/(mockvec.shape[0]-1)
	X = np.transpose(np.mat(output[:,0]))
	Y = np.transpose(np.mat(output[:,1]))
	
	popt,pcov = curve_fit(ampl,output[:,0],output[:,1],sigma=np.sqrt(np.diag(mockQ)),absolute_sigma=True)
	print "OLS estimate from QPM mocks: A_ISW = %0.3f +\- %0.3f (%0.2f sigma)" %(popt,np.sqrt(pcov),popt/np.sqrt(pcov))
	mockbeta = np.array(np.linalg.inv(X.T*np.linalg.inv(mockQ)*X)*(X.T*np.linalg.inv(mockQ)*Y))[0,0]
	mockbetaerr = np.array(np.sqrt(np.linalg.inv(X.T*np.linalg.inv(mockQ)*X)))[0,0]
	print "GLS estimate from QPM mocks: A_ISW = %0.3f +/- %0.3f (%0.2f sigma)" %(mockbeta,mockbetaerr, mockbeta/mockbetaerr)
	gls_model = sm.GLS(Y,X,sigma=mockQ)
	gls_results = gls_model.fit()
	print "statsmodel GLS estimate from QPM mocks: A_ISW = %0.3f +/- %0.3f (%0.2f sigma)"\
	     %(gls_results.params,gls_results.bse,gls_results.params/gls_results.bse)
	
	x = np.linspace(-6,5) 
	plt.plot(x,x,c='k',ls='--')
	plt.plot(x,mockbeta*x,c='r')
	plt.plot(x,popt*x,c='r',ls='--')
	plt.axhline(0,c='k',ls=':')
	plt.axvline(0,c='k',ls=':')
	plt.xlim([-6,5])
	plt.tick_params(labelsize=16)
	plt.xlabel(r'$T_0^\mathrm{expected}\;[\mu\mathrm{K}]$',fontsize=24)
	plt.ylabel(r'$T_0^\mathrm{measured}\;[\mu\mathrm{K}]$',fontsize=24)
	
	rand_temps = np.loadtxt('/Users/seshadri/Workspace/structures/QPM_DR12/mf_temp_data_smica/randomCMB/'\
		+'CMASS_DR12_Random2000_MF_temps.txt') 
	
	cmbvec = np.zeros((rand_temps.shape[1]-1,rand_temps.shape[0]))
	cmbvec = rand_temps.T[1:,:] - np.mean(rand_temps.T[1:,:],axis=0)
#	cmbvec = np.zeros((800,rand_temps.shape[0]))
#	cmbvec = rand_temps.T[1:801,:] - np.mean(rand_temps.T[1:801,:],axis=0)
	cmbQ = np.dot(cmbvec.T,cmbvec)/(cmbvec.shape[0]-1)
	output[:,1] = output[:,1] + np.hstack([mock_void_mean,mock_clust_mean]) - np.mean(rand_temps[:,1:],axis=1)
	X = np.transpose(np.mat(output[:,0]))
	Y = np.transpose(np.mat(output[:,1]))
	
	popt,pcov = curve_fit(ampl,output[:,0],output[:,1],sigma=np.sqrt(np.diag(cmbQ)),absolute_sigma=True)
	print "OLS estimate from mock CMB maps: A_ISW = %0.3f +\- %0.3f (%0.2f sigma)" %(popt,np.sqrt(pcov),popt/np.sqrt(pcov))
	cmbbeta = np.array(np.linalg.inv(X.T*np.linalg.inv(cmbQ)*X)*(X.T*np.linalg.inv(cmbQ)*Y))[0,0]
	cmbbetaerr = np.array(np.sqrt(np.linalg.inv(X.T*np.linalg.inv(cmbQ)*X)))[0,0]
	print "GLS estimate from mock CMB maps: A_ISW = %0.3f +/- %0.3f (%0.2f sigma)" %(cmbbeta,cmbbetaerr, cmbbeta/cmbbetaerr)
	gls_model = sm.GLS(Y,X,sigma=cmbQ)
	gls_results = gls_model.fit()
	print "statsmodel GLS estimate from mock CMB maps: A_ISW = %0.3f +/- %0.3f (%0.2f sigma)"\
		 %(gls_results.params,gls_results.bse,gls_results.params/gls_results.bse)
	
	plt.subplot(1,2,2)
	(_,caps,_) = plt.errorbar(output[:nbins,0],output[:nbins,1],yerr=np.sqrt(np.diag(cmbQ))[:nbins],\
			fmt='o',markersize=8,color='b',elinewidth=1.5,markeredgecolor='none')
	for cap in caps: cap.set_markeredgewidth(2)
	(_,caps,_) = plt.errorbar(output[nbins:,0],output[nbins:,1],yerr=np.sqrt(np.diag(cmbQ))[nbins:],\
			fmt='s',markersize=8,color='g',elinewidth=1.5,markeredgecolor='none')
	for cap in caps: cap.set_markeredgewidth(2)
	
	x = np.linspace(-6,5) 
	plt.plot(x,x,c='k',ls='--')
	plt.plot(x,cmbbeta*x,c='r')
	plt.plot(x,popt*x,c='r',ls='--')
	plt.axhline(0,c='k',ls=':')
	plt.axvline(0,c='k',ls=':')
	plt.xlim([-6,5])
	plt.tick_params(labelsize=16)
	plt.xlabel(r'$T_0^\mathrm{expected}\;[\mu\mathrm{K}]$',fontsize=24)
	plt.ylabel(r'$T_0^\mathrm{measured}\;[\mu\mathrm{K}]$',fontsize=24)
	plt.tight_layout(w_pad=2)
#	plt.savefig(figdir+'ISWproject/CMASS-DR12-mf-data.pdf',bbox_inches='tight')
	
	plt.figure(figsize=(16,6))
	plt.subplot(1,2,1)
	D = np.sqrt(np.diag(mockQ))
	mockcorr = np.fromfunction(lambda i, j: mockQ[i,j]/(D[i]*D[j]), (16, 16), dtype=int)
#	midpt = 1 - np.max(mockQ)/(np.max(mockQ) + 5)
	cmap = mpl.cm.RdBu_r
	scmap = shiftedColorMap(cmap,midpoint=0.5) 
#	plt.pcolormesh(mockQ[-1::-1,:],cmap=scmap,vmax=np.max(mockQ),vmin=-5)
	pcol = plt.pcolormesh(mockcorr[-1::-1,:],cmap=scmap,vmax=1,vmin=-1)
	pcol.set_edgecolor("face")
	cbar = plt.colorbar()
	cbar.ax.tick_params(labelsize=16)
	plt.tick_params(axis='both',which='both',bottom='off',top='off',left='off',right='off',labelbottom='off',labelleft='off') 
	plt.subplot(1,2,2)
	D = np.sqrt(np.diag(cmbQ))
	cmbcorr = np.fromfunction(lambda i, j: cmbQ[i,j]/(D[i]*D[j]), (16, 16), dtype=int)
	midpt = 1 - np.max(cmbQ)/(np.max(cmbQ))
#	scmap = shiftedColorMap(cmap,midpoint=midpt)
#	plt.pcolormesh(cmbQ[-1::-1,:],cmap=scmap,vmax=np.max(cmbQ),vmin=0)
	cmap = ListedColormap(np.loadtxt("/Users/seshadri/Workspace/colormaps/Planck_Parchment_RGB.txt")/255.)
	pcol = plt.pcolormesh(cmbcorr[-1::-1,:],cmap=cmap,vmax=1,vmin=0)
	pcol.set_edgecolor("face")
	cbar = plt.colorbar()
	cbar.ax.tick_params(labelsize=16)
	plt.tick_params(axis='both',which='both',bottom='off',top='off',left='off',right='off',labelbottom='off',labelleft='off')
	plt.tight_layout(w_pad=2)	
#	plt.savefig(figdir+'ISWproject/CMASS-DR12-mf-corrmat.pdf',bbox_inches='tight')
	
#	plt.figure(figsize=(20,8))
#	plt.subplot(1,2,1)
#	(_,caps,_) = plt.errorbar(bias[:nbins,0],bias[:nbins,1],yerr=bias[:nbins,2],fmt='o',markersize=8,color='b',\
#				elinewidth=1.5,markeredgecolor='none')
#	for cap in caps: cap.set_markeredgewidth(2)
#	(_,caps,_) = plt.errorbar(bias[nbins:,0],bias[nbins:,1],yerr=bias[nbins:,2],fmt='s',markersize=8,color='g',\
#				elinewidth=1.5,markeredgecolor='none')
#	for cap in caps: cap.set_markeredgewidth(2)
#	plt.axhline(c='k',ls=':'); plt.axvline(c='k',ls=':')
#	plt.tick_params(labelsize=16)
#	plt.xlabel(r'$T_0^\mathrm{expected}\;[\mu\mathrm{K}]$',fontsize=24)
#	plt.ylabel(r'$\overline{T_0^\mathrm{mocks}}\;[\mu\mathrm{K}]$',fontsize=24)
#	plt.title('Mock structures + Planck CMB',fontsize=22)
#	
#	plt.subplot(1,2,2)
#	(_,caps,_) = plt.errorbar(rand_temps[:nbins,0],np.mean(rand_temps[:nbins,1:],axis=1),yerr=np.std(rand_temps[:nbins,1:],axis=1)\
#				/np.sqrt(cmbvec.shape[0]),fmt='o',markersize=8,color='b',elinewidth=1.5,markeredgecolor='none')
#	for cap in caps: cap.set_markeredgewidth(2)
#	(_,caps,_) = plt.errorbar(rand_temps[nbins:,0],np.mean(rand_temps[nbins:,1:],axis=1),yerr=np.std(rand_temps[nbins:,1:],axis=1)\
#				/np.sqrt(cmbvec.shape[0]),fmt='s',markersize=8,color='g',elinewidth=1.5,markeredgecolor='none')
#	for cap in caps: cap.set_markeredgewidth(2)
#	plt.axhline(c='k',ls=':'); plt.axvline(c='k',ls=':')
#	plt.tick_params(labelsize=16)
#	plt.xlabel(r'$T_0^\mathrm{expected}\;[\mu\mathrm{K}]$',fontsize=24)
#	plt.ylabel(r'$\overline{T_0^\mathrm{mocks}}\;[\mu\mathrm{K}]$',fontsize=24)
#	plt.title('CMASS structures + synthetic CMB maps',fontsize=22)
#	plt.savefig(figdir+'ISWproject/CMASS-DR12-mf-offsets.pdf',bbox_inches='tight')
	
	return mockQ, cmbcorr

def cth_binned_plot(galaxy_sample='CMASS'):

	def ampl(x,A):
		return A*x
	if galaxy_sample=='CMASS': 
		nbins = 8 
	elif galaxy_sample=='LOWZ': nbins=6
	output = np.zeros((2*nbins,3))
	bias = np.zeros((2*nbins,3))
	
	plt.figure(figsize=(18,8))
	plt.subplot(1,2,1)
	
	filenames = glob.glob('/Users/seshadri/Workspace/structures/QPM_DR12/binned_cth_temp_data_smica/voids/CMASS_Voids_*')
	filenames = np.asarray(filenames)[np.argsort(filenames)]
	mock_void_temps = np.empty((len(filenames),nbins,6))
	data_temps = np.loadtxt('/Users/seshadri/Workspace/structures/QPM_DR12/binned_cth_temp_data_smica/voids/CMASS_DR12Voids_MF.txt')
	for i in range(len(filenames)):
	    mock_void_temps[i] = np.loadtxt(filenames[i])
	mock_void_mean = np.mean(mock_void_temps[:,:,4],axis=0)
	mock_void_errors = np.empty((mock_void_temps.shape[1],2))
	for i in range(mock_void_temps.shape[1]):
		mock_void_errors[i,0] = mock_void_mean[i] - mquantiles(mock_void_temps[:,i,4],0.1585)
		mock_void_errors[i,1] = mquantiles(mock_void_temps[:,i,4],0.8415) - mock_void_mean[i]	
	(_,caps,_) = plt.errorbar(data_temps[:,3],data_temps[:,4]-mock_void_mean,yerr=[mock_void_errors[:,0],mock_void_errors[:,1]],\
			fmt='o',markersize=8,color='b',elinewidth=1.5,markeredgecolor='none')
	for cap in caps: cap.set_markeredgewidth(2)
	output[:nbins,0] = data_temps[:,3]
	output[:nbins,1] = data_temps[:,4]-mock_void_mean
	output[:nbins,2] = 0.5*(mock_void_errors[:,0]+mock_void_errors[:,1])
	bias[:nbins,0] = np.mean(mock_void_temps[:,:,3],axis=0)
	bias[:nbins,1] = mock_void_mean
	bias[:nbins,2] = np.std(mock_void_temps[:,:,4],axis=0)/np.sqrt(len(filenames))
	
	filenames = glob.glob('/Users/seshadri/Workspace/structures/QPM_DR12/binned_cth_temp_data_smica/clusters/CMASS_Clusters_*')
	filenames = np.asarray(filenames)[np.argsort(filenames)]
	mock_clust_temps = np.empty((len(filenames),nbins,6))
	data_temps = np.loadtxt('/Users/seshadri/Workspace/structures/QPM_DR12/binned_cth_temp_data_smica/clusters/CMASS_DR12Clusters_MF.txt')
	for i in range(len(filenames)):
	    mock_clust_temps[i] = np.loadtxt(filenames[i])
	mock_clust_mean = np.mean(mock_clust_temps[:,:,4],axis=0)
	mock_clust_errors = np.empty((mock_clust_temps.shape[1],2))
	for i in range(mock_clust_temps.shape[1]):
		mock_clust_errors[i,0] = mock_clust_mean[i] - mquantiles(mock_clust_temps[:,i,4],0.1585)
		mock_clust_errors[i,1] = mquantiles(mock_clust_temps[:,i,4],0.8415) - mock_clust_mean[i]	
	(_,caps,_) = plt.errorbar(data_temps[:,3],data_temps[:,4]-mock_clust_mean,yerr=[mock_clust_errors[:,0],mock_clust_errors[:,1]],\
			fmt='s',markersize=8,color='g',elinewidth=1.5,markeredgecolor='none')
	for cap in caps: cap.set_markeredgewidth(2)
	output[nbins:,0] = data_temps[:,3]
	output[nbins:,1] = data_temps[:,4]-mock_clust_mean
	output[nbins:,2] = 0.5*(mock_clust_errors[:,0]+mock_clust_errors[:,1])
	bias[nbins:,0] = np.mean(mock_clust_temps[:,:,3],axis=0)
	bias[nbins:,1] = mock_clust_mean
	bias[nbins:,2] = np.std(mock_clust_temps[:,:,4],axis=0)/np.sqrt(len(filenames))
	
	mockvec = np.zeros((len(filenames),2*nbins))
	mockvec[:,:nbins] = mock_void_temps[:,:,4] - mock_void_mean
	mockvec[:,nbins:] = mock_clust_temps[:,:,4] - mock_clust_mean
	mockQ = np.dot(mockvec.T,mockvec)/(mockvec.shape[0]-1)
	X = np.transpose(np.mat(output[:,0]))
	Y = np.transpose(np.mat(output[:,1]))
	
	popt,pcov = curve_fit(ampl,output[:,0],output[:,1],sigma=np.sqrt(np.diag(mockQ)),absolute_sigma=True)
	print "OLS estimate from QPM mocks: A_ISW = %0.3f +\- %0.3f (%0.2f sigma)" %(popt,np.sqrt(pcov),popt/np.sqrt(pcov))
	mockbeta = np.array(np.linalg.inv(X.T*np.linalg.inv(mockQ)*X)*(X.T*np.linalg.inv(mockQ)*Y))[0,0]
	mockbetaerr = np.array(np.sqrt(np.linalg.inv(X.T*np.linalg.inv(mockQ)*X)))[0,0]
	print "GLS estimate from QPM mocks: A_ISW = %0.3f +/- %0.3f (%0.2f sigma)" %(mockbeta,mockbetaerr, mockbeta/mockbetaerr)
	gls_model = sm.GLS(Y,X,sigma=mockQ)
	gls_results = gls_model.fit()
	print "statsmodel GLS estimate from QPM mocks: A_ISW = %0.3f +/- %0.3f (%0.2f sigma)"\
	     %(gls_results.params,gls_results.bse,gls_results.params/gls_results.bse)
	
	x = np.linspace(-1.5,1.5) 
	plt.plot(x,x,c='k',ls='--')
	plt.plot(x,mockbeta*x,c='r')
	plt.plot(x,popt*x,c='r',ls='--')
	plt.axhline(0,c='k',ls=':')
	plt.axvline(0,c='k',ls=':')
	plt.xlim([-1.5,1.5])
	plt.tick_params(labelsize=16)
	plt.xlabel(r'$T_0^\mathrm{expected}\;[\mu\mathrm{K}]$',fontsize=24)
	plt.ylabel(r'$T_0^\mathrm{measured}\;[\mu\mathrm{K}]$',fontsize=24)
	
	rand_temps = np.loadtxt('/Users/seshadri/Workspace/structures/QPM_DR12/binned_cth_temp_data_smica/randomCMB/'\
		+'CMASS_DR12_Random500_CTH_temps.txt') 
	
	cmbvec = np.zeros((rand_temps.shape[1]-1,rand_temps.shape[0]))
	cmbvec = rand_temps.T[1:,:] - np.mean(rand_temps.T[1:,:],axis=0)
	cmbQ = np.dot(cmbvec.T,cmbvec)/(cmbvec.shape[0]-1)
	output[:,1] += np.hstack([mock_void_mean,mock_clust_mean])-np.mean(rand_temps[:,1:],axis=1)
	X = np.transpose(np.mat(output[:,0]))
	Y = np.transpose(np.mat(output[:,1]))
	
	popt,pcov = curve_fit(ampl,output[:,0],output[:,1],sigma=np.sqrt(np.diag(cmbQ)),absolute_sigma=True)
	print "OLS estimate from mock CMB maps: A_ISW = %0.3f +\- %0.3f (%0.2f sigma)" %(popt,np.sqrt(pcov),popt/np.sqrt(pcov))
	cmbbeta = np.array(np.linalg.inv(X.T*np.linalg.inv(cmbQ)*X)*(X.T*np.linalg.inv(cmbQ)*Y))[0,0]
	cmbbetaerr = np.array(np.sqrt(np.linalg.inv(X.T*np.linalg.inv(cmbQ)*X)))[0,0]
	print "GLS estimate from mock CMB maps: A_ISW = %0.3f +/- %0.3f (%0.2f sigma)" %(cmbbeta,cmbbetaerr, cmbbeta/cmbbetaerr)
	gls_model = sm.GLS(Y,X,sigma=cmbQ)
	gls_results = gls_model.fit()
	print "statsmodel GLS estimate from mock CMB maps: A_ISW = %0.3f +/- %0.3f (%0.2f sigma)"\
		 %(gls_results.params,gls_results.bse,gls_results.params/gls_results.bse)
	
	plt.subplot(1,2,2)
	(_,caps,_) = plt.errorbar(output[:nbins,0],output[:nbins,1],yerr=np.sqrt(np.diag(cmbQ))[:nbins],\
			fmt='o',markersize=8,color='b',elinewidth=1.5,markeredgecolor='none')
	for cap in caps: cap.set_markeredgewidth(2)
	(_,caps,_) = plt.errorbar(output[nbins:,0],output[nbins:,1],yerr=np.sqrt(np.diag(cmbQ))[nbins:],\
			fmt='s',markersize=8,color='g',elinewidth=1.5,markeredgecolor='none')
	for cap in caps: cap.set_markeredgewidth(2)
	
	x = np.linspace(-1.5,1.5) 
	plt.plot(x,x,c='k',ls='--')
	plt.plot(x,cmbbeta*x,c='r')
	plt.plot(x,popt*x,c='r',ls='--')
	plt.axhline(0,c='k',ls=':')
	plt.axvline(0,c='k',ls=':')
	plt.xlim([-1.5,1.5])
	plt.tick_params(labelsize=16)
	plt.xlabel(r'$T_0^\mathrm{expected}\;[\mu\mathrm{K}]$',fontsize=24)
	plt.ylabel(r'$T_0^\mathrm{measured}\;[\mu\mathrm{K}]$',fontsize=24)
	plt.tight_layout(w_pad=2)
	plt.savefig(figdir+'ISWproject/CMASS-DR12-cth-data.pdf',bbox_inches='tight')
	
	plt.figure(figsize=(16,6))
	plt.subplot(1,2,1)
	D = np.sqrt(np.diag(mockQ))
	mockcorr = np.fromfunction(lambda i, j: mockQ[i,j]/(D[i]*D[j]), (16, 16), dtype=int)
#	midpt = 1 - np.max(mockQ)/(np.max(mockQ) + 5)
	cmap = mpl.cm.RdBu_r
	scmap = shiftedColorMap(cmap,midpoint=0.5) 
#	plt.pcolormesh(mockQ[-1::-1,:],cmap=scmap,vmax=np.max(mockQ),vmin=-5)
	pcol = plt.pcolormesh(mockcorr[-1::-1,:],cmap=scmap,vmax=1,vmin=-1)
	pcol.set_edgecolor("face")
	cbar = plt.colorbar()
	cbar.ax.tick_params(labelsize=16)
	plt.tick_params(axis='both',which='both',bottom='off',top='off',left='off',right='off',labelbottom='off',labelleft='off') 
	plt.subplot(1,2,2)
	D = np.sqrt(np.diag(cmbQ))
	cmbcorr = np.fromfunction(lambda i, j: cmbQ[i,j]/(D[i]*D[j]), (16, 16), dtype=int)
	midpt = 1 - np.max(cmbQ)/(np.max(cmbQ))
#	scmap = shiftedColorMap(cmap,midpoint=midpt)
#	plt.pcolormesh(cmbQ[-1::-1,:],cmap=scmap,vmax=np.max(cmbQ),vmin=0)
	pcol = plt.pcolormesh(cmbcorr[-1::-1,:],cmap=scmap,vmax=1,vmin=-1)
	pcol.set_edgecolor("face")
	cbar = plt.colorbar()
	cbar.ax.tick_params(labelsize=16)
	plt.tick_params(axis='both',which='both',bottom='off',top='off',left='off',right='off',labelbottom='off',labelleft='off')
	plt.tight_layout(w_pad=2)	
	plt.savefig(figdir+'ISWproject/CMASS-DR12-cth-corrmat.pdf',bbox_inches='tight')
	
	plt.figure(figsize=(20,8))
	plt.subplot(1,2,1)
	(_,caps,_) = plt.errorbar(bias[:nbins,0],bias[:nbins,1],yerr=bias[:nbins,2],fmt='o',markersize=8,color='b',\
				elinewidth=1.5,markeredgecolor='none')
	for cap in caps: cap.set_markeredgewidth(2)
	(_,caps,_) = plt.errorbar(bias[nbins:,0],bias[nbins:,1],yerr=bias[nbins:,2],fmt='s',markersize=8,color='g',\
				elinewidth=1.5,markeredgecolor='none')
	for cap in caps: cap.set_markeredgewidth(2)
	plt.axhline(c='k',ls=':'); plt.axvline(c='k',ls=':')
	plt.tick_params(labelsize=16)
	plt.xlabel(r'$T_0^\mathrm{expected}\;[\mu\mathrm{K}]$',fontsize=24)
	plt.ylabel(r'$\overline{T_0^\mathrm{mocks}}\;[\mu\mathrm{K}]$',fontsize=24)
	plt.title('Mock structures + Planck CMB',fontsize=22)
	
	plt.subplot(1,2,2)
	(_,caps,_) = plt.errorbar(rand_temps[:nbins,0],np.mean(rand_temps[:nbins,1:],axis=1),yerr=np.std(rand_temps[:nbins,1:],axis=1)\
				/np.sqrt(cmbvec.shape[0]),fmt='o',markersize=8,color='b',elinewidth=1.5,markeredgecolor='none')
	for cap in caps: cap.set_markeredgewidth(2)
	(_,caps,_) = plt.errorbar(rand_temps[nbins:,0],np.mean(rand_temps[nbins:,1:],axis=1),yerr=np.std(rand_temps[nbins:,1:],axis=1)\
				/np.sqrt(cmbvec.shape[0]),fmt='s',markersize=8,color='g',elinewidth=1.5,markeredgecolor='none')
	for cap in caps: cap.set_markeredgewidth(2)
	plt.axhline(c='k',ls=':'); plt.axvline(c='k',ls=':')
	plt.tick_params(labelsize=16)
	plt.xlabel(r'$T_0^\mathrm{expected}\;[\mu\mathrm{K}]$',fontsize=24)
	plt.ylabel(r'$\overline{T_0^\mathrm{mocks}}\;[\mu\mathrm{K}]$',fontsize=24)
	plt.title('CMASS structures + synthetic CMB maps',fontsize=22)
	plt.savefig(figdir+'ISWproject/CMASS-DR12-cth-offsets.pdf',bbox_inches='tight')
	
	
def average_profiles(voidType='Minimal'):

	#load up the mock catalogue and profile data
	catalogue = np.loadtxt(structdir+'MDR1_LOWZ/'+voidType+'Voids_metrics.txt',skiprows=1)
	catalogue = catalogue[catalogue[:,3]<1]
	n_profile = np.loadtxt(structdir+'MDR1_LOWZ/profiles/differential/VTFE/'+voidType+'V_0A1',skiprows=2)
	dm_profile = np.loadtxt(structdir+'MDR1_LOWZ/profiles/differential/DM/res1024_'+voidType+'V_0A1',skiprows=2)
	phi_profile = np.loadtxt(structdir+'MDR1_LOWZ/profiles/differential/Phi/res1024_'+voidType+'V_0A1',skiprows=2)

	#figure 1: compare average profile to simplest Finelli postulates
	min_dens = np.mean(catalogue[:,2])
	bias = 2.0
	alpha = 1.0
	xvec = np.linspace(0,3)
	FA = (min_dens-1)/bias*(1-2*xvec**2/3.)*np.exp(-xvec**2)
	FB = (min_dens-1)/bias*(1-(2+7*alpha)/(3+3*alpha)*xvec**2+(2*alpha)/(3+3*alpha)*xvec**4)*np.exp(-xvec**2)
	plt.figure(figsize=(11,8))
	plt.errorbar(dm_profile[:,0],dm_profile[:,1]-1,yerr=dm_profile[:,2],fmt=pointstyles[0],color=kelly_colours[7],\
		elinewidth=2,markersize=8,markeredgecolor='none',label=r'measured $\delta(r)$')
	plt.errorbar(n_profile[:,0],n_profile[:,1]-1,yerr=n_profile[:,2],fmt=pointstyles[1],color=kelly_colours[4],\
		elinewidth=2,markersize=9,markeredgecolor='none',label=r'measured $\delta_n(r)$')
	plt.plot(xvec,FA,color=kelly_colours[9],linewidth=2,label='model A')
	plt.plot(xvec,FB,'--',color=kelly_colours[9],linewidth=2,label='model B')
	plt.axhline(0,linestyle=':',color='k')
	plt.legend(loc='lower right',numpoints=1,prop={'size':16})
	plt.xlabel(r'$r/R_v\,[h^{-1}\mathrm{Mpc}]$',fontsize=24,fontweight='extra bold')
	plt.ylabel(r"$\delta_n(r),\;\delta(r)$",fontsize=24,fontweight='extra bold')
 	ax = plt.gca(); prettify_plot(ax)
	plt.savefig(figdir+'/ISWproject/simple_profiles.pdf',bbox_inches='tight')

	#figure 2: best-fit versions of Finelli forms
	maxfev = 10000
	poptA, pcovA = curve_fit(FinelliA_dens,dm_profile[:,0],dm_profile[:,1]-1,sigma=dm_profile[:,2],p0=[-1,1],absolute_sigma=True,maxfev=maxfev)
	poptB, pcovB = curve_fit(FinelliB_dens,dm_profile[:,0],dm_profile[:,1]-1,sigma=dm_profile[:,2],p0=[-1,1,1],absolute_sigma=True,maxfev=maxfev)
	plt.figure(figsize=(11,8))
	plt.errorbar(dm_profile[:,0],dm_profile[:,1]-1,yerr=dm_profile[:,2],fmt=pointstyles[0],color=kelly_colours[7],\
		elinewidth=2,markersize=8,markeredgecolor='none',label=r'measured $\delta(r)$')
	fitlabel = '$b^\\ast=%0.2f,\,r_0=%0.2f R_v$' %((min_dens-1)/poptA[0],poptA[1])
	plt.plot(xvec,FinelliA_dens(xvec,poptA[0],poptA[1]),color=kelly_colours[9],linewidth=2,label=fitlabel)
	#plt.plot(xvec,FinelliB_dens(xvec,poptB[0],poptB[1],poptB[2]),'--',color=kelly_colours[9],linewidth=2,label='best-fit model B')
	plt.axhline(0,linestyle=':',color='k')
	plt.legend(loc='lower right',numpoints=1,prop={'size':16})
	plt.xlabel(r'$r/R_v\,[h^{-1}\mathrm{Mpc}]$',fontsize=24,fontweight='extra bold')
	plt.ylabel(r"$\delta(r)$",fontsize=24,fontweight='extra bold')
 	ax = plt.gca(); prettify_plot(ax)
	plt.savefig(figdir+'/ISWproject/best_profiles.pdf',bbox_inches='tight')

	#figure 3: actual Phi vs. Finelli prediction
	avg_size = np.mean(catalogue[:,1])
	FP, FPbest = np.zeros_like(xvec), np.zeros_like(xvec)
	bstar = (min_dens-1)/poptA[0]
	for i in range(len(catalogue)):
		FP += FinelliA_Phi(xvec*catalogue[i,1],(catalogue[i,2]-1)/bias,catalogue[i,1],Om_m=0.27)*10**5	
		FPbest += FinelliA_Phi(xvec*catalogue[i,1],(catalogue[i,2]-1)/bstar,poptA[1]*catalogue[i,1],Om_m=0.27)*10**5	
	FP /= len(catalogue)
	FPbest /= len(catalogue)
	plt.figure(figsize=(11,8))
	plt.errorbar(phi_profile[:,0],phi_profile[:,1],yerr=phi_profile[:,2],fmt=pointstyles[0],color=kelly_colours[7],\
		elinewidth=2,markersize=8,markeredgecolor='none',label=r'measured $\Phi(r)$')
	fitlabel = 'model given $b^\\ast=%0.2f,\,r_0=%0.2f R_v$' %(bstar,poptA[1])
	plt.plot(xvec,FP,'--',color=kelly_colours[9],linewidth=2,label='naive model')
	plt.plot(xvec,FPbest,color=kelly_colours[9],linewidth=2,label=fitlabel)
	plt.axhline(0,linestyle=':',color='k')
	plt.legend(loc='upper right',numpoints=1,prop={'size':16})
	plt.xlabel(r'$r/R_v\,[h^{-1}\mathrm{Mpc}]$',fontsize=24,fontweight='extra bold')
	plt.ylabel(r"$10^5\Phi(r)$",fontsize=24,fontweight='extra bold')
 	ax = plt.gca(); prettify_plot(ax)
	plt.savefig(figdir+'/ISWproject/best_Phi.pdf',bbox_inches='tight')

def match_centres(catA,catB,xcol=6,ycol=4,isBox=False,boxLen=0):

	if isBox:
		bounds = np.asarray([boxLen,boxLen,boxLen])
		Tree = PeriodicCKDTree(bounds,catA[:,1:4])
	else:
		Tree = cKDTree(catA[:,1:4])

	nnind, nndist, scaled_nndist, scaled_DeltaR = np.zeros((len(catB))), np.zeros((len(catB))), np.zeros((len(catB))), np.zeros((len(catB)))

	for i in xrange(len(catB)):
		nndist[i], nnind[i] = Tree.query(catB[i,1:4],k=1)
		scaled_nndist[i] = nndist[i]/catA[nnind[i],6]
		scaled_DeltaR[i] = (catB[i,6] - catA[nnind[i],6])/catA[nnind[i],6]

	print "Scaled NN dist:\n\tMean = %0.2f\n\tMedian = %0.2f" %(np.mean(scaled_nndist),np.median(scaled_nndist))
	print "Scaled DeltaR:\n\tMean = %0.2f\n\tMedian = %0.2f" %(np.mean(scaled_DeltaR),np.median(scaled_DeltaR))

	plt.figure(figsize=(10,8))
	H = plt.hist(scaled_nndist,color=kelly_RdYlGn[6],alpha=0.7,normed=True,bins=31)
	ax = plt.gca(); prettify_plot(ax)
	plt.xlabel("$r_{NN}/R_v$",fontsize=24,fontweight='extra bold')
	plt.ylabel(r'PDF',fontsize=24)
	plt.savefig(figdir+'/DES/NNmatch-distribution.png',bbox_inches='tight')
	

#	plt.figure(figsize=(10,8))
#	H = plt.hist(scaled_DeltaR,color=kelly_RdYlGn[0],normed=True,bins=20)
#	ax = plt.gca(); prettify_plot(ax)

	cm = plt.cm.get_cmap('RdYlGn')

	plt.figure(figsize=(10,8))
	max_c = min(np.max(scaled_nndist),3)
	plt.scatter(catB[:,xcol],catB[:,ycol],c=scaled_nndist,s=35,edgecolors='none',vmin=0,vmax=max_c,cmap=cm)
	plt.ylim([0,1.01])
	plt.xlabel("$R_v\,[h^{-1}\mathrm{Mpc}]$",fontsize=24,fontweight='extra bold')
	plt.ylabel(r'$n_\mathrm{min}/\overline{n}$',fontsize=24,fontweight='extra bold')
	ax = plt.gca(); prettify_plot(ax)
	cbar = plt.colorbar()
#	cbar.ax.set_yticklabels(ax.get_yticks(),fontsize=12)
	cbar.ax.get_yaxis().labelpad=30
	cbar.set_label("$r_{NN}/R_v$",fontsize=20,fontweight='extra bold',rotation=270)

#	plt.savefig(figdir+'/DES/NNmatch_spec-to-photo.png',bbox_inches='tight')

	plt.figure(figsize=(10,8))
	max_c = min(np.max(abs(scaled_DeltaR)),2)
	plt.scatter(catB[:,xcol],catB[:,ycol],c=scaled_DeltaR,s=35,edgecolors='none',vmin=-max_c,vmax=max_c,cmap=cm)
	plt.ylim([0,1.01])
	plt.xlabel("$R_v\,[h^{-1}\mathrm{Mpc}]$",fontsize=24,fontweight='extra bold')
	plt.ylabel(r'$n_\mathrm{min}/\overline{n}$',fontsize=24,fontweight='extra bold')
	ax = plt.gca(); prettify_plot(ax)
	cbar = plt.colorbar()
	cbar.ax.get_yaxis().labelpad=30
#	cbar.ax.set_yticklabels(ax.get_yticks(),fontsize=12)
	cbar.set_label("$\Delta R_v/R_v$",fontsize=20,fontweight='extra bold',rotation=270)

#	plt.savefig(figdir+'/DES/Rmatch_spec-to-photo.png',bbox_inches='tight')

def nRhist_comparison(Voids1,Voids2,meanNNsep1,meanNNsep2,nbins=50,Rmin=3,Rmax=100,usePoisson=False,PoissonFile='',PoissMInd=False):
	
	#make the histogram: 7th column is radius, 5th column is minimum tracer density
	xbins = np.linspace(Rmin,Rmax,nbins)
	ybins = np.linspace(0,1,nbins)
	H1, xedges1, yedges1 = np.histogram2d(Voids1[:,6],Voids1[:,4],bins=[xbins,ybins])
	ms1 = np.max([np.max(H1),100])
	masked_hist1 = np.ma.masked_where(H1==0,H1)
	H2, xedges2, yedges2 = np.histogram2d(Voids2[:,6],Voids2[:,4],bins=[xbins,ybins])
	ms2 = np.max([np.max(H2),ms1])
	masked_hist2 = np.ma.masked_where(H2==0,H2)

	if usePoisson:
		fig, axes = plt.subplots(1,3,sharex=False,sharey=False,figsize=(28,6))
	else:
		fig, axes = plt.subplots(1,2,sharex=False,sharey=False,figsize=(20,6))
	props = dict(edgecolor='none',facecolor='none')
	ax = axes.flat[0]
	im = ax.pcolor(xedges1,yedges1,masked_hist1.transpose(),cmap='YlOrRd',norm=LogNorm(),vmax=ms2)
	x = np.linspace(0.6*meanNNsep1,1.8*meanNNsep1,25)
	ax.plot(x,(3/(4*np.pi))*(meanNNsep1/x)**3,'k--')
	ax.set_xlim([0,Rmax])
	ax.set_ylim([0,1])
	ax.text(0.6,0.8,"CMASS spec-z",transform=ax.transAxes,fontsize=16,bbox=props)
	ax.set_xlabel("$R_v\,[h^{-1}\mathrm{Mpc}]$",fontsize=24,fontweight='extra bold')
	ax.set_ylabel(r'$n_\mathrm{min}/\overline{n}$',fontsize=24,fontweight='extra bold')
	prettify_plot(ax)

	ax = axes.flat[1]
	im = ax.pcolor(xedges2,yedges2,masked_hist2.transpose(),cmap='YlOrRd',norm=LogNorm(),vmax=ms2)
	x = np.linspace(0.6*meanNNsep2,1.8*meanNNsep2,25)
	ax.plot(x,(3/(4*np.pi))*(meanNNsep2/x)**3,'k--')
	ax.set_xlim([0,Rmax])
	ax.set_ylim([0,1])
	ax.text(0.6,0.8,"+ DES photo-z err",transform=ax.transAxes,fontsize=16,bbox=props)
	ax.set_xlabel("$R_v\,[h^{-1}\mathrm{Mpc}]$",fontsize=24,fontweight='extra bold')
	ax.set_ylabel(r'$n_\mathrm{min}/\overline{n}$',fontsize=24,fontweight='extra bold')
	prettify_plot(ax)

	if usePoisson:
		Poisson = np.loadtxt(PoissonFile,skiprows=2)
		H3, xedges3, yedges3 = np.histogram2d(Poisson[:,6],Poisson[:,4],bins=[xbins,ybins])
		ms3 = np.max([np.max(H3),100])
		masked_hist3 = np.ma.masked_where(H3==0,H3)

		ax = axes.flat[2]
		im = ax.pcolor(xedges3,yedges3,masked_hist3.transpose(),cmap='YlOrRd',norm=LogNorm(),vmax=ms2)
		x = np.linspace(0.6*meanNNsep1,1.8*meanNNsep1,25)
		ax.plot(x,(3/(4*np.pi))*(meanNNsep1/x)**3,'k--')
		ax.set_xlim([0,Rmax])
		ax.set_ylim([0,1])
		ax.text(0.6,0.8,"pure Poisson noise",transform=ax.transAxes,fontsize=16,bbox=props)
		ax.set_xlabel("$R_v\,[h^{-1}\mathrm{Mpc}]$",fontsize=24,fontweight='extra bold')
		ax.set_ylabel(r'$n_\mathrm{min}/\overline{n}$',fontsize=24,fontweight='extra bold')
		prettify_plot(ax)
	
	plt.tight_layout(w_pad=2)
	#add the colorbar
	cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
	cbar = plt.colorbar(im,cax=cax,ticks=[1,10,100])
	cbar.solids.set_edgecolor("face")
	cbar.ax.get_yaxis().labelpad=25
	cbar.ax.set_yticklabels(['$10^0$','$10^1$','$10^2$'],fontsize=16,fontweight='extra bold')
	cbar.set_label("$\mathrm{counts}$",fontsize=24,fontweight='extra bold',rotation=270)

	plt.savefig(figdir+'/DES/nRdistribution.png',bbox_inches='tight')

def abundance_comparison(data1,data2,zlim=[0.43,0.7],fSky=0.162,Om_m=0.3156,useCumul=False,nbins=31):

	vol = fSky*4*np.pi*(comovr(zlim[1],Om_m)**3.0 - comovr(zlim[0],Om_m)**3.0)/3
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
			dNdR1[i,1] = np.sum(hist1[i:])/vol
			dNdR2[i,1] = np.sum(hist2[i:])/vol
			dNdR1[i,2] = np.sqrt(np.sum(hist1[i:]))/vol
			dNdR2[i,2] = np.sqrt(np.sum(hist2[i:]))/vol
		else:
			dNdR1[i,1] = hist1[i]/(bins1[i+1]-bins1[i])/vol
			dNdR2[i,1] = hist2[i]/(bins2[i+1]-bins2[i])/vol
			dNdR1[i,2] = np.sqrt(hist1[i])/(bins1[i+1]-bins1[i])/vol
			dNdR2[i,2] = np.sqrt(hist2[i])/(bins2[i+1]-bins2[i])/vol

	plt.figure(figsize=(10,8))
	ax = plt.gca()
	ax.set_xscale('log')
	ax.set_yscale('log',nonposy='clip')
	ax.errorbar(dNdR1[:,0],dNdR1[:,1],yerr=dNdR1[:,2],fmt=pointstyles[0],color=kelly_RdYlGn[0],markeredgecolor='none',\
		markersize=8,elinewidth=2,label='DR11 CMASS (spec-z)')
	ax.errorbar(dNdR2[:,0],dNdR2[:,1],yerr=dNdR2[:,2],fmt=pointstyles[1],color=kelly_RdYlGn[7],markeredgecolor='none',\
		markersize=10,elinewidth=2,label=r"+ DES photo-z errors")
		
	ax.set_xlim([8,120])
	ax.set_xlabel(r'$R_v\,[h^{-1}\mathrm{Mpc}]$',fontsize=24,fontweight='extra bold')
	ax.set_xticks([10,50,100])
	if useCumul: 
		ax.set_ylim([1e-10,1e-5])
		ax.set_ylabel(r'$n(>R_v)\,[h^3\mathrm{Mpc}^{-3}]$',fontsize=24,fontweight='extra bold')
		ax.set_yticks([1e-10,1e-9,1e-8,1e-7,1e-6,1e-5])
	else:
		ax.set_ylim([1e-11,1e-7])
		ax.set_ylabel(r'$dn/dR_v\,[h^4\mathrm{Mpc}^{-4}]$',fontsize=24,fontweight='extra bold')
		ax.set_yticks([1e-11,1e-10,1e-9,1e-8,1e-7])

	ax.legend(loc='lower left',numpoints=1,prop={'size':16})
 	ax.set_xticklabels(ax.get_xticks(), {'family':'serif'})
	ax.tick_params(axis='both', labelsize=16)

	plt.savefig(figdir+'/DES/abundances-diff.png',bbox_inches='tight')

