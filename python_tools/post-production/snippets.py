# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 14:46:16 2016

@author: seshadri
"""

#-------------------------------------------------------------#
#--CMASS-like survey in BigMDPl: replace void centres in box--#
directory = '/Users/seshadri/Workspace/structures/BigMDPl/BigMDPl_CMASS_South/'
obspos = np.array([0,np.sqrt(1./2)*3*2500./2,2500./2+115])
filelist = np.asarray(['Clusters_info.txt','IsolatedVoids_info.txt','MinimalVoids_info.txt',\
	'barycentres/Isolated_baryC_Voids_info.txt','barycentres/Minimal_baryC_Voids_info.txt'])
parms = imp.load_source("name", directory+'sample_info.dat')
globals().update(vars(parms))

for name in filelist:
	with open(directory+name,'r') as F:
		header = F.readline()+F.readline()
	data = np.loadtxt(directory+name,skiprows=2)
	cmd = ["mkdir", directory+'obs_centres/']
	subprocess.call(cmd)
	cmd = ["cp", directory+name,directory+'obs_centres/'+name.replace('barycentres/','')]
	subprocess.call(cmd)
	#change from ZOBOV box coords to observer coords
	data[:,1:4] -= parms.boxLen/2
	#change from observer coordinates to shifted BigMDPl box coordinates
	data[:,1:4] += obspos
	#rotate back to original BigMDPl x-y axes
	rotmat = np.array([[np.cos(np.deg2rad(45)),np.sin(np.deg2rad(45)),0],[-np.sin(np.deg2rad(45)),np.cos(np.deg2rad(45)),0],[0,0,1]])
	for i in range(len(data)):
	    data[i,1:4] = np.dot(rotmat,data[i,1:4]) 
	#finally shift back to fiducial box using PBCs
	data[data[:,1]>2500,1] -= 2500
	data[data[:,1]<0,1] += 2500
	data[data[:,2]>2500,2] -= 2500
	data[data[:,2]<0,2] += 2500
	with open(directory+name,'w') as F:
		F.write(header)
		for i in range(len(data)):
			F.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d\t%f\n" %(int(data[i,0]), data[i,1], data[i,2], \
					data[i,3], data[i,4], data[i,5], data[i,6], data[i,7], int(data[i,8]), data[i,9]))
#-------------------------------------------------------------#

#-------------------------------------------------------------#
#--LOWZ-like survey in BigMDPl: replace void centres in box--#
directory = '/Users/seshadri/Workspace/structures/BigMDPl/BigMDPl_LOWZ_South1/'
obspos = np.array([0,2500./2,80])
filelist = np.asarray(['Clusters_info.txt','IsolatedVoids_info.txt','MinimalVoids_info.txt',\
	'barycentres/Isolated_baryC_Voids_info.txt','barycentres/Minimal_baryC_Voids_info.txt'])
parms = imp.load_source("name", directory+'sample_info.dat')
globals().update(vars(parms))

for name in filelist:
	with open(directory+name,'r') as F:
		header = F.readline()+F.readline()
	data = np.loadtxt(directory+name,skiprows=2)
	cmd = ["mkdir", directory+'obs_centres/']
	subprocess.call(cmd)
	cmd = ["cp", directory+name,directory+'obs_centres/'+name.replace('barycentres/','')]
	subprocess.call(cmd)
	#change from ZOBOV box coords to observer coords
	data[:,1:4] -= parms.boxLen/2
	#change from observer coordinates to BigMDPl box coordinates
	data[:,1:4] += obspos
	with open(directory+name,'w') as F:
		F.write(header)
		for i in range(len(data)):
			F.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d\t%f\n" %(int(data[i,0]), data[i,1], data[i,2], \
					data[i,3], data[i,4], data[i,5], data[i,6], data[i,7], int(data[i,8]), data[i,9]))
#-------------------------------------------------------------#


#---------------------------------------------------------------------#
#--CMASS-like survey in BigMDPl: create barycentre _metrics.txt file--#
outputFolder = '/Users/seshadri/Workspace/structures/BigMDPl/BigMDPl_CMASS_North/barycentres/'
sampleHandle = 'BigMDPl_CMASS_North'
prefixV = 'Isolated_baryC_Voids'
boxLen = 2500.0
tracerDens = 2.007000e-04
InfoFile = outputFolder.replace('barycentres/','obs_centres/') + prefixV + "_info.txt"
ListFile = outputFolder.replace('barycentres/','') + prefixV.replace('_baryC_','') + "_list.txt"
OutFile = outputFolder + prefixV + "_metrics.txt"
CatArray = np.loadtxt(InfoFile,skiprows=2)
ListArray = np.loadtxt(ListFile,skiprows=2)
centres = CatArray[:,1:4]
radii = CatArray[:,6]
vid = CatArray[:,0].astype(int)

print "Loading tracer particle data..."
posnFile = outputFolder.replace('barycentres/','') + sampleHandle + "_pos.dat"
File = file(posnFile)
Np = np.fromfile(File, dtype=np.int32,count=1)
Posns = np.empty([Np,3])
Posns[:,0] = np.fromfile(File, dtype=np.float64,count=Np)
Posns[:,1] = np.fromfile(File, dtype=np.float64,count=Np)
Posns[:,2] = np.fromfile(File, dtype=np.float64,count=Np)
File.close()
CentPart_inds = ListArray[:,1].astype(int)	#indices of central particles (min/max density particles for voids/clusters)
#	
## full cubic box, periodic boundary conditions - so use PeriodicCKDTree
#bounds = np.array([boxLen, boxLen, boxLen])
#print "Building the tracer kd-tree ...\n"
#Tree = PeriodicCKDTree(bounds, Posns)
##also make sure that dist from barycentre to centre particle is correctly calculated
#shiftXInds = abs(CatArray[:,1]-Posns[CentPart_inds[:],0])>boxLen/2.0
#shiftYInds = abs(CatArray[:,2]-Posns[CentPart_inds[:],1])>boxLen/2.0
#shiftZInds = abs(CatArray[:,3]-Posns[CentPart_inds[:],2])>boxLen/2.0
#shiftVec = np.zeros((len(CatArray),3))
#shiftVec[shiftXInds,0] = -np.copysign(boxLen,(CatArray[:,1]-Posns[CentPart_inds[:],0])[shiftXInds])
#shiftVec[shiftYInds,1] = -np.copysign(boxLen,(CatArray[:,2]-Posns[CentPart_inds[:],1])[shiftYInds])
#shiftVec[shiftZInds,2] = -np.copysign(boxLen,(CatArray[:,3]-Posns[CentPart_inds[:],2])[shiftZInds])
#	
# no periodic boundary conditions - so use cKDTree
print "Building the tracer kd-tree ...\n"
Tree = cKDTree(Posns)

#calculate enclosed tracer densities
DeltaN = np.zeros((len(radii),2))
print "Calculating enclosed tracer number densities ...\n"
for i in range(DeltaN.shape[0]):
	small_vol = (4*np.pi*(radii[i])**3)/3.0
	big_vol = (4*np.pi*(3.0*radii[i])**3)/3.0
	small_nums = len(getBall(Tree,centres[i],radii[i]))
	big_nums = len(getBall(Tree,centres[i],3.0*radii[i]))
	DeltaN[i,0] = (small_nums+1)/(small_vol*tracerDens) - 1
	DeltaN[i,1] = (big_nums+1)/(big_vol*tracerDens) - 1

with open(OutFile,'w') as Fout:
	Fout.write("StructID R_eff(Mpc/h) CentNumDens WtdAvgNumDens CentDMDens Phi_cent*10^5 DeltaN(Rv) Delta(Rv) DeltaN(3Rv) Delta(3Rv)\n")
	for i in range(CatArray.shape[0]):
		Fout.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" %(vid[i], CatArray[i,6], CatArray[i,4], CatArray[i,5], \
			0, 0, DeltaN[i,0], 0, DeltaN[i,1], 0))
#---------------------------------------------------------------------#


path = '/Users/seshadri/Workspace/structures/SDSS_DR11/CMASS_North/profiles/differential/VTFE/'
newpath = '/Users/seshadri/Workspace/structures/SDSS_DR11/joint_VTFEprofiles/CMASS/differential/'
filelist = glob.glob(path+'*')
for name in filelist:
	north = np.loadtxt(name)
	south = np.loadtxt(name.replace('North','South'))
	combined = np.hstack([north[1:],south[1:]])
	Nstruct = combined.shape[1]
	nbins = combined.shape[0]
	Dens_Jack = np.empty((nbins,Nstruct))	#create jackknife samples from the individual structures directly
	for i in range(Nstruct):
		Dens_Jack[:,i] = (np.sum(combined,axis=1)-combined[:,i])/(Nstruct-1)
	Jack_mean = np.mean(Dens_Jack,axis=1)
	Jack_err = np.std(Dens_Jack,axis=1)*np.sqrt(Nstruct)
	output = np.hstack([combined[:,0],Jack_mean,Jack_err])
	np.savetxt(name.replace(path,newpath)+'.txt',output,fmt='%0.5f',header='%d structures'%Nstruct)

#-----------------------------------------#
#---plot ellipsoidal void delta and Phi---#
r0 = 76
d0 = -0.5
q = 1.5
H0=100.0/(3e5)
Omegam=0.3
redz = 0.4
az = 1./(1+redz)

x = np.arange(-3,3.01,0.05)
y = np.arange(-3,3.01,0.05)
z = np.arange(-3,3.01,0.05)
Ngrid = len(x)
boxLen = r0*(x[-1]-x[0])

X,Y,Z = np.meshgrid(x*r0,y*r0,z*r0)
delta = deltaJGB(X,Y,Z,d0,q,r0)

XX,ZZ = np.meshgrid(x,z)

plt.figure(figsize=(10,10))
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
pcol = plt.pcolormesh(x*r0,z*r0,delta[:,Ngrid/2,:].transpose(),cmap='viridis_r',vmax=np.max(delta),vmin=np.min(delta))
cbar = plt.colorbar()
levels = [-0.4,-0.2,-0,0.03]
fmt = {}; strs = r'$\delta=$'
for l in levels: fmt[l] = strs+'$%0.2f$' %l
CS = plt.contour(x*r0,z*r0,delta[:,Ngrid/2,:].transpose(),levels=levels,colors='w',lw=2)
manual_locations = [(25,0),(50,00),(70,0),(150,0),(-150,0)]
plt.clabel(CS, inline=1, fontsize=12,fmt=fmt,manual=manual_locations)
plt.xlim([x[0]*r0,x[-1]*r0])
plt.ylim([z[0]*r0,z[-1]*r0])
plt.title('$q=%0.2f,\;\delta_0=%0.2f,\;r_0=%0.1f\,h^{-1}\mathrm{Mpc}$' %(q,d0,r0),fontsize=22)
plt.tick_params(labelsize=14)
plt.xlabel(r'$r_\perp\;[h^{-1}\mathrm{Mpc}]$',size=24)
plt.ylabel(r'$r_{||}\;[h^{-1}\mathrm{Mpc}]$',fontsize=24)
pcol.set_edgecolor("face")
cbar.ax.get_yaxis().labelpad=30
cbar.ax.tick_params(labelsize=12)
cbar.set_label("$\delta(r_{||},r_\perp)$",fontsize=22,rotation=270)
plt.savefig(figdir+'DES/ellipsoidal_void_1_delta.pdf',bbox_inches='tight')

deltak=np.fft.rfftn(delta)

kvec=2*np.pi*np.fft.fftfreq(Ngrid,boxLen/Ngrid)
#krealvec=2*np.pi*np.fft.rfftfreq(Ngrid,boxLen/Ngrid)
kxarray=np.ones((Ngrid,1,1))
kyarray=np.ones((1,Ngrid,1))
kzarray=np.ones((1,1,Ngrid/2+1)) # last dimension smaller because of reality condition
kxarray[:,0,0]=kvec
kyarray[0,:,0]=kvec
#kzarray[0,0,:]=krealvec
kzarray[0,0,:]=kvec[:Ngrid/2+1]

karray=(kxarray**2+kyarray**2+kzarray**2)**(1.0/2)
karray[0,0,0]=1
phik=-(3.0/2)*(H0*H0*Omegam*deltak/az)/karray**2
phik[0,0,0]=0 #mean over universe should be zero in rhok anyway (up to numerical error)

phi=np.fft.irfftn(phik,(Ngrid,Ngrid,Ngrid))

plt.figure(figsize=(10,10))
pcol = plt.pcolormesh(x*r0,z*r0,1e5*phi[:,Ngrid/2,:].transpose(),cmap='viridis',vmax=1e5*np.max(phi),vmin=1e5*np.min(phi))
pcol.set_edgecolor("face")
cbar = plt.colorbar()
levels=[3.5,2.5,1.5,0.5]
fmt = {0.5: r'$\Phi=5\times10^{-6}$',1.5: r'$\Phi=1.5\times10^{-5}$',2.5: r'$\Phi=2.5\times10^{-5}$',3.5: r'$\Phi=3.5\times10^{-5}$',}
manual_locations = [(-10,0),(50,0),(70,0),(100,0)]
CS = plt.contour(x*r0,z*r0,1e5*phi[:,Ngrid/2,:].transpose(),levels=levels,colors='w',)
plt.clabel(CS, inline=1, fontsize=12,fmt=fmt,manual=manual_locations)
plt.xlim([x[0]*r0,x[-1]*r0])
plt.ylim([z[0]*r0,z[-1]*r0])
plt.title('$q=%0.2f,\;\delta_0=%0.2f,\;r_0=%0.1f\,h^{-1}\mathrm{Mpc}$' %(q,d0,r0),fontsize=22)
plt.tick_params(labelsize=14)
plt.xlabel(r'$r_\perp\;[h^{-1}\mathrm{Mpc}]$',size=24)
plt.ylabel(r'$r_{||}\;[h^{-1}\mathrm{Mpc}]$',fontsize=24)
cbar.ax.get_yaxis().labelpad=30
cbar.ax.tick_params(labelsize=12)
cbar.set_label("$10^5\\times\Phi(r_{||},r_\perp)$",fontsize=22,rotation=270)

plt.savefig(figdir+'DES/ellipsoidal_void_1_Phi.pdf',bbox_inches='tight')
#-----------------------------#

#----------------------------------------------#
#---make xcorr plots for varying delta_avg-----#
data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/xcorr/LOWZIsolated0A0.8_x_Phi.txt')
plt.figure(figsize=(12,8))
plt.xscale('log')
plt.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=kelly_RdYlGn[0],alpha=0.7)
plt.plot(data[:,0],data[:,1],c=kelly_RdYlGn[0],lw=2,label=r'$\bar\delta_g<-0.2$')
data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/xcorr/LOWZIsolated0.8A0.9_x_Phi.txt')
plt.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=kelly_RdYlGn[2],alpha=0.7)
plt.plot(data[:,0],data[:,1],c=kelly_RdYlGn[2],ls='--',lw=2,label=r'$-0.2<\bar\delta_g<-0.1$')
data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/xcorr/LOWZIsolated0.9A1.0_x_Phi.txt')
plt.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=kelly_RdYlGn[5],alpha=0.7)
plt.plot(data[:,0],data[:,1],c=kelly_RdYlGn[5],ls=':',lw=2,label=r'$-0.1<\bar\delta_g<0$')
data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/xcorr/LOWZIsolated1.0A1.1_x_Phi.txt')
plt.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=kelly_RdYlGn[7],alpha=0.7)
plt.plot(data[:,0],data[:,1],c=kelly_RdYlGn[7],ls='-.',lw=2,label=r'$0<\bar\delta_g<0.1$')
plt.tick_params(labelsize=14)
plt.xlim([3,200]); plt.ylim([0,25])
plt.xticks([10,100],['10','100'])
plt.xlabel(r'$r\;[h^{-1}\mathrm{Mpc}]$',fontsize=24)
plt.ylabel(r'$\mathrm{CCF}(r)$',fontsize=24)
plt.legend(loc='upper right',numpoints=1,prop={'size':16},borderpad=0.5)
plt.savefig(figdir+'BigMDPl/LOWZ_Isolated_Phi_xcorr_Avar.pdf',bbox_inches='tight')
#----------------------------------------------#

#----------------------------------------------#
#---make xcorr plots for circumcentre vs barycentre-----#
data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/alt_xcorr/LOWZIsolated_all.txt')
plt.figure(figsize=(12,8))
plt.xscale('log')
plt.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=kelly_RdYlGn[0],alpha=0.7)
plt.plot(data[:,0],data[:,1],c=kelly_RdYlGn[0],lw=2,label='circumcentre')
data = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/PhiVoids/alt_xcorr/LOWZIsolated_baryC_all.txt')
plt.fill_between(data[:,0],data[:,1]-data[:,2],data[:,1]+data[:,2],color=kelly_RdYlGn[7],alpha=0.7)
plt.plot(data[:,0],data[:,1],c=kelly_RdYlGn[7],ls='--',lw=2,label='barycentre')
plt.tick_params(labelsize=14)
plt.xlim([3,200]); plt.ylim([-1,6])
plt.xticks([10,100],['10','100'])
plt.xlabel(r'$r\;[h^{-1}\mathrm{Mpc}]$',fontsize=24)
plt.ylabel(r'$\xi_{\mathrm{v}\Phi}(r)$',fontsize=24)
plt.legend(loc='upper right',numpoints=1,prop={'size':16},borderpad=0.5)
plt.axhline(0,c='k',ls=':')
plt.savefig(figdir+'BigMDPl/LOWZ_Isolated0A1.1_Phi_xcorr_cvsb.pdf',bbox_inches='tight')
#----------------------------------------------#

jub_info = np.loadtxt('/Users/seshadri/Workspace/structures/Jubilee1/JubDim/IsolatedVoids_info.txt',skiprows=2)
jub_sky = np.loadtxt('/Users/seshadri/Workspace/structures/Jubilee1/JubDim/IsolatedVoids_skypos.txt',skiprows=1)
select = ((jub_info[:,5]-1)*jub_info[:,6]**1.2<-30)&((jub_info[:,5]-1)*jub_info[:,6]**1.2<-20)
print sum(select)
mean_map = np.mean(proj[select],axis=0)
x = np.linspace(-15,15,size)
plt.figure(figsize=(10,8))
pcol = plt.pcolormesh(x,x,mean_map*1e6,vmax=-np.min(mean_map*1e6),vmin=np.min(mean_map*1e6))
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
pcol.set_edgecolor("face")
cbar.ax.get_yaxis().labelpad=15
cbar.set_label("$\Delta T\;[\mu K]$",fontsize=24,fontweight='extra bold',rotation=90)
plt.xlim([x[0],x[-1]])
plt.ylim([x[0],x[-1]])
plt.title(r'$\delta_g \left(\frac{R_v}{1\;h^{-1}\mathrm{Mpc}}\right)^{1.2}<-30$',fontsize=22,y=1.05)
ax = plt.gca()
ax.set_xticklabels([r'$-15^\circ$',r'$-10^\circ$',r'$-5^\circ$',r'$0^\circ$',r'$5^\circ$',r'$10^\circ$',r'$15^\circ$'],fontsize=16)
ax.set_yticklabels([r'$-15^\circ$',r'$-10^\circ$',r'$-5^\circ$',r'$0^\circ$',r'$5^\circ$',r'$10^\circ$',r'$15^\circ$'],fontsize=16)
#plt.savefig('/Users/seshadri/Workspace/structures/Jubilee1/JubDim/stackedISW_-infARsc-30_voids.pdf',bbox_inches='tight')

fov_deg = 30
size = 128
cmass_planck_proj = np.empty((len(cmass_sky),size,size))
cmass_smooth_proj = np.empty((len(cmass_sky),size,size))
lowz_planck_proj = np.empty((len(lowz_sky),size,size))
lowz_smooth_proj = np.empty((len(lowz_sky),size,size))
for i in range(len(cmass_sky)):
    cmass_planck_proj[i] = hp.gnomview(planck,rot=[cmass_sky[i,0],cmass_sky[i,1],0],xsize=size,reso=60.*fov_deg/size,return_projected_map=True)
    plt.close()


sample = 'CMASS'
proftype = 'alt_Phi_res1175'
lambda_limits = [-100,-40,-35,-30,-25,-20,-15,-10,-5,0]
suffixes = ['-infL-40','-40L-35','-35L-30','-30L-25','-25L-20','-20L-15','-15L-10','-10L-5','-5L0']
fit_params = np.empty((len(suffixes),6))
voids = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/'+sample+'/IsolatedVoids_metrics.txt',skiprows=1)
profdata = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/'+sample+'/profiles/differential/'+proftype+'/IsolatedV_all')
profiles = profdata[:,1:]
for i in range(len(suffixes)):
	Lbin = profiles[:,((voids[:,3]-1)*voids[:,1]**1.2>lambda_limits[i])&((voids[:,3]-1)*voids[:,1]**1.2<lambda_limits[i+1])]
	output = np.array([[profdata[j,0],np.mean(Lbin,axis=1)[j],np.std(Lbin,axis=1)[j]/np.sqrt(Lbin.shape[1])] for j in range(profiles.shape[0])])
	np.savetxt('/Users/seshadri/Workspace/structures/BigMDPl/'+sample+'/profiles/differential/'+proftype+'/IsolatedV_'+suffixes[i]+'.txt',\
		output,fmt='%0.3e',header='%d voids'%Lbin.shape[1])
	popt, pcov = curve_fit(phi1,output[:,0],output[:,1],sigma=output[:,2],absolute_sigma=True,p0=[3,100],maxfev=10000)
	Lbin = voids[((voids[:,3]-1)*voids[:,1]**1.2>lambda_limits[i])&((voids[:,3]-1)*voids[:,1]**1.2<lambda_limits[i+1])]
	fit_params[i,0] = np.mean((Lbin[:,3]-1)*Lbin[:,1]**1.2)
	fit_params[i,1] = np.mean(Lbin[:,1])
	fit_params[i,2:4] = popt
	fit_params[i,4:6] = np.sqrt(np.diag(pcov))
				
print 'CMASS'
from astropy.coordinates import SkyCoord
import astropy.units as u
for i in range(1,2):
	if i%100==0:
		print '%04d'%i
	filename = '/Users/seshadri/Workspace/structures/skycatalogues/CMASS_Clusters_%04d_cat.txt' %i
	data = np.loadtxt(filename)
	coords = SkyCoord(ra=data[:,1]*u.deg, dec=data[:,2]*u.deg, frame='icrs')	
	gal_l = np.asarray(coords.galactic.l)
	gal_b = np.asarray(coords.galactic.b)	
	data[:,1] = gal_l
	data[:,2] = gal_b
	np.savetxt(filename,data,fmt='%d %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f %d',\
			header='ID Galactic_l(deg) Galactic_b(deg) z Rv(Mpc/h) Theta_eff(deg) delta_g delta_max gamma EdgeFlag')
			
			
# make the binned matched filter figures for DR11 CMASS data			
filenames = glob.glob('/Users/seshadri/Workspace/structures/skycatalogues/mf_temp_data/CMASS_Voids_*')
filenames = np.asarray(filenames)[np.argsort(filenames)]
mfdata = np.empty((len(filenames),8,6))
for i in range(len(filenames)):
    mfdata[i] = np.loadtxt(filenames[i])
excl = np.logical_not(np.array(['0500' in filenames[i] for i in range(len(filenames))]))
l_m = np.mean(mfdata[excl,:,1],axis=0)
l_e = np.std(mfdata[excl,:,1],axis=0)
eT_m = np.mean(mfdata[excl,:,3],axis=0)
eT_e = np.std(mfdata[excl,:,3],axis=0)
mT_m = np.mean(mfdata[excl,:,4],axis=0)
mT_e = np.std(mfdata[excl,:,4],axis=0)
plt.figure(figsize=(10,8))
(_,caps,_) = plt.errorbar(mfdata[499,:,3],mfdata[499,:,4]-mT_m,yerr=mT_e,xerr=eT_e,fmt='o',elinewidth=1.5,markeredgecolor='none')
for cap in caps: cap.set_markeredgewidth(2)
x = np.linspace(-6,0)
plt.plot(x,x,lw=1.5)
plt.axhline(c='k',ls=':')
plt.tick_params(labelsize=14)
plt.xlabel(r'$T_0^\mathrm{predicted}$',fontsize=24)
plt.ylabel(r'$T_0^\mathrm{measured}$',fontsize=24)
plt.savefig('/Users/seshadri/Workspace/structures/skycatalogues/CMASS-DR11-mf-data.pdf',bbox_inches='tight')


# Jubilee ISW analysis with Andras's voids #
# ---------------------------------------- #
iswmap = hp.read_map('/Users/seshadri/Papers/JubileeISW/ISW/nodipole/ISW_All_nodipole.fits')
nside_isw = hp.get_nside(iswmap)
nside_cmb = 1024
nonisw_power = np.loadtxt('/Users/seshadri/software/class_public-2.5.0/output/wmap_noisw00_cl.dat')
ls = nonisw_power[:,0]
nonisw_cls = np.hstack([0,0,2*np.pi*nonisw_power[:,1]/(ls*(ls+1))])

# create highpass version of Jubilee ISW map by removing l<=10
isw_alms = hp.map2alm(iswmap)
highpass_isw_alms = isw_alms
for l in range(11):
	alm_id = hp.sphtfunc.Alm.getidx(3*nside_isw-1,l,np.arange(l))
	highpass_isw_alms[alm_id] = 0
highpass_iswmap = hp.alm2map(highpass_isw_alms,nside_isw)

# create mock CMB map and highpass version of it
bckgd = hp.synfast(nonisw_cls,nside_cmb)
iswmap = hp.ud_grade(iswmap,nside_out=nside_cmb)
cmbmap = iswmap + bckgd
cmb_alms = hp.map2alm(cmbmap)
highpass_cmb_alms = cmb_alms
for l in range(11):
	alm_id = hp.sphtfunc.Alm.getidx(3*nside_cmb-1,l,np.arange(l))
	highpass_cmb_alms[alm_id] = 0
highpass_cmbmap = hp.alm2map(highpass_cmb_alms,nside_cmb)

# load the void data
smth_scale = '20mpch'
voids = np.loadtxt('/Users/seshadri/Workspace/structures/Jubilee1/jubilee_data_allvoids_smoothing'+smth_scale+'.txt')

# step 1: stacked void images in the two ISW maps
fov_deg = 30
size = 128
sky_coords = np.rad2deg(voids[:,:2])
basic_stack = np.empty((len(voids),size,size))
highpass_stack = np.empty((len(voids),size,size))
for i in range(len(voids)):
	basic_stack[i] = hp.gnomview(iswmap,rot=[sky_coords[i,0],sky_coords[i,1]],xsize=size,reso=60.*fov_deg/size,return_projected_map=True)
	plt.close()
	highpass_stack[i] = hp.gnomview(highpass_iswmap,rot=[sky_coords[i,0],sky_coords[i,1]],xsize=size,reso=60.*fov_deg/size,return_projected_map=True)
	plt.close()

plt.figure(figsize=(10,8))
x = np.linspace(-15,15,size)
pcol = plt.pcolormesh(x,x,np.mean(basic_stack,axis=0)*1e6,vmax=20,vmin=-20)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
pcol.set_edgecolor("face")
cbar.ax.get_yaxis().labelpad=15
cbar.set_label("$\Delta T\;[\mu K]$",fontsize=24,fontweight='extra bold',rotation=90)
plt.xlim([x[0],x[-1]])
plt.ylim([x[0],x[-1]])
plt.title(r'$\ell\geq2;\;\mathrm{voids;\;ISW\;only}$',fontsize=22)
ax = plt.gca()
ax.set_xticklabels([r'$-15^\circ$',r'$-10^\circ$',r'$-5^\circ$',r'$0^\circ$',r'$5^\circ$',r'$10^\circ$',r'$15^\circ$'],fontsize=16)
ax.set_yticklabels([r'$-15^\circ$',r'$-10^\circ$',r'$-5^\circ$',r'$0^\circ$',r'$5^\circ$',r'$10^\circ$',r'$15^\circ$'],fontsize=16)
plt.savefig('/Users/seshadri/Workspace/structures/Jubilee1/DES-like_voidstack_'+smth_scale+'_isw-only_l>=2.pdf',bbox_inches='tight')

plt.figure(figsize=(10,8))
x = np.linspace(-15,15,size)
pcol = plt.pcolormesh(x,x,np.mean(highpass_stack,axis=0)*1e6,vmax=4,vmin=-4)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
pcol.set_edgecolor("face")
cbar.ax.get_yaxis().labelpad=15
cbar.set_label("$\Delta T\;[\mu K]$",fontsize=24,fontweight='extra bold',rotation=90)
plt.xlim([x[0],x[-1]])
plt.ylim([x[0],x[-1]])
plt.title(r'$\ell>10;\;\mathrm{voids;\;ISW\;only}$',fontsize=22)
ax = plt.gca()
ax.set_xticklabels([r'$-15^\circ$',r'$-10^\circ$',r'$-5^\circ$',r'$0^\circ$',r'$5^\circ$',r'$10^\circ$',r'$15^\circ$'],fontsize=16)
ax.set_yticklabels([r'$-15^\circ$',r'$-10^\circ$',r'$-5^\circ$',r'$0^\circ$',r'$5^\circ$',r'$10^\circ$',r'$15^\circ$'],fontsize=16)
plt.savefig('/Users/seshadri/Workspace/structures/Jubilee1/DES-like_voidstack_'+smth_scale+'_isw-only_l>10.pdf',bbox_inches='tight')

# step 2: rescaled stacked void images in the two ISW maps
fov_deg = 3.*np.rad2deg(voids[:,3]) # void angular widths as calculated by Andras
size = 128
sky_coords = np.rad2deg(voids[:,:2])
basic_stack = np.empty((len(voids),size,size))
highpass_stack = np.empty((len(voids),size,size))
for i in range(len(voids)):
	basic_stack[i] = hp.gnomview(iswmap,rot=[sky_coords[i,0],sky_coords[i,1]],xsize=size,reso=60.*fov_deg[i]/size,return_projected_map=True)
	plt.close()
	highpass_stack[i] = hp.gnomview(highpass_iswmap,rot=[sky_coords[i,0],sky_coords[i,1]],xsize=size,reso=60.*fov_deg[i]/size,return_projected_map=True)
	plt.close()

plt.figure(figsize=(10,8))
x = np.linspace(-15,15,size)
pcol = plt.pcolormesh(x,x,np.mean(basic_stack,axis=0)*1e6,vmax=15,vmin=-15)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
pcol.set_edgecolor("face")
cbar.ax.get_yaxis().labelpad=15
cbar.set_label("$\Delta T\;[\mu K]$",fontsize=24,fontweight='extra bold',rotation=90)
plt.xlim([x[0],x[-1]])
plt.ylim([x[0],x[-1]])
plt.title(r'$\ell\geq2;\;\mathrm{voids;\;ISW\;only}$',fontsize=22)
ax = plt.gca()
ax.set_xticklabels([r'$-3$',r'$-2$',r'$-1$',r'$0$',r'$1$',r'$2$',r'$3$'],fontsize=16)
ax.set_yticklabels([r'$-3$',r'$-2$',r'$-1$',r'$0$',r'$1$',r'$2$',r'$3$'],fontsize=16)
ax.set_xlabel(r'$\theta/\Theta_v$',fontsize=22); ax.set_ylabel(r'$\theta/\Theta_v$',fontsize=22)
plt.savefig('/Users/seshadri/Workspace/structures/Jubilee1/DES-like_voidrescaled_'+smth_scale+'_isw-only_l>=2.pdf',bbox_inches='tight')

plt.figure(figsize=(10,8))
x = np.linspace(-15,15,size)
pcol = plt.pcolormesh(x,x,np.mean(highpass_stack,axis=0)*1e6,vmax=3,vmin=-3)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
pcol.set_edgecolor("face")
cbar.ax.get_yaxis().labelpad=15
cbar.set_label("$\Delta T\;[\mu K]$",fontsize=24,fontweight='extra bold',rotation=90)
plt.xlim([x[0],x[-1]])
plt.ylim([x[0],x[-1]])
plt.title(r'$\ell>10;\;\mathrm{voids;\;ISW\;only}$',fontsize=22)
ax = plt.gca()
ax.set_xticklabels([r'$-3$',r'$-2$',r'$-1$',r'$0$',r'$1$',r'$2$',r'$3$'],fontsize=16)
ax.set_yticklabels([r'$-3$',r'$-2$',r'$-1$',r'$0$',r'$1$',r'$2$',r'$3$'],fontsize=16)
ax.set_xlabel(r'$\theta/\Theta_v$',fontsize=22); ax.set_ylabel(r'$\theta/\Theta_v$',fontsize=22)
plt.savefig('/Users/seshadri/Workspace/structures/Jubilee1/DES-like_voidrescaled_'+smth_scale+'_isw-only_l>10.pdf',bbox_inches='tight')

# step 3: stacked random images in the two ISW maps
fov_deg = 30
size = 128
sky_coords = np.empty((len(voids),2))
sky_coords[:,0] = np.random.uniform(5,100,len(voids))
sky_coords[:,1] = np.random.uniform(-58,-42,len(voids))
basic_stack = np.empty((len(voids),size,size))
highpass_stack = np.empty((len(voids),size,size))
for i in range(len(voids)):
	basic_stack[i] = hp.gnomview(iswmap,rot=[sky_coords[i,0],sky_coords[i,1]],xsize=size,reso=60.*fov_deg/size,return_projected_map=True)
	plt.close()
	highpass_stack[i] = hp.gnomview(highpass_iswmap,rot=[sky_coords[i,0],sky_coords[i,1]],xsize=size,reso=60.*fov_deg/size,return_projected_map=True)
	plt.close()

plt.figure(figsize=(10,8))
x = np.linspace(-15,15,size)
pcol = plt.pcolormesh(x,x,np.mean(basic_stack,axis=0)*1e6,vmax=20,vmin=-20)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
pcol.set_edgecolor("face")
cbar.ax.get_yaxis().labelpad=15
cbar.set_label("$\Delta T\;[\mu K]$",fontsize=24,fontweight='extra bold',rotation=90)
plt.xlim([x[0],x[-1]])
plt.ylim([x[0],x[-1]])
plt.title(r'$\ell\geq2;\;\mathrm{random\;locations;\;ISW\;only}$',fontsize=22)
ax = plt.gca()
ax.set_xticklabels([r'$-15^\circ$',r'$-10^\circ$',r'$-5^\circ$',r'$0^\circ$',r'$5^\circ$',r'$10^\circ$',r'$15^\circ$'],fontsize=16)
ax.set_yticklabels([r'$-15^\circ$',r'$-10^\circ$',r'$-5^\circ$',r'$0^\circ$',r'$5^\circ$',r'$10^\circ$',r'$15^\circ$'],fontsize=16)
plt.savefig('/Users/seshadri/Workspace/structures/Jubilee1/DES-like_randomstack_'+smth_scale+'_isw-only_l>=2.pdf',bbox_inches='tight')

plt.figure(figsize=(10,8))
x = np.linspace(-15,15,size)
pcol = plt.pcolormesh(x,x,np.mean(highpass_stack,axis=0)*1e6,vmax=4,vmin=-4)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
pcol.set_edgecolor("face")
cbar.ax.get_yaxis().labelpad=15
cbar.set_label("$\Delta T\;[\mu K]$",fontsize=24,fontweight='extra bold',rotation=90)
plt.xlim([x[0],x[-1]])
plt.ylim([x[0],x[-1]])
plt.title(r'$\ell>10;\;\mathrm{random\;locations;\;ISW\;only}$',fontsize=22)
ax = plt.gca()
ax.set_xticklabels([r'$-15^\circ$',r'$-10^\circ$',r'$-5^\circ$',r'$0^\circ$',r'$5^\circ$',r'$10^\circ$',r'$15^\circ$'],fontsize=16)
ax.set_yticklabels([r'$-15^\circ$',r'$-10^\circ$',r'$-5^\circ$',r'$0^\circ$',r'$5^\circ$',r'$10^\circ$',r'$15^\circ$'],fontsize=16)
plt.savefig('/Users/seshadri/Workspace/structures/Jubilee1/DES-like_randomstack_'+smth_scale+'_isw-only_l>10.pdf',bbox_inches='tight')

# step 4: run rescaled CTH filters on voids, with the mock CMB maps and the ISW maps
alpha = np.linspace(0.1,1.5,15)
sky_coords = voids[:,:2]	# need coords in radians now
void_filt_iswtemps = np.empty((len(alpha),len(voids),2))
void_filt_cmbtemps = np.empty((len(alpha),len(voids),2))
for j in range(len(alpha)):
	for i in range(len(voids)):
		void_filt_cmbtemps[j,i,0] = cth_filter(cmbmap,sky_coords[i,0],sky_coords[i,1],np.rad2deg(alpha[j]*voids[i,3]))
		void_filt_cmbtemps[j,i,1] = cth_filter(highpass_cmbmap,sky_coords[i,0],sky_coords[i,1],np.rad2deg(alpha[j]*voids[i,3]))
		void_filt_iswtemps[j,i,0] = cth_filter(iswmap,sky_coords[i,0],sky_coords[i,1],np.rad2deg(alpha[j]*voids[i,3]))
		void_filt_iswtemps[j,i,1] = cth_filter(highpass_iswmap,sky_coords[i,0],sky_coords[i,1],np.rad2deg(alpha[j]*voids[i,3]))
		
#step 5: run rescaled CTH filters on random lines of sight within same window
num_random_runs = 500
random_filt_temps = np.empty((num_random_runs,len(alpha),len(voids)))
for i in range(num_random_runs):
	sky_coords = np.empty((len(voids),2))
	sky_coords[:,0] = np.deg2rad(np.random.uniform(5,100,len(voids)))
	sky_coords[:,1] = np.deg2rad(np.random.uniform(-58,-42,len(voids)))
	for j in range(len(alpha)):
		for k in range(len(voids)):
			random_filt_temps[i,j,k] = cth_filter(cmbmap,sky_coords[k,0],sky_coords[k,1],np.rad2deg(alpha[j]*voids[k,3]))
two_sigma = np.empty((len(alpha),2))
for i in range(len(alpha)):
    two_sigma[i] = mquantiles(np.mean(random_filt_temps*1e6,axis=2)[:,i],[0.0225,0.9775])

plt.figure(figsize=(10,8))
plt.plot(alpha,np.mean(void_filt_iswtemps*1e6,axis=1)[:,0],c='#C10020',ls='--',lw=1.5,label=r'$\mathrm{ISW\;only;}\;\ell>=2$')
plt.plot(alpha,np.mean(void_filt_iswtemps*1e6,axis=1)[:,1],c='#007D34',ls='--',lw=1.5,label=r'$\mathrm{ISW\;only;}\;\ell>10$')
plt.plot(alpha,np.mean(void_filt_cmbtemps*1e6,axis=1)[:,0],c='#C10020',lw=1.5,label=r'$\mathrm{full\;CMB;}\;\ell>=2$')
plt.plot(alpha,np.mean(void_filt_cmbtemps*1e6,axis=1)[:,1],c='#007D34',lw=1.5,label=r'$\mathrm{full\;CMB;}\;\ell>10$')
plt.fill_between(alpha,two_sigma[:,0],two_sigma[:,1],color='b',alpha=0.5)
plt.plot([],[],color='b',alpha=0.3,linewidth=10,label=r'$\mathrm{full\;CMB\;95\%\;c.l.}$')
plt.xlim([0.1,1.5])
plt.ylim([-5,5])
plt.axhline(0,c='k',ls=':')
plt.tick_params(labelsize=14)
plt.xlabel(r'$\mathrm{filter\;rescaling\;factor}\;\alpha$',fontsize=22)
plt.ylabel(r'$\overline{\Delta T_\mathrm{filt}}\;[\mu\mathrm{K}]$',fontsize=22)
plt.legend(loc='upper right',numpoints=1,prop={'size':16},borderpad=0.5)
plt.savefig('/Users/seshadri/Workspace/structures/Jubilee1/DES-like_rescaling_alpha.pdf',bbox_inches='tight')

# step 6: plot a figure to satisfy Andras
signal = np.cumsum(void_filt_iswtemps[:,np.argsort(-voids[:,2]),1],axis=1)/np.cumsum(1+np.arange(len(voids)))
noise = random_filt_temps[:,:,np.argsort(-voids[:,2])]
mean_noise = np.cumsum(noise,axis=2)/np.cumsum(1+np.arange(len(voids)))
noise_one_sigma = np.empty((len(alpha),len(voids),2))
for i in range(len(alpha)):
    for j in range(len(voids)):
        noise_one_sigma[i,j] = mquantiles(mean_noise[:,i,j],[0.1585,0.8415])
avg_noise_sigma = 0.5*(noise_one_sigma[:,:,1]-noise_one_sigma[:,:,0])
plt.figure(figsize=(10,8))
for i in range(len(alpha)/2):
    plt.plot(1+np.arange(len(voids)),signal[2*i+1]/avg_noise_sigma[2*i+1],c=kelly_colours[i],lw=2,label=r'$\alpha=%0.1f$'%alpha[2*i+1])
plt.legend(loc='upper right',numpoints=1,prop={'size':16},borderpad=0.5)
plt.tick_params(labelsize=14)
plt.xlabel('$N_v$',fontsize=22)
plt.ylabel('$S/N$',fontsize=22)
plt.savefig('/Users/seshadri/Workspace/structures/Jubilee1/DES-like_SNR_scaling_example.pdf',bbox_inches='tight')


#--------testing Jubilee voids using CTH filter--------------#
jub_voids = np.loadtxt('/Users/seshadri/Workspace/structures/Jubilee1/JubCMASS/Voids_cat.txt')
lon = np.deg2rad(jub_voids[:,1])
lat = np.deg2rad(jub_voids[:,2])

cmb_power = np.loadtxt('/Users/seshadri/software/class_public-2.5.0/output/wmap_noisw00_cl.dat')
ls = cmb_power[:,0]
cmb_cls = np.hstack([0,0,2*np.pi*cmb_power[:,1]/(ls*(ls+1))])*1e-12

jub_isw = hp.read_map('/Users/seshadri/Workspace/Sims/Jubilee1/ISW_All_nodipole.fits')

nside = 512
npix = hp.nside2npix(nside)
jub_isw = hp.ud_grade(jub_isw,nside_out=nside)

num_mocks = 200
width_ratio = 0.7
cthT = np.vectorize(cth_filter,excluded=[0])
filt_temps = np.zeros((len(jub_voids),num_mocks+1))
filt_temps[:,0] = cthT(jub_isw,lon,lat,jub_voids[:,5]*width_ratio)
for i in range(num_mocks):

	if i%50==0: print "Done %d maps" %i
	# generate synthetic CMB map to filter
	syn_map = hp.synfast(cmb_cls,nside,verbose=False)
	cmb_map = jub_map + syn_map
	filt_temps[:,i+1] = cthT(cmb_map,lon,lat,jub_voids[:,5]*width_ratio)


#-------------getting Delta T(theta) profiles for Jubilee voids-------------#
jub_voids = np.loadtxt('/Users/seshadri/Workspace/structures/Jubilee1/JubCMASS/Voids_cat.txt')
jub_isw = hp.read_map('/Users/seshadri/Workspace/Sims/Jubilee1/ISW_All_nodipole.fits')
lon = np.deg2rad(jub_voids[:,1])
lat = np.deg2rad(jub_voids[:,2])
nside = 512
npix = hp.nside2npix(nside)
jub_isw = hp.ud_grade(jub_isw,nside_out=nside)

angles = np.deg2rad(np.linspace(0.1,31.1,32))
angles = 0.5*(angles[1:]+angles[:-1])
discT = np.zeros((len(jub_voids),len(angles)))
discNos = np.zeros((len(jub_voids),len(angles)))
for i in range(len(jub_voids)):
	for j in range(len(angles)):
		neighbours = hp.query_disc(nside,hp.ang2vec(np.pi/2-lat[i],lon[i]),angles[j])
		discT[i,j] = np.sum(jub_isw[neighbours])
		discNos[i,j] = len(neighbours)
ringT = np.fromfunction(lambda i,j: discT[i,j+1]-discT[i,j],(len(jub_voids),len(angles)-1),dtype=int)
ringT = np.insert(ringT,0,discT[:,0],axis=1)
ringNos = np.fromfunction(lambda i,j: discNos[i,j+1]-discNos[i,j],(len(jub_voids),len(angles)-1),dtype=int)
ringNos = np.insert(ringNos,0,discNos[:,0],axis=1)
ringT = ringT/ringNos
output = np.zeros((ringT.shape[1],ringT.shape[0]+1))
output[:,0] = np.rad2deg(angles)
output[:,1:] = ringT.T


#----------------
lambda_bins = np.array([-1000,-31,-26,-23,-19,-14,-10,-5,0])

profdata = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/CMASS/profiles/differential/alt_Phi_res1175/IsolatedV_all')
sim_voids = np.loadtxt('/Users/seshadri/Workspace/structures/BigMDPl/CMASS/IsolatedVoids_metrics.txt',skiprows=1)
profiles = profdata[:,1:]
void_fit_params = np.empty((len(lambda_bins)-1,2))
zsim = 0.52; OmM = 0.308; nside = 128; npix = hp.nside2npix(nside)
for i in range(lambda_bins.shape[0]-1):
	Lbin = profiles[:,((sim_voids[:,3]-1)*sim_voids[:,1]**1.2>lambda_bins[i])&    ((sim_voids[:,3]-1)*sim_voids[:,1]**1.2<lambda_bins[i+1])]
	output = np.array([[profdata[j,0],np.mean(Lbin,axis=1)[j],np.std(Lbin,axis=1)[j]    /np.sqrt(Lbin.shape[1])] for j in range(profiles.shape[0])])
	popt, pcov = curve_fit(phi1,output[:,0],output[:,1],sigma=output[:,2],    absolute_sigma=True,p0=[3,100],maxfev=10000)
	void_fit_params[i] = popt
	
voids = np.loadtxt('/Users/seshadri/Workspace/structures/QPM_DR12/void_catalogues/CMASS/CMASS_DR12Voids_cat.txt')
void_isw_map = np.zeros(npix)
cmb_power = np.loadtxt('/Users/seshadri/software/class_public-2.5.0/output/planck_noisw00_cl_lensed.dat')
ls = cmb_power[:,0]
cmb_cls = np.hstack([0,0,2*np.pi*cmb_power[:,1]/(ls*(ls+1))])*1e-12
syn_map = hp.synfast(cmb_cls,1024,verbose=False)


for j in range(lambda_bins.shape[0] - 1):
	print j
	Lbin = voids[(voids[:,8]>lambda_bins[j])&(voids[:,8]<lambda_bins[j+1])]
	# mean Phi0, r0 and z0 values for this bin
	Phi0 = void_fit_params[j,0]*1e-5; r0= void_fit_params[j,1]
	z0 = np.mean(Lbin[:,3])			
	Phi0 = Phi0 * (1+z0)*Dgrowth(z0,OmegaM=OmM)/((1+zsim)*Dgrowth(zsim,OmegaM=OmM))
	# generate the template map 
	theta, phi = hp.pix2ang(nside,np.arange(npix))
	angles = np.deg2rad(np.linspace(0,180,2*180))
	normTemp = np.zeros_like(angles)
	for i in range(len(angles)):
	    normTemp[i] = T_ISW_exact(angles[i],Phi0,r0,z0,OmegaM=OmM)
	T_interp = interp1d(angles,normTemp,kind='cubic')
	template_map = T_interp(theta)
	
	
	for i in range(len(Lbin)):
		template_alms = hp.map2alm(template_map,lmax=300)
		hp.rotate_alm(template_alms,psi=0,theta=np.deg2rad(Lbin[i,2]),phi=np.deg2rad(Lbin[i,1]),lmax=300)
		void_isw_map += hp.alm2map(template_alms,nside,verbose=False)
		
		
#-----------------
cmap = ListedColormap(np.loadtxt("/Users/seshadri/Workspace/colormaps/Planck_Parchment_RGB.txt")/255.)
midpt = 1 - np.max(cmb_map)/(np.max(cmb_map) - np.min(cmb_map))
scmap = shiftedColorMap(cmap,midpoint=midpt)
scmap.set_under('white')
hp.mollview(cmb_map*1e6,title='',xsize=2400,cmap=scmap,cbar=None,max=np.max(cmb_map*1e6),min=np.min(cmb_map*1e6))
fig = plt.gcf()
ax = plt.gca()
image = ax.get_images()[0]
cbaxes = fig.add_axes([0.1, 0.02, 0.8, 0.03]) 
cbar = fig.colorbar(image, cax = cbaxes, ax=ax, orientation='horizontal')
cbar.ax.tick_params(labelsize=16)
cbar.set_label("$\Delta T\;[\mu\mathrm{K}]$",fontsize=20,fontweight='extra bold')
plt.savefig('/Users/seshadri/Workspace/figures/ISWproject/Jubilee_CMBrealization.pdf',dpi=100,bbox_inches='tight')


#---------------building survey cutouts-----------------#
#-------------------------------------------------------#
box = np.loadtxt('/Users/seshadri/Workspace/Sims/MultiDark/BigMDPl/HOD/BigMDPl_CMASS.dat',skiprows=6)
real_boxLen = 2500.
rfar = comovr(0.7,OmegaM=0.307115)
rnear = comovr(0.43,OmegaM=0.307115)
offset = 500
phi_max = np.rad2deg(np.arccos(offset/rnear))
ra_min = 180 - phi_max
ra_max = 180 + phi_max
theta_max = np.rad2deg(np.arccos(offset/rnear))
print phi_max, ra_min, ra_max, theta_max

obs1 = np.array([-offset,real_boxLen/2,real_boxLen/2])
obspos1 = box[:,1:4] - obs1
losvel1 = np.empty(len(obspos1))
for i in range(len(obspos1)):
    losvel1[i] = np.dot(shiftedbox[i,4:7],obspos1[i])/np.linalg.norm(obspos1[i])

#convert Cartesian to sky positions
skypos1 = np.empty((len(obspos1),5))

rdist = np.linalg.norm(obspos1,axis=1)
Dec = 90 - np.rad2deg(np.arccos(obspos1[:,2]/rdist))
RA = np.rad2deg(np.arctan2(obspos1[:,1],obspos1[:,0]))
#RA[RA<0] += 360 #to ensure RA is in the range 0 to 360
rrange = np.linspace(0,max(rdist)+1,40)
zvals = np.asarray([brentq(lambda x: comovr(x,0.307115) - rr, 0.0, 4.0) for rr in rrange])
zinterp = interp1d(rrange,zvals)
true_redshifts = zinterp(rdist)
obs_redshifts = true_redshifts + losvel1/LIGHT_SPEED
sigma_z = 0.02*(1+obs_redshifts)
photoz_err = np.random.normal(scale=sigma_z,size=len(sigma_z))
photoz = obs_redshifts + photoz_err

skypos1[:,0] = RA
skypos1[:,1] = Dec
skypos1[:,2] = obs_redshifts
skypos1[:,3] = true_redshifts
skypos1[:,4] = photoz

select = (RA>ra_min)&(RA<ra_max)&(np.abs(Dec)<theta_max)&(obs_redshifts>0.43)&(obs_redshifts<0.7)
select_real = (RA>ra_min)&(RA<ra_max)&(np.abs(Dec)<theta_max)&(true_redshifts>0.43)&(true_redshifts<0.7)
select_photoz = (RA>ra_min)&(RA<ra_max)&(np.abs(Dec)<theta_max)&(photoz>0.43)&(photoz<0.7)