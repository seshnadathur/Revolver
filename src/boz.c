/* boz.c by Mark Neyrinck */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "voz.h"

#define PRNTHR 1000
#define NPTOTOL 3000
#define NGRID 10
#define PI 3.14159265359
/* #define NSIM 512. */

typedef struct Zone {
  realT prob; /* Halo probability */
  int *p; /* Particles in the central zone */
  int np; /* Number of them in the central zone */
  int *z; /* Other zones attached to the zone */
  int nz; /* Number of them */
  int nptot; /* Total number of them */
  int npunb; /* Total number of them after unbinding */
  int core;
  realT corevol;
  int mpp; /* Particle with deepest potential */
  char mppincentral; /* Whether or not mpp is in central zone */
  int mbp; /* Most bound particle */
  char mbpincentral; /* Whether or not mbp is in central zone */
  realT potmin;
  realT pottot;
  realT bndmin;
  realT bndtot;
  realT v[3];
} ZONE;

void findrtop(realT *a, int na, int *iord, int nb);

int compar(const void * n1, const void * n2) {
  int i1,i2;

  i1 = *(int *)n1;
  i2 = *(int *)n2;
  return 2*(i1 > i2) - 1 + (i1 == i2);
}

realT sq(realT x) {
  return(x*x);
}

void dbg(char *c) {
  printf("%s\n",c); fflush(stdout);
  return;
}

int posread(char *posfile, realT ***p, realT fact);

int velread(char *velfile, realT ***v, realT fact);

int main(int argc,char **argv) {

  FILE *pos, *inz, *outz, *outztmp, *outlist;
  int i, j, h, h2,h3,c, np, nhaloes, nhunb;
  int nsim;
  int *hp, niter;
  char *unb;
  realT e1;

  ZONE *z;

  realT hubble,pot;
  realT **p, **v;
  realT potconstreal, potconst;
  realT boundness, boundnesslow, boundnesshigh, maxboundness;
  realT dist, distp,v2;
  int numunbound, oldnumunbound, unbindthis;
  char *posfile, *velfile, *inzfile, *outzfile, *outlistfile;
  char systemstr[80];
  realT boxsize, boxsize2;
  
  int p1,p2;
  realT *sorter, maxprob;
  realT *sortermass;
  int *tot2unb, *iord;

  realT **hr,**gc,xg,yg,zg;
  int d,skippy,maxnptot,***gn,**gm,g0,g1,g2,gcrit[3];
  realT cell,a,omegam;
  char ok, existold;
  int halready,nhaloesoutz,nread;

  realT potfact,ke;
  realT *koverp,koverpmax,multiplier;
  realT *bndarr,*potarr;

  e1 = exp(1.)-1.;

  halready = 0;
  
  if (argc != 11) {
    printf("Wrong number of arguments.\n");
    printf("arg1: box size\n");
    printf("arg2: nsim\n");
    printf("arg3: Omega_matter\n");
    printf("arg4: scale factor a\n");
    printf("arg5: position file\n");
    printf("arg6: velocity file\n");
    printf("arg7: input zone file\n");
    printf("arg8: output bound zone file\n");
    printf("arg9: output text file\n");
    printf("arg10: unbinding f\n\n");
    exit(0);
  }
  if (sscanf(argv[1],"%"vozRealSym,&boxsize) != 1) {
    printf("%s is not a valid boxsize\n",argv[1]);
    exit(0);
  }
  boxsize2 = boxsize/2.;
  if (sscanf(argv[2],"%d",&nsim) != 1) {
    printf("%s is not a valid Nsim\n",argv[2]);
    exit(0);
  }
  if (sscanf(argv[3],"%"vozRealSym,&omegam) != 1) {
    printf("%s is not a valid Omega_matter\n",argv[3]);
    exit(0);
  }
  if (sscanf(argv[4],"%"vozRealSym,&a) != 1) {
    printf("%s is not a valid scale factor\n",argv[4]);
    exit(0);
  }  
  posfile = argv[5];
  velfile = argv[6];
  inzfile = argv[7];
  outzfile = argv[8];
  outlistfile = argv[9];
  if (sscanf(argv[10],"%"vozRealSym,&potfact) != 1) {
    printf("%s is not a valid potfact\n",argv[10]);
    exit(0);
  }

  cell = boxsize/nsim;
  hubble = 100. * sqrt(omegam/(a*a*a) + 1. - omegam);
  /* Assumes Omega_matter + Omega_lambda = 1 */
  /* There's no little_h because distance measures are already in Mpc/h */
  printf("a = %g => Multiplying H0 by %e\n",a,hubble/100.);
  /* When A != 1 */
  potconstreal = omegam * 3.*cell*cell*cell*100.*100./(8.*PI);
  /* GM_particle, plus conversion factors to (km/sec)^2 */

  printf("cell, potconst = %g,%e\n",cell,potconstreal);

  /* Read positions & velocities */
  np = posread(posfile,&p,boxsize);
  if (velread(velfile,&v,1.0) != np) {
    printf("Numbers of particles don't match! Exiting.\n");
    exit(0);
  }

  /* Pre-unbinding halo list */
  inz = fopen(inzfile, "r");
  if (inz == NULL) {
    printf("Unable to open %s\n\n",inzfile);
  }
  fread(&np,1,sizeof(int),inz);
  fread(&nhaloes,sizeof(int),1,inz);
  printf("np = %d, nhaloes = %d\n",np,nhaloes);fflush(stdout);
  z = (ZONE *)malloc(nhaloes*sizeof(ZONE));
  for (h=0; h<nhaloes; h++) {
    fread(&(z[h].np),sizeof(int),1,inz);
    z[h].p = (int *)malloc(z[h].np*sizeof(int));
    fread(z[h].p,sizeof(int),z[h].np,inz);
  }
  maxprob = 0.;
  for (h=0; h<nhaloes;h++) {
    fread(&(z[h].prob),sizeof(realT),1,inz);
    if (z[h].prob > maxprob) maxprob = z[h].prob;
    fread(&(z[h].core),sizeof(int),1,inz);
    fread(&(z[h].corevol),sizeof(realT),1,inz);
    fread(&(z[h].nz),sizeof(int),1,inz);
    z[h].z = (int *)malloc(z[h].nz*sizeof(int));
    fread(z[h].z,sizeof(int),z[h].nz,inz);
  }
  fclose(inz);

  /* Assign cores, nptots, determine maximum */
  
  maxnptot = 0;
  for (h=0; h<nhaloes; h++) {
    z[h].nptot = 0;
    z[h].npunb = -1;
    for (h2 = 0; h2 < z[h].nz; h2++)
      z[h].nptot += z[z[h].z[h2]].np;
    if (z[h].nptot > maxnptot) maxnptot = z[h].nptot;
  }

  hp = (int *)malloc(maxnptot*sizeof(int));
  koverp = (realT *)malloc(maxnptot*sizeof(realT));
  bndarr = (realT *)malloc(maxnptot*sizeof(realT));
  potarr = (realT *)malloc(maxnptot*sizeof(realT));
  unb = (char *)malloc(maxnptot*sizeof(char));
  /* Stores boundedness; 0 = bound, 1 = freshly unbound, 2 = unbound in last iteration*/
  hr = (realT **)malloc(3*sizeof(realT *));
 
  h = 0;
  /* Read stuff from ubz file if it exists */
  outz = fopen(outzfile, "r");

  if (outz == NULL) {
    existold = 0;
  } else {
    fclose(outz);
    printf("Previous output bound zone file found.\n");
    printf("Moving it to boz.tmp and reading from it ...\n");
    sprintf(systemstr,"mv %s boz.tmp",outzfile);
    system(systemstr);
    existold = 1;
  }

  outz = fopen(outzfile,"w");
  fwrite(&nhaloes,sizeof(int),1,outz);
  if (existold == 1) {
    outztmp = fopen("boz.tmp", "r");
    fread(&nhaloesoutz,sizeof(int),1,outztmp);
    if (nhaloesoutz != nhaloes) {
      printf("Numbers of haloes (%d,%d) don't match!\n",nhaloes,nhaloesoutz);
      printf("Not using previous outz file\n");
    } else {
      ok = 1;
      while (ok) {
	nread = fread(&(z[h].npunb),sizeof(int),1,outztmp);
	if (nread != 1) {
	  printf("End/corruption of outz file encountered, at h=%d.\n",h); 
	  ok = 0;
	} else {
	  if (z[h].npunb > maxnptot) {
	    printf("Fatal error reading halo %d:\n",h); 
	    printf("Number of bound particles (%d) exceeds maxnptot.",z[h].npunb);fflush(stdout);
	    exit(0);
	  }
	  if (z[h].npunb > 0) {
	    nread = fread(hp,sizeof(int),z[h].npunb,outztmp);
	    if (nread != z[h].npunb) {
	      printf("End/corruption of outz file encountered, at h=%d.\n",h); fflush(stdout);
	      ok = 0;
	    } else {
	      fwrite(&(z[h].npunb),sizeof(int),1,outz);
	      fwrite(hp,sizeof(int),z[h].npunb,outz); 
	      /* Calculate v for halo */
	      DL z[h].v[d] = 0.;
	      for (i=0; i<z[h].npunb; i++) {
		DL z[h].v[d] += v[d][hp[i]];
	      }
	      DL z[h].v[d] /= (realT)z[h].npunb;
	    }
	  } else {
	    fwrite(&(z[h].npunb),sizeof(int),1,outz);
	  }	    
	}
	if (ok) h++;
	else z[h].npunb = -1; /* h!++ because we have to redo it */
      } 
    }
    fclose(outztmp);
    halready = h;
    printf("%d haloes read from file.\n",halready-1);
  }

  sorter = (realT *)malloc(maxnptot*sizeof(realT));

  if (NGRID > 0) {
    gc = (realT **)malloc(3*sizeof(realT));
    gm = (int **)malloc(3*sizeof(int));
    DL {
      hr[d] = (realT *)malloc(maxnptot*sizeof(realT));
      gm[d] = (int *)malloc(maxnptot*sizeof(int));
      gc[d] = (realT *)malloc((NGRID+1)*sizeof(realT));
    }
    gn = (int ***)malloc(NGRID*sizeof(int **));
    for (i=0; i<NGRID; i++) {
      gn[i] = (int **)malloc(NGRID*sizeof(int *));
      for (j=0; j<NGRID; j++)
	gn[i][j] = (int *)malloc(NGRID*sizeof(int));
    }
  }

  /* Unbinding loop */
  for (h=halready; h<nhaloes; h++) {
    if (z[h].nptot > PRNTHR) {
      printf("Halo %d: %d ->",h,z[h].nptot); fflush(stdout);
    }
    j = 0;
    for (h2 = 0; h2 < z[h].nz; h2++) {
      h3 = z[h].z[h2];
      for (i = 0; i<z[h3].np; i++) {
	if ((h2 > 0) && (z[h3].p[i] == z[h].core))
	  printf("Core:%d,%d\n",z[h].np,j);
	hp[j] = z[h3].p[i];
	DL hr[d][j] = p[hp[j]][d];
	j++;
      }
    }

    for (i=0; i<z[h].nptot; i++) {
      unb[i] = 0;
      /* Periodic Boundary Conditions -- comment out if no PBC's */
      DL {
	if ((hr[d][i] - p[z[h].core][d]) > boxsize2) hr[d][i] -= boxsize;
	if ((hr[d][i] - p[z[h].core][d]) < -boxsize2) hr[d][i] += boxsize;
      }
      /* End of PBC handling */
    }

    if (j != z[h].nptot) printf("j = %d, znt = %d!\n",j,z[h].nptot);
    z[h].npunb = z[h].nptot;
    
    if (z[h].nptot == 1) { /* Singleton halo */
      z[h].npunb = 0;
    } else if (z[h].nptot < NPTOTOL) { /* If it's small enough to be 
					  treated normally */
      numunbound = 1;
      /* 1 by 1 unbinding:*/
      /*while ((numunbound > 0) && (z[h].npunb > 1)) {
	potconst = potconstreal;
	maxboundness = 0.; unbindthis = -1;*/

      /* start lowthresh unbinding */
      niter = 0;
      multiplier = 2.;
      
      while (((numunbound > 0) || (multiplier > 1.)) && (z[h].npunb > 1)) {
	niter++;
	if (multiplier > potfact) multiplier /= potfact;
	else multiplier = 1.;
	potconst = potconstreal*multiplier;

	/* end lowthresh unbinding */
	numunbound = 0;

	/* Calculate velocity centroids */
	j = 0;
	for (i=0; i<z[h].np; i++) 
	  if (unb[i] < 2) j++;

	if (j > 0) {
	  DL z[h].v[d] = 0.;
	  
	  DL z[h].v[d] = 0.;
	  for (i=0; i<z[h].np; i++) /* only include the core zone (np instead of nptot) */	
	    if (unb[i] < 2) {/* if not already unbound */
	      DL z[h].v[d] += v[d][hp[i]];
	    }
	  
	  
	  if (j > 0) DL z[h].v[d] /= (realT)j;
	} /* If no particles in the original zone are bound, 
	     use z[h].v[] from the last iteration */
	
#pragma omp parallel for default(none) shared(z,v,hp,unb,p,hr,potconst,potconstreal,hubble,h,maxboundness,unbindthis,koverp,niter,potarr,bndarr) private (p1,pot,boundness,d,i,j,ke)
	for (i=0; i< z[h].nptot; i++) {
	  p1 = hp[i];
	  if (unb[i] < 2) {
	    ke = 0.;
	    DL ke += sq(hubble*(p[p1][d]-p[z[h].core][d]) +
			(v[d][p1]-z[h].v[d]));
	    ke *= 0.5;
	    pot = 0.;
	    for (j=0; j< z[h].nptot; j++)
	      if (unb[j] < 2)
		if (i != j) {
		  pot += 1./sqrt(sq(hr[0][i]-hr[0][j]) + sq(hr[1][i]-hr[1][j])
				 + sq(hr[2][i]-hr[2][j]));
		  /* This is where the potential is calculated */
		}
	    
	    boundness = ke - potconst * pot;
	    /*if (boundness > maxboundness) {
		unbindthis = i;
		maxboundness = boundness;
		}*//* 1 by 1 unbinding */
	    koverp[i] = ke / (potconstreal*pot);
	    if ((boundness > 0.) && (niter > 1)) {
	      unb[i] = 1;
	    } /* lowthresh unbinding */
	  }
	  if (unb[i] == 0) {
	    potarr[i] = -pot;
	    bndarr[i] = boundness;
	  }
	  else {
	    potarr[i] = 0.;
	    bndarr[i] = 0.;
	  }
	}
	koverpmax = 1.;
	for (i=0;i<z[h].nptot; i++)
	  if ((unb[i] < 2) && (koverp[i] > koverpmax))
	    koverpmax = koverp[i];
	if (niter == 1) {
	  multiplier = koverpmax;
	  if (z[h].nptot > PRNTHR) printf("m:%g ",multiplier);
	} else {
	  for (i=0;i<z[h].nptot;i++)
	    if (unb[i] == 1) {
	      numunbound++;
	      unb[i] = 2;
	    }
	  if ((numunbound == 0) && (multiplier > potfact)) {
	    /*printf("%g %g\n",koverpmax,multiplier);fflush(stdout);*/
	    multiplier = koverpmax;
	  }
	  /* lowthresh unbinding */
	  /*if (unbindthis > -1) { 
	    numunbound = 1;
	    unb[unbindthis] = 2;
	    }*/ /* 1 by 1 unbinding  -- remove niter if statement, too */
	  z[h].npunb -= numunbound;
	  if (z[h].nptot > PRNTHR) { /* see how many were unbound */
	    printf(" %d",z[h].npunb); fflush(stdout);
	  }
	}
      }
    } else {
      /* Order particles in x,y,z to find gridpoints */
      printf("G");fflush(stdout);
      DL {
#pragma omp parallel for default(none) shared(sorter,hr,z,h,d) private (i)
	for (i=0;i<z[h].nptot;i++)
	  sorter[i] = hr[d][i];

	qsort(sorter, z[h].nptot, 4, &compar);
	skippy = floor((realT)z[h].nptot/NGRID);
#pragma omp parallel for default(none) shared(sorter,gc,z,h,d,skippy) private (i)
	for (i=skippy; i<z[h].nptot; i += skippy)
	  gc[d][i/skippy] = 0.5*(sorter[i]+sorter[i+1]);
	gc[d][0] = sorter[0];
	gc[d][NGRID] = sorter[z[h].nptot - 1];
      }
      /* Place each particle, count particles in each grid point */
#pragma omp parallel for default(none) shared(gn) private (g0,g1,g2)
      for (g0=0;g0<NGRID;g0++) 
	for (g1=0;g1<NGRID;g1++) 
	  for (g2=0;g2<NGRID;g2++) 
	    gn[g0][g1][g2] = 0;
      
#pragma omp parallel for default(none) shared(z,h,gm,hr,gc) private (i,d,j)
      for (i=0; i<z[h].nptot; i++)
	DL {
	  for (j=1; (hr[d][i] > gc[d][j]) && (j <= NGRID); j++);
	  if (j > NGRID) j--;
	  gm[d][i] = j-1;
	}

      for (i=0; i<z[h].nptot; i++) {
	gn [gm[0][i]] [gm[1][i]] [gm[2][i]] ++;
      }
      /* Start unbinding */
      numunbound = 1;
      /* 1 by 1 unbinding:*/
      /*while ((numunbound > 0) && (z[h].npunb > 1)) {
	potconst = potconstreal;
	maxboundness = 0.; unbindthis = -1;*/

      /* start lowthresh unbinding */
      niter = 0;
      multiplier = 2.;
      while (((numunbound > 0) || (multiplier > 1.)) && (z[h].npunb > 1)) {
	niter++;
	if (multiplier > potfact) multiplier /= potfact;
	else multiplier = 1.;
	potconst = potconstreal*multiplier;
	/* end lowthresh unbinding */

	numunbound = 0;
	/* Calculate velocity centroids */
	j = 0;
	for (i=0; i<z[h].np; i++) 
	  if (unb[i] < 2) j++;

	if (j > 0) {
	  DL z[h].v[d] = 0.;
	  
	  for (i=0; i<z[h].np; i++) /* only include the core zone (np instead of nptot) */	
	    if (unb[i] < 2) {/* if not already unbound */
	      DL z[h].v[d] += v[d][hp[i]];
	    }
	  
	  
	  if (j > 0) DL z[h].v[d] /= (realT)j;
	} /* If no particles in the original zone are bound, 
	     use z[h].v[] from the last iteration */

#pragma omp parallel for default(none) shared(unb,hp,hubble,p,v,z,h,gm,hr,gc,gn,potconst,potarr,bndarr,unbindthis,koverp,potconstreal,niter) private (i,p1,gcrit,g0,g1,g2,pot,boundness,boundnesshigh,boundnesslow,xg,yg,zg,j,d,ke)
	for (i=0; i< z[h].nptot; i++) {
	  p1 = hp[i];
	  if (unb[i] < 2) {
	    ke = 0.;
	    DL {
	      ke += sq(hubble*(p[p1][d]-p[z[h].core][d])
		       + (v[d][p1]-z[h].v[d]));
	      gcrit[d] = gm[d][i] +
		(hr[d][i] > 0.5*(gc[d][gm[d][i]]+gc[d][gm[d][i]+1]));
	    }
	    ke *= 0.5;

	    pot = 0.;
	    /* First see if it's bound, using a shallower, easier potential */
	    for (g0=0; g0<NGRID; g0++) {
	      xg = gc[0][g0 + (g0 >= gcrit[0])];
	      for (g1=0; g1<NGRID; g1++) {
		yg = gc[1][g1 + (g1 >= gcrit[1])];
		for (g2=0; g2<NGRID; g2++) {
		  zg = gc[2][g2 + (g2 >= gcrit[2])]; /* Could be speeded up*/
		  pot += (realT)(gn[g0][g1][g2])/sqrt(sq(hr[0][i] - xg) +
				  sq(hr[1][i] - yg) + sq(hr[2][i] - zg));
		}
	      }
	    }
	    /* Take out the self-pair */
	    pot -= 1./sqrt(sq(hr[0][i] - gc[0][gm[0][i]+(gm[0][i]>=gcrit[0])]) +
			   sq(hr[1][i] - gc[1][gm[1][i]+(gm[1][i]>=gcrit[1])]) +
			   sq(hr[2][i] - gc[2][gm[2][i]+(gm[2][i]>=gcrit[2])]));

	    boundness = ke - potconst * pot;
	    boundnesslow = boundness;
	    
	    if (boundness > 0.) { 
	      /* Not bound by this criterion; try a potential deeper than
		 the true one */
	      pot = 0.;
	      for (g0=0; g0<NGRID; g0++) {
		xg = (g0 == gm[0][i]) ? hr[0][i] : gc[0][g0 + (g0<gcrit[0])];
		for (g1=0; g1<NGRID; g1++) {
		  yg = (g1 == gm[1][i]) ? hr[1][i] : gc[1][g1 + (g1<gcrit[1])];
		  for (g2=0; g2<NGRID; g2++) {
		    zg = (g2 == gm[2][i]) ? hr[2][i] : gc[2][g2 + (g2<gcrit[2])];
		    if ((g0 != gm[0][i]) || (g1 != gm[1][i]) || (g2 != gm[2][i])) {
		      /* Unless we're in the same cell */
		      pot += (realT)gn[g0][g1][g2] /
			sqrt(sq(hr[0][i] - xg) + sq(hr[1][i] - yg) +
			     sq(hr[2][i] - zg));
		    }
		  }
		}
	      }
	      /* Now do the ones in the same cell by brute force */
	      for (j=0; j< z[h].nptot; j++)
		if (gm[0][i] == gm[0][j])
		 if (gm[1][i] == gm[1][j])
		  if (gm[2][i] == gm[2][j])
		    if (unb[j] < 2)
		      if (i != j)
			pot += 1./sqrt(sq(hr[0][i]-hr[0][j]) + sq(hr[1][i]-hr[1][j])
				       + sq(hr[2][i]-hr[2][j]));
		      
	      boundness = ke - potconst * pot;
	      boundnesshigh = boundness;
	      if (boundness < 0.) { 
		/* The true boundness is too close to zero to use our 
		   easy estimates; we need the brute-force potential */
		pot = 0.;
		for (j=0; j< z[h].nptot; j++)
		  if (unb[j] < 2)
		    if (i != j)
		      pot += 1./sqrt(sq(hr[0][i]-hr[0][j]) + sq(hr[1][i]-hr[1][j])
				     + sq(hr[2][i]-hr[2][j]));
		boundness = ke - potconst * pot;
	      }
	    }
	    /*if (boundness > maxboundness) {
	      unbindthis = i;
	      maxboundness = boundness;
	    }*/ /* 1 by 1 unbinding */
	    koverp[i] = ke / (potconstreal*pot);
	    if ((boundness > 0.) && (niter > 1)) {
	      unb[i] = 1;
	    } /* lowthresh unbinding */
	  }
	  if (unb[i] == 0) {
	    bndarr[i] = boundness;
	    potarr[i] = -pot;
	  }
	  else {
	    bndarr[i] = 0.;
	    potarr[i] = 0.;
	  }
	}
	koverpmax = 1.;
	for (i=0;i<z[h].nptot; i++)
	  if ((unb[i] < 2) && (koverp[i] > koverpmax))
	    koverpmax = koverp[i];
	if (niter == 1) {
	  multiplier = koverpmax;
	  if (z[h].nptot > PRNTHR) printf("m:%g ",multiplier);
	} else {
	  for (i=0; i<z[h].nptot; i++)
	    if (unb[i] == 1) {
	      numunbound++;
	      unb[i] = 2;
	      gn[gm[0][i]][gm[1][i]][gm[2][i]] --;
	    }
	  if ((numunbound == 0) && (multiplier > potfact))
	    multiplier = koverpmax;
	  /* lowthresh unbinding */
	  /*if (unbindthis > -1) { 
	    numunbound = 1;
	    unb[unbindthis] = 2;
	    gn[gm[0][unbindthis]][gm[1][unbindthis]][gm[2][unbindthis]] --;
	    }*/ /* 1 by 1 unbinding */
	  z[h].npunb -= numunbound;
	  if (z[h].nptot > PRNTHR) { /* see how many were unbound */
	    printf(" %d",z[h].npunb); fflush(stdout);
	    /*printf(",%g",maxboundness);*/
	  }
	}
      }
    }
    if (z[h].nptot > PRNTHR) printf("\n");

    if (z[h].npunb == 1) z[h].npunb = 0;

    /* Find most bound particle, total boundness, max boundness */
    z[h].potmin = 0.;
    z[h].pottot = 0.;
    z[h].bndmin = 0.;
    z[h].bndtot = 0.;
    for (i = 0; i<z[h].nptot; i++) {
      if (unb[i] == 0) {
	if (potarr[i] < z[h].potmin) {
	  z[h].potmin = potarr[i];
	  z[h].mpp = hp[i];
	  z[h].mppincentral = (i<z[h].np);
	  distp = 0.;
	  ke = 0.;
	  v2 = 0.;
	  DL {
	    distp += sq(p[hp[i]][d]-p[z[h].core][d]);
	  }
	  ke *= 0.5;
	  distp = sqrt(dist);
	}
	z[h].pottot += potarr[i];
	if (bndarr[i] < z[h].bndmin) {
	  z[h].bndmin = bndarr[i];
	  z[h].mbp = hp[i];
	  z[h].mbpincentral = (i<z[h].np);
	  dist = 0.;
	  ke = 0.;
	  v2 = 0.;
	  DL {
	    v2 += sq(v[d][hp[i]]-z[h].v[d]);
	    ke += sq(hubble*(p[hp[i]][d]-p[z[h].core][d])
			     + (v[d][hp[i]]-z[h].v[d]));
	    dist += sq(p[hp[i]][d]-p[z[h].core][d]);
	  }
	  ke *= 0.5;
	  dist = sqrt(dist);
	}
	z[h].bndtot += bndarr[i];
      }
    }
    /*if (z[h].nptot > PRNTHR) {
      printf("m?p: b:%1d,%f %1d,%f\n",
	     (int)z[h].mbpincentral, distp,(int)z[h].mppincentral,dist);
	     }*/
    /* Output the bound particles */
    fwrite(&(z[h].npunb),sizeof(int),1,outz);
    if (z[h].npunb > 0)
      for (i=0; i<z[h].nptot; i++) {
	if (unb[i] < 2) {
	  fwrite(&hp[i],sizeof(int),1,outz);
	}
      }
    fflush(outz);
  }
  fclose(outz);
  printf("Done! Outputting ...\n");fflush(stdout);
  
  nhunb = 0;
  for (i=0; i<nhaloes; i++)
    if (z[i].npunb > 0)
      nhunb++;

  tot2unb = (int *)malloc(nhunb*sizeof(int));

  /* Sort the haloes by bound mass, then pre-unbound mass, then prob */
  maxnptot++;
  h = 0;

  sortermass = (realT *)malloc(nhunb*sizeof(realT));

  for (i=0; i<nhaloes; i++) {
    if (z[i].npunb > 0) {
      tot2unb[h] = i;
      sortermass[h] = (realT)z[i].npunb + 
	log((e1*((realT)z[i].nptot + log((e1*(realT)z[i].prob+(realT)maxprob)/(realT)maxprob)) + 
	     (realT)maxnptot)/(realT)maxnptot);
      /*printf("%f, %d, %d, %e\n",sortermass[h],z[i].npunb,z[i].nptot,z[i].prob);*/
      h++;
    }
  }

  iord = (int *)malloc(nhunb*sizeof(int));

  findrtop(sortermass, nhunb, iord, nhunb);

  outlist = fopen(outlistfile, "w");
  printf("Nhunb = %d\n",nhunb);
  fprintf(outlist, "%d\t%d\n",nhaloes,nhunb);
  for (i=0; i<nhunb; i++) {
    h = tot2unb[iord[i]];
    if (z[h].npunb > 0) {
      c = z[h].core;
      /* %d Halo number, 
	 %d # bound particles,
	 %d # total particles, 
	 %e peak-to-"strongest link" density ratio, 
	 %e volume of peak's cell (inverse of its density), 
	 %e sum of boundnesses for all particles, 
	 %e deepest boundness,
	 (Could also return deepest value of potential, sum of all potentials)
	 %d "core" (peak particle ID)
	 %d most bound particle ID
	 %d particle ID at potential minimum
	 %d is most bound particle in central zone? (1 if yes, 0 if no)
	 %d is deepest-potential particle in central zone? (1 if yes, 0 if no)
          (if one of these is zero, COULD indicate weird/spurious/duplicate halo)
	 %f%f%f x,y,z coords of central zone (in units s.t. 1 is the boxsize)
	 %e%e%e vx,vy,vy velocity centroid in km/sec */
      fprintf(outlist,
	      "%d \t%d \t%d \t%e %e %e %e \t%d\t%d\t%d\t%1d %1d %8.6f %8.6f %8.6f %e %e %e\n",
	      h, z[h].npunb, z[h].nptot,z[h].prob,z[h].corevol,
	      z[h].bndtot,z[h].bndmin, c, z[h].mbp, z[h].mpp,
	      (int)z[h].mbpincentral, (int)z[h].mppincentral,
	      p[c][0]/boxsize/cell, p[c][1]/boxsize/cell, p[c][2]/boxsize/cell, 
	      z[h].v[0],z[h].v[1],z[h].v[2]);
    }
  }
  fclose(outlist);

  return(0);
}
