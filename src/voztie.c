#include <stdio.h>
#include <stdlib.h>
#include "voz.h"

#define EQUALTHRESHOLD 1.52587890625e-5 /* 2^-16 */

int main(int argc, char *argv[]) {

  FILE *part, *adj, *vol;
  char partfile[80], *suffix, adjfile[80], volfile[80];
  realT *vols, volstemp;
  
  PARTADJ *adjs;

  int numdiv,np,np2,na;

  int i,j,k,p,nout;
  int nvp,npnotdone,nvpmax, nvpsum, *orig;
  realT avgnadj, avgvol;
  
  if (argc != 3) {
    printf("Wrong number of arguments.\n");
    printf("arg1: number of divisions (default 2)\n");
    printf("arg2: suffix describing this run\n\n");
    exit(0);
  }
  if (sscanf(argv[1],"%d",&numdiv) != 1) {
    printf("That's no number of divisions; try again.\n");
    exit(0);
  }
  if (numdiv < 2) {
    printf("Cannot have a number of divisions less than 2.  Resetting to 2:\n");
    numdiv = 2;
  }
  suffix = argv[2];
  
  np = -1; nvpmax = -1; nvpsum = 0;

  for (i = 0; i < numdiv; i++) {
   for (j = 0; j < numdiv; j++) {
    for (k = 0; k < numdiv; k++) {
      sprintf(partfile,"part.%s.%02d.%02d.%02d",suffix,i,j,k);
      part = fopen(partfile,"r");
      if (part == NULL) {
	printf("Unable to open file %s.\n\n",partfile);
	exit(0);
      }
      fread(&np2,1,sizeof(int),part);
      fread(&nvp,1,sizeof(int),part);
      if (np == -1)
	np = np2;
      else 
	if (np2 != np) {
	  printf("Incompatible total particle numbers: %d,%d\n\n",np,np2);
	  exit(0);
	}
      if (nvp > nvpmax) nvpmax = nvp;
      fclose(part);
    }
   }
  }
  printf("We have %d particles to tie together.\n",np); fflush(stdout);
  printf("The maximum number of particles in a file is %d.\n",nvpmax);

  adjs = (PARTADJ *)malloc(np*sizeof(PARTADJ));
  if (adjs == NULL) printf("Couldn't allocate adjs.\n");
  vols = (realT *)malloc(np*sizeof(realT));
  if (vols == NULL) printf("Couldn't allocate vols.\n");
  orig = (int *)malloc(nvpmax*sizeof(int));
  if (orig == NULL) printf("Couldn't allocate orig.\n");
  if ((vols == NULL) || (orig == NULL) || (adjs == NULL)) {
    printf("Not enough memory to allocate. Exiting.\n");
    exit(0);
  }
  for (p=0;p<np;p++)
    vols[p] = -1.;

  for (i = 0; i < numdiv; i++) {
   for (j = 0; j < numdiv; j++) {
    for (k = 0; k < numdiv; k++) {
      sprintf(partfile,"part.%s.%02d.%02d.%02d",suffix,i,j,k);
      part = fopen(partfile,"r");
      if (part == NULL) {
	printf("Unable to open file %s.\n\n",partfile);
	exit(0);
      }
      fread(&np2,1,sizeof(int),part);
      fread(&nvp,1,sizeof(int),part);
      nvpsum += nvp;

      fread(orig,nvp,sizeof(int),part);
      for (p=0;p<nvp;p++) {
 	fread(&volstemp,1,sizeof(realT),part);
 	if (vols[orig[p]] > -1.)
 	  if (fabs(vols[orig[p]] - volstemp)/volstemp < EQUALTHRESHOLD) {
 	    printf("Warning: different vols measured for p.%d (%g,%g). Ignore if close enough.\n",
		   orig[p],vols[orig[p]],volstemp);
 	      volstemp = 0.5*(volstemp + vols[orig[p]]);
	  }
	vols[orig[p]] = volstemp;
      }
      
      for (p=0;p<nvp;p++) {
	fread(&na,1,sizeof(int),part);
	if (na > 0) {
	  adjs[orig[p]].nadj = na;
	  adjs[orig[p]].adj = (int *)malloc(na*sizeof(int));
	  if (adjs[orig[p]].adj == NULL) {
	    printf("Couldn't allocate adjs[orig[%d]].adj.\n",p);
	    exit(0);
	  }
	  fread(adjs[orig[p]].adj,na,sizeof(int),part);
	} else {
	  printf("0"); fflush(stdout);
	}
      }
      fclose(part);
      printf("%d ",k);
    }
   }
  }
  printf("\n");
  npnotdone = 0; avgnadj = 0.; avgvol = 0.;
  for (p=0;p<np;p++) {
    if (vols[p] == -1.) npnotdone++;
    avgnadj += (realT)(adjs[p].nadj);
    avgvol += (realT)(vols[p]);
  }
  if (npnotdone > 0)
    printf("%d particles not done!\n", npnotdone);
  printf("%d particles done more than once.\n",nvpsum-np);
  avgnadj /= (realT)np;
  avgvol /= (realT)np;
  printf("Average # adjacencies = %g (%f for Poisson)\n",avgnadj,
	 48.*3.141593*3.141593/35.+2.);
  printf("Average volume = %g\n",avgvol);
    
  /* Now the output! */

  sprintf(adjfile,"%s.adj",suffix);
  sprintf(volfile,"%s.vol",suffix);

  printf("Outputting to %s, %s\n\n",adjfile,volfile);

  adj = fopen(adjfile,"w");
  if (adj == NULL) {
    printf("Unable to open %s\n",adjfile);
    exit(0);
  }
  fwrite(&np,1, sizeof(int),adj);
  /* Adjacencies: first the numbers of adjacencies, 
     and the number we're actually going to write per particle */
  for (i=0;i<np;i++)
    fwrite(&adjs[i].nadj,1,sizeof(int),adj);
    
  /* Now the lists of adjacencies (without double counting) */
  for (i=0;i<np;i++)
    if (adjs[i].nadj > 0) {
      nout = 0;
      for (j=0;j<adjs[i].nadj; j++) if (adjs[i].adj[j] > i) nout++;
      fwrite(&nout,1,sizeof(int),adj);      
      for (j=0;j<adjs[i].nadj; j++) 
	if (adjs[i].adj[j] > i) 
	  fwrite(&(adjs[i].adj[j]),1,sizeof(int),adj);
    }

  fclose(adj);
  
  /* Volumes */
  vol = fopen(volfile,"w");
  if (vol == NULL) {
    printf("Unable to open %s\n",volfile);
    exit(0);
  }
  fwrite(&np,1, sizeof(int),vol);
  fwrite(vols,sizeof(realT),np,vol);

  fclose(vol);

  return(0);
}
