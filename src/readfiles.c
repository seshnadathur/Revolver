#include <stdio.h>
#include <stdlib.h>
#include "voz.h"

/* Open File */
/* Returns number of particles read */
int openfile(char *filename, FILE **f) {
  int np;

  *f = fopen(filename, "rb");
  if (f == NULL) {
    printf("Unable to open position file %s\n\n",filename);
    exit(0);
  }

  /* Read number of particles */
  fread(&np,sizeof(int),1, *f); 

  return(np);
}

/* Chunked Positions */
/* Read in a chunk of particles */
/* Return number actually read in */
int posread_chunk(FILE *f, realT **p, realT fact, int np, int nread){

  int i, d, ntoread = N_CHUNK;
  realT *ptemp;

  if((np - nread) < N_CHUNK){
    ntoread = np - nread;
  }

  ptemp = (realT *)malloc(ntoread*sizeof(realT));

  fseek(f,sizeof(int) +  nread*sizeof(realT), SEEK_SET);
  fread(ptemp,sizeof(realT),ntoread,f);   
  for (i=0; i<ntoread; i++) p[i][0] = ptemp[i];   

  fseek(f,sizeof(int) +  (np+nread)*sizeof(realT), SEEK_SET);
  fread(ptemp,sizeof(realT),ntoread,f);   
  for (i=0; i<ntoread; i++) p[i][1] = ptemp[i];

  fseek(f,sizeof(int) +  (2*np+nread)*sizeof(realT), SEEK_SET);
  fread(ptemp,sizeof(realT),ntoread,f);
  for (i=0; i<ntoread; i++) p[i][2] = ptemp[i];

  free(ptemp);

  for (i=0; i< ntoread; i++) DL p[i][d] *= fact;

  return ntoread;
}

int posread_isol(char *posfile, float ***p, float fact, int *np, int *npreal) {

  FILE *pos;
  int dum,d,i;
  float xt,yt,zt;
  int npread, npreadreal;
  float xmin,xmax,ymin,ymax,zmin,zmax;

  pos = fopen(posfile, "r");
  if (pos == NULL) {
    printf("Unable to open position file %s\n\n",posfile);
    exit(0);
  }

  /* Read number of particles */
  if (fscanf(pos,"%d%d\n",&npread,&npreadreal) != 2) {
    printf("Problem reading no. particles\n");
    exit(0);
  }

  /* Allocate the arrays */
  (*p) = (float **)malloc(npread*sizeof(float *));
  /* Fill the arrays */
  for (i=0; (fscanf(pos,"%f%f%f\n",&xt,&yt,&zt) == 3); i++) {
    if (i >= npread) {
      printf("More particles in %s than first line claims!  Exiting.\n\n",posfile);
      exit(0);
    }
    (*p)[i] = (float *)malloc(3*sizeof(float));
    if ((*p)[i] == NULL) {
      printf("Unable to allocate particle array in readfiles!\n");
      fflush(stdout);
      exit(0);
    }
    (*p)[i][0]=xt*fact; (*p)[i][1]=yt*fact; (*p)[i][2] = zt*fact;
  }
  fclose(pos);
  printf("%d\n",i);
  if (npread != i) {
    printf("Read %d particles (not %d, as the first line claims)!  Exiting.\n\n", i, npread);
    exit(0);
  }
  /* Test range -- can comment out */
  xmin = BF; xmax = -BF; ymin = BF; ymax = -BF; zmin = BF; zmax = -BF;
  for (i=0; i<npread;i++) {
    if ((*p)[i][0]<xmin) xmin = (*p)[i][0]; if ((*p)[i][0]>xmax) xmax = (*p)[i][0];
    if ((*p)[i][1]<ymin) ymin = (*p)[i][1]; if ((*p)[i][1]>ymax) ymax = (*p)[i][1];
    if ((*p)[i][2]<zmin) zmin = (*p)[i][2]; if ((*p)[i][2]>zmax) zmax = (*p)[i][2];
  }
  printf("npread: %d, x: %f,%f; y: %f,%f; z: %f,%f\n",npread,xmin,xmax, ymin,ymax, zmin,zmax); fflush(stdout);

  *np = npread;
  *npreal = npreadreal;
  return(0);
}

/* Positions */
/* Returns number of particles read */
int posread(char *posfile, realT ***p, realT fact) {

  FILE *pos;
  int npr,dum,d,i;
  realT xmin,xmax,ymin,ymax,zmin,zmax;
  realT *ptemp;

  pos = fopen(posfile, "rb");
  if (pos == NULL) {
    printf("Unable to open position file %s\n\n",posfile);
    exit(0);
  }
  /* Fortran77 4-byte headers and footers */
  /* Delete "dum" statements if you don't need them */

  /* Read number of particles */
   fread(&npr,sizeof(int),1,pos); 

  /* Allocate the arrays */
  (*p) = (realT **)malloc(npr*sizeof(realT *));
  ptemp = (realT *)malloc(npr*sizeof(realT));

  printf("np = %d\n",npr);

  /* Fill the arrays */

  for (i=0; i<npr; i++) {
    (*p)[i] = (realT *)malloc(3*sizeof(realT));
    if ((*p)[i] == NULL) {
      printf("Unable to allocate particle array in readfiles!\n");
      fflush(stdout);
      exit(0);
    }
  }

  fread(ptemp,sizeof(realT),npr,pos);   
  for (i=0; i<npr; i++) (*p)[i][0] = ptemp[i];   

  fread(ptemp,sizeof(realT),npr,pos);
  for (i=0; i<npr; i++) (*p)[i][1] = ptemp[i];

  fread(ptemp,sizeof(realT),npr,pos);
  for (i=0; i<npr; i++) (*p)[i][2] = ptemp[i];
   

  fclose(pos);
  free(ptemp);

  /* Get from physical units (Mpc/h) into box units (range 0-1)*/
  for (i=0; i<npr; i++) DL (*p)[i][d] *= fact;


  /* Test range -- can comment out */
  xmin = BF; xmax = -BF; ymin = BF; ymax = -BF; zmin = BF; zmax = -BF;
  for (i=0; i<npr;i++) {
    if ((*p)[i][0]<xmin) xmin = (*p)[i][0]; if ((*p)[i][0]>xmax) xmax = (*p)[i][0];
    if ((*p)[i][1]<ymin) ymin = (*p)[i][1]; if ((*p)[i][1]>ymax) ymax = (*p)[i][1];
    if ((*p)[i][2]<zmin) zmin = (*p)[i][2]; if ((*p)[i][2]>zmax) zmax = (*p)[i][2];
  }
  printf("np: %d, x: %g,%g; y: %g,%g; z: %g,%g\n",npr,xmin,xmax, ymin,ymax, zmin,zmax); fflush(stdout);

  return(npr);
}

/* Velocities */
/* Returns number of particles read */
int velread(char *velfile, realT ***v, realT fact) {

  FILE *vel;
  int npr,dum,d,i;
  realT xmin,xmax,ymin,ymax,zmin,zmax;

  vel = fopen(velfile, "rb");
  if (vel == NULL) {
    printf("Unable to open velocity file %s\n\n",velfile);
    exit(0);
  }
  /* Fortran77 4-byte headers and footers */
  /* Delete "dum" statements if you don't need them */

  /* Read number of particles */
   fread(&npr,sizeof(int),1,vel); 

  /* Allocate the arrays */
  (*v) = (realT **)malloc(3*sizeof(realT*));
  for (i=0;i<3;i++) (*v)[i] = (realT *)malloc(npr*sizeof(realT));

  /* Fill the arrays */
  fread((*v)[0],sizeof(realT),npr,vel); 
  fread((*v)[1],sizeof(realT),npr,vel); 
  fread((*v)[2],sizeof(realT),npr,vel); 

  fclose(vel);

  /* Convert from code units into physical units (km/sec) */
  
  for (i=0; i<npr; i++) DL (*v)[d][i] *= fact;

  /* Test range -- can comment out */
  xmin = BF; xmax = -BF; ymin = BF; ymax = -BF; zmin = BF; zmax = -BF;
  for (i=0; i<npr;i++) {
    if ((*v)[0][i] < xmin) xmin = (*v)[0][i]; if ((*v)[0][i] > xmax) xmax = (*v)[0][i];
    if ((*v)[1][i] < ymin) ymin = (*v)[1][i]; if ((*v)[1][i] > ymax) ymax = (*v)[1][i];
    if ((*v)[2][i] < zmin) zmin = (*v)[2][i]; if ((*v)[2][i] > zmax) zmax = (*v)[2][i];
  }
  printf("vx: %g,%g; vy: %g,%g; vz: %g,%g\n",xmin,xmax, ymin,ymax, zmin,zmax);
  
  return(npr);
}
