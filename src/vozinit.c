#include "voz.h"

#define DL for (d=0;d<3;d++)
#define BF REALmax

int openfile(char *filename, FILE **f);
int posread(char *posfile, realT ***p, realT fact);
int posread_chunk(FILE *f, realT **p, realT fact, int np, int nread);

int main(int argc, char *argv[]) {
  int i, np, np_current, np_tot = 0;
  realT **rfloat = NULL, rtemp[3];
  FILE *pos, *scr;
  char *posfile, scrfile[80], systemstr[90], *suffix;
  realT xmin,xmax,ymin,ymax,zmin,zmax;
  
  int isitinbuf;
  char isitinmain, d;
  int numdiv;
  int nvp, nvpall, nvpbuf, nvpmin, nvpmax, nvpbufmin, nvpbufmax; /* yes, the insurance */
  realT width, width2, totwidth, totwidth2, bf, s, g;
  realT border, boxsize;
  realT c[3];
  int b[3];

  if (argc != 6) {
    printf("Wrong number of arguments.\n");
    printf("arg1: position file\n");
    printf("arg2: buffer size (default 0.1)\n");
    printf("arg3: box size\n");
    printf("arg4: number of divisions (default 2)\n");
    printf("arg5: suffix describing this run\n\n");
    exit(0);
  }
  posfile = argv[1];
  suffix = argv[2];

  if (sscanf(suffix,"%"vozRealSym,&border) != 1) {
    printf("That's no border size; try again.\n");
    exit(0);
  }
  suffix = argv[3];
  if (sscanf(suffix,"%"vozRealSym,&boxsize) != 1) {
    printf("That's no boxsize; try again.\n");
    exit(0);
  }
  suffix = argv[4];
  if (sscanf(suffix,"%d",&numdiv) != 1) {
    printf("That's no number of divisions; try again.\n");
    exit(0);
  }
  if (numdiv < 2) {
    printf("Cannot have a number of divisions less than 2.  Resetting to 2:\n");
    numdiv = 2;
  }

  suffix = argv[5];

  /* chunked read doesn't allocate memory */
  rfloat = (realT **)malloc(N_CHUNK*sizeof(realT *));
  for (i = 0; i < N_CHUNK; i++) {
    rfloat[i] = (realT *)malloc(3*sizeof(realT));
    if (rfloat[i] == NULL) {
      printf("Unable to allocate particle array!\n");
      fflush(stdout);
      exit(0);
    }
  }

  /* Open the position file */
  np = openfile(posfile, &pos);

  /* Boxsize should be the range in r, yielding a range 0-1 */

  width = 1./(realT)numdiv;
  width2 = 0.5*width;
  if (border > 0.) bf = border;
  else bf = 0.1;

  /* In units of 0-1, the thickness of each subregion's buffer*/
  totwidth = width+2.*bf;
  totwidth2 = width2 + bf;
  
  s = width/(realT)NGUARD;
  if ((bf*bf - 2.*s*s) < 0.) {
    printf("Not enough guard points for given border.\nIncrease guards to >= %g\n.",
	   totwidth/sqrt(0.5*bf*bf));
    printf("bf = %g\n",bf);
    exit(0);
  }
  g = (bf / 2.)*(1. + sqrt(1 - 2.*s*s/(bf*bf)));
  printf("s = %g, bf = %g, g = %g.\n",s,bf,g);
  
  nvpmax = 0; nvpbufmax = 0; nvpmin = np; nvpbufmin = np;
  
  for (b[0] = 0; b[0] < numdiv; b[0]++) {
   c[0] = ((realT)b[0]+0.5)*width;
   for (b[1] = 0; b[1] < numdiv; b[1]++) {
    c[1] = ((realT)b[1]+0.5)*width;
    for (b[2] = 0; b[2] < numdiv; b[2]++) {
      c[2] = ((realT)b[2]+0.5)*width;

      nvp = 0; /* Number of particles excluding buffer */
      nvpbuf = 0; /* Number of particles to tesselate, including
		     buffer */
      xmin = BF; xmax = -BF; ymin = BF; ymax = -BF; zmin = BF; zmax = -BF;

      /* put chunked read here */
      np_tot = 0;
      while(np_tot < np){
	np_current = posread_chunk(pos, rfloat, 1./boxsize, np, np_tot);
	for (i=0; i< np_current; i++) {
	  isitinbuf = 1; isitinmain = 1;
	  for (d=0; d<3; d++) {
	    rtemp[d] = rfloat[i][d] - c[d];
	    if (rtemp[d] > 0.5) rtemp[d] --;
	    if (rtemp[d] < -0.5) rtemp[d] ++;
	    isitinbuf = isitinbuf && (fabs(rtemp[d]) < totwidth2);
	    isitinmain = isitinmain && (fabs(rtemp[d]) <= width2);
	  }
	  if (isitinbuf) {
	    nvpbuf++;
	  }
	  if (isitinmain) {
	    nvp++;
	    if (rtemp[0] < xmin) xmin = rtemp[0];
	    if (rtemp[0] > xmax) xmax = rtemp[0];
	    if (rtemp[1] < ymin) ymin = rtemp[1];
	    if (rtemp[1] > ymax) ymax = rtemp[1];
	    if (rtemp[2] < zmin) zmin = rtemp[2];
	    if (rtemp[2] > zmax) zmax = rtemp[2];
	  }
	}
	np_tot += np_current;
      }
      /* end chunked read here */

      if (nvp > nvpmax) nvpmax = nvp;
      if (nvpbuf > nvpbufmax) nvpbufmax = nvpbuf;
      if (nvp < nvpmin) nvpmin = nvp;
      if (nvpbuf < nvpbufmin) nvpbufmin = nvpbuf;

      printf("b=(%d,%d,%d), c=(%g,%g,%g), nvp=%d, nvpbuf=%d\n",
	     b[0],b[1],b[2],c[0],c[1],c[2],nvp,nvpbuf);
    }
   }
  }

  for (i = 0; i < N_CHUNK; i++) {
    free(rfloat[i]);
  }
  free(rfloat);

  fclose(pos);
  printf("Nvp range: %d,%d\n",nvpmin,nvpmax);
  printf("Nvpbuf range: %d,%d\n",nvpbufmin,nvpbufmax);

  /* Output script file */
  sprintf(scrfile,"scr%s",suffix);
  printf("Writing script file to %s.\n",scrfile);fflush(stdout);
  scr = fopen(scrfile,"w");
  if (scr == NULL) {
    printf("Problem opening script file.\n");
    fflush(stdout);
    exit(0);
  }
  fprintf(scr,"#!/bin/bash -f\n");
  for (b[0]=0;b[0]<numdiv; b[0]++) {
   for (b[1] = 0; b[1] < numdiv; b[1]++) {
    for (b[2] = 0; b[2] < numdiv; b[2]++) {
      fprintf(scr,"bin/voz1b1 %s %g %g %s %d %d %d %d\n",
	     posfile,border,boxsize,suffix,numdiv,b[0],b[1],b[2]);
    }
   }
  }
  fprintf(scr,"bin/voztie %d %s\n",numdiv,suffix);
  fclose(scr);

  sprintf(systemstr,"chmod u+x %s",scrfile);
  system(systemstr);

  return(0);
}
