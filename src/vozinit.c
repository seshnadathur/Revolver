#include "voz.h"

#define DL for (d=0;d<3;d++)
#define BF REALmax
#define FNL 1024 /* Max length of filenames */

int openfile(char *filename, FILE **f);
int posread(char *posfile, realT ***p, realT fact);
int posread_chunk(FILE *f, realT **p, realT fact, int np, int nread);

int main(int argc, char *argv[]) {
  int i, np, np_current, np_tot = 0;
  realT **rfloat = NULL, rtemp[3];
  FILE *pos, *scr;
  char *posfile, scrfile[FNL], systemstr[FNL], *suffix;
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
