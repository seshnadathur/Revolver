#include "voz.h"

int delaunadj (coordT *points, int nvp, int nvpbuf, int nvpall, PARTADJ **adjs);
int vorvol (coordT *deladjs, coordT *points, pointT *intpoints, int numpoints, realT *vol);

int openfile(char *filename, FILE **f);
int posread(char *posfile, realT ***p, realT fact);
int posread_chunk(FILE *f, realT **p, realT fact, int np, int nread);

void voz1b1(char *posfile, realT border, realT boxsize,
	    int numdiv, int b[], char *suffix);

int main(int argc, char *argv[]) {
  int exitcode;
  int i, j, np, np_current, np_tot;
  realT **r;
  coordT rtemp[3], *parts;
  coordT deladjs[3*MAXVERVER], points[3*MAXVERVER];
  pointT intpoints[3*MAXVERVER];
  FILE *pos, *out;
  char *posfile, outfile[80], *suffix;
  PARTADJ *adjs;
  realT *vols;
  realT predict, xmin,xmax,ymin,ymax,zmin,zmax;
  int *orig;
  
  int isitinbuf;
  char isitinmain, d;
  int numdiv, nvp, nvpall, nvpbuf;
  realT width, width2, totwidth, totwidth2, bf, s, g;
  realT border, boxsize;
  realT c[3];
  int b[3];
  realT totalvol;

  if (argc != 9) {
    printf("Wrong number of arguments.\n");
    printf("arg1: position file\n");
    printf("arg2: border size\n");
    printf("arg3: box size\n");
    printf("arg4: suffix\n");
    printf("arg5: number of divisions\n");
    printf("arg6-8: b[0-2]\n\n");
    exit(0);
  }
  posfile = argv[1];
  if (sscanf(argv[2],"%"vozRealSym,&border) != 1) {
    printf("That's no border size; try again.\n");
    exit(0);
  }
  if (sscanf(argv[3],"%"vozRealSym,&boxsize) != 1) {
    printf("That's no boxsize; try again.\n");
    exit(0);
  }
  suffix = argv[4];
  if (sscanf(argv[5],"%d",&numdiv) != 1) {
    printf("%s is no number of divisions; try again.\n",argv[5]);
    exit(0);
  }
  if (numdiv == 1) {
    printf("Only using one division; should only use for an isolated segment.\n");
  }
  if (numdiv < 1) {
    printf("Cannot have a number of divisions less than 1.  Resetting to 1.\n");
    numdiv = 1;
  }
  if (sscanf(argv[6],"%d",&b[0]) != 1) {
    printf("That's no b index; try again.\n");
    exit(0);
  }
  if (sscanf(argv[7],"%d",&b[1]) != 1) {
    printf("That's no b index; try again.\n");
    exit(0);
  }
  if (sscanf(argv[8],"%d",&b[2]) != 1) {
    printf("That's no b index; try again.\n");
    exit(0);
  }

  voz1b1(posfile,border,boxsize,numdiv,b, suffix);

  return 0;
}

