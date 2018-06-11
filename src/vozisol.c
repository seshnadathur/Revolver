/*changes made to original version:*/
/*1. all floats changed to realT for consistency with rest of code*/
/*2. used posread() instead of posread_isol() to read particle positions*/
/*3. npreal is taken as an input argument, changed other input arguments too*/
/*4. stores true Voronoi volumes and assigned densities separately*/
#include <ctype.h>
#include "voz.h"

#define DL for (d=0;d<3;d++)
#define PRINTFREQ 10000
/* print out particle volume every PRINTFREQ particles */

int delaunadj (coordT *points, int nvp, int nvpbuf, int nvpall, PARTADJ **adjs);
int vorvol (coordT *deladjs, coordT *points, pointT *intpoints, int numpoints,  realT *vol);
int posread(char *posfile, realT ***p, realT fact);

int main(int argc, char *argv[]) {
  int exitcode;
  int i, j, k, np, npreal;
  realT **r;
  coordT rtemp[3], *parts;
  coordT deladjs[3*MAXVERVER], points[3*MAXVERVER];
  pointT intpoints[3*MAXVERVER];
  FILE *pos, *adj, *vol;
  char adjfile[256], trvolfile[256], volfile[256], *prefix, *posfile;
  char asciiadjfile[261], asciitrvolfile[261], asciivolfile[261];
  PARTADJ *adjs;
  double *vols, *dens;
  realT volley;
  realT predict, xmin,xmax,ymin,ymax,zmin,zmax;
  
  realT width, width2, totwidth, totwidth2, bf, s, g;
  realT border, boxsize, boundarydens;
  realT c[3];
  int hasfewadj;
  int numborder, numfewadj, numinside, nout,maxver;
  double totalvol;

  char d;

  if (argc != 6) {
    printf("Wrong number of arguments.\n");
    printf("arg1: file with positions data\n");
    printf("arg2: file prefix\n");
    printf("arg3: boxsize\n");
    printf("arg4: number of real particles (excluding buffer mocks)\n");
    printf("arg5: density to assign to boundary and edge particles\n");
    exit(0);
  }
  posfile = argv[1];
  prefix = argv[2];
  if (sscanf(argv[3],"%lf",&boxsize) != 1) {
    printf("That's no boxsize; try again.\n");
    exit(0);
  }
  if ((sscanf(argv[4],"%d",&npreal) != 1)||(npreal<1)) {
    printf("That's not a valid number of particles; try again.\n");
    exit(0);
  } 
  if (sscanf(argv[5],"%lf",&boundarydens) == 0) { 
    printf("Bad choice of boundary density.\n");
    exit(0);
  }

  printf("posfile = %s\nprefix = %s\nboxsize = %f\nnpreal = %d\nbounddens = %e\n",posfile,prefix,boxsize,npreal,boundarydens);
  maxver = MAXVERVER;

  /* Boxsize should be the range in r, yielding a range 0-1 */
  np = posread(posfile,&r,1./boxsize);

  /*some standard output checks*/
  printf("%d particles; %d of them real\n",np,npreal); fflush(stdout);
  xmin = BF; xmax = -BF; ymin = BF; ymax = -BF; zmin = BF; zmax = -BF;
  for (i=0; i<np;i++) {
    if (r[i][0]<xmin) xmin = r[i][0]; if (r[i][0]>xmax) xmax = r[i][0];
    if (r[i][1]<ymin) ymin = r[i][1]; if (r[i][1]>ymax) ymax = r[i][1];
    if (r[i][2]<zmin) zmin = r[i][2]; if (r[i][2]>zmax) zmax = r[i][2];
  }
  printf("np: %d, x: %f,%f; y: %f,%f; z: %f,%f\n",np,xmin,xmax, ymin,ymax, zmin,zmax); fflush(stdout);

  /*allocate memory for adjacencies*/
  adjs = (PARTADJ *)malloc(np*sizeof(PARTADJ));
  if (adjs == NULL) {
    printf("Unable to allocate adjs\n");
    exit(0);
  }

  /*move position data from r to parts*/
  parts = (coordT *)malloc(3*np*sizeof(coordT));  
  for (i=0; i<np; i++) {
    parts[3*i] = r[i][0];
    parts[3*i+1] = r[i][1];
    parts[3*i+2] = r[i][2];

    if (r[i][0] < xmin) xmin = r[i][0];
    if (r[i][0] > xmax) xmax = r[i][0];
    if (r[i][1] < ymin) ymin = r[i][1];
    if (r[i][1] > ymax) ymax = r[i][1];
    if (r[i][2] < zmin) zmin = r[i][2];
    if (r[i][2] > zmax) zmax = r[i][2];
  }
  for (i=0;i<np;i++) free(r[i]);
  free(r);
  
  /* Do the tesselation*/
  printf("File read.  Tessellating ...\n"); fflush(stdout);
  exitcode = delaunadj(parts, np, np, np, &adjs);
  
  /* Now calculate Voronoi volumes and densities*/
  printf("\nNow finding volumes ...\n"); fflush(stdout);
  vols = (double *)malloc(np*sizeof(double));
  dens = (double *)malloc(np*sizeof(double));
  numborder = 0;
  numfewadj = 0;
  for (i=0; i<npreal; i++) { /* Cycle over real particles only, ignoring buffer mocks*/
		
    hasfewadj = 0;
    dens[i] = 0.;
    vols[i] = 0.;

    /*check for particles which may have too few adjacencies*/
    if (adjs[i].nadj < 4) {
	hasfewadj = 1;
	numfewadj ++; 
	vols[i] = 1e-30; /*will not be able to calculate Voronoi volumes for such particles*/
	dens[i] = boundarydens;
	if (adjs[i].nadj < 0) { /*for weird case of no boundary it could be -1*/
	  printf("#adj(%d)=%d; on the boundary. Expect warning in jo?o?.\n",i,adjs[i].nadj);
	}
    }

    if (adjs[i].nadj >= 4) {
	/*enough adjacencies, so calculate the Voronoi volume*/
	for (j = 0; j < adjs[i].nadj; j++) {
	  DL {
	    deladjs[3*j + d] = parts[3*adjs[i].adj[j]+d] - parts[3*i+d];
	  }
	}
	exitcode = vorvol(deladjs, points, intpoints, adjs[i].nadj, &(vols[i]));
	if (exitcode) {
		printf("	Error in i=%d, vols[%d]=%f\n",i,i,vols[i]);
	}
	vols[i] *= (double)np;	/*put it in units of the mean*/
	dens[i] = 1.0/vols[i];
    }

    /*check if it is an edge particle adjacent to buffer mocks*/
    for (j = 0; j < adjs[i].nadj; j++) {
	if ((adjs[i].adj[j] >= npreal) || (adjs[i].adj[j] < 0)) {
	  if (hasfewadj == 0) {
	    numborder++;	
	  }
	  /*leave volume unchanged, but set density flag*/
	  dens[i] = boundarydens; 
	  /* Get rid of the adjacencies to buffer mocks outside the boundary*/
	  adjs[i].nadj = j;
	  if (adjs[i].adj[j] < 0) { /*for weird case of no boundary it could be -1*/
	    printf("Negative adj: %d %d %d\n",i,j,adjs[i].adj[j]);
	  }
	  break;
	}
    }
    
    if ((i % PRINTFREQ) == 0) {
      printf("%d: %d, %g\n",i,adjs[i].nadj,vols[i]);fflush(stdout);
    }
  }

  totalvol = 0.;
  numinside=0;
  for (i=0;i<npreal; i++) {
    if (dens[i] != boundarydens) {
      totalvol += vols[i];
      numinside++;
    }
  }
  printf("numborder = %d, numfewadj = %d, numinside = %d\n",numborder,numfewadj, numinside);
  printf("Total %d out of %d\n",numborder+numfewadj+numinside,npreal);

  /*sanity check*/
  if (numborder+numfewadj+numinside != npreal) {
    printf("It doesn't add up!\n");
  }
  printf("Total volume of non-edge = %g\n",totalvol);
  printf("Average volume of non-edge = %g\n",totalvol/(double)numinside);
  
  /* Now write the output to file */

  sprintf(adjfile,"%s.adj",prefix);
  sprintf(asciiadjfile,"%s.ascii.adj",prefix);
  sprintf(trvolfile,"%s.trvol",prefix);
  sprintf(asciitrvolfile,"%s.ascii.trvol",prefix);
  sprintf(volfile,"%s.vol",prefix);
  sprintf(asciivolfile,"%s.ascii.vol",prefix);

  printf("Outputting to %s, %s, %s\n\n",adjfile,trvolfile,volfile);

  adj = fopen(adjfile,"w");
  if (adj == NULL) {
    printf("Unable to open %s\n",adjfile);
    exit(0);
  }
  fwrite(&npreal,1, sizeof(int),adj);

  /* Adjacencies: first the numbers of adjacencies, 
     and the number we're actually going to write per particle */
  for (i=0;i<npreal;i++) {
    if (adjs[i].nadj < 0) {
      adjs[i].nadj = 0; /* In a weird case of no boundary, it could be -1 */
    }
    fwrite(&adjs[i].nadj,1,sizeof(int),adj);
  }
  /* Now the lists of adjacencies (without double counting) */
  for (i=0;i<npreal;i++) {
    nout = 0;
    for (j=0;j<adjs[i].nadj; j++) if (adjs[i].adj[j] > i) nout++;
    fwrite(&nout,1,sizeof(int),adj);      
    for (j=0;j<adjs[i].nadj; j++) 
      if (adjs[i].adj[j] > i) 
	fwrite(&(adjs[i].adj[j]),1,sizeof(int),adj);
  }
  fclose(adj);

  /* Now the true Voronoi volumes */
  vol = fopen(trvolfile,"w");
  if (vol == NULL) {
    printf("Unable to open %s\n",trvolfile);
    exit(0);
  }
  fwrite(&npreal,1, sizeof(int),vol);
  for (i=0;i<npreal;i++) {
    volley = (realT)vols[i];
    fwrite(&volley,sizeof(realT),1,vol);
  }
  fclose(vol);

  /* And the volumes with edge particles masked*/
  vol = fopen(volfile,"w");
  if (vol == NULL) {
    printf("Unable to open %s\n",volfile);
    exit(0);
  }
  fwrite(&npreal,1, sizeof(int),vol);
  for (i=0;i<npreal;i++) {
    volley = (realT)1.0/dens[i];
    fwrite(&volley,sizeof(realT),1,vol);
  }
  fclose(vol);

  return(0);
}

