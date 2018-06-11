/*  checkedges.c written by Seshadri Nadathur
	Code to read output of Voronoi tessellations, 
	delete buffer mocks and identify edge particles
	for later use by jozov.c
*/
#include <ctype.h>
#include "voz.h"

#define FF fflush(stdout)
#define FNL 256 /* Max length of filenames */

typedef struct Particle {
  int nadj;
  int nadj_count;
  int *adj;
} PARTICLE;

int main(int argc, char *argv[]) {
  int exitcode;
  int i, j, k, nin, np, npreal;
  FILE *pos, *adj, *vol;
  PARTICLE *p;
  char adjfile[256], trvolfile[256], volfile[256], *prefix;
  PARTADJ *adjs;
  double *vols, *dens;
  realT volley;
  
  realT boundarydens;
  int numborder, numfewadj, numinside, nout,maxver;
  double totalvol;

  char d;

  if (argc != 4) {
    printf("Wrong number of arguments.\n");
    printf("arg1: file prefix\n");
    printf("arg2: number of real particles (excluding buffer mocks)\n");
    printf("arg3: density to assign to boundary and edge particles\n");
    exit(0);
  }
  prefix = argv[1];
  sprintf(adjfile,"%s.adj",prefix);
  sprintf(volfile,"%s.vol",prefix);
  sprintf(trvolfile,"%s.trvol",prefix);
  if ((sscanf(argv[2],"%d",&npreal) != 1)||(npreal<1)) {
    printf("That's not a valid number of particles; try again.\n");
    exit(0);
  } 
  if (sscanf(argv[3],"%lf",&boundarydens) == 0) { 
    printf("Bad choice of boundary density.\n");
    exit(0);
  }

  /*open adjacency file and read number of particles*/
  adj = fopen(adjfile, "r");
  if (adj == NULL) {
    printf("Unable to open %s\n",adjfile);
    exit(0);
  }
  fread(&np,1, sizeof(int),adj);

  /*allocate memory for adjacencies*/
  adjs = (PARTADJ *)malloc(np*sizeof(PARTADJ));
  if (adjs == NULL) {
    printf("Unable to allocate adjs\n");
    exit(0);
  }

  p = (PARTICLE *)malloc(np*sizeof(PARTICLE));
  /* Read and assign adjacencies*/
  for (i=0;i<np;i++) {
    fread(&p[i].nadj,1,sizeof(int),adj); 
    /* The number of adjacencies per particle */
    if (p[i].nadj > 0) {
      p[i].adj = (int *)malloc(p[i].nadj*sizeof(int));
    }
    p[i].nadj_count = 0;
  }
  for (i=0;i<np;i++) {
    fread(&nin,1,sizeof(int),adj);
    if (nin > 0)
      for (k=0;k<nin;k++) {
	fread(&j,1,sizeof(int),adj);
	if (j < np) {
	  /* Set both halves of the pair */
	  if ((p[i].nadj_count < p[i].nadj) && (p[j].nadj_count < p[j].nadj)){
	    p[i].adj[p[i].nadj_count] = j;
	    p[j].adj[p[j].nadj_count] = i;
	    p[i].nadj_count++; p[j].nadj_count++;
	  } else{
	    printf("weird#adj,p %d or %d\t",i,j);
	  }
	} else {
	  printf("adj(%d)=%d>np\n",i,j); FF;
	}
      }
  }
  fclose(adj);
  printf("\n");

  /* Check that we got all the pairs */
  adj = fopen(adjfile, "r");
  fread(&np,1, sizeof(int),adj);
  for (i=0;i<np;i++) {
    fread(&nin,1,sizeof(int),adj); /* actually nadj */
    if (nin != p[i].nadj) {
      printf("We didn't get all of %d's adj's; %d != %d.\n",i,nin,p[i].nadj);
      /*exit(0);*/
    }
  }
  fclose(adj);
 
  /*read in the true volume information*/
  printf("Identifying edge particles\n");
  vols = (double *)malloc(npreal*sizeof(double));
  dens = (double *)malloc(npreal*sizeof(double));
  vol = fopen(trvolfile, "r");
  if (vol == NULL) {
    printf("Unable to open %s\n",trvolfile);
    exit(0);
  }
  fread(&nin,1, sizeof(int),vol);
  if (nin!=np) {
    printf("Np=%d in .trvol file does not match Np=%d in .adj file!",nin,np);
    exit(0);
  }
  for (i=0;i<npreal;i++) {
    fread(&vols[i],1,sizeof(realT),vol);
  }
  fclose(vol);
 
  /* Now check adjacencies and flag edge particles*/
  numborder = 0;
  numfewadj = 0;
  for (i=0; i<npreal; i++) { /* Cycle over real particles only, ignoring buffer mocks*/
		
    dens[i] = 1./vols[i];
    /*check if it is an edge particle adjacent to buffer mocks*/
    for (j = 0; j < p[i].nadj; j++) {
	if (p[i].adj[j] >= npreal) {
	  numborder++;	
	  /*leave volume unchanged, but set density flag*/
	  dens[i] = boundarydens; 
	  /* Get rid of the adjacencies to buffer mocks outside the boundary*/
	  p[i].nadj = j;
	}
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
  printf("numborder = %d, numinside = %d\n",numborder,numinside);
  printf("Total %d out of %d\n",numborder+numinside,npreal);
  printf("Total volume of non-edge = %g\n",totalvol);
  printf("Average volume of non-edge = %g\n",totalvol/(double)numinside);
  
  /* Now write the output to file */

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
    if (p[i].nadj < 0) {
      p[i].nadj = 0; /* In a weird case of no boundary, it could be -1 */
    }
    fwrite(&p[i].nadj,1,sizeof(int),adj);
  }
  /* Now the lists of adjacencies (without double counting) */
  for (i=0;i<npreal;i++) {
    nout = 0;
    for (j=0;j<p[i].nadj; j++) if (p[i].adj[j] > i) nout++;
    fwrite(&nout,1,sizeof(int),adj);      
    for (j=0;j<p[i].nadj; j++) 
      if (p[i].adj[j] > i) 
	fwrite(&(p[i].adj[j]),1,sizeof(int),adj);
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

