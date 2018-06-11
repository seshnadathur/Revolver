/* jozov.c by Mark Neyrinck */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "voz.h"

#define NLINKS 1000 /* 100 Number of possible links with the same rho_sl */

typedef struct Particle {
  realT vol;
  int nadj;
  int *adj;
} PARTICLE;

typedef struct Zone {
  int core; /* Identity of peak particle */
  realT corevol;
  int np; /* Number of particles in zone */
  int npjoin; /* Number of particles in the joined zone */
  int nadj;
  realT leak; /* Volume of last leak zone*/
  int *adj; /* Each adjacent zone, with ... */
  realT *slv; /* Smallest Linking Volume */
  realT prob; /* probability that it is fake */
} ZONE;

typedef struct ZoneT {
  int nadj; /* Number of particles on border */
  int *adj; /* Each adjacent zone, with ... */
  realT *slv; /* Smallest Linking Volume */
} ZONET;

/* None of these are currently used */
#define NJIGSJOIN 2048
#define NJIGSUNB 32
#define NPROC 256 
#define SAVEMOD 1
#define REMDUP 16384

void findrtop(realT *a, int na, int *iord, int nb);

int compar(const void * n1, const void * n2) {
  int i1,i2;

  i1 = *(int *)n1;
  i2 = *(int *)n2;
  return 2*(i1 > i2) - 1 + (i1 == i2);
}

int main(int argc,char **argv) {

  FILE *adj, *vol, *zon, *txt;
  PARTICLE *p;
  ZONE *z;
  ZONET *zt;
  char *adjfile, *volfile, *zonfile, *txtfile;
  int i, j,k,l, h, h2,hl,n,np, np2, nhaloes, nhl, nhlcount, nhl2;
  int *jumper, *jumped, *numinh;
  int *halonum, *halolist, *halolist2;
  int link[NLINKS], link2, nl;
  realT lowvol, voltol;

  int za, nin;
  int testpart;
  char already, interior, *inyet, *inyet2, added, beaten;
  int *nm, **m;

  realT maxvol, minvol;
  realT *sorter, e1,maxprob;
  int *iord;

  e1 = exp(1.)-1.;

  if (argc != 6) {
    printf("Wrong number of arguments.\n");
    printf("arg1: adjacency file\n");
    printf("arg2: volume file\n");
    printf("arg3: output zone membership file\n");
    printf("arg4: output text file\n");
    printf("arg5: volume tolerance (e.g. 1)\n\n");
    exit(0);
  }
  adjfile = argv[1];
  volfile = argv[2];
  zonfile = argv[3];
  txtfile = argv[4];
  if (sscanf(argv[5],"%"vozRealSym,&voltol) == 0) {
    printf("Bad volume tolerance.\n");
    exit(0);
  }

  adj = fopen(adjfile, "r");
  if (adj == NULL) {
    printf("Unable to open %s\n",adjfile);
    exit(0);
  }
  fread(&np,1, sizeof(int),adj);
  
  p = (PARTICLE *)malloc(np*sizeof(PARTICLE));
  /* Adjacencies*/
  for (i=0;i<np;i++) {
    fread(&p[i].nadj,1,sizeof(int),adj); 
    /* The number of adjacencies per particle */
    if (p[i].nadj > 0)
      p[i].adj = (int *)malloc(p[i].nadj*sizeof(int));
    p[i].nadj = 0; /* Temporarily, it's an adj counter */
  }
  for (i=0;i<np;i++) {
    fread(&nin,1,sizeof(int),adj);
    if (nin > 0)
      for (k=0;k<nin;k++) {
	fread(&j,1,sizeof(int),adj);
	
	/* Set both halves of the pair */
	p[i].adj[p[i].nadj] = j;
	p[j].adj[p[j].nadj] = i;
	p[i].nadj++; p[j].nadj++;
      }
  }
  fclose(adj);

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

  /* Volumes */
  vol = fopen(volfile, "r");
  if (vol == NULL) {
    printf("Unable to open volume file %s.\n\n",volfile);
    exit(0);
  }
  fread(&np2,1, sizeof(int),adj);
  if (np != np2) {
    printf("Number of particles doesn't match! %d != %d\n",np,np2);
    exit(0);
  }
  for (i=0;i<np;i++) {
    fread(&p[i].vol,1,sizeof(realT),vol);
    if ((p[i].vol < 1e-30) || (p[i].vol > 1e30)) {
      printf("Whacked-out volume found, of particle %d: %g\n",i,p[i].vol);
      p[i].vol = 1.;
    }
  }
  fclose(vol);

  jumped = (int *)malloc(np*sizeof(int));  
  jumper = (int *)malloc(np*sizeof(int));
  numinh = (int *)malloc(np*sizeof(int));

  /* find jumper */
  for (i = 0; i < np; i++) {
    minvol = p[i].vol; jumper[i] = -1;
    for (j=0; j< p[i].nadj; j++) {
      if (p[p[i].adj[j]].vol < minvol) {
	jumper[i] = p[i].adj[j];
	minvol = p[jumper[i]].vol;
      }
    }
    numinh[i] = 0;
  }

  printf("About to jump ...\n"); fflush(stdout);
  
  /* Jump */
  for (i = 0; i < np; i++) {
    jumped[i] = i;
    while (jumper[jumped[i]] > -1)
      jumped[i] = jumper[jumped[i]];
    numinh[jumped[i]]++;
  }
  printf("Post-jump ...\n"); fflush(stdout);
  
  nhaloes = 0;
  for (i = 0; i < np; i++)
    if (numinh[i] > 0) nhaloes++;
  printf("%d initial haloes found\n", nhaloes);

  z = (ZONE *)malloc(nhaloes*sizeof(ZONE));
  if (z == NULL) {
    printf("Unable to allocate z\n");
    exit(0);
  }
  zt = (ZONET *)malloc(nhaloes*sizeof(ZONET));
  if (zt == NULL) {
    printf("Unable to allocate zt\n");
    exit(0);
  }
  for (h=0;h<nhaloes;h++) {
    zt[h].nadj = 0;
  }

  halonum = (int *)malloc(np*sizeof(int));
  if (halonum == NULL) {
    printf("Unable to allocate halonum\n");
    exit(0);
  }

  h = 0;
  for (i = 0; i < np; i++)
    if (numinh[i] > 0) {
      z[h].core = i;
      halonum[i] = h;
      h++;
    } else {
      halonum[i] = -1;
    }
 
  /* Count border particles */
  for (i = 0; i < np; i++)
    for (j = 0; j < p[i].nadj; j++) {
      testpart = p[i].adj[j];
      if (jumped[i] != jumped[testpart]) {
	zt[halonum[jumped[i]]].nadj++;
      }
    }
  
  for (h=0;h<nhaloes;h++) {
    zt[h].adj = (int *)malloc(zt[h].nadj*sizeof(int));
    if (zt[h].adj == NULL) {
      printf("Unable to allocate %d adj's of zone %d\n",zt[h].nadj,h);
      exit(0);
    }
    zt[h].slv = (realT *)malloc(zt[h].nadj*sizeof(realT));
    if (zt[h].slv == NULL) {
      printf("Unable to allocate %d slv's of zone %d\n",zt[h].nadj,h);
      exit(0);
    }
    zt[h].nadj = 0;
  }

  /* Find "weakest links" */
  for (i = 0; i < np; i++) {
    h = halonum[jumped[i]];
    for (j = 0; j < p[i].nadj; j++) {
      testpart = p[i].adj[j];
      if (h != halonum[jumped[testpart]]) {
	if (p[testpart].vol > p[i].vol) {
	  /* there could be a weakest link through testpart */
	  already = 0;
	  for (za = 0; za < zt[h].nadj; za++)
	    if (zt[h].adj[za] == halonum[jumped[testpart]]) {
	      already = 1;
	      if (p[testpart].vol < zt[h].slv[za]) {
		zt[h].slv[za] = p[testpart].vol;
	      }
	    }
	  if (already == 0) {
	    zt[h].adj[zt[h].nadj] = halonum[jumped[testpart]];
	    zt[h].slv[zt[h].nadj] = p[testpart].vol;
	    zt[h].nadj++;
	  }
	} else { /* There could be a weakest link through i */
	  already = 0;
	  for (za = 0; za < zt[h].nadj; za++)
	    if (zt[h].adj[za] == halonum[jumped[testpart]]) {
	      already = 1;
	      if (p[i].vol < zt[h].slv[za]) {
		zt[h].slv[za] = p[i].vol;
	      }
	    }
	  if (already == 0) {
	    zt[h].adj[zt[h].nadj] = halonum[jumped[testpart]];
	    zt[h].slv[zt[h].nadj] = p[i].vol;
	    zt[h].nadj++;
	  }
	}
      }
    }
  }
  printf("Found zone adjacencies\n"); fflush(stdout);

  /* Free particle adjacencies */
  for (i=0;i<np; i++) free(p[i].adj);

  /* Use z instead of zt */
  for (h=0;h<nhaloes;h++) {
    /*printf("%d ",zt[h].nadj);*/
    z[h].nadj = zt[h].nadj;
    z[h].adj = (int *)malloc(zt[h].nadj*sizeof(int));
    z[h].slv = (realT *)malloc(zt[h].nadj*sizeof(realT));
    for (za = 0; za<zt[h].nadj; za++) {
      z[h].adj[za] = zt[h].adj[za];
      z[h].slv[za] = zt[h].slv[za];
    }
    free(zt[h].adj);
    free(zt[h].slv);
    z[h].np = numinh[z[h].core];
  }
  free(zt);
  free(numinh);

  m = (int **)malloc(nhaloes*sizeof(int *));
  nm = (int *)malloc(nhaloes*sizeof(int));
  for (h=0; h<nhaloes; h++) {
    m[h] = (int *)malloc(z[h].np*sizeof(int));
    nm[h] = 0;
  }
  for (i=0; i<np; i++) {
    h = halonum[jumped[i]];
    m[h][nm[h]] = i;
    nm[h] ++;
  }
  free(nm);

  zon = fopen(zonfile,"w");
  if (zon == NULL) {
    printf("Problem opening zonefile %s.\n\n",zonfile);
    exit(0);
  }
  fwrite(&np,1,sizeof(int),zon);
  fwrite(&nhaloes,1,sizeof(int),zon);
  for (h=0; h<nhaloes; h++) {
    fwrite(&(z[h].np),1,sizeof(int),zon);
    fwrite(m[h],z[h].np,sizeof(int),zon);
    free(m[h]);
  }
  free(m);

  inyet = (char *)malloc(nhaloes*sizeof(char));
  inyet2 = (char *)malloc(nhaloes*sizeof(char));
  halolist = (int *)malloc(nhaloes*sizeof(int));
  halolist2 = (int *)malloc(nhaloes*sizeof(int));
  sorter = (realT *)malloc((nhaloes+1)*sizeof(realT));

  for (h = 0; h< nhaloes; h++) {
    inyet[h] = 0;
    inyet2[h] = 0;
  }

  nhl = 0; 

  maxvol = 0.;
  minvol = BF;
  maxprob = 0.;
  for(i=0;i<np; i++){
    if (p[i].vol > maxvol) maxvol = p[i].vol;
    if (p[i].vol < minvol) minvol = p[i].vol;
  }
  printf("Volumes range from %e to %e.\n",minvol,maxvol);fflush(stdout);

  for (h = 0; h<nhaloes; h++) {
    nhlcount = 0;
    for (hl = 0; hl < nhl; hl++)
      inyet[halolist[hl]] = 0;

    halolist[0] = h;
    inyet[h] = 1;
    nhl = 1;
    z[h].npjoin = z[h].np;
    do {
      /* Find the lowest-volume (highest-density) adjacency */
      lowvol = BF; nl = 0; beaten = 0;
      for (hl = 0; hl < nhl; hl++) {
	h2 = halolist[hl];
	if (inyet[h2] == 1) { /* If it's not already identified as 
				 an interior zone, with inyet=2 */
	  interior = 1;
	  for (za = 0; za < z[h2].nadj; za ++) {
	    if (inyet[z[h2].adj[za]] == 0) {
	      interior = 0;
	      if (z[h2].slv[za] == lowvol) {
		link[nl] = z[h2].adj[za];
		nl ++;
		if (nl == NLINKS) {
		  printf("Too many links with the same rho_sl!  Increase NLINKS from %d\n",nl);
		  exit(0);
		}
	      }
	      if (z[h2].slv[za] < lowvol) {
		lowvol = z[h2].slv[za];
		link[0] = z[h2].adj[za];
		nl = 1;
	      }
	    }
	  }
	  if (interior == 1) inyet[h2] = 2; /* No bordering exter. zones */
	}
      }

      if (nl == 0) {
	beaten = 1;
	z[h].leak = maxvol;
	continue;
      }
	
      if (lowvol > voltol) {
	beaten = 1;
	z[h].leak = lowvol;
	continue;
      }

      for (l=0; l < nl; l++)
	if (p[z[link[l]].core].vol < p[z[h].core].vol)
	  beaten = 1;
      if (beaten == 1) {
	z[h].leak = lowvol;
	continue;
      }
      /* Add everything linked to the link(s) */
      nhl2 = 0;
      for (l=0; l < nl; l++) {
	if (inyet2[link[l]] == 0) {
	  halolist2[nhl2] = link[l];
	  inyet2[link[l]] = 1;
	  nhl2 ++;
	  added = 1;
	  while ((added == 1) && (beaten == 0)) {
	    added = 0;
	    for (hl = 0; (hl < nhl2) && (beaten == 0); hl++) {
	      h2 = halolist2[hl];
	      if (inyet2[h2] == 1) {
		interior = 1; /* Guilty until proven innocent */
		for (za = 0; za < z[h2].nadj; za ++) {
		  link2 = z[h2].adj[za];
		  if ((inyet[link2]+inyet2[link2]) == 0) {
		    interior = 0;
		    if (z[h2].slv[za] <= lowvol) {
		      if (p[z[link2].core].vol < p[z[h].core].vol) {
			beaten = 1;
			break;
		      }
		      halolist2[nhl2] = link2;
		      inyet2[link2] = 1;
		      nhl2++;
		      added = 1;
		    }
		  }
		}
		if (interior == 1) inyet2[h2] = 2;
	      }
	    }
	  }
	}
      }
      for (hl = 0; hl < nhl2; hl++)
	inyet2[halolist2[hl]] = 0;
      
      /* See if there's a beater */
      if (beaten == 1) {
	z[h].leak = lowvol;
      } else {
	for (h2 = 0; h2 < nhl2; h2++) {
	  halolist[nhl] = halolist2[h2];
	  inyet[halolist2[h2]] = 1;
	  nhl++;
	  z[h].npjoin += z[halolist2[h2]].np;
	}
      }
      if (nhl/10000 > nhlcount) {
	nhlcount = nhl/10000;
	printf(" %d",nhl); fflush(stdout);
      }
    } while((lowvol < BF) && (beaten == 0));
    
    z[h].prob = z[h].leak/p[z[h].core].vol;
    if (z[h].prob < 1.) z[h].prob = 1.;
    
    /* find biggest prob */
    if (z[h].prob > maxprob) {
      maxprob = (realT)z[h].prob;
    }

    /* Sort them, just for the hell of it */
    /* No, don't!  Want the core zone to be first 
       qsort((void *)halolist, nhl, sizeof(int), &compar); */

    if (nhlcount > 0) {
      printf(" h%d:%d\n",h,nhl);
      fflush(stdout);
    }

    /* output to file */
    fwrite(&z[h].prob,1,sizeof(realT),zon);
    fwrite(&z[h].core,1,sizeof(int),zon);
    fwrite(&(p[z[h].core].vol),1,sizeof(realT),zon);
    fwrite(&nhl,1,sizeof(int),zon);
    fwrite(halolist,nhl,sizeof(int),zon);
  }
  fclose(zon);

  printf("Maxprob = %g.\n",maxprob);

  /* Assign sorter : first by halo size, then probability  */
  for (h=0; h< nhaloes; h++)
    sorter[h] = (realT)z[h].npjoin + log((e1*(realT)z[h].prob+maxprob)/maxprob);
    
  /* Text output file */

  printf("about to sort ...\n");fflush(stdout);

  iord = (int *)malloc(nhaloes*sizeof(int));

  findrtop(sorter, nhaloes, iord, nhaloes);

  txt = fopen(txtfile,"w");
  fprintf(txt,"%d\n", nhaloes);
  for (h=0; h<nhaloes; h++) {
    fprintf(txt,"%d\t%d\t%e\t%e\n",iord[h],z[iord[h]].npjoin,
	    z[iord[h]].prob,z[iord[h]].leak);
  }
  fclose(txt);

  return(0);
}
