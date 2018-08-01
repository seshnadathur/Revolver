/* jozov-grid.c written by Sesh Nadathur
    This code examines a density field on a grid, identifies
    locations of density minima and finds watershed void
    regions around them. Has an option to invert algorithm
    to find maxima (and 'supercluster' regions). Based on
    the original jozov watershed code by Mark Neyrinck,
    which applied to tessellation-estimated densities.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define NLINKS 300000 /* Number of possible links with the same dens_sl */
#define BF 100000000 /* This is simply a number very much larger than the range in density */
#define FF fflush(stdout)
#define FNL 256 /* Max length of filenames */

typedef struct Voxel {
  float dens;
  long *adj;
  
} VOXEL;

typedef struct Zone {
  long core; /* Identity of minimum-density voxel */
  long np; /* Number of voxels in zone */
  long npjoin; /* Number of voxels in the joined structure */
  int nadj; /* Number of adjacent zones */
  int nhl; /* Number of zones in final joined structure */
  float leak; /* density of last leak zone*/
  int *adj; /* Each adjacent zone, with ... */
  float *slv; /* lowest linking density */
  long *vox; /* a list of voxels directly contained in this zone */
  float densratio; /* density ratio */
  int edge;
} ZONE;

typedef struct ZoneT {
  int nadj; /* Number of zones on border */
  int *adj; /* Each adjacent zone, with ... */
  float *slv; /* lowest linking density */

} ZONET;


int main(int argc,char **argv) {

  FILE *dens, *zon, *vod, *txt; 
  VOXEL *p;
  ZONE *z;
  ZONET *zt;
  char *prefix, *densfile, zonefile[FNL], voidfile[FNL], txtfile[FNL]; 
  int h, h2, hl, n, nzones, nrealzones, nhl, nhlcount, nhl2, minusone;
  long i, Nvox; 
  int ix, iy, iz, j, l, ind, Nside;
  int neighbours[18] = {0,0,1,0,0,-1,0,1,0,0,-1,0,1,0,0,-1,0,0};
  int nbrs[18];
  long *jumper, *jumped, *numinh, *realnuminh;
  int *zonenum, *zonelist, *zonelist2, *voxcounter;
  int link[NLINKS], link2, nl;
  float lowdens, borderdens;

  int za;
  long testpart;
  char already, interior, *inyet, *inyet2, added, beaten, obo;

  float maxdens, mindens;
  float maxdensratio;

  if (argc != 5) {
    printf("Wrong number of arguments.\n");
    printf("arg1: voids (dens minima) or clusters (dens maxima)? (v/c)\n");
    printf("arg2: filename for file with dens data\n");
    printf("arg3: prefix to use for filenames\n");
    printf("arg4: dens grid resolution N (for NxNxN voxels)\n");
    exit(0);
  }

  if (sscanf(argv[1],"%c",&obo) == 0) {
    printf("Bad choice of (v)oids or (c)lusters.\n");
    exit(0);
  }  
  obo = tolower(obo);
  if (obo == 'v'){
    printf("\nFinding density minima\n");
    borderdens = 0.9e30;
  }
  else if (obo == 'c'){
    printf("\nFinding density maxima\n");
    borderdens = 0.9e30;
  }
  else{
    printf("1st arg: please enter v (for finding dens. minima) or c (for dens. maxima).\n\n");
    exit(0);
  }
  densfile = argv[2];
  prefix = argv[3];
  if (sscanf(argv[4],"%d",&Nside) == 0) {
    printf("Bad grid size.\n");
    exit(0);
  }
  Nvox = Nside*Nside*Nside;	/* the total number of voxels (grid cells) */  
  printf("Total number of voxels: %ld\n",Nvox);

  if (obo == 'v'){
    sprintf(zonefile,"%s.zone",prefix);
    sprintf(voidfile,"%s.void",prefix);
    sprintf(txtfile,"%s.txt",prefix);
  }
  if (obo == 'c'){
    sprintf(zonefile,"%sc.zone",prefix);
    sprintf(voidfile,"%sc.void",prefix);
    sprintf(txtfile,"%sc.txt",prefix);
  }

  /* Read the density data from file */
  printf("Reading in the dens data from file ...\n"); FF;
  p = (VOXEL *)malloc(Nvox*sizeof(VOXEL));
  dens = fopen(densfile, "r");
  if (dens == NULL) {
    printf("Unable to open dens file %s.\n\n",densfile);
    exit(0);
  }
  for (i=0;i<Nvox;i++) {
    fread(&p[i].dens,1,sizeof(float),dens); 
    /* if finding minima, invert the densities and then algorithm proceeds the same */
    if (obo=='c') p[i].dens = 1./p[i].dens;
  }
  fclose(dens);
  printf("Debug: dens[0] = %0.4e, dens[%ld] = %0.4e\n",p[0].dens,Nvox-1,p[Nvox-1].dens);

  /* Set the adjacencies */
  printf("Setting the voxel adjacencies ...\n"); FF;
  for (i=0;i<Nvox;i++) {
    if (p[i].dens < borderdens) {
      /* each non-border voxel is adjacent to 6 other voxels */
      p[i].adj = (long *)malloc(6*sizeof(long));
      /* convert central voxel index to 3D */
      iz = i%Nside;
      iy = i%(Nside*Nside)/Nside;
      ix = i/(Nside*Nside);
      /* find the 3D indices of neighbouring voxels */
      for (j=0;j<6;j++) {
        nbrs[j*3] = neighbours[j*3] + ix;
        nbrs[j*3+1] = neighbours[j*3+1] +iy;
        nbrs[j*3+2] = neighbours[j*3+2] +iz;
      }
      /* adjust for periodic boundary conditions */
      for (j=0;j<18;j++) {
        if (nbrs[j]>=Nside) nbrs[j] -= Nside;
        if (nbrs[j]<0) nbrs[j] += Nside;
      }
      /* convert these to flattened indices and set adjacencies */
      for (j=0;j<6;j++) {
        ind = 3*j;
        p[i].adj[j] = nbrs[ind]*Nside*Nside + nbrs[ind+1]*Nside + nbrs[ind+2];
      }
    }
  }

  jumped = (long *)malloc(Nvox*sizeof(long));  
  jumper = (long *)malloc(Nvox*sizeof(long));
  numinh = (long *)malloc(Nvox*sizeof(long));
  realnuminh = (long *)malloc(Nvox*sizeof(long));

  /* find jumper - every voxel jumps to its lowest neighbour */
  printf("Finding jumper for each voxel\n"); FF;
  for (i = 0; i < Nvox; i++) {
    mindens = p[i].dens; jumper[i] = -1;
    if (p[i].dens < borderdens) {
      for (j=0; j<6; j++) {
        if (p[p[i].adj[j]].dens < mindens) {
          jumper[i] = p[i].adj[j];
	      mindens = p[jumper[i]].dens;
        }
      }
    }
    numinh[i] = 0;
    realnuminh[i] = 0;
  }

  /* Jump along the chain */
  printf("About to jump ...\n"); FF;
  for (i = 0; i < Nvox; i++) {
    jumped[i] = i;
    while (jumper[jumped[i]] > -1)
      jumped[i] = jumper[jumped[i]];
    numinh[jumped[i]]++;	/* count all voxels in the basin */
    if (p[i].dens < borderdens) realnuminh[jumped[i]]++;
  }
  printf("Post-jump ...\n"); FF;
  
  /* count the number of zones */
  nzones = 0; nrealzones = 0;
  for (i = 0; i < Nvox; i++) 
    if (numinh[i] > 0) {
      nzones++;	
      if (realnuminh[i] > 0) nrealzones++;	
    }
  printf("%d initial zones found...\n  of which %d are 'real'\n",nzones,nrealzones);

  /* allocate the zone variables */
  z = (ZONE *)malloc(nzones*sizeof(ZONE));
  if (z == NULL) {
    printf("Unable to allocate z\n");
    exit(0);
  }
  zt = (ZONET *)malloc(nzones*sizeof(ZONET));
  if (zt == NULL) {
    printf("Unable to allocate zt\n");
    exit(0);
  }
  
  /* initialize zone adjacencies */
  for (h=0;h<nzones;h++) {
    zt[h].nadj = 0;
  }
  zonenum = (int *)malloc(Nvox*sizeof(int));
  if (zonenum == NULL) {
    printf("Unable to allocate zonenum\n");
    exit(0);
  }

  h = 0;
  for (i = 0; i < Nvox; i++)
    if (numinh[i] > 0) {  /* here we are counting all zones */
      z[h].core = i;
      zonenum[i] = h;
      h++;
    } else {	
      zonenum[i] = -1;
    }
 
  /* Finding particles on zone borders */ 
  printf("Finding zone borders\n");
  for (i = 0; i < Nvox; i++)
    if (p[i].dens < borderdens) {
      for (j = 0; j < 6; j++) {
        testpart = p[i].adj[j];
        if (jumped[i] != jumped[testpart])
	      /* two neighbouring voxels jump to different zones */
	      zt[zonenum[jumped[i]]].nadj++;
	  }
    }
  
  printf("Allocating zone adjacencies and links\n");
  for (h=0;h<nzones;h++) {
    zt[h].adj = (int *)malloc(zt[h].nadj*sizeof(int));
    if (zt[h].adj == NULL) {
      printf("Unable to allocate %d adj's of zone %d\n",zt[h].nadj,h);
      exit(0);
    }
    zt[h].slv = (float *)malloc(zt[h].nadj*sizeof(float));
    if (zt[h].slv == NULL) {
      printf("Unable to allocate %d slv's of zone %d\n",zt[h].nadj,h);
      exit(0);
    }
    zt[h].nadj = 0;	/* resets number of adjacencies to zero */
  }

  /* Find "weakest links" - i.e. the lowest-density link */
  printf("Finding weakest links\n");
  for (i = 0; i < Nvox; i++) {
    h = zonenum[jumped[i]];
    if (p[i].dens < borderdens) {
      for (j = 0; j < 6; j++) {
        testpart = p[i].adj[j];
        if (h != zonenum[jumped[testpart]]) {
	      if (p[testpart].dens < p[i].dens) {
	        /* there could be a weakest link through testpart */
	        already = 0;
	        for (za = 0; za < zt[h].nadj; za++)
	          if (zt[h].adj[za] == zonenum[jumped[testpart]]) {
	            already = 1;
	            if (p[testpart].dens < zt[h].slv[za]) {
		          zt[h].slv[za] = p[testpart].dens;
	            }
	          }
	        if (already == 0) {
	          zt[h].adj[zt[h].nadj] = zonenum[jumped[testpart]];
	          zt[h].slv[zt[h].nadj] = p[testpart].dens;
	          zt[h].nadj++;
	        }
	      }
	      else {
	        /* There could be a weakest link through i */
	        already = 0;
	        for (za = 0; za < zt[h].nadj; za++)
	          if (zt[h].adj[za] == zonenum[jumped[testpart]]) {
	            already = 1;
	            if (p[i].dens < zt[h].slv[za]) {
		          zt[h].slv[za] = p[i].dens;
	            }
	          }
	        if (already == 0) {
	          zt[h].adj[zt[h].nadj] = zonenum[jumped[testpart]];
	          zt[h].slv[zt[h].nadj] = p[i].dens;
	          zt[h].nadj++;
	        }
	      }
        }
      }
    }
  }
  printf("Found zone adjacencies\n"); FF;

  /* Free voxel adjacencies */ 
  for (i=0;i<Nvox; i++) free(p[i].adj);

  /* Use z instead of zt */ 
  for (h=0;h<nzones;h++) {
    z[h].nadj = zt[h].nadj;
    z[h].adj = (int *)malloc(zt[h].nadj*sizeof(int));
    z[h].slv = (float *)malloc(zt[h].nadj*sizeof(float));
    for (za = 0; za<zt[h].nadj; za++) {
      z[h].adj[za] = zt[h].adj[za];
      z[h].slv[za] = zt[h].slv[za];
    }
    free(zt[h].adj);
    free(zt[h].slv);
    z[h].np = realnuminh[z[h].core]; /* only count member voxels that are not flagged (using borderdens) */
    z[h].vox = (long *)malloc(realnuminh[z[h].core]*sizeof(long));
    if (realnuminh[z[h].core] == numinh[z[h].core]) {
      z[h].edge = 0;  /* no border flagged voxels in this zone so edge flag is zero */
    }
    else z[h].edge = 1;  /* else flag it as an edge zone */
  }
  free(zt);
  free(numinh);
  free(realnuminh);

  inyet = (char *)malloc(nzones*sizeof(char));
  inyet2 = (char *)malloc(nzones*sizeof(char));
  zonelist = (int *)malloc(nzones*sizeof(int));
  zonelist2 = (int *)malloc(nzones*sizeof(int));
  voxcounter = (int *)malloc(nzones*sizeof(int));

  for (h = 0; h< nzones; h++) {
    inyet[h] = 0;
    inyet2[h] = 0;
  }

  nhl = 0; 

  /* Find the maximum and minimum values of dens */
  maxdens = -BF;
  mindens = BF;
  maxdensratio = 0.;
  for(i=0;i<Nvox; i++){
    if ((p[i].dens > maxdens) && (p[i].dens < borderdens)) maxdens = p[i].dens;
    if ((p[i].dens < mindens) && (p[i].dens > 1./borderdens)) mindens = p[i].dens;
  }
  printf("density ranges from %e to %e.\n",mindens,maxdens); FF;


  /* Write the zone merger information to voidfile */
  vod = fopen(voidfile,"w");
  if (vod == NULL) {
    printf("Problem opening voidfile %s.\n\n",voidfile);
    exit(0);
  }
  fprintf(vod,"%d\n",nrealzones);

  for (h = 0; h<nzones; h++) {
    if (p[z[h].core].dens >= borderdens) continue;	/* completely skip 'non-real' zones */
    fprintf(vod,"%d ",h);
    nhlcount = 0;
    for (hl = 0; hl < nhl; hl++)
      inyet[zonelist[hl]] = 0;

    zonelist[0] = h;
    inyet[h] = 1;
    nhl = 1;
    z[h].npjoin = z[h].np;
    do {
      /* Find the highest-density adjacency */
      lowdens = BF; nl = 0; beaten = 0;
      for (hl = 0; hl < nhl; hl++) {	/* loop over all the zones in the current list */
	    h2 = zonelist[hl];
	    if (inyet[h2] == 1) {
	      /* If it's not already identified as an interior zone, with inyet=2 */
	      interior = 1;
 	      for (za = 0; za < z[h2].nadj; za ++) {  /*loop over the adjacencies of this zone */
	        if (inyet[z[h2].adj[za]] == 0) {
 	          /* this adjacent zone not yet included in the current list */
	          interior = 0;  /* therefore current zone h2 is not an interior zone */
	          if (z[h2].slv[za] == lowdens) {
		        /* there is a link via zone h2 which equals current weakest link */
		        link[nl] = z[h2].adj[za];
		        nl ++;
		        if (nl == NLINKS) {
		          printf("Too many links with the same linking density!  Increase NLINKS from %d\n",nl);
		          exit(0);
		        }
	          }
	          if (z[h2].slv[za] < lowdens) {
		        /* there is a link via zone h2 which betters previous weakest link */
		        lowdens = z[h2].slv[za];
		        link[0] = z[h2].adj[za];
		        nl = 1;
	          }
	        }
	      }
	      if (interior == 1) inyet[h2] = 2; /* this zone does not border any zones not in current zonelist */
	    }
      }

      if (nl == 0) { 
	    /* either (a) this zonelist encompasses all zones or (b) something else went wrong */
	    printf("	h=%d, nl=0\n",h);
 	    beaten = 1;
	    z[h].leak = maxdens;
	    continue;	/* skip the step adding links, break out of do-while loop */
      }
	
      for (l=0; l < nl; l++)
	    if (p[z[link[l]].core].dens < p[z[h].core].dens) /* linked zone is superior, so don't add these links */
	      beaten = 1;

      if (lowdens >= borderdens) {
	    /* the weakest link is via a border voxel, so don't add these links */
	    beaten = 1;
      }

      if (beaten == 1) {
        z[h].leak = lowdens;
	    continue;	/* skip the step adding links, break out of do-while loop */
      }

      /* Add everything linked to the link(s) */
      nhl2 = 0;
      for (l=0; l < nl; l++) { /* loop over all the zones linked by weakest link dens */
	    if (inyet2[link[l]] == 0) {
	      /* linked zone link[l] not yet counted as part of current zonelist */
	      /* so we add its 'watershed basin' to current zonelist */
	      zonelist2[nhl2] = link[l];
	      inyet2[link[l]] = 1;
	      nhl2 ++;
	      added = 1;
	      while ((added == 1) && (beaten == 0)) { /* build up the watershed basin */
	        added = 0;
	        for (hl = 0; (hl < nhl2) && (beaten == 0); hl++) {
	          h2 = zonelist2[hl];
	          if (inyet2[h2] == 1) { /* zone h2 already part of link[l]'s zonelist */
		        interior = 1; /* Guilty until proven innocent */
		        for (za = 0; za < z[h2].nadj; za ++) {
		          /* loop over adjacencies of h2 */
		          link2 = z[h2].adj[za];
		          if ((inyet[link2]+inyet2[link2]) == 0) {
		            /* zone link2 neither already part of h's zonelist nor link[l]'s zonelist */
		            interior = 0;
		            if (z[h2].slv[za] <= lowdens) {
		              /* weakest link from h2 to link2 lies 'below current water level' so the two zonelists */
		              /* should be added together */
		              if (p[z[link2].core].dens < p[z[h].core].dens) {
			            /*but link2 is the superior, so h's zonelist beaten at this link level */
			            beaten = 1;
			            break;
		              }
		              zonelist2[nhl2] = link2;
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
      for (hl = 0; hl < nhl2; hl++) /*reset inyet2 for next round */
	    inyet2[zonelist2[hl]] = 0;
      
      /* See if there's a beater */
      if (beaten == 1) {
        /* h was beaten, so nothing added to its zonelist */
	    z[h].leak = lowdens;
      }
      else {
	    /* h not yet beaten, so add all the zones that entered at this link level */
	    fprintf(vod,"%d %lf ",nhl2, lowdens/p[z[h].core].dens);
	    for (h2 = 0; h2 < nhl2; h2++) {
	      zonelist[nhl] = zonelist2[h2];
          inyet[zonelist2[h2]] = 1;
          nhl++;
	      z[h].npjoin += z[zonelist2[h2]].np;
	      fprintf(vod,"%d ",zonelist2[h2]);
	    }
	    fflush(vod);
      }
      if (nhl/10000 > nhlcount) {
	    nhlcount = nhl/10000;
	    printf(" %d",nhl); FF;
      }
    } while((lowdens < BF) && (beaten == 0)); /* end of do-while loop */
   
    z[h].densratio = z[h].leak/p[z[h].core].dens;

    fprintf(vod,"0 %lf\n",z[h].densratio); /* Mark the end of the line */

    if (nhlcount > 0) { /* Outputs the number of zones in very large voids */
      printf(" h%d:%d\n",h,nhl);
      FF;
    }
    
    z[h].nhl = nhl;	/* total number of zones in this void */
  }
  fclose(vod);

  /* Text output file */
  txt = fopen(txtfile,"w");
  fprintf(txt,"%dx%dx%d voxels, %d vloidsters\n", Nside, Nside, Nside, nrealzones);
  fprintf(txt,"Zone# EdgeFlag CoreVoxel CoreDens ignore Zone#Vox ignore Vloid#Zones Vloid#Vox DensLeakRatio ignore\n");
  for (h=0; h<nzones; h++) {
    if (obo == 'c') { /* reinvert dens back to correct value */
      p[z[h].core].dens = 1./p[z[h].core].dens;
    }
    if (z[h].np>0) { /* only record 'real zones' */
      fprintf(txt,"%d %d %ld %e %d %ld %d %d %ld %e %d\n",
	      h, z[h].edge, z[h].core, p[z[h].core].dens, 0, z[h].np, 0, z[h].nhl, z[h].npjoin, z[h].densratio, 0);
    }
  }
  fclose(txt);

  for (h=0; h<nzones; h++) {
    voxcounter[h] = 0;
  }
  for (i=0; i<Nvox; i++) {
    h = zonenum[jumped[i]];
    z[h].vox[voxcounter[h]] = i;
    voxcounter[h]++;
  }
  /* Record which voxel membership for each zone */
  printf("Writing the zone memberships\n"); FF;
  zon = fopen(zonefile,"w");
  if (zon == NULL) {
    printf("Problem opening zonefile %s.\n\n",zonefile);
    exit(0);
  }
  for (h=0; h<nzones; h++) {
    if (z[h].np>0) {
      fprintf(zon, "%d ", h);
      for (i=0; i<voxcounter[h]; i++) {
        fprintf(zon, "%ld ", z[h].vox[i]);
      }
      fprintf(zon, "\n");
    }
  }
  close(zon);

  return(0);
}
