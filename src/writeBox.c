#include <stdio.h>
#include <stdlib.h>
#include "user.h"
#include <string.h>

int main(int argc,char **argv){

  FILE *fin, *fout;
  
  int i;
  int counter=0;
  int np, nignore, ignint, shiftFlag;
  char inFile[200], outFile[200], fileType[200], dummyStr[200];
  realT ignore;
  realT *x_buf, *y_buf, *z_buf;
  realT *ra_buf, *dec_buf, *redshift_buf, *mag_buf;
  realT x, y, z, Lmax;
  realT ra, dec, redshift, mag;

  if (argc < 2) {
    printf("You did not specify a configuration file for writeBox.c!\n");
    exit(0);
  }

  fin = fopen(argv[1],"r");
  fscanf(fin,"%s %s %s %d %d %lf %d",inFile, outFile, fileType, &nignore, &np, &Lmax, &shiftFlag);
  fclose(fin);

  fin = fopen(inFile,"r");
  fout = fopen(outFile,"w");
  
  fwrite(&np,sizeof(int),1,fout);
  
  x_buf = (realT *)malloc(np*sizeof(realT));
  y_buf = (realT *)malloc(np*sizeof(realT));
  z_buf = (realT *)malloc(np*sizeof(realT));
  ra_buf = (realT *)malloc(np*sizeof(realT));
  dec_buf = (realT *)malloc(np*sizeof(realT));
  redshift_buf = (realT *)malloc(np*sizeof(realT));
  mag_buf = (realT *)malloc(np*sizeof(realT));

  for (i = 0; i < nignore; i++){
    fgets(dummyStr,sizeof(dummyStr),fin);
  }

  for (i = 0; i < np; i++){
    if(strcmp(fileType,"MD_Box")==0){
	fscanf(fin,"%d %lf %lf %lf %lf %lf %lf %lf", &ignint, &x, &y, &z, &ignore, &ignore, &ignore, &ignore);
    }
    else if(strcmp(fileType,"Survey")==0){
	fscanf(fin,"%lf %lf %lf %lf %lf %lf %lf %lf", &x, &y, &z, &ra, &dec, &redshift, &mag, &ignore);
        ra_buf[counter] = ra;
        dec_buf[counter] = dec;
        redshift_buf[counter] = redshift;
	mag_buf[counter] = mag;
    }

    if(shiftFlag){
	x_buf[counter] = x+0.5*Lmax;
   	y_buf[counter] = y+0.5*Lmax;
    	z_buf[counter++] = z+0.5*Lmax;
    }else{
	x_buf[counter] = x;
   	y_buf[counter] = y;
    	z_buf[counter++] = z;
    }

  }

  fwrite(x_buf,sizeof(realT),np,fout);
  fwrite(y_buf,sizeof(realT),np,fout);
  fwrite(z_buf,sizeof(realT),np,fout);
  if(strcmp(fileType,"Survey")==0){
    fwrite(ra_buf,sizeof(realT),np,fout);
    fwrite(dec_buf,sizeof(realT),np,fout);
    fwrite(redshift_buf,sizeof(realT),np,fout);
    fwrite(mag_buf,sizeof(realT),np,fout);
  }

  fclose(fout);
  fclose(fin);

  free(x_buf);
  free(y_buf);
  free(z_buf);
  free(ra_buf);
  free(dec_buf);
  free(redshift_buf);
  free(mag_buf);
 
  return 0;
}
