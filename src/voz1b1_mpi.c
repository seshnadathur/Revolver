#include <mpi.h>
#include "voz.h"

void voz1b1(char *posfile, realT border, realT boxsize,
	    int numdiv, int b[], char *suffix);

int main(int argc, char *argv[]) {
  int i, MyProcessorNumber, NumberOfProcessors;
  char *posfile, *suffix;
  int numdiv;
  realT border, boxsize;
  int b[3];

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &MyProcessorNumber);
  MPI_Comm_size(MPI_COMM_WORLD, &NumberOfProcessors);

  if (argc != 6) {
    printf("Wrong number of arguments.\n");
    printf("arg1: position file\n");
    printf("arg2: border size\n");
    printf("arg3: box size\n");
    printf("arg4: suffix\n");
    printf("arg5: number of divisions\n");
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

  i = 0;
  for (b[0]=0;b[0]<numdiv; b[0]++) {
    for (b[1] = 0; b[1] < numdiv; b[1]++) {
      for (b[2] = 0; b[2] < numdiv; b[2]++) {
	if((i % NumberOfProcessors) == MyProcessorNumber){
	  voz1b1(posfile,border,boxsize,numdiv,b, suffix);
	}
	i++;
      }
    }
  }

  MPI_Finalize();

  return(0);
}
