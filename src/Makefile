# Compiler choice:
CC    = /usr/local/Cellar/clang-omp/2015-04-01/bin/clang-omp
#CC    = gcc
MPICC = mpicc
CFLAGS = -Wall -O3 -fopenmp -Wunused-variable

MLIBS	=   -lm

QLIB = -L../qhull/src/ -lqhull
QINC = -I../qhull/src/

EXES = vozinit voz1b1 voztie jovoz boz jozov jozovtrvol jozov-grid vozisol voz1b1_mpi checkedges
EXES_NOMPI = vozinit voz1b1 voztie jovoz boz jozov jozovtrvol jozov-grid vozisol checkedges

all: $(EXES)

all_nompi: $(EXES_NOMPI)

jovoz: jovoz.o findrtop.o voz.h
	$(CC) $(CFLAGS) -o jovoz jovoz.o findrtop.o $(MLIBS)

jozov: jozov.o findrtop.o
	$(CC) $(CFLAGS) -o jozov jozov.o findrtop.o $(MLIBS)

jozovtrvol: jozovtrvol.o findrtop.o
	$(CC) $(CFLAGS) -o jozovtrvol jozovtrvol.o findrtop.o $(MLIBS)

jozov-grid: jozov-grid.o
	$(CC) $(CFLAGS) -o jozov-grid jozov-grid.o $(MLIBS)

checkedges: checkedges.o voz.h
	$(CC) $(CFLAGS) -o checkedges checkedges.o $(MLIBS)

boz: boz.o readfiles.o findrtop.o voz.h
	$(CC) $(CFLAGS) -o boz boz.o readfiles.o findrtop.o $(MLIBS)

voz1b1: voz1b1_serial.o voz1b1.o readfiles.o vozutil.o  voz.h
	$(CC)  -o voz1b1 $(CFLAGS) voz1b1_serial.o voz1b1.o readfiles.o vozutil.o -L. $(QLIB) $(MLIBS)

vozinit: vozinit.o readfiles.o voz.h
	$(CC)  -o vozinit $(CFLAGS) vozinit.o readfiles.o -L. $(MLIBS) 

voztie: voztie.o readfiles.o voz.h
	$(CC)  -o voztie $(CFLAGS) voztie.o readfiles.o

vozisol: vozisol.o readfiles.o vozutil.o  voz.h
	$(CC)  -o vozisol $(CFLAGS) vozisol.o readfiles.o vozutil.o -L. $(QLIB) $(MLIBS)

voz1b1_mpi: voz1b1_mpi.o voz1b1.o readfiles.o vozutil.o voz.h
	$(MPICC) -o voz1b1_mpi.o $(CFLAGS) -c voz1b1_mpi.c $(QINC)
	$(MPICC) $(CFLAGS) -o voz1b1_mpi voz1b1_mpi.o voz1b1.o readfiles.o vozutil.o -L. $(QLIB) $(MLIBS)

install: $(EXES_NOMPI)
	mv $(EXES_NOMPI) ../bin

.c.o:
	$(CC) -c -o $@ $(CFLAGS) $(QINC) $*.c

clean:
	rm -f $(EXES) *.o *~
