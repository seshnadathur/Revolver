export C_INCLUDE_PATH=/usr/lib/openmpi/include

# compiler choice
CC    = gcc

# set python and numpy paths
PYTHONINC = -I/Users/seshadri/anaconda/include/python2.7
NUMPYINC = -I/Users/seshadri/anaconda/lib/python2.7/site-packages/numpy/core/include
PYTHONLIB = -L/Users/seshadri/anaconda/lib -lpython2.7

all: qhull voboz cic

.PHONY : qhull voboz cic

qhull:
	make -C qhull/src

voboz:
	make -C src install

cic:
	swig -python python_tools/cic.i
	$(CC) -g -fPIC -c python_tools/cic.c python_tools/cic_wrap.c $(PYTHONINC) $(NUMPYINC)
	mv cic.o python_tools/cic.o
	mv cic_wrap.o python_tools/cic_wrap.o
	$(CC) -g -shared python_tools/cic.o python_tools/cic_wrap.o -o python_tools/_cic.so $(PYTHONLIB)

clean:
	make -C src clean
	make -C qhull/src cleanall
	rm -f bin/*
	rm -f python_tools/*.o
	rm -f python_tools/cic_wrap.c
	rm -f python_tools/cic.py*
