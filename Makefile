export C_INCLUDE_PATH=/usr/lib/openmpi/include

# compiler choice
# GNU compiler
#CC   = gcc

# Intel compiler
CC    = icpc

all: qhull voboz fastmodules

.PHONY : qhull voboz fastmodules

qhull:
	make -C qhull/src

voboz:
	make -C src all

fastmodules:
	python2.7 python_tools/setup.py build_ext --inplace
	mv fastmodules.so python_tools/.

clean:
	make -C src clean
	make -C qhull/src cleanall
	rm -f bin/*
	rm -f python_tools/*.*o
	rm -f python_tools/fastmodules.c
	rm -f python_tools/fastmodules.so