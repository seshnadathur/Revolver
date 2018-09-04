export C_INCLUDE_PATH=/usr/lib/openmpi/include

# compiler choice
CC    = gcc

all: qhull voboz fastmodules

.PHONY : qhull voboz fastmodules

qhull:
	make -C qhull/src

voboz:
	make -C src all

fastmodules:
	python2.7 python_tools/setup.py build_ext --inplace

clean:
	make -C src clean
	make -C qhull/src cleanall
	rm -f bin/*
	rm -f python_tools/*.*o
	rm -f python_tools/fastmodules.c