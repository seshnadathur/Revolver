export C_INCLUDE_PATH=/usr/lib/openmpi/include

all: qhull voboz

.PHONY : qhull voboz examples

qhull:
	make -C qhull/src

voboz: 
	make -C src install

examples:
	make -C examples
	make -C examples data

examplestest:
	make -C examplestest
	make -C examplestest data

test:
	make -C LRGs

clean:
	make -C src clean
	make -C qhull/src cleanall
#	make -C examples clean
	rm -f bin/* 
