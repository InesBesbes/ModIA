MPICC=smpicc
CFLAGS=-g -O4

all: scatter_stride

clean:
	rm -rf *.o scatter_stride

%.o: %.c
	echo $@
	$(MPICC) -c -Wall -o $@ $<

scatter_stride: scatter_stride.o
	$(MPICC) -o $@ $^  -lm
