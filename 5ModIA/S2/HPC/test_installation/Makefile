MPICC=smpicc
CFLAGS=-g -O4

all: hello_world

clean:
	rm -f *.o hello_world

%.o: %.c
	echo $@
	$(MPICC) -c -Wall -o $@ $< 

hello_world: hello_world.o
	$(MPICC) -o $@ $^ 

