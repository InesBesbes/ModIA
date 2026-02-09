#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]) {

  int x, y;
  MPI_Status status;

  MPI_Init(&argc, &argv);

  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  // processor 1 sends its integer data x(=5) to processor 3
  if (my_rank == 1) {
    x=5;
    MPI_Ssend(&x, 1, MPI_INT , 3, 0, MPI_COMM_WORLD);
  }

  // processor 3 receives a integer from processor 1 in its data y and displays it
  if (my_rank == 3) {
    MPI_Recv(&y, 1, MPI_INT , 1, 0, MPI_COMM_WORLD , &status);
    printf("je suis le processus %d et j'ai reçu la valeur %d\n", my_rank, y);
    printf("je suis le processus %d et ma valeur de x est  %d\n", my_rank, x);
  }

  MPI_Finalize();

  return EXIT_SUCCESS;
}

