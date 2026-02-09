#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]) {

  int x[10];
  int *y;
  MPI_Status status;

  MPI_Init(&argc, &argv);

  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  // processor 1 sends 10 integers stored in the vector x to processor 3
  if (my_rank == 1) {
    for(int i=0;i<10;i++)
    {
      x[i] = 2*i;
      printf("x[%d] = %d\n", i, x[i]);
    }
    MPI_Ssend(x, 10, MPI_INT , 3, 0, MPI_COMM_WORLD);
  } 

  // if (my_rank == 0) {
  //   for(int i=0;i<10;i++)
  //   {
  //     x[i] = 2*i;
  //     printf("x[%d] = %d\n", i, x[i]);
  //   }
  //   MPI_Ssend(x, 10, MPI_INT , 3, 1, MPI_COMM_WORLD);
  // } 

  // processor 3 receives 10 integers from processor 1 in its vector y and displays y
  if (my_rank == 3) {
    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD , &status);
    printf("je suis le processus %d et j'ai reçu un message depuis le processus %d avec le tag %d\n", my_rank, status.MPI_SOURCE, status.MPI_TAG);

    int count;
    MPI_Get_count(&status, MPI_INT, &count);

    y = (int *) malloc(count*sizeof(int));

    MPI_Recv(y, count, MPI_INT , status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD , &status);


    for(int i=0;i<count;i++)
    {
      printf("y[%d] = %d\n", i, y[i]);
    }
  }

  MPI_Finalize();

  return EXIT_SUCCESS;
}

