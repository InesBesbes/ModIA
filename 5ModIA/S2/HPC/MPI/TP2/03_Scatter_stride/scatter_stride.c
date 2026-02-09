#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

// perform the dot product between the two vectors x and y of size n
int dot(int x[], int y[], int n);

int main( int argc, char *argv[] ) {
 
  // comment this line, if you want the same vector for each run
  //srand( time( NULL ) );

  MPI_Init(&argc, &argv);

  // Get number of processes
  int nb_process;
  MPI_Comm_size(MPI_COMM_WORLD, &nb_process);

  // Fix root's rank
  int root_rank = 0;

  // Get my rank
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  // global size (only the root know its value)
  int global_size = 0;
  // local size (we fix this value in order to be regular)
  int local_size = 3;
  // local vector
  int *local_vectorX = NULL;
  int *global_vectorX = NULL;
  int *local_vectorY = NULL;
  int *global_vectorY = NULL;

  int stride = nb_process;

  // root process
  if(my_rank == root_rank) {
    global_size = nb_process*local_size; // to be able to split
                                         // the global vector into sub-vectors
                                         // with the same size
    printf("global_size = %d\n", global_size); 
    global_vectorX = (int *) malloc(sizeof(int)*global_size);
    global_vectorY = (int *) malloc(sizeof(int)*global_size);
    for(int i = 0; i < global_size; i++) {
      global_vectorX[i] = i;
      global_vectorY[i] = i;
      //global_vectorX[i] = rand() % 101;
      //global_vectorY[i] = rand() % 101;
      printf("global_vectorX[%d] = %d\n", i, global_vectorX[i]); 
      printf("global_vectorY[%d] = %d\n", i, global_vectorY[i]); 
    }
  }

  local_vectorX = (int *) malloc(sizeof(int)*local_size);
  local_vectorY = (int *) malloc(sizeof(int)*local_size);

  // Initialisation of local vector with dummy values to be sure to receive the
  // right components
  for(int j = 0; j < local_size; j++) {
    local_vectorX[j] = -my_rank;
    local_vectorY[j] = -my_rank;

  }

  // n : the number of components in the global vectors (global_size in the code)
  // N : number of process (nb_process in the code)
  //
  // Vector X - n peer to peer communications
  // we send the components of global vector X one by one
  if(my_rank==0)
  {
    for(int i=0;i<global_size;i++)
    {
      if(i%stride == 0){
        local_vectorX[i/stride] = global_vectorX[i];
      }
      else 
      {
        MPI_Ssend(global_vectorX+i,1,MPI_INT,i%stride,0,MPI_COMM_WORLD);
      }
    }
  }
  else 
  {
    for(int j=0; j<global_size/stride;j++)
    {
      MPI_Recv(local_vectorX+j,1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
  }

  // Display local X
  for(int i = 0; i < local_size; i++) {
    printf("[%d] local_vectorX[%d] = %d\n", my_rank, i, local_vectorX[i]); 
  }

  // Vector Y - n/N collective communications
  // we consider n/N sub-vectors of the global vector Y
  // for each sub-vector we perform a collective communication to scatter each
  // component of this slice all over the N process
  //
  // ...
  // TODO
  // ...


  // Display Y
  for(int i = 0; i < local_size; i++) {
    printf("[%d] local_vectorY[%d] = %d\n", my_rank, i, local_vectorY[i]); 
  }

  // local dot
  int local_dot;
  local_dot  = dot(local_vectorX, local_vectorY, local_size);
  printf("[%d] local_dot = %d\n", my_rank, local_dot);

  // Perform a collective communication to get the global dot
  // on process root_rank
  int global_dot = -999;

  // ...
  // TODO
  // ...

  if(my_rank == root_rank) printf("[%d] global_dot = %d\n", my_rank, global_dot);

  MPI_Finalize();

  return EXIT_SUCCESS;

}

int dot(int x[], int y[], int n){
  int res = 0;

  for(int i = 0; i < n; i++) {
    res += x[i]*y[i];
  }

  return res;
}
