#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

// perform the dot product between the two vectors x and y of size n
float dot(float x[], float y[], int n);

int main( int argc, char *argv[] ) {
  int n;
  float *x = NULL;
  float *y= NULL;
  float *local_x, *local_y;
  float local_dot, global_dot1, global_dot2, reference;
  float start, end;
  // int borne;

  int my_rank, size;

  MPI_Init (NULL, NULL);

  MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);

  if(my_rank==0)
  {
    // récupération de la taille du problème (en argument de la ligne de commande)
    n = atoi(argv[1]);
    // s'assurer que n%size == 0
    x = (float *)malloc(n*sizeof(float));
    y = (float *)malloc(n*sizeof(float));

    // Initialization of both local vectors with the same values
  // the global vectors would be [0, 1, ..., local_data_size -1]
    for(int i = 0; i < n; i++) {
      x[i] = (float) (i+1);
      y[i] = (float) (i+1);
      //printf("[MPI process %d] value[%d]: %f\n", my_rank, i, local_x[i]);
    }
  }

  start = MPI_Wtime();
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  reference = (float) (n*(n + 1)*(2*n+1)/6);
  int local_n = n/size;
  local_x = (float *)malloc(local_n*sizeof(float));
  local_y = (float *)malloc(local_n*sizeof(float));


  MPI_Scatter(x, local_n, MPI_FLOAT, local_x, local_n, MPI_FLOAT, 0, MPI_COMM_WORLD );
  MPI_Scatter(y, local_n, MPI_FLOAT, local_y, local_n, MPI_FLOAT, 0, MPI_COMM_WORLD );
  
  end = MPI_Wtime();

  if(my_rank==0)
  {
    printf("[%d] time elapsed during the scatter: %fs. \n", my_rank, end-start);
  }

  start = MPI_Wtime();
  local_dot  = dot(local_x, local_y, local_n);
  end = MPI_Wtime();
  printf("[%d] time elapsed during the scatter: %fs. \n", my_rank, end-start);


  printf("[MPI process %d] my local dot product: %f\n", my_rank, local_dot);

  /* Two-step operation */
  start = MPI_Wtime();

  global_dot1 = 0.0;

  // Step 1
  // Use a collective communication to compute the global dot product 
  // in such a way that the node 0 gets this value
  MPI_Reduce(&local_dot, &global_dot1, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

  // Node 0 displays the global value and the reference (sum of first integer ^ 2)
  if(my_rank == 0) {
    printf("[MPI process %d] *Two-step collective operation* global dot product: %f == %f\n", my_rank, global_dot1, reference);
  }

  // Step 2
  // Use a collective communication to broadcast the global value on each node
  MPI_Bcast(&global_dot1, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // A node i (i different from 0) displays the global value
  if(my_rank == size-1) {
    printf("[MPI process %d] *Two-step collective operation* global dot product: %f == %f\n", my_rank, global_dot1, reference);
  }
  
  /* One-step operation */

  global_dot2 = 0;

  // Step 3
  // Now use one single collective communication to perfom both step 1 and 2
  MPI_Allreduce(&local_dot, &global_dot2, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  

  // Another node displays the global value
  if(my_rank == size/2) {
    printf("[MPI process %d] *One-step collective operation* global dot product: %f == %f\n", my_rank, global_dot2  , reference);
  }

  end = MPI_Wtime();

  if(my_rank==0)
  {
    printf("[%d] time elapsed during the computation: %fs. \n", my_rank, end-start);
  }

  MPI_Finalize();

  return EXIT_SUCCESS;

}

float dot(float x[], float y[], int n){
  float res = 0.0;

  for(int i = 0; i < n; i++) {
    res += x[i]*y[i];
  }

  return res;
}
