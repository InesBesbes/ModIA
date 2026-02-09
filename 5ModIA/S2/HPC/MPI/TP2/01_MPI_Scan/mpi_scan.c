/**
 * @author Ronan Guivarch from RookieHPC MPI_Scan example
 * @brief Original source code at https://rookiehpc.org/mpi/docs/mpi_scan/index.html
 **/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/**
 * @brief Illustrate how to use an MPI_Scan.
 * @details This program uses MPI_Scan to compute a progressive sum of ranks. It
 * can be visualized as follows for first element of the data arrays:
 *
 * +---------------+   +---------------+   +---------------+   +---------------+
 * | MPI process 0 |   | MPI process 1 |   | MPI process 2 |   | MPI process 3 |
 * +---------------+   +---------------+   +---------------+   +---------------+
 * |       0       |   |      10       |   |      20       |   |      30       |
 * +-------+-------+   +-------+-------+   +-------+-------+   +-------+-------+
 *         |                   |                   |                   |
 *         |                +--+--+                |                   |
 *         +----------------| SUM |                |                   |
 *         |                +--+--+                |                   |
 *         |                   |                +--+--+                |
 *         |                   +----------------| SUM |                |
 *         |                   |                +--+--+                |
 *         |                   |                   |                +--+--+
 *         |                   |                   +----------------| SUM |
 *         |                   |                   |                +--+--+
 *         |                   |                   |                   |
 * +-------+-------+   +-------+-------+   +-------+-------+   +-------+-------+
 * |       0       |   |      10       |   |      30       |   |      60       |
 * +---------------+   +---------------+   +---------------+   +---------------+
 * | MPI process 0 |   | MPI process 1 |   | MPI process 2 |   | MPI process 3 |
 * +---------------+   +---------------+   +---------------+   +---------------+
 *                                       
 **/
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    // Get my rank
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Data from a process
    const int n = 10;
    int data[n];

    for(int i = 0; i < n; i++)
    {
        data[i] = 10*my_rank + i;
    }

    // Display initial data
    printf("[%d] data = [", my_rank);
    for(int i = 0; i < n-1; i++)
    {
        printf(" %d;", data[i]);
    }
    printf(" %d ]\n", data[n-1]);

    // initialize result with minus ones
    int result[n];
    for(int i = 0; i < n; i++)
    {
        result[i] = -1;
    }

    // Partial reduction (sum)
    MPI_Scan(data, result, n, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Display partial reduction result
    printf("[%d] result (mpi_scan) = [", my_rank);
    for(int i = 0; i < n-1; i++)
    {
        printf(" %d;", result[i]);
    }
    printf(" %d ]\n", result[n-1]);

    // Perform the same operation using only peer to peer communications
    // You must also print the "result2" array of each process

    // initialize result2 with minus ones
    int result2[n];
    for(int i = 0; i < n; i++)
    {
        result2[i] = -1;
    }

    // ...

    // Compare the two results
    int norm = 0;
    for(int i = 0; i < n; i++)
    {
        norm += abs(result[i] - result2[i]);
    }
    printf("[%d] norme = %d\n", my_rank, norm);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
