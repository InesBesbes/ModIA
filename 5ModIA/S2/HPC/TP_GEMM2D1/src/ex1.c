#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cblas.h>
#include "utils.h"
#include "dsmat.h"
#include "gemms.h"

void p2p_transmit_A(int p, int q, Matrix *A, int i, int l) {
    int j;
    int me, my_row, my_col;
    MPI_Status status;

    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    node_coordinates_2i(p,q,me,&my_row,&my_col); // coordonnées du processus dans la grille des processus

    Block *Ail;
    int node, tag, b;

    Ail =  & A->blocks[i][l];
    b = A->b;
    tag=0;

    /* TODO : transmit A[i,l] using MPI_Ssend & MPI_Recv */

    if (me == Ail -> owner) { // récupérer propriétaire du bloc Ail
      for(j = 0; j < q; j++){ // boucle suivant les processus de la même ligne
        node = get_node(p, q, my_row, j); // renvoie numéro du processus (réciproque de node_coordinates_2i)
        // my_row : ligne du processus 
        if(node!=me){ // on ne s'envoie pas à soi même 
          /* MPI_Ssend A[i,l] to my row */
          MPI_Ssend(Ail->c, b*b, MPI_FLOAT, node, tag, MPI_COMM_WORLD); // synchrone , on a 1 seul communicateur
        }
      }
      /* A[i,l] is stored on my row */
    } else if (my_row == Ail ->row) { //propriétaire de Ail est à la même ligne que moi 
      Ail->c = malloc(b*b*sizeof(float));
      /* MPI_Recv A[i,l] */
      MPI_Recv(Ail->c, b*b, MPI_FLOAT, Ail->owner, tag, MPI_COMM_WORLD, &status);   // qui m'envoie = Ail-> owner qui est le propriétaire du bloc   
    }
}

void p2p_transmit_B(int p, int q, Matrix *B, int l, int j) {
    int i;
    int me, my_row, my_col;
    MPI_Status status;

    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    node_coordinates_2i(p,q,me,&my_row,&my_col);

    int node, tag, b;
    Block *Blj;

    Blj =  & B->blocks[l][j];
    b = B->b;
    tag=1;

        /* I owned B[l,j]*/
    if (me== Blj->owner) { 
      for(i=0;i<p;i++){
        node=get_node(p,q,i,my_col);
        if(node!=me){
          /* MPI_Ssend B[l,j] to my column */
          MPI_Ssend(Blj->c, b*b, MPI_FLOAT, node, tag, MPI_COMM_WORLD);
        }
      }
    } else if (my_col == Blj ->col) {
      Blj->c = malloc(b*b*sizeof(float));
      /* MPI_Recv B[l,j] */
      MPI_Recv(Blj->c, b*b, MPI_FLOAT, Blj->owner, tag, MPI_COMM_WORLD, &status);      

    }
}

// run avec :
//smpirun -trace -platform platforms/cluster_crossbar.xml -hostfile hostfiles/cluster_hostfile.txt -np 4 ./build/bin/main -p 2 -q 2 -n 10 -b 5 --algorithm p2p --niter 2 -c -v
// pour avoir la trace : rajouter -trace après smpirun 
// puis vite nom_du_fichier.trace