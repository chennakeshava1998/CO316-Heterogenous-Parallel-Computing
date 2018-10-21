#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  // Initialize the MPI environment
  MPI_Init(NULL, NULL);
  // Find out rank, size
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int half = world_size, Pn = world_rank, sum = 0;

  int Sum[world_size];

  // fill in data into the Sum array
  for(int i=0;i<world_size;++i)
    Sum[i] = i;

  MPI_Status *status;
  MPI_Request request; 

  do
  {
    /* sync partial sum calculation */

    // handle the odd number of processors case
    if(half%2 != 0)
    {
      // send this value to the root process
      MPI_Isend(&Sum[half], 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &request);
      MPI_Waitall(1,&request,status);
    }

    if (half%2 != 0 && Pn == 0)
    {
      int partial_sum;
      MPI_Irecv(&partial_sum, 1, MPI_INT, half, 0, MPI_COMM_WORLD, &request);
      MPI_Waitall(1,&request,status);

      Sum[0] += partial_sum;
    }

    half /= 2;

    if (Pn < half)
    {
      int partial_sum;
      MPI_Irecv(&partial_sum, 1, MPI_INT, half + Pn, 0, MPI_COMM_WORLD, &request);
      MPI_Waitall(1,&request,status);

      Sum[Pn] += partial_sum;
    }
    else
    {
      // send this partial sum to the appropriate process
      MPI_Isend(&Sum[Pn], 1, MPI_INT, Pn - half, 0, MPI_COMM_WORLD, &request);
      MPI_Waitall(1,&request,status);
    }


  }while(half > 1);

  if(Pn == 0)
    printf("Total Sum = %d\n", Sum[0]);
  
  MPI_Finalize();
}