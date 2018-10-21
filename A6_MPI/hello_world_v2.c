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

  // We are assuming at least 2 processes for this task
  if (world_size < 2) {
    fprintf(stderr, "World size must be greater than 1 for %s\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int number;

  if (world_rank != 0) {
        char buff[] = "Hello World !!";
        
        MPI_Send(&buff, sizeof(buff), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        // printf("Rank: %d\t cnt = %d\n", world_rank, cnt);
  }

  for(int i=1;i<world_size && world_rank == 0;++i)
  {

    char buff[20];
    MPI_Recv(&buff, sizeof(buff), MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Message: %s recieved from  process: %d\n", buff, i);

  }
  
  MPI_Finalize();
}