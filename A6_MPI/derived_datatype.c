#include <mpi.h>
#include <stdio.h>

struct object {
  char c;
  int i[2];
  float f[4];

}myobject;

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if(world_rank == 0)
    printf("\n\nProgram to illustrate creation of derived datatypes, collective and point-to-point transmission of derived data\n\n");

  myobject.c = 'c';
  myobject.i[0] = 12;
  myobject.i[1] = 17;
  myobject.f[0] = 17.84;
  myobject.f[1] = 11.81;
  myobject.f[2] = 99.73;
  myobject.f[3] = 171.123;


  MPI_Datatype newstructuretype;
  int structlen = 3;
  int blocklengths[structlen]; MPI_Datatype types[structlen];
  MPI_Aint displacements[structlen];
  // where are the components relative to the structure?

  blocklengths[0] = 1; types[0] = MPI_CHAR;
  displacements[0] = (size_t)&(myobject.c) - (size_t)&myobject;

  blocklengths[1] = 2; types[1] = MPI_INT;
  displacements[1] = (size_t)&(myobject.i[0]) - (size_t)&myobject;

  blocklengths[2] = 4; types[2] = MPI_FLOAT;
  displacements[2] = (size_t)&(myobject.f[0]) - (size_t)&myobject;

  MPI_Type_create_struct(structlen,blocklengths,displacements,types,&newstructuretype);
  MPI_Type_commit(&newstructuretype);

  MPI_Bcast(&myobject, 1, newstructuretype, 0, MPI_COMM_WORLD);


  printf(":\n");
  printf("Broadcast Communication: Process-%d: Contents of struct: %c \t %d %d \t %lf %lf %lf %lf\n\n", world_rank, myobject.c, myobject.i[0], myobject.i[1], myobject.f[0], myobject.f[1], myobject.f[2], myobject.f[3]);


  if (world_rank != 0) {

    MPI_Recv(&myobject, 1, newstructuretype, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Point-to-Point Communication: Process-%d: Contents of struct: %c \t %d %d \t %lf %lf %lf %lf\n\n", world_rank, myobject.c, myobject.i[0], myobject.i[1], myobject.f[0], myobject.f[1], myobject.f[2], myobject.f[3]);
  }
  else
  {
    for(int i=1;i<world_size;++i)
    {
      MPI_Send(&myobject, 1, newstructuretype, i, 0, MPI_COMM_WORLD);
    }

  }

  


    

  MPI_Type_free(&newstructuretype); 
  MPI_Finalize();
}