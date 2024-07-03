#ifndef MPI_H
#define MPI_H

#ifdef  __cplusplus
extern "C" {
#endif

//struct MPI_Comm_s;
//struct MPI_Comm_s {
//  int id; // id = 0 -> MPI_COMM_WORLD (the only supported)
//  uint32_t size;
//  uint32_t barrier_counter;
//  uint32_t barrier_generation_counter;
//  uint32_t *nodes_infos;
//};
typedef struct MPI_Comm_s *MPI_Comm;
MPI_Comm MPI_COMM_WORLD;

enum MPI_Datatype_e {
  MPI_CHAR,
  MPI_SHORT,
  MPI_INT,
  MPI_LONG,
  MPI_LONG_LONG_INT,
  MPI_LONG_LONG,
  MPI_SIGNED_CHAR,
  MPI_UNSIGNED_CHAR,
  MPI_UNSIGNED_SHORT,
  MPI_UNSIGNED,
  MPI_UNSIGNED_LONG,
  MPI_UNSIGNED_LONG_LONG,
  MPI_FLOAT,
  MPI_DOUBLE,
  MPI_LONG_DOUBLE,
  MPI_WCHAR,
  MPI_C_BOOL,
  MPI_INT8_T,
  MPI_INT16_T,
  MPI_INT32_T,
  MPI_INT64_T,
  MPI_UINT8_T,
  MPI_UINT16_T,
  MPI_UINT32_T,
  MPI_UINT64_T,
  MPI_C_COMPLEX,
  MPI_C_FLOAT_COMPLEX,
  MPI_C_DOUBLE_COMPLEX,
  MPI_C_LONG_DOUBLE_COMPLEX,
  MPI_BYTE,
//MPI_PACKED,
//MPI_AINT,
//MPI_OFFSET,
//MPI_COUNT,
  MPI_CXX_BOOL
//MPI_CXX_FLOAT_COMPLEX,
//MPI_CXX_DOUBLE_COMPLEX,
//MPI_CXX_LONG_DOUBLE_COMPLEX,
};
typedef enum MPI_Datatype_e MPI_Datatype;

struct MPI_Status_s {
  int MPI_SOURCE;
  int MPI_TAG;
  int MPI_ERROR;
};
typedef struct MPI_Status_s MPI_Status;

typedef struct MPI_Request_s *MPI_Request;

//const MPI_Status MPI_STATUS_IGNORE;
//const MPI_Status MPI_STATUSES_IGNORE;

// global variables
const int MPI_ANY_SOURCE = -1;
const int MPI_ANY_TAG = -1;
const int MPI_SUCCESS = 0;

MPI_Status MPI_STATUS_IGNORE;
MPI_Status MPI_STATUSES_IGNORE;
MPI_Request MPI_REQUEST_NULL = 0; // nullptr in cpp and NULL in c

// Common  functions
int MPI_Init(int *argc, char **argv);
int MPI_Barrier(MPI_Comm comm);
int MPI_Finalize(void);

int MPI_Comm_rank(MPI_Comm comm, int *rank);
int MPI_Comm_size(MPI_Comm comm, int *size);

int MPI_Type_size(MPI_Datatype datatype, int *size);

// Blocking Comm
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
//int MPI_Rsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
int MPI_Ssend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
int MPI_Bsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);

// Non Blocking Comm
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
//int MPI_Irsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
int MPI_Issend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
int MPI_Ibsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request);

// Non Blocking Wait
int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status);
int MPI_Wait(MPI_Request *request, MPI_Status *status);
int MPI_Testall(int count, MPI_Request array_of_requests[], int *flag, MPI_Status array_of_statuses[]);
int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]);
int MPI_Testany(int count, MPI_Request array_of_requests[], int *index, int *flag, MPI_Status *status);
int MPI_Waitany(int count, MPI_Request array_of_requests[], int *index, MPI_Status *status);
int MPI_Testsome(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[], MPI_Status array_of_statuses[]);
int MPI_Waitsome(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[], MPI_Status array_of_statuses[]);

// Persistent Communications setup
int MPI_Send_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
//int MPI_Rsend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
int MPI_Ssend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
int MPI_Bsend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
int MPI_Recv_init(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request);

// Persistent Communications start and end
int MPI_Start(MPI_Request *request);
int MPI_Startall(int count, MPI_Request array_of_requests[]);
int MPI_Request_free(MPI_Request *request);

#ifdef  __cplusplus
}
#endif

#endif  // MPI_H

