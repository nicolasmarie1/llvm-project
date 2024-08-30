#ifndef MPI_H
#define MPI_H

#ifdef  __cplusplus
extern "C" {
#endif

typedef uint64_t MPI_Aint;

typedef struct MPI_Comm_s *MPI_Comm;

typedef enum MPI_Datatype_e {
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
  MPI_CXX_BOOL,
//MPI_CXX_FLOAT_COMPLEX,
//MPI_CXX_DOUBLE_COMPLEX,
//MPI_CXX_LONG_DOUBLE_COMPLEX,

// Reduce Data Type
  MPI_FLOAT_INT,
  MPI_DOUBLE_INT,
  MPI_LONG_INT,
  MPI_2INT,
  MPI_SHORT_INT,
  MPI_LONG_DOUBLE_INT

} MPI_Datatype;

typedef enum MPI_Intent_e {
  MPI_THREAD_SINGLE,
  MPI_THREAD_FUNNELED,
  MPI_THREAD_SERIALIZED,
  MPI_THREAD_MULTIPLE
} MPI_Intent;

typedef void MPI_User_function(const void *invec, void *inoutvec, int *len, MPI_Datatype *datatype);
typedef void MPI_Loc_User_function(const void *invec, int index, const void *current, int *outvec, int *len, MPI_Datatype * datatype);

typedef struct MPI_Op_s {
  MPI_User_function *func_user;
  MPI_Loc_User_function *func_loc;
} MPI_Op;

typedef struct MPI_Status_s {
  int MPI_SOURCE;
  int MPI_TAG;
  int MPI_ERROR;
  int count;
} MPI_Status;

typedef struct MPI_Request_s *MPI_Request;

typedef enum MPI_Errhandler_e {
  MPI_ERRORS_ARE_FATAL,
  MPI_ERRORS_ABORT,
  MPI_ERRORS_RETURN
} MPI_Errhandler;

typedef struct MPI_Info_s {
  int info; //TODO implement MPI Infos
} MPI_Info;

// global variables[MaQ
extern const int MPI_ANY_SOURCE;
extern const int MPI_ANY_TAG;
extern const int MPI_SUCCESS;

extern MPI_Comm MPI_COMM_WORLD;

extern MPI_Status *MPI_STATUS_IGNORE;
extern MPI_Status *MPI_STATUSES_IGNORE;

extern MPI_Op MPI_MIN;
extern MPI_Op MPI_MINLOC;
extern MPI_Op MPI_MAX;
extern MPI_Op MPI_MAXLOC;
extern MPI_Op MPI_SUM;

extern MPI_Request MPI_REQUEST_NULL;

extern MPI_Info MPI_INFO_NULL;

// Common  functions
int MPI_Init(int *argc, char ***argv);
int MPI_Init_thread(int *argc, char ***argv, int required, int *provided);
int MPI_Finalize(void);
int MPI_Abort(MPI_Comm comm, int errorcode);

int MPI_Barrier(MPI_Comm comm);

int MPI_Comm_rank(MPI_Comm comm, int *rank);
int MPI_Comm_size(MPI_Comm comm, int *size);

int MPI_Type_size(MPI_Datatype datatype, int *size);

// Blocking Comm
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
//int MPI_Rsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
int MPI_Ssend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
int MPI_Bsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);

int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
int MPI_Sendrecv_replace(void *buf, int count, MPI_Datatype datatype, int dest, int sendtag, int source, int recvtag, MPI_Comm comm, MPI_Status *status);



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

int MPI_Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count);

// Collective Operations
int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

// Others
double MPI_Wtime(void);
int MPI_Get_version(int *version, int *subversion);

// Memory Allocations
int MPI_Alloc_mem(MPI_Aint size, MPI_Info info, void **baseptr);
int MPI_Free_mem(void *base);

#ifdef  __cplusplus
}
#endif

#endif  // MPI_H

