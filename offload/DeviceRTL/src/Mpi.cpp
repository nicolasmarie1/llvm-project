#pragma omp begin declare target device_type(nohost)

#include "Utils.h"
#include "Synchronization.h"
#include "Interface.h"
#include "Memory.h"
#include "Debug.h"

#include "Mpi.h"

// TODO; replace with libc headers
extern "C" {
  void *memcpy(void *dest, const void *src, size_t n);
}

using namespace ompx;
namespace mpiutils {

template <typename T>
struct LinkListNode {
  template <typename R> friend struct LinkList;

public:
  T *getPrev(){ return Prev; };
  T *getNext(){ return Next; };

private:
  T *Prev = nullptr;
  T *Next = nullptr;
};


template <typename T>
struct LinkList {

public:
  T *getHead(){return Head;};
  T *getTail(){return Tail;};

  void push(T *Node){ insertImpl(Tail, Node); }

  T *remove(T *Node){ return removeImpl(Node); }

  void lock(){ listlock.lock(); }

  void unlock(){ listlock.unlock(); }

private:
  T *Head = nullptr;
  T *Tail = nullptr;
  mutex::TicketLock listlock;

  void insertImpl(T *Prev, T *Node) {
    lock();

    // Set current Node link
    Node->Prev = Prev;
    if (Node->Prev)
      Node->Next = Node->Prev->Next;
    else
      Node->Next = Head;

    // Set adjacent nodes links
    if (Node->Prev)
      Node->Prev->Next = Node;
    if (Node->Next)
      Node->Next->Prev = Node;

    // Set head & tail
    if (!Node->Prev)
      Head = Node;
    if (!Node->Next)
      Tail = Node;

    unlock();
  }

  T *removeImpl(T *Node) {
    lock();

    // Set adjacent nodes links
    if (Node->Prev)
      Node->Prev->Next = Node->Next;
    if (Node->Next)
      Node->Next->Prev = Node->Prev;

    // Set Head and tail
    if (!Node->Prev)
      Head = Node->Next;
    if (!Node->Next)
      Tail = Node->Prev;

    unlock();

    // Set current nodes links
    Node->Prev = nullptr;
    Node->Next = nullptr;

    return Node;
  }
};
}

extern "C" {

union dataptr {
  const void *dataorig;
  void *databuf;
};

enum MPI_Request_type_e {
  MPI_SEND,
  MPI_RECV
};

//struct MPI_Message_s final : mpiutils::LinkListNode<MPI_Message> {
//  struct MPI_Request_s *send_info;
//  union dataptr data;
//  uint32_t status;
//  // 0 send done (may be waiting for recv)
//  // 1 recv done (done, sender can clean)
//};

struct MPI_Request_s {
  int count;
  MPI_Datatype datatype;
  int src_rank;
  int dst_rank;
  int tag;
  struct MPI_Comm_s *comm;
  // request status
  uint32_t status;
  // 0 not finish
  // 1 finish (recv done)
};

struct MPI_Send_Request_s : MPI_Request_s, mpiutils::LinkListNode<struct MPI_Send_Request_s> {
  const void *send_data;
  void *buf_data;
  bool buffered;
  bool blocking;
  bool persistent;
};

struct MPI_Recv_Request_s : MPI_Request_s {
  void *recv_data;
};

//struct MPI_Send_request_s : mpiutils::LinkListNode<struct MPI_Request_s>{
//  const void *send_data;
//  void *buf_data
//  int count;
//  enum MPI_Datatype dtype;
//  int dst_rank;
//  int tag;
//  struct MPI_Comm_s *comm;
//  // request status
//  uint32_t status;
//  // 0 send done (may be waiting for recv)
//  // 1 recv done (done, sender can clean)
//  // extra infos
//  int src_rank;
//  bool buffered;
//  bool blocking;
//  bool persistent;
//  bool will_callback;
//};

//struct MPI_Recv_request_s {
//  void *data,
//  int count;
//  enum MPI_Datatype dtype;
//  int src_rank;
//  int tag;
//  struct MPI_Comm_s *comm;
//};
//
//union MPI_Request_s : 

struct MPI_Comm_s {
  int id; // id = 0 -> MPI_COMM_WORLD (the only supported)
  uint32_t size;
  uint32_t barrier_counter;
  uint32_t barrier_generation_counter;
  int *ranks; // map teams to ranks
  struct mpiutils::LinkList<struct MPI_Send_Request_s> *messagebox;
};

}



namespace impl {

void yield(void){
  // split kernel here
  //__ompx_split();
}

void barrier(uint32_t *counter, uint32_t *gen_counter, uint32_t size){
  int previous_gen = atomic::load(gen_counter, atomic::seq_cst);
  int current = atomic::inc(counter, size - 1,
                  atomic::seq_cst, atomic::MemScopeTy::device);
  if (current + 1 == size)
    atomic::add(gen_counter, 1, atomic::seq_cst);
  while(atomic::load(gen_counter, atomic::seq_cst) <= previous_gen){
    yield();
  }
}

size_t mpi_type_size(MPI_Datatype datatype){
  switch(datatype){
    case MPI_CHAR                   : return sizeof(char)                     ;
    case MPI_SHORT                  : return sizeof(signed short int)         ;
    case MPI_INT                    : return sizeof(signed int)               ;
    case MPI_LONG                   : return sizeof(signed long int)          ;
    case MPI_LONG_LONG_INT          : return sizeof(signed long long int)     ;
    case MPI_LONG_LONG              : return sizeof(signed long long int)     ;
    case MPI_SIGNED_CHAR            : return sizeof(signed char)              ;
    case MPI_UNSIGNED_CHAR          : return sizeof(unsigned char)            ;
    case MPI_UNSIGNED_SHORT         : return sizeof(unsigned short int)       ;
    case MPI_UNSIGNED               : return sizeof(unsigned int)             ;
    case MPI_UNSIGNED_LONG          : return sizeof(unsigned long int)        ;
    case MPI_UNSIGNED_LONG_LONG     : return sizeof(unsigned long long int)   ;
    case MPI_FLOAT                  : return sizeof(float)                    ;
    case MPI_DOUBLE                 : return sizeof(double)                   ;
    case MPI_LONG_DOUBLE            : return sizeof(long double)              ;
    case MPI_WCHAR                  : return sizeof(wchar_t)                  ;
    case MPI_C_BOOL                 : return sizeof(bool)                     ; // should be `_Bool` but it is not defined
    case MPI_INT8_T                 : return sizeof(int8_t)                   ;
    case MPI_INT16_T                : return sizeof(int16_t)                  ;
    case MPI_INT32_T                : return sizeof(int32_t)                  ;
    case MPI_INT64_T                : return sizeof(int64_t)                  ;
    case MPI_UINT8_T                : return sizeof(uint8_t)                  ;
    case MPI_UINT16_T               : return sizeof(uint16_t)                 ;
    case MPI_UINT32_T               : return sizeof(uint32_t)                 ;
    case MPI_UINT64_T               : return sizeof(uint64_t)                 ;
    case MPI_C_COMPLEX              : return sizeof(float _Complex)           ;
    case MPI_C_FLOAT_COMPLEX        : return sizeof(float _Complex)           ;
    case MPI_C_DOUBLE_COMPLEX       : return sizeof(double _Complex)          ;
    case MPI_C_LONG_DOUBLE_COMPLEX  : return sizeof(long double _Complex)     ;
    case MPI_BYTE                   : return 8                                ;
  //case MPI_PACKED                 : return //unsported                      ;
  //case MPI_AINT                   : return sizeof(MPI_Aint)                 ;
  //case MPI_OFFSET                 : return sizeof(MPI_Offset)               ;
  //case MPI_COUNT                  : return sizeof(MPI_Count)                ;
    case MPI_CXX_BOOL               : return sizeof(bool)                     ;
  //case MPI_CXX_FLOAT_COMPLEX      : return sizeof(std::complex<float>)      ;
  //case MPI_CXX_DOUBLE_COMPLEX     : return sizeof(std::complex<double>)     ;
  //case MPI_CXX_LONG_DOUBLE_COMPLEX: return sizeof(std::complex<long double>);
    default:
      __builtin_unreachable();
  }
  __builtin_unreachable();
}

int mpi_rank(MPI_Comm comm) {
  return comm->ranks[omp_get_team_num()];
}

void mpi_req_init(int count, MPI_Datatype datatype, int tag, MPI_Comm comm,
    struct MPI_Request_s *req)
{
  req->count = count;
  req->datatype = datatype;
  req->tag = tag;
  req->comm = comm;

  atomic::store(&req->status, 0, atomic::seq_cst);
}


struct MPI_Send_Request_s *mpi_send_init(
        const void *buf, int count, MPI_Datatype datatype,
        int recv_rank, int tag, MPI_Comm comm,
        bool buffered, bool blocking, bool persistent)
{
  struct MPI_Send_Request_s *req =
      reinterpret_cast<struct MPI_Send_Request_s *>(
          malloc(sizeof(struct MPI_Send_Request_s)));

  mpi_req_init(count, datatype, tag, comm, req);

  req->src_rank = mpi_rank(comm);
  req->dst_rank = recv_rank;

  req->send_data = buf;
  req->buf_data = nullptr;

  req->buffered = buffered;
  req->blocking = blocking;
  req->persistent = persistent;

  return req;
}

struct MPI_Recv_Request_s *mpi_recv_init(
        void *buf, int count, MPI_Datatype datatype,
        int send_rank, int tag, MPI_Comm comm)
{
  struct MPI_Recv_Request_s *req =
      reinterpret_cast<struct MPI_Recv_Request_s *>(
          malloc(sizeof(struct MPI_Recv_Request_s)));

  mpi_req_init(count, datatype, tag, comm, req);

  req->src_rank = send_rank;
  req->dst_rank = mpi_rank(comm);

  req->recv_data = buf;

  return req;
}


//
//  req->comm = comm;
//  req->rt = MPI_SEND;
//  req->persistent = persistent;
//
//  message->sender = source_rank;
//  message->tag = tag;
//  atomic::store(&message->status, 0, atomic::seq_cst);
//  message->buffered = buffered;
//  message->blocking = blocking;
//  message->count = count;
//  message->datatype = datatype;
//
//  if (buffered){
//    message->data.databuf = malloc(data_size);
//    memcpy(message->data.databuf, buf, data_size);
//  } else {
//    message->data.dataorig = buf;
//  }
//
//  return req;


void mpi_req_free(struct MPI_Request_s *req){
  free(req);
}

//void mpi_exec_request(struct MPI_Request_s *req){
//   switch (req->rtype) {
//    case (MPI_SEND):
//      mpi_send(req);
//      break;
//    case (MPI_RECV):
//      mpi_recv(req);
//      break;
//    default:
//      __builtin_unreachable();
//  }
//}

bool mpi_send_test(struct MPI_Send_Request_s *req)
{
  return atomic::load(&req->status, atomic::seq_cst) == 1;
}

void mpi_send_wait(struct MPI_Send_Request_s *req)
{
  while (! mpi_send_test(req))
    yield();
}

void mpi_send(struct MPI_Send_Request_s *req)
{
  int data_size = mpi_type_size(req->datatype) * req->count;
  if(req->buffered) {
    if (req->buf_data == nullptr)
      req->buf_data = malloc(data_size);
    memcpy(req->buf_data, req->send_data, data_size);
  }

  req->comm->messagebox[req->dst_rank].push(req);

  if (req->blocking && !req->buffered) { // ssend
    mpi_send_wait(req);
  }
}




//void mpi_send(const void *buf, int count, MPI_Datatype datatype, int dest_rank,
//                  int tag, MPI_Comm comm, bool buffered, bool blocking){
//  int data_size;
//  MPI_Type_size(datatype, &data_size);
//  data_size *= count;
//  int source_rank = comm->ranks[omp_get_team_num()];
//
//  MPI_Message *message = reinterpret_cast<struct MPI_Message *>(
//          malloc(sizeof(struct MPI_Message)));
//
//  message->sender = source_rank;
//  message->tag = tag;
//  atomic::store(&message->status, 0, atomic::seq_cst);
//  message->buffered = buffered;
//  message->blocking = blocking;
//  message->count = count;
//  message->datatype = datatype;
//
//  if (buffered){
//    message->data.databuf = malloc(data_size);
//    memcpy(message->data.databuf, buf, data_size);
//  } else {
//    message->data.dataorig = buf;
//  }
//
//  comm->messagebox[dest_rank].push(message);
//
//  if (!buffered && blocking) {
//    while(atomic::load(&message->status, atomic::seq_cst) == 0)
//      yield();
//    comm->messagebox[dest_rank].remove(message);
//    free(message);
//  }
//
//  return;
//}

//void mpi_recv(struct MPI_Recv_Request_s *req){
//  struct mpiutils::LinkList<struct MPI_Send_Request_s> *messages =
//      &comm->messagebox[req->dst_rank];
//  bool rcv = false;
//  while(!rcv) {
//    messages->lock();
//    for (struct MPI_Request_s *rsend = messages->getHead();
//        rsend != nullptr; rsend = m->getNext()){
//      if ((req->src_rank == MPI_ANY_SOURCE || req->src_rank == rsend->src_rank)
//          && (req->tag == MPI_ANY_TAG || req->tag == rsend->tag)){
//        assert(rsend->count == req->count && rsend->dtype == req->dtype
//            && "[MPI_recv]: count or datatype invalide");
//        messages->unlock();
//        recv = true;
//
//        req->src_rank = rsend->src_rank;
//        req->tag = rsend->tag;
//
//        memcpy(req->data,
//            rsend->buffered ? rsend->data : rsend->cst_data, data_size);
//
//        atomic::store(&rsend->status, 1, atomic::seq_cst);
//        req->comm->messagebox[req->dst_rank].remove(rsend);
//
//        if (rsend->will_callback)
//          break;
//
//        mpi_send_finish(rsend);
//        break;
//      }
//    }
//    if (!recv) {
//      messages->unlock();
//      yield();
//    } else {
//      atomic::store(&req->status, 1, atomic::seq_cst);
//    }
//  }
//}E

struct MPI_Send_Request_s *mpi_recv_test(struct MPI_Recv_Request_s *recv)
{
    struct mpiutils::LinkList<struct MPI_Send_Request_s> *messages =
        &recv->comm->messagebox[recv->dst_rank];
    messages->lock();
    for (struct MPI_Send_Request_s *send = messages->getHead();
        send != nullptr; send = send->getNext()){
      if ((recv->src_rank == MPI_ANY_SOURCE || recv->src_rank == send->src_rank)
          && (recv->tag == MPI_ANY_TAG || recv->tag == send->tag)){
        assert(send->count == recv->count && send->datatype == recv->datatype
            && "[MPI_recv]: count or datatype invalide");
        messages->unlock();
        return send;
      }
    }
    messages->unlock();
    return nullptr;
}

struct MPI_Send_Request_s *mpi_recv_wait(struct MPI_Recv_Request_s *recv)
{
  struct MPI_Send_Request_s *send = nullptr;
  while ((send = mpi_recv_test(recv)) == nullptr){
    // we did not reciev any messages
    yield();
  }
  return send;
}

void mpi_recv_finish(struct MPI_Recv_Request_s *recv, struct MPI_Send_Request_s *send){
  recv->src_rank = send->src_rank;
  recv->tag = send->tag;

  int data_size = mpi_type_size(recv->datatype) * recv->count;
  if (send->buffered) {
    memcpy(recv->recv_data, send->buf_data, data_size);
    if (! send->persistent)
      free(send->buf_data);
  } else {
    memcpy(recv->recv_data, send->send_data, data_size);
  }

  atomic::store(&send->status, 1, atomic::seq_cst);
  atomic::store(&recv->status, 1, atomic::seq_cst);
  recv->comm->messagebox[recv->dst_rank].remove(send);
}

void mpi_recv(struct MPI_Recv_Request_s *req){
  struct MPI_Send_Request_s *send = mpi_recv_wait(req);
  mpi_recv_finish(req, send);
}

bool mpi_recv_try(struct MPI_Recv_Request_s *req){
  struct MPI_Send_Request_s *send = mpi_recv_test(req);
  if (send == nullptr)
    return false;
  mpi_recv_finish(req, send);
  return true;
}

void mpi_get_req_status(struct MPI_Request_s *req, struct MPI_Status_s *status){
  status->MPI_SOURCE = req->src_rank;
  status->MPI_TAG = req->tag;
  status->MPI_ERROR = 0;
}


//void mpi_recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
//             MPI_Comm comm, MPI_Status *status){
//  int data_size;
//  MPI_Type_size(datatype, &data_size);
//  data_size *= count;
//  int dest_rank = comm->ranks[omp_get_team_num()];
//
//  struct mpiutils::LinkList<MPI_Message> *messages =
//      &comm->messagebox[dest_rank];
//  bool recv = false;
//  while(!recv) {
//    messages->lock();
//    for (MPI_Message *m = messages->getHead(); m != nullptr; m = m->getNext()){
//      if (m->sender == source && m->tag == tag){
//        assert(m->count == count && m->datatype == datatype
//            && "[MPI_recv]: count or datatype invalide");
//        messages->unlock();
//        recv = true;
//
//        status->MPI_SOURCE = m->sender;
//        status->MPI_TAG = m->tag;
//        status->MPI_ERROR = 0;
//        memcpy(buf, m->data.dataorig, data_size);
//        if (m->buffered)
//          free(m->data.databuf);
//        if (!m->buffered && m->blocking) {
//          atomic::store(&m->status, 1, atomic::seq_cst);
//        } else {
//          messages->remove(m);
//          free(m);
//        }
//        return;
//      }
//    }
//    if (!recv) {
//      messages->unlock();
//      yield();
//    }
//  }
//
//  return;
//}


} // namespace impl

extern "C" {

// global used for atomics
uint32_t global_counter = 0;
uint32_t global_generation_counter = 0;


int MPI_Init(int *argc, char **argv){
  (void) argc;
  (void) argv;

  int size = omp_get_num_teams();
  int rank = omp_get_team_num();

  if (omp_get_team_num() == 0){
    MPI_COMM_WORLD = reinterpret_cast<MPI_Comm>(
        malloc(sizeof(struct MPI_Comm_s)));
    MPI_COMM_WORLD->id = 0;
    MPI_COMM_WORLD->size = size;
    MPI_COMM_WORLD->barrier_counter = 0;
    MPI_COMM_WORLD->barrier_generation_counter = 0;
    MPI_COMM_WORLD->ranks = reinterpret_cast<int *>(
            malloc(MPI_COMM_WORLD->size * sizeof(int)));
    MPI_COMM_WORLD->messagebox =
            reinterpret_cast<struct mpiutils::LinkList<struct MPI_Send_Request_s> *>(
            malloc(MPI_COMM_WORLD->size
                * sizeof(struct mpiutils::LinkList<struct MPI_Send_Request_s>)));
  }

  impl::barrier(&global_counter, &global_generation_counter,
          omp_get_num_teams());

  MPI_COMM_WORLD->ranks[omp_get_team_num()] = rank;
  MPI_COMM_WORLD->messagebox[rank] = mpiutils::LinkList<struct MPI_Send_Request_s>();

  impl::barrier(&global_counter, &global_generation_counter,
          omp_get_num_teams());
  return 0;
}

int MPI_Finalize(void){
  impl::barrier(&global_counter, &global_generation_counter,
                omp_get_num_teams());
  if(omp_get_team_num() == 0){
    free(MPI_COMM_WORLD->ranks);
    free(MPI_COMM_WORLD->messagebox);
    free(MPI_COMM_WORLD);
  }
  return 0;
}

int MPI_Barrier(MPI_Comm comm) {
  impl::barrier(&comm->barrier_counter, &comm->barrier_generation_counter,
    comm->size);
  return 0;
}

int MPI_Comm_rank(MPI_Comm comm, int *rank){
  *rank = impl::mpi_rank(comm);
  return 0;
}

int MPI_Comm_size(MPI_Comm comm, int *size){
  *size = comm->size;
  return 0;
}

int MPI_Type_size(MPI_Datatype datatype, int *size){
  *size = impl::mpi_type_size(datatype);
  return 0;
}

// Blocking Communications

int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{
  struct MPI_Send_Request_s *req = impl::mpi_send_init(buf, count, datatype, dest, tag, comm, true, true, false);
  impl::mpi_send(req);
  // req will be free by the reciver
  return 0;
}

int MPI_Bsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{
  struct MPI_Send_Request_s *req = impl::mpi_send_init(buf, count, datatype, dest, tag, comm, true, true, false);
  impl::mpi_send(req);
  // req will be free by the reciver
  return 0;
}

int MPI_Ssend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{
  struct MPI_Send_Request_s *req = impl::mpi_send_init(buf, count, datatype, dest, tag, comm, false, true, false);
  impl::mpi_send(req);
  impl::mpi_req_free(req);
  return 0;
}

int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
{
  struct MPI_Recv_Request_s *req = impl::mpi_recv_init(buf, count, datatype, source, tag, comm);
  impl::mpi_recv(req);
  impl::mpi_get_req_status(req, status);
  impl::mpi_req_free(req);
  return 0;
}

// Non-Blocking Communications

int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
  struct MPI_Send_Request_s *req = impl::mpi_send_init(buf, count, datatype, dest, tag, comm, true, false, false);
  impl::mpi_send(req);
  *request = req;
  return 0;
}

int MPI_Ibsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
  struct MPI_Send_Request_s *req = impl::mpi_send_init(buf, count, datatype, dest, tag, comm, true, false, false);
  impl::mpi_send(req);
  *request = req;
  return 0;
}

int MPI_Issend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
  struct MPI_Send_Request_s *req = impl::mpi_send_init(buf, count, datatype, dest, tag, comm, false, false, false);
  impl::mpi_send(req);
  *request = req;
  return 0;
}

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request)
{
  struct MPI_Recv_Request_s *req = impl::mpi_recv_init(buf, count, datatype, source, tag, comm);
  impl::mpi_recv_try(req);
  *request = req;
  return 0;
}

// Test & Wait

int MPI_Wait(MPI_Request *request, MPI_Status *status)
{

  return 0;
}

int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status)
{

  return 0;
}

int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[])
{

  return 0;
}

int MPI_Testall(int count, MPI_Request array_of_requests[], int *flag, MPI_Status array_of_statuses[])
{

  return 0;
}

int MPI_Waitany(int count, MPI_Request array_of_requests[], int *index, MPI_Status *status)
{

  return 0;
}

int MPI_Testany(int count, MPI_Request array_of_requests[], int *index, int *flag, MPI_Status *status)
{

  return 0;
}

int MPI_Waitsome(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[], MPI_Status array_of_statuses[])
{

  return 0;
}

int MPI_Testsome(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[], MPI_Status array_of_statuses[])
{

  return 0;
}

// Persistent Communications setup

int MPI_Send_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
  return 0;
}

int MPI_Rsend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
  return 0;
}

int MPI_Ssend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
  return 0;
}

int MPI_Bsend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
  return 0;
}

int MPI_Recv_init(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request)
{
  return 0;
}

// Persistent Communications start and end

int MPI_Start(MPI_Request *request)
{
  return 0;
}

int MPI_Startall(int count, MPI_Request array_of_requests[])
{
  return 0;
}

int MPI_Request_free(MPI_Request *request)
{
  return 0;
}

} // extern "C"

#pragma omp end declare target

