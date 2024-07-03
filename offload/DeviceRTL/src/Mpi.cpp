#pragma omp begin declare target device_type(nohost)

#include "Utils.h"
#include "Synchronization.h"
#include "Interface.h"
#include "Memory.h"
#include "Debug.h"
#include "LibC.h"

#include "Mpi.h"

//using namespace ompx;

// forward declaration of new
inline void *operator new(__SIZE_TYPE__ size, void *ptr) { return ptr; }

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
  ompx::mutex::TicketLock listlock;

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

const int MPI_MAX_BUF_SEND = 8 * 512 * 1024;

enum MPI_Request_type_e {
  MPI_SEND,
  MPI_RECV
};

struct MPI_Message_info_s {
  int count;                // nb. of items
  MPI_Datatype datatype;    // type of items
  int rank;                 // sender or recv rank
  int tag;                  // tag
  struct MPI_Comm_s *comm;  // communicator
};

struct MPI_Message_s : MPI_Message_info_s, mpiutils::LinkListNode<struct MPI_Message_s> {
  const void *send_data;
  void *buf_data;
  uint32_t status;
  // 0 send done (may be waiting for recv)
  // 1 recv done (done, sender can clean)
  bool buffered;
};

struct MPI_Request_s : MPI_Message_info_s {
  enum MPI_Request_type_e req_type;
  MPI_Status mpi_status;
  bool persistent;
  bool enable;
  // true:  running (not finish)
  // false: finish (send / recv done)
};

struct MPI_Send_Request_s : MPI_Request_s {
  const void *send_data;
  bool buffered;
  struct MPI_Message_s *msg;
};

struct MPI_Recv_Request_s : MPI_Request_s {
  void *recv_data;
};

struct MPI_Comm_s {
  int id; // id = 0 -> MPI_COMM_WORLD (the only supported)
  uint32_t size;
  uint32_t barrier_counter;
  uint32_t barrier_generation_counter;
  int *ranks; // map teams to ranks
  struct mpiutils::LinkList<struct MPI_Message_s> *messagebox;
};

}



namespace impl {

void yield(void){
  // split kernel here
  //__ompx_split();
  __builtin_amdgcn_s_sleep(1);
}

void barrier(uint32_t *counter, uint32_t *gen_counter, uint32_t size){
  int previous_gen = ompx::atomic::load(gen_counter, ompx::atomic::seq_cst);
  int current = ompx::atomic::inc(counter, size - 1,
                  ompx::atomic::seq_cst, ompx::atomic::MemScopeTy::device);
  if (current + 1 == size)
    ompx::atomic::add(gen_counter, 1, ompx::atomic::seq_cst);
  while(ompx::atomic::load(gen_counter, ompx::atomic::seq_cst) <= previous_gen){
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

void mpi_req_init(int count, MPI_Datatype datatype,
    int rank, int tag, MPI_Comm comm,
    enum MPI_Request_type_e req_type, struct MPI_Request_s *req)
{
  req->req_type = req_type;
  req->count    = count;
  req->datatype = datatype;
  req->rank     = rank;
  req->tag      = tag;
  req->comm     = comm;
  req->enable   = false;
}

struct MPI_Send_Request_s *mpi_send_init(
        const void *buf, int count, MPI_Datatype datatype,
        int recv_rank, int tag, MPI_Comm comm,
        bool buffered, bool persistent)
{
  struct MPI_Send_Request_s *req =
      reinterpret_cast<struct MPI_Send_Request_s *>(
          malloc(sizeof(struct MPI_Send_Request_s)));

  req->send_data  = buf;
  req->buffered   = buffered;
  req->persistent = persistent;

  mpi_req_init(count, datatype, recv_rank, tag, comm, MPI_SEND, req);

  return req;
}

struct MPI_Recv_Request_s *mpi_recv_init(
        void *buf, int count, MPI_Datatype datatype,
        int send_rank, int tag, MPI_Comm comm)
{
  struct MPI_Recv_Request_s *req =
      reinterpret_cast<struct MPI_Recv_Request_s *>(
          malloc(sizeof(struct MPI_Recv_Request_s)));

  req->recv_data              = buf;
  req->mpi_status.MPI_SOURCE  = MPI_ANY_SOURCE;
  req->mpi_status.MPI_TAG     = MPI_ANY_TAG;
  req->mpi_status.MPI_ERROR   = MPI_SUCCESS;

  mpi_req_init(count, datatype, send_rank, tag, comm, MPI_RECV, req);

  return req;
}

void mpi_req_free(struct MPI_Request_s **req){
  free(*req);
  *req = MPI_REQUEST_NULL;
}

bool mpi_msg_test(struct MPI_Message_s *msg)
{
  return ompx::atomic::load(&msg->status, ompx::atomic::seq_cst) == 1;
}

void mpi_msg_wait(struct MPI_Message_s *msg)
{
  while (! mpi_msg_test(msg))
    yield();
}

void mpi_msg_free(struct MPI_Message_s *msg)
{
  free(msg);
}

struct MPI_Message_s *mpi_send(
    const void *buf, int count, MPI_Datatype datatype, int recv_rank,
    int tag, MPI_Comm comm, bool buffered, bool blocking)
{
  int data_size = mpi_type_size(datatype) * count;
  int send_rank = mpi_rank(comm);

  struct MPI_Message_s *msg = reinterpret_cast<struct MPI_Message_s *>(
          malloc(sizeof(struct MPI_Message_s)));

  msg->datatype   = datatype;
  msg->count      = count;
  msg->rank       = send_rank;
  msg->tag        = tag;
  msg->buffered   = buffered;
  ompx::atomic::store(&msg->status, 0, ompx::atomic::seq_cst);

  if (buffered){
    msg->buf_data = malloc(data_size);
    memcpy(msg->buf_data, buf, data_size);
  } else {
    msg->send_data = buf;
  }

  comm->messagebox[recv_rank].push(msg);

  if (blocking && !buffered) {// ssend
    mpi_msg_wait(msg);
    mpi_msg_free(msg);
  }

  return msg;
}

void mpi_send_start(struct MPI_Send_Request_s *req)
{
  req->enable = true;
  struct MPI_Message_s *msg = mpi_send(req->send_data, req->count, req->datatype,
      req->rank, req->tag, req->comm, req->buffered, false);
  req->msg = msg;
}

bool mpi_send_test(struct MPI_Send_Request_s *req)
{
  if (req->buffered)
    return true;
  return mpi_msg_test(req->msg);
}

void mpi_send_wait(struct MPI_Send_Request_s *req)
{
  if (req->buffered)
    return;
  mpi_msg_wait(req->msg);
}

struct MPI_Message_s *__mpi_recv_test(int count, MPI_Datatype datatype,
        int source, int tag, MPI_Comm comm)
{
    struct mpiutils::LinkList<struct MPI_Message_s> *messages =
        &comm->messagebox[mpi_rank(comm)];
    messages->lock();
    for (struct MPI_Message_s *msg = messages->getHead();
        msg != nullptr; msg = msg->getNext()){
      if ((source == MPI_ANY_SOURCE || source == msg->rank)
          && (tag == MPI_ANY_TAG    || tag == msg->tag)){
        assert(count == msg->count && datatype == msg->datatype
            && "[MPI_recv]: count or datatype invalide");
        messages->unlock();
        // TODO: fixe race condition here
        messages->remove(msg);
        return msg;
      }
    }
    messages->unlock();
    return nullptr;
}

struct MPI_Message_s *__mpi_recv_wait(int count, MPI_Datatype datatype,
        int source, int tag, MPI_Comm comm)
{
  struct MPI_Message_s *msg = nullptr;
  while ((msg = __mpi_recv_test(count, datatype, source, tag, comm)) == nullptr){
    // we did not reciev any messages
    yield();
  }
  return msg;
}

void __mpi_recv_do(struct MPI_Message_s *msg, void *buf, MPI_Status *status)
{
  int data_size = mpi_type_size(msg->datatype) * msg->count;

  if (msg->buffered) {
    memcpy(buf, msg->buf_data, data_size);
    free(msg->buf_data);
  } else {
    memcpy(buf, msg->send_data, data_size);
  }

  if (status != &MPI_STATUS_IGNORE && status != &MPI_STATUSES_IGNORE) {
    status->MPI_SOURCE = msg->rank;
    status->MPI_TAG = msg->tag;
  }

  if (msg->buffered)
    mpi_msg_free(msg);
  else
    ompx::atomic::store(&msg->status, 1, ompx::atomic::seq_cst);
}

bool mpi_recv_test(void *buf, int count, MPI_Datatype datatype,
    int source, int tag, MPI_Comm comm, MPI_Status *status)
{
  struct MPI_Message_s *msg = __mpi_recv_test(count, datatype, source, tag, comm);
  if (msg == nullptr)
    return false;
  __mpi_recv_do(msg, buf, status);
  return true;
}

void mpi_recv_wait(void *buf, int count, MPI_Datatype datatype,
    int source, int tag, MPI_Comm comm, MPI_Status *status)
{
  struct MPI_Message_s *msg = __mpi_recv_wait(count, datatype, source, tag, comm);
  __mpi_recv_do(msg, buf, status);
}

void mpi_recv(void *buf, int count, MPI_Datatype datatype,
    int source, int tag, MPI_Comm comm, MPI_Status *status)
{
  mpi_recv_wait(buf, count, datatype, source, tag, comm, status);
}

void mpi_recv_start(struct MPI_Recv_Request_s *req){
  bool res = mpi_recv_test(req->recv_data, req->count, req->datatype,
          req->rank, req->tag, req->comm, &req->mpi_status);
}

bool mpi_recv_test(struct MPI_Recv_Request_s *req){
  bool res = mpi_recv_test(req->recv_data, req->count, req->datatype,
          req->rank, req->tag, req->comm, &req->mpi_status);
  return res;
}

void mpi_recv_wait(struct MPI_Recv_Request_s *req){
  mpi_recv_wait(req->recv_data, req->count, req->datatype,
          req->rank, req->tag, req->comm, &req->mpi_status);
}

void mpi_req_start(struct MPI_Request_s **reqp){
  if (reqp == &MPI_REQUEST_NULL)
    return;

  struct MPI_Request_s *req = *reqp;
  if (!req->enable)
    return;

  switch (req->req_type) {
   case (MPI_SEND):
     mpi_send_start(static_cast<struct MPI_Send_Request_s *>(req));
     break;
   case (MPI_RECV):
     mpi_recv_start(static_cast<struct MPI_Recv_Request_s *>(req));
     break;
   default:
     __builtin_unreachable();
  }
}

bool mpi_req_test(struct MPI_Request_s **reqp){
  if (reqp == &MPI_REQUEST_NULL)
    return false;

  struct MPI_Request_s *req = *reqp;
  if (!req->enable)
    return false;

  bool res = false;
  switch (req->req_type) {
    case (MPI_SEND):
      res =  mpi_send_test(static_cast<struct MPI_Send_Request_s *>(req));
      break;
    case (MPI_RECV):
      res = mpi_recv_test(static_cast<struct MPI_Recv_Request_s *>(req));
      break;
    default:
      __builtin_unreachable();
  }

  return res;
}

void mpi_req_wait(struct MPI_Request_s **reqp){
  if (reqp == &MPI_REQUEST_NULL)
    return;

  struct MPI_Request_s *req = *reqp;
  if (!req->enable)
    return;

  switch (req->req_type) {
   case (MPI_SEND):
     mpi_send_wait(static_cast<struct MPI_Send_Request_s *>(req));
     break;
   case (MPI_RECV):
     mpi_recv_wait(static_cast<struct MPI_Recv_Request_s *>(req));
     break;
   default:
     __builtin_unreachable();
  }
}

void mpi_req_deactivte(struct MPI_Request_s **reqp, MPI_Status *status) {
  struct MPI_Request_s *req = *reqp;
  if (status != &MPI_STATUS_IGNORE) {
    status->MPI_SOURCE  = req->mpi_status.MPI_SOURCE;
    status->MPI_TAG     = req->mpi_status.MPI_TAG;
    status->MPI_ERROR   = req->mpi_status.MPI_ERROR;
  }
  if (req->persistent) {
    req->enable = false;
  } else {
    mpi_req_free(reqp);
  }
}


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
            reinterpret_cast<struct mpiutils::LinkList<struct MPI_Message_s> *>(
            malloc(MPI_COMM_WORLD->size
                * sizeof(struct mpiutils::LinkList<struct MPI_Message_s>)));
  }

  impl::barrier(&global_counter, &global_generation_counter,
          omp_get_num_teams());

  MPI_COMM_WORLD->ranks[omp_get_team_num()] = rank;
  new (&MPI_COMM_WORLD->messagebox[rank]) mpiutils::LinkList<struct MPI_Message_s>();

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
  impl::mpi_send(buf, count, datatype, dest, tag, comm,
      (count * impl::mpi_type_size(datatype)) < MPI_MAX_BUF_SEND, true);
  return 0;
}

int MPI_Bsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{
  impl::mpi_send(buf, count, datatype, dest, tag, comm, true, true);
  return 0;
}

int MPI_Ssend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{

  impl::mpi_send(buf, count, datatype, dest, tag, comm, false, true);
  return 0;
}

int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
{
  impl::mpi_recv(buf, count, datatype, source, tag, comm, status);
  return 0;
}

// Non-Blocking Communications

int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
  struct MPI_Send_Request_s *req = 
    impl::mpi_send_init(buf, count, datatype, dest, tag, comm,
      (count * impl::mpi_type_size(datatype)) < MPI_MAX_BUF_SEND, false);
  impl::mpi_send_start(req);
  *request = req;
  return 0;
}

int MPI_Ibsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
  struct MPI_Send_Request_s *req =
    impl::mpi_send_init(buf, count, datatype, dest, tag, comm, true, false);
  impl::mpi_send_start(req);
  *request = req;
  return 0;
}

int MPI_Issend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
  struct MPI_Send_Request_s *req =
    impl::mpi_send_init(buf, count, datatype, dest, tag, comm, false, false);
  impl::mpi_send_start(req);
  *request = req;
  return 0;
}

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request)
{
  struct MPI_Recv_Request_s *req =
    impl::mpi_recv_init(buf, count, datatype, source, tag, comm);
  impl::mpi_recv_start(req); // try to recive early to reduce deadlock probability
  *request = req;
  return 0;
}

// Test & Wait

int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status)
{
  *flag = impl::mpi_req_test(request);
  if (*flag)
    impl::mpi_req_deactivte(request, status);
  return 0;
}

int MPI_Wait(MPI_Request *request, MPI_Status *status)
{
  impl::mpi_req_wait(request);
  impl::mpi_req_deactivte(request, status);
  return 0;
}

int MPI_Testall(int count, MPI_Request array_of_requests[], int *flag, MPI_Status array_of_statuses[])
{
  int finished = 0;
  for (int i = 0; i < count; ++i)
    finished += impl::mpi_req_test(&array_of_requests[i]);
  *flag = (finished == count);
  if (*flag)
    for (int i = 0; i < count; ++i)
      impl::mpi_req_deactivte(&array_of_requests[i], &array_of_statuses[i]);
  return 0;
}

int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[])
{
  // we don't want to wait for the request one after the others
  while (true) {
    int finished = 0;
    for (int i = 0; i < count; ++i)
      finished += impl::mpi_req_test(&array_of_requests[i]);
    if (finished >= count)
      break;
    impl::yield();
  }
  for (int i = 0; i < count; ++i)
    impl::mpi_req_deactivte(&array_of_requests[i], &array_of_statuses[i]);
  return 0;
}

int MPI_Testany(int count, MPI_Request array_of_requests[], int *index, int *flag, MPI_Status *status)
{
  for (int i = 0; i < count; ++i) {
    if (impl::mpi_req_test(&array_of_requests[i])) {
      *flag = true;
      *index = i;
      impl::mpi_req_deactivte(&array_of_requests[i], status);
      return 0;
    }
  }
  *flag = false;
  return 0;
}

int MPI_Waitany(int count, MPI_Request array_of_requests[], int *index, MPI_Status *status)
{
  while (true) {
    for (int i = 0; i < count; ++i) {
      if (impl::mpi_req_test(&array_of_requests[i])) {
        *index = i;
        impl::mpi_req_deactivte(&array_of_requests[i], status);
        return 0;
      }
    }
    impl::yield();
  }
  return 0;
}

int MPI_Testsome(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[], MPI_Status array_of_statuses[])
{
  *outcount = 0;
  for (int i = 0; i < incount; ++i) {
    if(impl::mpi_req_test(&array_of_requests[i])) {
      array_of_indices[*outcount] = i;
      impl::mpi_req_deactivte(&array_of_requests[i], &array_of_statuses[*outcount]);
      (*outcount)++;
    }
  }
  return 0;
}

int MPI_Waitsome(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[], MPI_Status array_of_statuses[])
{
  while (true) {
    for (int i = 0; i < incount; ++i)
      if (impl::mpi_req_test(&array_of_requests[i]))
        return MPI_Testsome(incount, array_of_requests, outcount, array_of_indices, array_of_statuses);
    impl::yield();
  }
  return 0;
}



// Persistent Communications setup

int MPI_Send_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
  struct MPI_Send_Request_s *req =
    impl::mpi_send_init(buf, count, datatype, dest, tag, comm,
      (count * impl::mpi_type_size(datatype)) < MPI_MAX_BUF_SEND, true);
  *request = req;
  return 0;
}

int MPI_Ssend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
  struct MPI_Send_Request_s *req =
    impl::mpi_send_init(buf, count, datatype, dest, tag, comm, true, true);
  *request = req;
  return 0;
}

int MPI_Bsend_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
  struct MPI_Send_Request_s *req =
    impl::mpi_send_init(buf, count, datatype, dest, tag, comm, false, true);
  *request = req;
  return 0;
}

int MPI_Recv_init(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request)
{
  struct MPI_Recv_Request_s *req =
    impl::mpi_recv_init(buf, count, datatype, source, tag, comm);
  *request = req;
  return 0;
}

// Persistent Communications start and end

int MPI_Start(MPI_Request *request)
{
  impl::mpi_req_start(request);
  return 0;
}

int MPI_Startall(int count, MPI_Request array_of_requests[])
{
  for (int i = 0; i < count; ++i)
    MPI_Start(&array_of_requests[i]);
  return 0;
}

int MPI_Request_free(MPI_Request *request)
{
  impl::mpi_req_free(request);
  return 0;
}

} // extern "C"

#pragma omp end declare target

