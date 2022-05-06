#include "SocketOnline.h"

SocketOnline::SocketOnline() {}

SocketOnline::~SocketOnline() {
    delete buffer;
    delete header;
}

SocketOnline::SocketOnline(int id, SOCK sock) {
    this->id = id;
    this->sock = sock;
    buffer = new char[BUFFER_MAX];
    header = new char[max(HEADER_LEN, HEADER_LEN_OPT) + 1];
    send_num = 0;
    recv_num = 0;
    init();
}

SocketOnline& SocketOnline::operator=(const SocketOnline &u) {
    id = u.id;
    sock = u.sock;
    buffer = new char[BUFFER_MAX];
    header = new char[max(HEADER_LEN, HEADER_LEN_OPT) + 1];
    send_num = 0;
    recv_num = 0;
}

void SocketOnline::init(int id, SOCK sock) {
    this->id = id;
    this->sock = sock;
    buffer = new char[BUFFER_MAX];
    header = new char[HEADER_LEN + 1];
    send_num = 0;
    recv_num = 0;
#ifdef UNIX_PLATFORM
    int flag, ret_flag, ret_sol;
        flag = 1024*1024*1024;
        ret_flag = setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(flag));
        DBGprint("ret sol: %d flag: %d\n", ret_sol, ret_flag);
#else
    int flag, ret_sol, ret_flag;
    flag = 1;
        ret_flag = setsockopt(sock, SOL_SOCKET, TCP_NODELAY, (const char*)&flag, sizeof(flag));
    DBGprint("ret sol: %d flag: %d\n", ret_sol, ret_flag);
#endif
}

void SocketOnline::init() {
#ifdef UNIX_PLATFORM
    int size, ret_flag;
    size = 1024*1024*1024;
    ret_flag = setsockopt(sock, SOL_SOCKET, SO_SNDBUF, (char*)&size, sizeof(size));
    ret_flag = setsockopt(sock, SOL_SOCKET, SO_RCVBUF, (char*)&size, sizeof(size));
    DBGprint("ret flag: %d size: %d\n", ret_flag, size);

    int snd_size = 0;
    socklen_t optlen;
    optlen = sizeof(snd_size);
    int err = getsockopt(sock, SOL_SOCKET, SO_SNDBUF,&snd_size, &optlen);
    cout << "snd buffer size: " << snd_size << endl;

    err = getsockopt(sock, SOL_SOCKET, SO_RCVBUF,&snd_size, &optlen);
    cout << "recv buffer size: " << snd_size << endl;

#else
    int flag, ret_sol, ret_flag;
    flag = 1;
    ret_flag = setsockopt(sock, SOL_SOCKET, TCP_NODELAY, (const char*)&flag, sizeof(flag));
    DBGprint("ret sol: %d flag: %d\n", ret_sol, ret_flag);
#endif
}

void SocketOnline::reset() {
    send_num = 0;
    recv_num = 0;
}

int SocketOnline::send_message(SOCK sock, char *p, int l) {
    int ret;
#ifdef UNIX_PLATFORM
    ret = write(sock, p, l);
#else
    ret = send(sock, p, l, NULL);
#endif
    return ret;
}

int SocketOnline::send_message_n(SOCK sock, char *p, int l) {
    int tot = 0, cur = 0;
    while (tot < l) {
#ifdef UNIX_PLATFORM
        cur = write(sock, p, l-tot);
#else
        cur = send(sock, p, l - tot, NULL);
#endif
        p += cur;
        tot += cur;
    }
    send_num+=l;
    return tot;
}

int SocketOnline::recv_message(SOCK sock, char *p, int l) {
    int ret;
#ifdef UNIX_PLATFORM
    ret = read(sock, p, l);
#else
    ret = recv(sock, p, l, NULL);
#endif
    return ret;
}

int SocketOnline::recv_message_n(SOCK sock, char *p, int l) {
    int tot = 0, cur = 0;
    while (tot < l) {
#ifdef UNIX_PLATFORM
        cur = read(sock, p, l-tot);
#else
        cur = recv(sock, p, l - tot, NULL);
#endif
        p += cur;
        tot += cur;
    }
    recv_num+=l;
    return tot;
}

void SocketOnline::send_message(const Mat &a) {
    int len_buffer;
    len_buffer = a.toString_pos(buffer);
    Constant::Util::int_to_header(header, len_buffer);
    send_message_n(sock, header, HEADER_LEN);
    send_message_n(sock, buffer, len_buffer);
}

void SocketOnline::send_message(Mat *a) {
    int len_buffer;
    len_buffer = a->toString_pos(buffer);
    Constant::Util::int_to_header(header, len_buffer);
    send_message_n(sock, header, HEADER_LEN);
    send_message_n(sock, buffer, len_buffer);
}

void SocketOnline::send_message(int b) {
    char* p = buffer;
    Constant::Util::int_to_char(p, b);
    send_message_n(sock, buffer, 4);
}

Mat SocketOnline::recv_message() {
    Mat ret;
    int len_header = recv_message_n(sock, header, HEADER_LEN);
    int l = Constant::Util::header_to_int(header);
    int len_buffer = recv_message_n(sock, buffer, Constant::Util::header_to_int(header));
    char* p = buffer;
    ret.getFrom_pos(p);
    return ret;
}

void SocketOnline::recv_message(Mat *a) {
    recv_message_n(sock, header, HEADER_LEN);
    recv_message_n(sock, buffer, a->getStringLen());
    char* p = buffer;
    a->addFrom_pos(p);
}

void SocketOnline::recv_message(Mat &a) {
    recv_message_n(sock, header, HEADER_LEN);
    recv_message_n(sock, buffer, Constant::Util::header_to_int(header));
    char* p = buffer;
    a.getFrom_pos(p);
}

int SocketOnline::recv_message_int() {
    recv_message_n(sock, buffer, 4);
    char* p = buffer;
    int ret = Constant::Util::char_to_int(p);
    return ret;
}

void SocketOnline::push(const Mat &a) {}

void SocketOnline::print() {
    DBGprint("socket online\n");
}