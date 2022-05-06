#ifndef MPC_ML_CONSTANT_H
#define MPC_ML_CONSTANT_H

#ifndef UNIX_PLATFORM
#define UNIX_PLATFORM
#endif

//#ifndef MAC_OS
//#define MAC_OS
//#endif

#include <iostream>
#include <cstdio>
#include <vector>
#include <fstream>
#include <queue>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <limits>
#ifdef UNIX_PLATFORM
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#else
#include <winsock2.h>
#endif

#define B 128

#define LEAKEY_RELU_BIAS IE/2
#define MOD 100000000000000003ll                            // nearly 2^56
// #define MOD 170141183460469231731687303715884105727ll    // 2^127-1
#define N 60000
#define M 3
#define D 784
#define L 2
// LSTM
#define D2 D/L
#define CH 128
#define IE_b 10000

#define TN 2
#define MAX_NODE_NUM 2001
#define MASTER 0
#define IE 1048576
#define NM 10000
#define BIT_LENGTH 64
#define REDUNDANCY 3
#define BIT_P_LEN 55
#define BUFFER_MAX 10000001
#define HEADER_LEN 4
#define ND 784
#define DECIMAL_PLACES 20
#define HEADER_LEN_OPT 2
#define TRAIN_ITE 10000
#define PRINT_PRE_ITE 100

#define MatColMajor 0
#define MatRowMajor 1

#define MM_NN 0
#define MM_NT 1
#define MM_TN 2
#define MM_TT 3

#define LOG_MESSAGES false

// Offline phase
#define OFFLINE_PHASE_ON true
#define THREAD_NUM 10

// SecGet Config
#define BENCHMARK false
#if BENCHMARK
#define SECGET_BATCH PREDICTION_BATCH_BENCHMARK
#define SECGET_DIM FEATURE_DIM_BENCHMARK
#else
#define SECGET_BATCH INFER_BATCH
#define SECGET_DIM KEY_DOMAIN_SIZE
#endif

// kv statistics
#define KV_BATCH 50
#define INPUT_DIM 1
#define KV_FEATURE_SIZE 50
#define USER_NUM 10000

// Markov training
#define NGRAM 3
#define KEY_NUM 1
#define KEY_BATCH 50

// the number of tests
#define TOTAL_SAMPLE_SIZE 10

// Markov inference
#define USING_BASIC false
#define USING_FOLD false
#define MAX_LEN 10
#define INFER_BATCH 5
#define CP_DOMAIN_SIZE  ((ALPHABET_SIZE*ALPHABET_SIZE*ALPHABET_SIZE) + 1)
#define IP_DOMAIN_SIZE  (ALPHABET_SIZE*ALPHABET_SIZE)
#define KEY_DOMAIN_SIZE (IP_DOMAIN_SIZE+CP_DOMAIN_SIZE)
#define MARKOV_INFER_LOG_BATCH 10000

#define MAX_SMOOTHING_LEVEL 10
#define ALPHABET_SIZE 62        // textual passwords alphabet size

// Time
#define CLOCK_MAIN 1
#define CLOCK_TRAIN 2

// Decision Tree
#define FEATURE_DIM 9
#define PREDICTION_BATCH 10
#define DEPTH 8
#define NODE_NUM ((1 << DEPTH) - 1)
#define PERMUTE_USING_BASIC true
#if (PERMUTE_USING_BASIC)
    #define FEATURE_VECTOR_SIZE FEATURE_DIM
#else
    #define FEATURE_VECTOR_SIZE (2 * ceil(sqrt(FEATURE_DIM)))
#endif

// Benchmark Constants
#define FEATURE_DIM_BENCHMARK 100
#define PREDICTION_BATCH_BENCHMARK 100


#define DEBUG
#ifdef DEBUG
#define DBGprint(...) printf(__VA_ARGS__)
#else
#define DBGprint(...)
#endif

using namespace std;
using namespace chrono;
typedef long long ll;       // 64-bit
typedef __int128_t ll128;   // 128-bit
typedef __float128 ff128;

extern int DBGtest;
class Constant {
public:
    static const ll SQRTINV = (ll128)(MOD+1>>2) * (MOD-2) % (MOD-1);
    static const ll inv2;
    static const ll inv2_m;

    static string getDateTime() {
        time_t t = std::time(nullptr);
        struct tm * now = localtime(&t);
        char buf[80];
        strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", now);
        return string(buf);
    }
    class Clock {
        int id;
        system_clock::time_point start;
        static ll global_clock[101];
    public:
        static double get_clock(int id);
        static void print_clock(int id);
        Clock(int id) : id(id) {
            start = system_clock::now();
        };
        ~Clock() {
            system_clock::time_point end = system_clock::now();
            decltype(duration_cast<microseconds>(end - start)) time_span = duration_cast<microseconds>(end - start);
            global_clock[id] += time_span.count();
        };
        double get() {
            system_clock::time_point end = system_clock::now();
            decltype(duration_cast<microseconds>(end - start)) time_span = duration_cast<microseconds>(end - start);
            return time_span.count() * 1.0 * microseconds::period::num / microseconds::period::den;
        }
        void print() {
            system_clock::time_point end = system_clock::now();
            decltype(duration_cast<microseconds>(end - start)) time_span = duration_cast<microseconds>(end - start);
            DBGprint("duration: %f\n", time_span.count() * 1.0 * microseconds::period::num / microseconds::period::den);
        }
    };
    class Util {
    public:
        static void int_to_char(char* &p, int u);
        static void ll_to_char(char* &p, ll u);
        static int char_to_int(char* &p);
        static ll char_to_ll(char* &p);
        static void int_to_header(char* p, int u);
        static int header_to_int(char* p);
        static int getint(char* &p);
        static ll getll(char* &p);
        static ll randomlong();
        static ll128 get_residual(ll128 a);
        static ll128 get_sign(ll128 a);
        static ll128 get_abs(ll128 a);
        static ll128 sqrt(ll128 a);
        static ll128 inverse(ll128 a, ll128 b);
        static ll128 power(ll128 a, ll128 b);
        static ll128 cal_perm(ll128 key[], int l, int k);
    };
};
std::ostream&
operator<<( std::ostream& dest, __int128_t value );

#endif //MPC_ML_CONSTANT_H
