#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <inttypes.h>
#include <time.h>
#include <getopt.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <sstream>

#include "../util/SocketManager.h"
#include "../util/IOManager.h"
#include "../Constant.h"
#include "../machine_learning/NN.h"
#include "../Player.h"
#include "../MathOp.h"
#include "TestMathOp.h"

int node_type;
SocketManager::SMMLF tel;
int globalRound;

void secgetper_benchmark()
{
    // test graph
    TestMathOp *tm = new TestMathOp();

    cout << (OFFLINE_PHASE_ON ? "on+off" : "on") << endl;
    int sample_time = TOTAL_SAMPLE_SIZE;
    cout << "Sample: " << sample_time << endl;

    int r = FEATURE_DIM_BENCHMARK;      // set 100/10000
    int c = PREDICTION_BATCH_BENCHMARK; // 50/100
    cout << "Domain M: " << r << endl;
    cout << "Batch: " << c << endl;
    // test mat
    // every column is same kvdata, different permutation for every column
    // [kvdata,kvdata,...,kvdata], kvdata = [v1,v2,...,vr]^T
    Mat kv_data(r, c);

    // test index mat:1*batch
    // for every perm column, there is one index, which means different permutation for every index
    // [index1,index2,...,indexc]
    int index_num = 1;
    Mat index(index_num, c);

    // 0 : PShuffleData
    // 1 : EShuffleData
    // 2 : ShuffleDataNoGP
    tm->test_secgetper_graph(r, c, index_num, 1);

    double sum_time = 0.0;
    globalRound = 0;
    for (size_t spt = 0; spt < sample_time; spt++)
    {
        for (int i = 0; i < r; i++)
        {
            ll128 temp = Constant::Util::randomlong();
            for (int j = 0; j < c; j++)
            {
                kv_data(i, j) = temp;
            }
        }

        for (int i = 0; i < c; i++)
        {
            index(0, i) = random() % r;
        }

        globalRound++;

        double one_time = tm->forward_secgetper(&kv_data, &index);
        cout << spt << " " << one_time << endl;
        sum_time += one_time;

        ll tot_send = 0, tot_recv = 0;
        for (int i = 0; i < M; i++) {
            if (node_type != i) {
                tot_send += socket_io[node_type][i]->send_num;
                tot_recv += socket_io[node_type][i]->recv_num;
            }
        }
        Constant::Clock *clock_train = new Constant::Clock(CLOCK_TRAIN);
        DBGprint("round: %d tot_time: %.3f ",
                spt, clock_train->get());
        DBGprint("tot_send: %lld tot_recv: %lld\n", tot_send, tot_recv);
    }
    // Total number of queries is c * sample_time
    cout << "Avg single query time: " << endl
         << sum_time / (c * sample_time) << " ms" << endl;
}

int main(int argc, char *argv[])
{
    srand(time(NULL));
    if (argc < 2)
    {
        DBGprint("Please enter node type:\n");
        scanf("%d", &node_type);
    }
    else
    {
        node_type = argv[1][0] - '0';
    }
    DBGprint("node type: %d\n", node_type);

    Player::init();

    tel.init();

    cout << "===========\tTests for secgetper\t===========" << endl;

    secgetper_benchmark();

    cout << "===========\t End Tests for secgetper\t===========" << endl;

    return 0;
}