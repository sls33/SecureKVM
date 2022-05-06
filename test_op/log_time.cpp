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

void log_benchmark()
{
    cout << (OFFLINE_PHASE_ON ? "on+ff" : "on") << endl;
    int sample_time = TOTAL_SAMPLE_SIZE;
    cout << "Sample: " << sample_time << endl;

    int r = PREDICTION_BATCH_BENCHMARK; // 50/100
    int c = FEATURE_DIM_BENCHMARK;      // FEATURE_DIM_BENCHMARK 1
    cout << "Batch: " << PREDICTION_BATCH_BENCHMARK << endl;
    cout << "log mat(r,c)" << r << " " << c << endl;

    Mat test_log_data(r, c);

    TestMathOp *tm = new TestMathOp();
    tm->test_log_graph(r, c);

    double sum_time = 0.0;
    globalRound = 0;
    for (size_t spt = 0; spt < sample_time; spt++)
    {
        globalRound++;

        for (int i = 0; i < test_log_data.rows(); ++i)
        {
            for (int j = 0; j < test_log_data.cols(); ++j)
            {
                test_log_data(i, j) = Constant::Util::randomlong();
            }
        }

        double one_time = tm->forward_log_time(&test_log_data);
        cout << spt << " " << one_time << endl;
        sum_time += one_time;
    }
    // Total number of log op is r * sample_time
    cout << "Avg single log time: " << endl
         << sum_time / (r * sample_time) << " ms" << endl;
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

    log_benchmark();

    return 0;
}