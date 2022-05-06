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

    // test mat
    int r = 10;
    int c = 1;
    // int total_samples
    Mat test_log_data(r, c); // set FEATURE_DIM_BENCHMARK  to 1
    for (int i = 0; i < test_log_data.rows(); ++i)
    {
        for (int j = 0; j < test_log_data.cols(); ++j)
        {
            test_log_data(i, j) = Constant::Util::randomlong();
            // test_log_data(i,j) = Constant::Util::randomlong();
        }
    }

    // test graph
    TestMathOp *tm = new TestMathOp();
    tm->test_graph_before(r, c);
    tm->getNN()->addOpLog_approximate(tm->get_test_output(), tm->get_test_input(), BIT_P_LEN, DECIMAL_PLACES);
    tm->test_graph_after();
    // forward
    tm->forward(&test_log_data);

    return 0;
}