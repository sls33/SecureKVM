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
    int r = 4;
    int c = 1;
    Mat test_m(r, c);

    // test graph
    TestMathOp *tm = new TestMathOp();
    tm->test_graph_before(r, c);
    tm->getNN()->addOpGenPerm(tm->get_test_output(), tm->get_test_input());
    tm->test_graph_after();

    // forward
    tm->forward(&test_m);

    return 0;
}