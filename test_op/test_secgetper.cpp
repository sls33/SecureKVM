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
#include <iomanip>

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
    int c = 4;

    Mat test_m(r, c);
    for (int i = 0; i < r; ++i)
    {
        for (size_t j = 0; j < c; j++)
        {
            test_m(i, j) = (i + 1) * IE;
        }
    }
    // test_m.print();

    Mat index_m(r, c);
    for (int i = 0; i < r; ++i)
    {
        for (size_t j = 0; j < c; j++)
        {
            index_m(i, j) = random()%r;
        }
    }
    index_m.print();

    // test graph
    TestMathOp *tm = new TestMathOp();
    tm->test_secgetper(&test_m, &index_m, 1);

    return 0;
}