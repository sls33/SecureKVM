#ifndef SMMLF_TESTMATHOP_H
#define SMMLF_TESTMATHOP_H

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

class TestMathOp
{
private:
    NN *nn;
    int test_input, test_output;
    Constant::Clock *clock_train;
    int test_kv, test_index;

public:
    TestMathOp();

    void test_graph_before(int r, int c);
    void test_graph_after();
    void forward(Mat *input_data);
    double forwardt(Mat *input_data);

    NN* getNN();
    int get_test_input();
    int get_test_output();

    void test_secgetper_graph(int r, int c, int index_num, int shuffle_type);
    double forward_secgetper(Mat *kv_data, Mat *index);
    void test_secgetper(Mat *kv_data, Mat *index, int shuffle_type);

    // test_graph_before + nn->op + test_graph_after = test_xxx_graph
    void test_genperm_graph(int r, int c);
    void forward_genperm(Mat *input_data);

    void test_permute_graph(int r,int c);
    void forward_permute(Mat *input_data);

    void test_shuffledata_graph(int r, int c);
    void forward_shuffledata(Mat *input_data);

    void test_log_graph(int r,int c);
    void forward_log(Mat *input_data);
    double forward_log_time(Mat *input_data);

    void test_reshare_graph(int r,int c, int n0, int n1);
    void forward_reshare(Mat *input_data);

    void next_batch(Mat &batch, int start, Mat *A, int mod);
};

#endif // SMMLF_TESTMATHOP_H