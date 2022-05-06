#ifndef SMMLF_TESTGRAPH_H
#define SMMLF_TESTGRAPH_H

#include "../Constant.h"
#include "../machine_learning/NN.h"
#include "../Player.h"
class TestGraph {
public:
    class Test {
        NN* nn;
        int size;
        Constant::Clock *clock_train;
//        Mat *ip_data, *cp_data, *ep_data, *ln_data;
        Mat* data;
        Mat *data_ip, *data_cp;
        int input[M], output;
        int inference_input, inference_nodes[MAX_LEN-NGRAM+1], inference_medium[MAX_LEN-NGRAM+1];
        int inference_ip, inference_cp, inference_ip_and_cp, output_cp, output_ip;
        int delta, ep;
        int count, seg_t, st_add_delta, st_add_ep, st_div, st_mul, st_add[M+1];
        int out_sig, re_out_sig;
        int total, tmp;
        int ltz, pow, exp;
    public:
        Test();
        Test(Mat *data);
        Test(Mat *data_ip, Mat *data_cp);
        void feed(NN* nn, Mat& x_batch, Mat& y_batch, int input, int output);
        void next_batch(Mat& batch, int start, int batch_size, Mat* A, int mod = NM);
        void forward_count(ll *total, Mat* count_data, Mat * seg_total);
        void count_graph();
        void test_log_graph();
        void smooth_level_graph(ll total_count, int level_adjust, bool isCp);
        void forward_smooth(Mat data, Mat seg_total, int delta, int ep, Mat *level_data);
        void forward_calculate(Mat *calculate_data, Mat* ip_data, Mat* cp_data, Mat *ip_and_cp_data);
        void calculate_graph();
        void test();
        void print_perd(int round);
        static Mat vector_to_mat(int* data, int r, int c);
    };
};


#endif //SMMLF_TESTGRAPH_H