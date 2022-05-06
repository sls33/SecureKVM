#ifndef SECUREKVM_GETBENCHMARK_H
#define SECUREKVM_GETBENCHMARK_H
#include <assert.h>

#include "../Constant.h"
#include "../machine_learning/NN.h"
#include "../Player.h"
class GetBenchmarkGraph {
public:
    class GetBenchmark {
        NN* nn;
        int size;
        Constant::Clock *clock_train;
        Mat* kv_data;
        int input[M], input_mean[M], output;
        int kv_data_input;
        int secget_input, secget_output;

        int data, reshare;
        int test_log_input,test_log;
        int dim;
    public:
        GetBenchmark();
        GetBenchmark(Mat* kv_data);
        void secget_graph();
        void secget_graph(Mat *perm_mat, Mat *perm_mat_plain);
        void forward_secget(Mat *input);
        void feed(NN* nn, Mat& x_batch, Mat& y_batch, int input, int output);
        void next_batch(Mat& batch, int start, Mat* A, int mod = NM);
        void test();
        void print_perd(int round);
        static Mat vector_to_mat(int* data, int r, int c);

        void test_graph();
        void forward_test(Mat *input);

        void test_log_graph();
        void forward_log(Mat *input);
    };
};


#endif //SECUREKVM_GETBENCHMARK_H
