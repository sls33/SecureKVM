#ifndef SMMLF_DT_GRAPH_H
#define SMMLF_DT_GRAPH_H


#include "../Constant.h"
#include "DTDAG.h"
#include "../Player.h"
#include <map>
class DT_graph {
public:
    class DT {
        DTDAG* dag;
        vector<map<string, ll>> encode_table;
        int size;
        Constant::Clock *clock_train;
        Mat* data;
        int input[M], output, input_table;
        int total_entropy;
        int split_finding;
        int entropy_gain_feature[FEATURE_DIM];
        int delta, ep;
        int count, seg_t, st_add_delta, st_add_ep, st_div, st_mul, st_add[M+1];
        int out_sig, re_out_sig;
        int total, tmp;
        int ltz;
        Mat *input_data;

        // predict
        int node1;
        int feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11;
        int internal1, internal2, internal3, internal4, internal5, internal6, internal7, internal8, internal9, internal10;
        int internal11, internal12, internal13, internal14, internal15, internal16, internal17, internal18, internal19;
        int internal20, internal21, internal22;
        int leaf1, leaf2, leaf3, leaf4, leaf5, leaf6, leaf7, leaf8, leaf9, leaf10, leaf11, leaf12;
        int res;
        vector<int> leafs;

        // Tree permutation
        int idxs_input, thresholds_input, perm_bits_input, raw_tree;
        int shuffled_tree;

        // Optimized DT inference
        int depth;
        // vector<int> *internal_nodes;
        int internal_node;
        int leaf_node;
    public:
        DT();
        DT(Mat *data, vector<map<string, ll>> encode_table);
        void feed(DTDAG* nn, Mat& x_batch, Mat& y_batch, int input, int output);
        void next_batch(Mat& batch, int start, Mat* A, int mod = NM);
        void forward_predict(ll *total, Mat* count_data, Mat * seg_total);
        void forward_train();
        void forward_predict();
        void predict_graph();
        void train_graph();
        void permute_tree_graph(int depth);
        void forward_permute_tree(NodeMat *res, Mat *idxs_perm, Mat *thresholds_perm);
        void optimized_predict_graph(int depth);
        void forward_optimized_graph(Mat *data, Mat *idxs_perm, Mat *thresholds_perm, Mat *perm_bits);
        void test();
        void print_perd(int round);
        static Mat vector_to_mat(int* data, int r, int c);
    };
};


#endif //SMMLF_DT_GRAPH_H
