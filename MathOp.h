#ifndef MPC_ML_MATHOP_H
#define MPC_ML_MATHOP_H

#include "Op.h"
#include "NeuronMat.h"
#include "decision_tree/NodeMat.h"
#include "util/SocketOnline.h"
#include "Player.h"
#include "malicious_lib/reed_solomn_reconstruct.h"
#include "util/IOManager.h"
#include "Constant.h"
#include "util/Permutation.h"

extern int node_type;
extern SocketOnline *socket_io[M][M];
extern Player player[M];
extern Mat metadata;
// extern Mat *perm_mat, *perm_mat_plain;
class MathOp
{
public:
    class PRandFld;
    class MulPub;
    class PRandBit;
    class PRandM;
    class Reveal;
    class Div2mP;
    class DegRed;
    class RevealD;
    class Log_approximate;
    class ReShare;
    class ShuffleData;
    class Add_Mat : public Op
    {
        NeuronMat *res, *a, *b;

    public:
        Add_Mat();
        Add_Mat(NeuronMat *res, NeuronMat *a, NeuronMat *b);
        void forward();
        void back();
    };
    class Minus_Mat : public Op
    {
        NeuronMat *res, *a, *b;

    public:
        Minus_Mat();
        Minus_Mat(NeuronMat *res, NeuronMat *a, NeuronMat *b);
        void forward();
        void back();
    };
    class SmoothLevel : public Op
    {
        Mat *res, *a;
        Reveal *reveal;
        Log_approximate *log_appr;

    public:
        SmoothLevel();
        SmoothLevel(Mat *res, Mat *a);
        void forward();
        void back();
    };
    class Mul_Mat : public Op
    {
        NeuronMat *res, *a, *b;
        Mat *temp_a, *temp_b;
        Div2mP *div2mP_f;
        Div2mP *div2mP_b_a, *div2mP_b_b;

    public:
        Mul_Mat();
        Mul_Mat(NeuronMat *res, NeuronMat *a, NeuronMat *b);
        void forward();
        void back();
    };
    class Hada_Mat : public Op
    {
        NeuronMat *res, *a, *b;
        Mat *temp_a, *temp_b;
        Div2mP *div2mP_f;
        Div2mP *div2mP_b_a, *div2mP_b_b;

    public:
        Hada_Mat();
        Hada_Mat(NeuronMat *res, NeuronMat *a, NeuronMat *b);
        void forward();
        void back();
    };
    class Mul_Const_Trunc : public Op
    {
        Mat *res, *a, *revlea1;
        //        ll128 b;
        double b;
        Div2mP *div2mP;
        Reveal *reveal, *reveal2;

    public:
        Mul_Const_Trunc();
        Mul_Const_Trunc(Mat *res, Mat *a, double b);
        void forward();
        void back();
    };
    class Div_Const_Trunc : public Op
    {
        Mat *res, *a;
        ll128 b;
        Div2mP *div2mP;

    public:
        Div_Const_Trunc();
        Div_Const_Trunc(Mat *res, Mat *a, ll128 b);
        void forward();
        void back();
    };
    class Div_Const_Trunc_Optimized : public Op
    {
        Mat *res, *a;
        ll b;
        int exponent;
        Div2mP *div2mP;

    public:
        Div_Const_Trunc_Optimized();
        Div_Const_Trunc_Optimized(Mat *res, Mat *a, ll b);
        void forward();
        void back();
    };
    class Div_Seg_Const_Trunc : public Op
    {
        Mat *res, *a;
        Mat *b;
        Div2mP *div2mP;

    public:
        Div_Seg_Const_Trunc();
        Div_Seg_Const_Trunc(Mat *res, Mat *a, Mat *b);
        void forward();
        void back();
    };
    class Via : public Op
    {
        NeuronMat *res, *a;

    public:
        Via();
        Via(NeuronMat *res, NeuronMat *a);
        void forward();
        void back();
    };
    class MeanSquaredLoss : public Op
    {
        NeuronMat *res, *a, *b;
        Reveal *reveal_a, *reveal_b, *reveal_res;
        Mat *tmp_a, *tmp_b, *tmp_res;

    public:
        MeanSquaredLoss();
        MeanSquaredLoss(NeuronMat *res, NeuronMat *a, NeuronMat *b);
        void forward();
        void back();
    };
    class Similar : public Op
    {
        NeuronMat *res, *a, *b;

    public:
        Similar();
        Similar(NeuronMat *res, NeuronMat *a, NeuronMat *b);
        void forward();
        void back();
    };
    class Concat : public Op
    {
        NeuronMat *res, *a, *b;

    public:
        Concat();
        Concat(NeuronMat *res, NeuronMat *a, NeuronMat *b);
        void forward();
        void back();
    };
    class Hstack : public Op
    {
        NeuronMat *res, *a, *b;

    public:
        Hstack();
        Hstack(NeuronMat *res, NeuronMat *a, NeuronMat *b);
        void forward();
        void back();
    };
    class Vstack : public Op
    {
        NeuronMat *res, *a, *b;

    public:
        Vstack();
        Vstack(NeuronMat *res, NeuronMat *a, NeuronMat *b);
        void forward();
        void back();
    };
    class Div2mP : public Op
    {
        Mat *res, *a;
        Mat *r_nd, *r_st, *r_B;
        Mat *r;
        Mat *b;
        int k, m;
        PRandM *pRandM;
        RevealD *pRevealD;

    public:
        Div2mP();
        Div2mP(Mat *res, Mat *a, int k, int m);
        void forward();
        void back();
        void reset_for_multi_call();
    };
    class Reveal : public Op
    {
        Mat *res, *a;
        Mat *b;
        reed_solomn *rs = new reed_solomn(MOD);

    public:
        Reveal();
        Reveal(Mat *res, Mat *a);
        void forward();
        void back();
    };
    class PRandM : public Op
    {
        Mat *r_nd, *r_st, *b_B;
        int k, m;
        PRandFld *pRandFld;
        PRandBit **pRandBit;
        Reveal *reveal, *reveal1;
        //        Mat *tmp, *tmp1;
    public:
        PRandM();
        PRandM(Mat *r_nd, Mat *r_st, Mat *b_B, int k, int m);
        PRandM(Mat *r_nd, Mat *r_st, int k, int m);
        PRandM(int r, int c, int k, int m);
        void PRandM_init(Mat *r_nd, Mat *r_st, Mat *b_B, int k, int m);
        void forward();
        void back();
        void reset_for_multi_call();
    };
    class PRandBit : public Op
    {
        Mat *res;
        PRandFld *pRandFld;
        MulPub *mulPub;
        Mat *a, *a2;
        Mat *a_r, *a2_r;

    public:
        PRandBit();
        PRandBit(Mat *res);
        void forward();
        void back();
        void reset_for_multi_call();
    };
    class MulPub : public Op
    {
        Mat *res, *a, *b;

    public:
        MulPub();
        MulPub(Mat *res, Mat *a, Mat *b);
        void forward();
        void back();
    };
    class PRandFld : public Op
    {
        Mat *res;
        ll128 range;

    public:
        PRandFld();
        PRandFld(Mat *res, ll range);
        void forward();
        void back();
    };
    class Mod2 : public Op
    {
        Mat *res, *a;
        Mat *r_nd, *r_st, *r_B;
        Mat *c;
        int k;
        PRandM *pRandM;
        Reveal *reveal;

    public:
        Mod2();
        Mod2(Mat *res, Mat *a, int k);
        void forward();
        void back();
    };
    class Mod2D : public Op
    {
        Mat *res, *a;
        Mat *r_nd, *r_st, *r_B;
        Mat *c;
        int k;
        PRandM *pRandM;
        Reveal *reveal;
        DegRed *degred;

    public:
        Mod2D();
        Mod2D(Mat *res, Mat *a, int k);
        void forward();
        void back();
    };
    class DegRed : public Op
    {
        Mat *res, *a;
        Mat *tmp;

    public:
        DegRed();
        DegRed(Mat *res, Mat *a);
        void forward();
        void back();
    };
    class PreMulC : public Op
    {
        Mat *res, *a;
        Mat *m;
        Mat *w, *z;
        Mat *r, *s, *u;
        int k;
        PRandFld **pRandFld_r, **pRandFld_s;
        MulPub **pMulPub_st;
        DegRed **pDegRed;
        MulPub **pMulPub;

    public:
        PreMulC();
        PreMulC(Mat *res, Mat *a, int k);
        void forward();
        void back();
    };
    class BitLT : public Op
    {
        Mat *res, *a;
        Mat *b_B;
        Mat *d_B, *p_B;
        Mat *d_B_inverse, *p_B_inverse;
        Mat *s;
        int k;
        PreMulC *preMulC;
        Mod2 *pMod2;
        Reveal *reveal, *reveal1, *reveal_a;
        Reveal **reveal_b;
        Mat *tmp_b;
        Mat *tmp, *tmp1, *tmp_a;

    public:
        BitLT();
        BitLT(Mat *res, Mat *a, Mat *b_B, int k);
        void forward();
        void back();
    };
    class RevealD : public Op
    {
        Mat *res, *a, *b;
        Reveal *pReveal;
        DegRed *pDegRed;

    public:
        RevealD();
        RevealD(Mat *res, Mat *a, Mat *b);
        RevealD(Mat *res, Mat *a);
        void forward();
        void back();
        void reset_for_multi_call();
    };
    class Mod2m : public Op
    {
        Mat *res, *a;
        Mat *r_nd, *r_st, *r_B;
        Mat *b;
        Mat *u;
        int k, m;
        PRandM *pRandM;
        RevealD *pRevealD;
        BitLT *pBitLT;

    public:
        Mod2m();
        Mod2m(Mat *res, Mat *a, int k, int m);
        void forward();
        void back();
    };
    class Div2m : public Op
    {
        Mat *res, *a;
        Mat *b;
        int k, m;
        Mod2m *pMod2m;
        Mat *test1, *test2, *test3;
        Reveal *reveal1, *reveal2, *reveal3;

    public:
        Div2m();
        Div2m(Mat *res, Mat *a, int k, int m);
        void forward();
        void back();
    };
    class LTZ : public Op
    {
        Mat *res, *a;
        int k;
        Div2m *pDiv2m;
        Reveal *reveal, *reveal_tmp, *reveal_a;
        Mat *tmp1, *tmp2, *tmp3;

    public:
        LTZ();
        LTZ(Mat *res, Mat *a, int k);
        void forward();
        void back();
    };
    class KOrCL : public Op
    {
        Mat *res, *d_B;
        Mat *r_nd, *r_st, *r_B;
        Mat *A, *C, *A_pow;
        Mat *b, *b_st, *B_pub, *B_mul;
        int k, m;
        PRandFld **pRandFld_b, **pRandFld_b_st;
        vector<ll> coefficients;
        MulPub **mul_pub_nd, **mul_pub;
        DegRed **pDegRed;
        PRandM *pRandM;
        Reveal *reveal, *reveal_tmp, *reveal_a;
        Reveal **reveal_a_pow;
        Mat *tmp1, *tmp2, *tmp3;

    public:
        KOrCL();
        KOrCL(Mat *res, Mat *a, int k);
        void forward();
        void back();
    };
    class EQZ : public Op
    {
        Mat *res, *a;
        Mat *r_nd, *r_st, *r_B;
        Mat *b;
        Mat *u;
        Mat *d_B;
        int k, m;
        PRandM *pRandM;
        Reveal *pReveal;
        Div2m *pDiv2m;
        Reveal *reveal, *reveal_tmp, *reveal_a;
        KOrCL *kOrCl;
        Mat *tmp1, *tmp2, *tmp3;

    public:
        EQZ();
        EQZ(Mat *res, Mat *a, int k);
        void forward();
        void back();
    };
    class EQZ_2LTZ : public Op
    {
        Mat *res, *a;
        Mat *u_st, *u_nd;
        Mat *u_st_res, *u_nd_res;
        LTZ *pLTZ_f_1, *pLTZ_f_2;

        int k, m;
        PRandM *pRandM;
        Div2m *pDiv2m;
        Reveal *reveal, *reveal_1, *reveal_2;
        KOrCL *kOrCl;
        Mat *tmp1, *tmp2, *tmp3;

    public:
        EQZ_2LTZ();
        EQZ_2LTZ(Mat *res, Mat *a, int k);
        void forward();
        void back();
    };
    class ReLU_Mat : public Op
    {
        NeuronMat *res, *a;
        LTZ *pLTZ;
        DegRed *pDegRed_f, *pDegRed_b;
        Div2mP *pDiv2mp_f, *pDiv2mp_b;
        Reveal *reveal_a, *reveal_res;
        Mat *tmp_a, *tmp_res;

    public:
        ReLU_Mat();
        ReLU_Mat(NeuronMat *res, NeuronMat *a);
        void forward();
        void back();
    };
    class Sigmoid_Mat : public Op
    {
        NeuronMat *res, *a;
        Mat *u_st, *u_nd;
        Mat *u_st_res, *u_nd_res;
        LTZ *pLTZ_f_1, *pLTZ_f_2;
        //        DegRed *pDegRed_res;
        Div2mP *pDegRed_res;
        //        Div2mP *pDiv2mP_aux, *pDiv2mP_a;
        Div2mP *pDiv2mP_b1, *pDiv2mP_b2;

        Mat *tmp;
        Reveal *reveal;

    public:
        Sigmoid_Mat();
        Sigmoid_Mat(NeuronMat *res, NeuronMat *a);
        void forward();
        void back();
    };
    class Argmax : public Op
    {
        NeuronMat *res, *a;

    public:
        Argmax();
        Argmax(NeuronMat *res, NeuronMat *a);
        void forward();
        void back();
    };
    class Equal : public Op
    {
        NeuronMat *res, *a, *b;

    public:
        Equal();
        Equal(NeuronMat *res, NeuronMat *a, NeuronMat *b);
        void forward();
        void back();
    };
    class CrossEntropy : public Op
    {
        NeuronMat *res, *a, *b;
        Reveal *reveal;

    public:
        CrossEntropy();
        CrossEntropy(NeuronMat *res, NeuronMat *a, NeuronMat *b);
        void forward();
        void back();
    };
    static void broadcast(Mat *a);
    static void broadcast_share(Mat *a, int target);
    static void receive_share(Mat *a, int from);
    static void broadcase_rep(Mat *a);
    static void receive(Mat *a);
    static void receive_add(Mat *a);
    static void receive_rep(Mat *a);
    static void random(Mat *a, ll range);
    class Tanh_ex : public Op
    {
        NeuronMat *a, *res;
        Mat *temp_f, *temp_b;

    public:
        Tanh_ex();
        Tanh_ex(NeuronMat *res, NeuronMat *a);
        void forward();
        void back();
    };
    class Hard_Tanh : public Op
    {
        NeuronMat *a, *res;
        Mat *temp_f, *temp_b;

    public:
        Hard_Tanh();
        Hard_Tanh(NeuronMat *res, NeuronMat *a);
        void forward();
        void back();
    };
    class Sigmoid : public Op
    {
        NeuronMat *a, *res;
        Mat *temp_f, *temp_b;

    public:
        Sigmoid();
        Sigmoid(NeuronMat *res, NeuronMat *a);
        void forward();
        void back();
    };
    class Tanh : public Op
    {
        NeuronMat *a, *res;
        Mat *a_r, *temp_f, *temp_b;
        // const ll128 coefficients_f[10]={10,0,-652,0,15392,0,-164163,0,971505,0};
        const ll128 coefficients_f[6] = {3342, 0, 99999999999909374, 0, 879282, 0};
        Div2mP *div2mP_f1;
        Div2mP *div2mP_f2;
        Div2mP *div2mP_f3;
        Div2mP *div2mP_f4;
        Div2mP *div2mP_f5;
        Div2mP *div2mP_f6;
        Div2mP *div2mP_f7;
        Div2mP *div2mP_f8;
        Div2mP *div2mP_b1;
        Div2mP *div2mP_b2;

    public:
        Tanh();
        Tanh(NeuronMat *res, NeuronMat *a);
        void forward();
        void back();
    };
    class Tanh_Mat : public Op
    {
        NeuronMat *a, *res;
        Mat *a_r;
        const ll128 coefficients[2] = {-246456, 1036312};
        Div2mP *div2mP_f1;
        Div2mP *div2mP_f2;
        Div2mP *div2mP_f3;
        Div2mP *div2mP_b1;
        Div2mP *div2mP_b2;
        Mat *u_st;
        Mat *u_nd;
        Mat *u_rd;
        Mat *u_st_res, *u_nd_res, *u_rd_res;
        Reveal *reveal;
        LTZ *pLTZ_f_1;
        LTZ *pLTZ_f_2;

    public:
        Tanh_Mat();
        Tanh_Mat(NeuronMat *res, NeuronMat *a);
        void forward();
        void back();
    };

    class Hybrid_tanh : public Op
    {
        NeuronMat *a, *res;
        Mat *a_2;
        const ll128 coefficients[2] = {-246456, 1036312};
    };
    class Tanh_change : public Op
    {
        NeuronMat *a, *res;

    public:
        Tanh_change();
        Tanh_change(NeuronMat *res, NeuronMat *a);
        void forward();
        void back();
    };

    class Raw_Tanh : public Op
    {
        NeuronMat *a, *res;
        Mat *temp_f, *temp_b;

    public:
        Raw_Tanh();
        Raw_Tanh(NeuronMat *res, NeuronMat *a);
        void forward();
        void back();
    };

    class SigmaOutput : public Op
    {
        //        NodeMat *node;
        Mat *res, **a;
        int size;

    public:
        SigmaOutput();
        SigmaOutput(Mat *node, Mat **a, int size);
        void forward();
        void back();
    };

    /** Non-parametric decision tree **/
    class DT_entropy : public Op
    {
        NodeMat *node;

    public:
        DT_entropy();
        DT_entropy(NodeMat *node);
        void forward(ll *res);
    };

    class entropy_gain : public Op
    {
        NodeMat *node;

    public:
        entropy_gain();
        entropy_gain(NodeMat *node);
        void forward(ll *res);
    };

    class feature_split_finding : public Op
    {
        NodeMat *node;

    public:
        feature_split_finding();
        feature_split_finding(NodeMat *node);
        void forward(ll *res);
    };

    /** Decision tree inference **/

    class TreePerm : public Op
    {
    private:
        NodeMat *res;
        Mat *feature_vector, *threshold;

        vector<int> perm_bits_prev, perm_bits_nxt, tree_permutation;
        Mat *shuffle_idxs, *shuffle_thresholds;
        ReShare **reshare_idxs1, **reshare_thresholds1;
        ReShare **reshare_idxs2, **reshare_thresholds2;
        ReShare **reshare_idxs3, **reshare_thresholds3;

        Mat *tmp_bits;
        Mat *first_bits, *second_bits;
        Mat *final_bits, *final_bits_reveal;
        DegRed *degRed;
        Reveal *reveal_final_bits;

        Mat *reveal1, *reveal2;
        Reveal **reveal_idxs, **reveal_thresholds;

    public:
        TreePerm();
        TreePerm(NodeMat *res, Mat *feature_vector, Mat *threshold);
        void forward();
        void back();
    };
    class OptInternalNode : public Op
    {
        NodeMat *node;
        Mat *product;
        Mat *res, *input;
        Mat *feature_vector, *threshold, *perm_bit;
        Mat *index;
        int fold_num;
        Mat *vector_x, *vector_y; //  Fold to perform feature selection
        Mat *rows;
        Mat *feature_val;
        Mat *cur_test, *comparison_res;
        Mat *xor_res;
        Div2mP *div2mp, *div2mp_row;
        Reveal *reveal_res, *reveal_feature, *reveal_threshold, *reveal_com_res, *reveal_feature_val;
        LTZ *ltz;
        Mat *feature_vec_reveal, *com_res_reveal, *res_reveal, *feature_val_reveal;

        // timing
        system_clock::time_point end, start;
        microseconds time_span;

    public:
        OptInternalNode();
        OptInternalNode(Mat *res, Mat *input, Mat *feature_vector, Mat *threshold, Mat *perm_bit);
        void forward();
        void back();
    };

    class Internal : public Op
    {
        NodeMat *node;
        Mat *product, *accumulate_product;
        Mat *val_mat;
        ll shared_val;
        ll shared_label;
        Mat *vector, *res, *cur_test, *test;
        Mat *tmp, *eqz_res;
        int label, feature_val;
        Div2mP *div2mp;
        Op *eqz;
        int sign;
        int sibling;
        Mat *sibling_test;
        //        EQZ_2LTZ *eqz;
        Reveal *reveal;
        Mat *tmp_reveal;

    public:
        Internal();
        Internal(Mat *res, Mat *vector, double feature_val, int sign, Mat *test, int sibling, Mat *sibling_test);
        void forward();
        void back();
    };

    class FeatureNode : public Op
    {
        NodeMat *node;
        Mat *product, *product_deg, *accumulate_product;
        Mat *bitmap;
        Mat *perm_mat_plain, *perm_mat, *table_data_shuffle, *query_plaintex;
        Mat *table_data, *res, *a;
        Mat *tmp;
        Mat *tmp_reveal;
        int feature_index;
        Div2mP *div2mp, *div2mp_shuffle;
        Reveal *reveal;

    public:
        FeatureNode();
        FeatureNode(Mat *res, Mat *table_data, Mat *a, int feature_index);
        void forward();
        void back();
    };

    class Leaf : public Op
    {
        NodeMat *node;
        Mat *product;
        Mat *label_mat;
        ll shared_label;
        Mat *a, *res;
        Mat *tmp, *eqz_res;
        int label;
        Div2mP *div2mp;
        EQZ *eqz;
        Reveal *reveal;
        Mat *tmp_reveal;

    public:
        Leaf();
        Leaf(Mat *node, Mat *eqz_res, int feature_val);
        void forward();
        void back();
    };

    /** Map primitives **/
    class Map_Get : public Op
    {
        Mat *node;
        Mat *a, *b;

    public:
        Map_Get();
        Map_Get(Mat *node, Mat *a, Mat *b);
        void forward();
        void back();
    };

    class Map_Set : public Op
    {
        Mat *node;
        Mat *a, *b;

    public:
        Map_Set();
        Map_Set(Mat *node, Mat *a, Mat *b);
        void forward();
        void back();
    };

    class SecGetBasic : public Op
    {
        Mat *node;
        Mat *res, *kv_data;
        Mat *query_paintext;
        // int batch_size, feature_dim;
        Mat *bitmap;
        Mat *product;
        Div2mP *div2mp;
        Reveal *reveal;
        Mat *tmp_reveal;

    public:
        SecGetBasic();
        SecGetBasic(Mat *res, Mat *kv_data, Mat *query_paintext);
        void forward();
        void back();
    };

    class SecGetFold : public Op
    {
        Mat *node;
        Mat *res, *kv_data;
        Mat *kv_data_reshape;
        Mat *query_paintext;
        // int batch_size, feature_dim;
        int fold_dim;
        Mat *bitmap;
        Mat bitmap_x, bitmap_y;
        Mat *row, *row_trunc;
        Mat *product;
        Div2mP *div2mp_row, *div2mp_res;
        Reveal *reveal;
        Mat *tmp_reveal;
        bool isMarkovInference = BENCHMARK ? false : true;
        int query_num;

    public:
        SecGetFold();
        SecGetFold(Mat *res, Mat *kv_data, Mat *query_paintext);
        void forward();
        void back();
    };

    class SecGetOnline : public Op
    {
        Mat *node;
        Mat *res, *kv_data, *kv_data_shuffle;
        Mat *query_paintext;
        // int batch_size, feature_dim;
        Mat *bitmap;
        Mat *product;
        Div2mP *div2mp_data;
        Reveal *reveal;
        Mat *tmp_reveal;
        Mat *perm_mat, *perm_mat_plain;

    public:
        SecGetOnline();
        SecGetOnline(Mat *res, Mat *kv_data, Mat *query_paintext, Mat *perm_mat, Mat *perm_mat_plain);
        ~SecGetOnline();
        void forward();
        void back();
    };

    // permutation kvdata and get value of index
    // both kv_data and index are rx1 mat
    class SecGetPer : public Op
    {
        Mat *res, *kv_data, *index_mat;
        // Mat *shuffe_kv;
        int kv_r;                 // row of kvdata
        int shuffle_type = 0;     // pshuffle = 0, eshuffle = 1, shufflenogp = 2
        ShuffleData *shuffledata; // shuffle kv_data
        Mat *index_shuffle;       // index after shuffle

        Mat *skv_plain;
        Reveal *skv_rev;

    public:
        SecGetPer(Mat *res, Mat *kv_data, Mat *index_mat, int shuffle_type);
        ~SecGetPer();
        void forward();
        void back();

    private:
        SecGetPer();
    };

    /** Complex non-linear functions **/
    class Pow_log : public Op
    {
        Mat *res, *a;
        int k, cur_k, last_k;
        Mat *tmp, *base;
        Div2mP *div2mp_res, *div2mp_tmp;
        bool perform_div2mp;
        Reveal *reveal, *reveal_tmp, *reveal_a;
        Mat *res_red, *tmp_red, *tmp3;

    public:
        Pow_log();
        Pow_log(Mat *res, Mat *a, int k);
        void forward();
        void back();
    };

    class Exp_approximate : public Op
    {
        Mat *res, *a, *cur;
        int exp_iterations;
        Mat *base;
        Div2m *pDiv2m;
        Div2mP *div2mp;
        Mat *base_tr;
        Reveal *reveal, *reveal_tmp, *reveal_a;
        Mat *tmp1, *tmp2, *tmp3;
        Pow_log *pow_log;

    public:
        Exp_approximate();
        Exp_approximate(Mat *res, Mat *a, int k);
        void forward();
        void back();
    };

    class SufMulInv : public Op
    {
        Mat *p_B, *p_inv_B, *p_rev_B, *p_inv_rev_B, *d_B, *d_rev_B, *d_reveal_B, *r_B, *c_B, *c_rev_B, *c_inv_rev_B, *c_rev, *c, *test_1, *test_2, *test_3;
        int k, m;
        Div2m *pDiv2m;
        Reveal **reveal, *reveal_tmp, *reveal_a, **reveal_d, **reveal_test3, **reveal_test1, **reveal_test2;
        Exp_approximate *exp_approximate;
        Mat *tmp1, *tmp2, *tmp3;
        PRandFld **prandFld_c;
        // Reveal *reveal;
        Div2mP **div2mp1, **div2mp2;
        PreMulC *preMulC;
        MulPub **mulpub;

    public:
        SufMulInv();
        SufMulInv(Mat *p_B, Mat *p_inv_B, Mat *d_B, int k);
        void forward();
        void back();
    };

    class PreBitLT : public Op
    {
        Mat *res, *a, *b_B, *test_1, *test_2, *test_3, *test_4, *test_5;
        Mat *p_B, *p_st_B, *d_B, *d_B_plus_one, *s_B, *tmp_mul_sp;
        int k;
        Reveal **reveal_test3, *reveal_test1, **reveal_test2, *reveal_test4, *reveal_test5;
        SufMulInv *sufMulInv;
        Div2mP **div2mp;
        Mod2 **mod2D;
        Mod2 *mod2;

    public:
        PreBitLT();
        PreBitLT(Mat *res, Mat *a, Mat *b_B, int k);
        void forward();
        void back();
    };

    class PreMod2m : public Op
    {
        Mat *res, *a, *a_reveal, *res_reveal;
        Mat *r_nd, *r_st, *r_B, *r, *b;
        int log_iterations, k, m;
        PRandM *pRandM;
        PreBitLT *preBitLT;
        Div2m *pDiv2m;
        Reveal *reveal, *reveal_a, **reveal_res;
        Mat *temp_c, *temp_r, *temp_c_dec, *temp_s_dec, *temp_u_dec;

    public:
        PreMod2m();
        PreMod2m(Mat *res, Mat *a, int k, int m);
        void forward();
        void back();
    };
    class BitDec : public Op
    {
        Mat *res, *a, *b_temp, *reveal_b_temp, *a_reveal;
        Reveal **reveal, *reveal_a;
        int k, m;
        PreMod2m *preMod2m;

    public:
        BitDec();
        BitDec(Mat *res, Mat *a, int k, int m);
        void forward();
        void back();
    };
    class SufOrC : public Op
    {
        Mat *res, *a, *a_rev, *b, *res_rev, *res_reveal;
        int k;
        PreMulC *preMulC;
        Reveal **reveal_res;
        Mod2 **mod2;

    public:
        SufOrC();
        SufOrC(Mat *res, Mat *a, int k);
        void forward();
        void back();
    };
    class Log_approximate : public Op
    {
        Mat *res, *a, *exp_res, *x_pie, *a_B, *c_B, *x_st, *b, *exp, *xb, *exp_reveal;
        const ll128 coeff_degree1[2] = {1050673, 99999999998999452}; // f(x) = 1.002x - 0.9452
        const ll128 coeff_degree2[3] = {99999999999646396, 2093384, 99999999998269374};
        const ll128 coeff_degree3[4] = {169974, 99999999998882221, 3212836, 99999999997735079}; // f(x) = 0.1621x^3 - 1.066x^2 + 3.064x - 2.16
        const ll128 coeff_degrees[3][4] = {
            {1050673, 0, 0, 99999999998999452},
            {99999999999646396, 2093384, 0, 99999999998269374},
            {169974, 99999999998882221, 3212836, 99999999997735079}};
        int k, f, temp;
        Div2m *pDiv2m;
        Div2mP *pDiv2mP, **div2mp_a_B, *Div2mp_1, *Div2mp_2;
        Mat *reveal_c_B, *reveal_a_B;
        Reveal **reveal, *reveal_tmp, **reveal_a, *reveal_exp;
        BitDec *bitDec;
        Exp_approximate *exp_approximate;
        SufOrC *sufOrC;
        Mat *tmp1, *tmp2, *tmp3;
        Mat *mul_a_B;
        Div2mP *divIE;
        Constant::Clock *clock_train;
        int depth = 1; // 1,2,3

    public:
        Log_approximate();
        Log_approximate(Mat *res, Mat *a, int k, int f);
        void forward();
        void back();
    };
    // class Div: public Op{
    //     Mat *res,*a,*b,*c,*rand;
    //     int k,f,temp;
    //     PRandFld *prandFld_c;
    //     Div2mP *pDiv2mP,*p_Div2mp_res;

    // public:
    //     Div();
    //     Div(Mat *res,Mat *a, Mat *b);
    //     void forward();
    //     void back();
    // };

    class DivMat_CTO : public Op
    {
        Mat *res, *a, *b, *division;
        int *exponent;
        Div2mP *div2mP;

    public:
        DivMat_CTO();
        DivMat_CTO(Mat *res, Mat *a, Mat *b);
        void forward();
        void back();
    };

    class ReShare : public Op
    {
        Mat *res, *a;
        Mat *b;
        Mat *share0, *share1;
        Mat *shares;
        int p0, p1;
        reed_solomn *rs = new reed_solomn(MOD);

        // Mat *a_p;
        // Reveal *rev;

    public:
        ReShare();
        ReShare(Mat *res, Mat *a, int p0, int p1);
        void forward();
        void back();
    };

    // ger random mat r and permutation r in three-party setting
    class GenPerm : public Op
    {
    private:
        int r, c;
        Mat *r_mat, *pr_mat; // random mat, permutation random mat
        PRandFld *pRandFld;  // generate random mat
        ReShare *reshare0, *reshare1, *reshare2;
        
        PermutationObj *permObjPrev;
        PermutationObj *permObjNext;
        // vector<int> perm_mat0;
        // vector<int> perm_mat1;

        // Mat *p1,*p2,*p3;
        // Reveal *rev1,*rev2,*rev3;

    public:
        GenPerm(Mat *pr_mat, Mat *r_mat);
        ~GenPerm();
        void piemulr(int n0, int n1);
        void forward();
        void back();
        
        PermutationObj* getPermObjPrev();
        PermutationObj* getPermObjNext();

    private:
        GenPerm();
    };

    class ShuffleData : public Op
    {
    protected:
        int r, c;
        Mat *r_mat, *pr_mat;
        Mat *res, *kv_data;
        Mat *x_r;
        Mat *p_xr, *p_xr_plain;

        // PermutationObj *permNext, *permPrev;

    public:
        ShuffleData();
        ShuffleData(Mat *res, Mat *kv_data);

        ~ShuffleData();

        virtual void forward() = 0;
        virtual void back() = 0;
        virtual void ShuffleIndex(Mat *index_mat) = 0;
    };

    // The first charm P in PShuffleData reprensents that party0 is the query party
    // so P0 know extra pie1, can permutation x-r locally
    class PShuffleData : public ShuffleData
    {
    private:
        Mat *xr_plain;
        Reveal *xr_reveal;

        GenPerm *genperm;
        PermutationObj *permExtra;

    public:
        PShuffleData(Mat *res, Mat *kv_data);
        ~PShuffleData();

        void forward();
        void back();
        void ShuffleIndex(Mat *index_mat);
    };

    // The first charm E in EShuffleData represents that every party is equal to each other
    // None of three parties is query party
    // so none of three parties know permutation r
    class EShuffleData : public ShuffleData
    {
    private:
        GenPerm *genperm;
        PermutationObj *permExtra; // for query party

    public:
        EShuffleData(Mat *res, Mat *kv_data);
        ~EShuffleData();

        void forward();
        void back();
        void ShuffleIndex(Mat *index_mat);
    };

    class ShuffleDataNoGP : public ShuffleData
    {
    private:
        PRandFld *pRandFld;
        Mat *p_mat_plain, *p_mat;
        Div2mP *div2mp;
        Mat *xr_plain;
        Reveal *xr_reveal;

        Mat *r_plain, *pr_plain;
        Reveal *r_reveal, *pr_reveal;
        Reveal *res_reveal;

    public:
        ShuffleDataNoGP(Mat *res, Mat *kv_data);
        ~ShuffleDataNoGP();

        void forward();
        void back();
        void ShuffleIndex(Mat *index_mat);
    };
};

#endif // MPC_ML_MATHOP_H
