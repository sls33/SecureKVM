#ifndef MPC_ML_NN_H
#define MPC_ML_NN_H

#include "../Constant.h"
#include "../NeuronMat.h"
#include "../MathOp.h"

class NN
{
    vector<vector<int>> adj;
    int tot;
    int cur;
    int curForward;
    int curGrad;
    int curUpdate;
    int curReveal;
    vector<NeuronMat *> neuron;
    vector<bool> vst;
    vector<int> q;
    vector<int> to;

public:
    NN();
    NN &operator=(NN &a);
    void global_variables_initializer();
    void epoch_init();
    void reveal_init(int u);
    void reveal_init();
    void addedge(int u, int v);
    int addnode(int r, int c, int k);
    NeuronMat *getNeuron(int u);
    void setOp(int u, Op *op);
    void addOpAdd_Mat(int res, int a, int b);
    void addOpMinus_Mat(int res, int a, int b);
    void addOpSmoothLevel(int res, int a);
    void addOpMul_Mat(int res, int a, int b);
    void addOpMul_Const_Mat(int res, int a, double b); // for multiply learning rate
                                                       //    void addOpMul_Const_Mat(int res, int a, ll128 b);
    void addOpTruncate(int res, int a);                // for truncate the f bits
    void addOpDiv_Const_Mat(int res, int a, ll128 b);
    void addOpDiv_Const_Optimized_Mat(int res, int a, ll b);
    void addOpDiv_Seg_Const_Mat(int res, int a, int b);
    void addOpHada_Mat(int res, int a, int b);
    void addOpMeanSquaredLoss(int res, int a, int b);
    void addOpSimilar(int res, int a, int b);
    void addOpConcat(int res, int a, int b);
    void addOpHstack(int res, int a, int b);
    void addOpVstack(int res, int a, int b);
    void addOpVia(int res, int a);
    void addOpTanh(int res, int a);
    void addOpHard_Tanh(int res, int a);
    void addOpHybrid_Tanh(int res, int a);
    void addOpRaw_Tanh(int res, int a);
    void addOpTanh_change(int res, int a);
    void addOpTanh_ex(int res, int a);
    void addOpSigmoid(int res, int a);
    void addOpReLU(int res, int a);
    void addOpArgmax(int res, int a);
    void addOpLTZ(int res, int a);
    void addOpEQZ(int res, int a);
    void addOpEQZ_2LTZ(int res, int a);
    void addOpEqual(int res, int a, int b);
    //void addOpSecGetPer(int res, int a);
    void toposort();
    void gradUpdate();
    bool forwardHasNext();
    void forwardNext();
    void backNext();
    bool backHasNext();
    bool updateHasNext();
    void update();
    bool revealHasNext();
    void reveal();
    int getTot();
    void print();

    /** KV related operators **/
    void addOpSecGetBasic(int res, int kv_data, int query);
    void addOpSecGetFold(int res, int kv_data, int query);
    void addOpSecGetOnline(int res, int kv_data, int query, Mat *perm_mat, Mat *perm_mat_plain);
    
    void addOpSecGetPer(int res, int kv_data, int query, int shuffle_type);

    /** Complex function implementation **/
    void addOpPow_Log(int res, int a, int k);
    void addOpExp_approximate(int res, int a, int iterations);
    void addOpLog_approximate(int res, int a, int k, int f);

    /** YL added operators **/
    void addOpSum(int res, int a[], int size);
    void addOpAvg_Plain(int res, int a, ll128 dividend);
    void addOpAvg_Secret(int res, int a);

    /** Reshare operators **/
    void addOpReshare(int res, int a, int p0, int p1);
    void addOpGenPerm(int res, int a);
    void addOpPShuffleData(int res, int a);
    void addOpEShuffleData(int res, int a);
    void addOpShuffleDataNoGP(int res, int a);
};

#endif // MPC_ML_NN_H
