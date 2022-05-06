#ifndef SMMLF_DTDAG_H
#define SMMLF_DTDAG_H

#include "../Constant.h"
#include "NodeMat.h"
#include "../MathOp.h"

class DTDAG {
    vector<vector<int> > adj;
    int tot;
    int cur;
    int curForward;
    int curGrad;
    vector<NodeMat*> nodes;
    vector<bool> vst;
    vector<int> q;
    vector<int> to;
public:
    DTDAG();
    DTDAG& operator=(DTDAG &a);
    void global_variables_initializer();
    void epoch_init();
    void reveal_init(int u);
    void reveal_init();
    void addedge(int u, int v);
    int addnode(int r, int c, int k);
    NodeMat* getNode(int u);
    void setOp(int u, Op* op);
    void addOpAdd_Mat(int res, int a, int b);
    void addOpMinus_Mat(int res, int a, int b);
    void addOpSmoothLevel(int res, int a);
    void addOpMul_Mat(int res, int a, int b);
    void addOpMul_Const_Mat(int res, int a, double b);   // for multiply learning rate
//    void addOpMul_Const_Mat(int res, int a, ll128 b);
    void addOpDiv_Const_Mat(int res, int a, ll128 b);
    void addOpDiv_Seg_Const_Mat(int res, int a, int b);
    void addOpMeanSquaredLoss(int res, int a, int b);
    void addOpSimilar(int res, int a, int b);
    void addOpConcat(int res, int a, int b);
    void addOpHstack(int res, int a, int b);
    void addOpVstack(int res, int a, int b);
    void addOpVia(int res, int a);
    void addOpTanh(int res, int a);
    void addOpHard_Tanh(int res, int a);
    void addOpTanh_ex(int res, int a);
    void addOpSigmoid(int res, int a);
    void addOpArgmax(int res, int a);
    void addOpLTZ(int res, int a);
    void addOpEqual(int res, int a, int b);
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

    // dt op
    void addOpDT_FeatureNode(int res, int a, int feature_index, Mat *input_table_data);
    void addOpDT_InternalNode(int res, int feature, double feature_val, int sign, int last_internal, int sibling);
    void addOpDT_Opt_InternalNode(int res, int input, Mat *feature_vector, Mat *threshold, Mat *perm_bit);
    void addOpDT_PermTree(int res, int raw_tree);

    void addOpDT_LeafNode(int res, int a, int label);
    void addOpDT_ResNode(int res, int a[], int size);

    // key-value op
    void addOpMapGet(int res, int a, int b);
    void addOpMapSet(int res, int a, int b);
};


#endif //SMMLF_DTDAG_H
