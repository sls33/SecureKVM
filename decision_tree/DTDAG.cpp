#include "DTDAG.h"

DTDAG::DTDAG() {
    tot = 0;
    cur = 0;
    adj.resize(MAX_NODE_NUM, vector<int>(0));
    nodes.resize(MAX_NODE_NUM);
    vst.resize(MAX_NODE_NUM);
    q.resize(MAX_NODE_NUM);
    to.resize(MAX_NODE_NUM);
}

DTDAG& DTDAG::operator=(DTDAG &a) {
    for (int i = 1; i <= tot; i++) {
        *getNode(i)->getForward() = *a.getNode(i)->getForward();
        *getNode(i)->getGrad() = *a.getNode(i)->getGrad();
    }
    return *this;
}

void DTDAG::global_variables_initializer() {
    for (int i = 1; i <= tot; i++) {
        if (nodes[i]->getForward() == nullptr) {
            nodes[i]->initForward();
        }
        if (nodes[i]->getGrad() == nullptr) {
            nodes[i]->initGrad();
        }
        /** init node mat **/
        if (nodes[i]->getTable() == nullptr) {
            nodes[i]->initTable();
        }
        // todo: delete or add new op
        if (nodes[i]->getIsNet()) {
            nodes[i]->setOpUpdate(new MathOp::Mul_Const_Trunc(getNode(i)->getGrad(), getNode(i)->getGrad(), 0.0001));
        }
    }
    curForward = 1;
    curGrad = tot;
}

void DTDAG::epoch_init() {
    for (int i = 1; i <= tot; i++) {
        if (!nodes[i]->getIsBack()) {
            nodes[i]->getGrad()->clear();
        }
    }
    for (int i = 1; i <= tot; i++) {
        nodes[i]->resetOp();
    }
    curForward = 1;
    curGrad = tot;
}

void DTDAG::reveal_init(int u) {
    nodes[u]->setOpReveal(new MathOp::Reveal(getNode(u)->getForward(), getNode(u)->getForward()));
}

void DTDAG::reveal_init() {
    for (int i = 1; i <= tot; i++) {
        nodes[i]->resetOp();
    }
}

void DTDAG::addedge(int u, int v) {
    adj[u].push_back(v);
    to[v]++;
}

int DTDAG::addnode(int r, int c, int k) {
    nodes[++tot] = new NodeMat(r, c, k);
    return tot;
}

NodeMat* DTDAG::getNode(int u) {
    return nodes[u];
}

void DTDAG::setOp(int u, Op *op) {
    nodes[u]->setOp(op);
}

void DTDAG::addOpAdd_Mat(int res, int a, int b) {
    addedge(a, res);
    addedge(b, res);
    setOp(res, new MathOp::Add_Mat(getNode(res), getNode(a), getNode(b)));
}

void DTDAG::addOpMinus_Mat(int res, int a, int b) {
    addedge(a, res);
    addedge(b, res);
    setOp(res, new MathOp::Minus_Mat(getNode(res), getNode(a), getNode(b)));
}

void DTDAG::addOpSmoothLevel(int res, int a) {
    addedge(a, res);
    setOp(res, new MathOp::SmoothLevel(getNode(res)->getForward(), getNode(a)->getForward()));
}

void DTDAG::addOpMul_Mat(int res, int a, int b) {
    addedge(a, res);
    addedge(b, res);
    setOp(res, new MathOp::Mul_Mat(getNode(res), getNode(a), getNode(b)));
}

//void DTDAG::addOpMul_Const_Mat(int res, int a, ll128 b) {
//    addedge(a, res);
//    setOp(res, new MathOp::Mul_Const_Trunc(getNode(res)->getForward(), getNode(a)->getForward(), b));
//}

// for multiply learning rate
void DTDAG::addOpMul_Const_Mat(int res, int a, double b) {
    addedge(a, res);
    setOp(res, new MathOp::Mul_Const_Trunc(getNode(res)->getForward(), getNode(a)->getForward(), b));
}

void DTDAG::addOpDiv_Const_Mat(int res, int a, ll128 b) {
    addedge(a, res);
    setOp(res, new MathOp::Div_Const_Trunc(getNode(res)->getForward(), getNode(a)->getForward(), b));
}

void DTDAG::addOpDiv_Seg_Const_Mat(int res, int a, int b) {
    addedge(a, res);
    setOp(res, new MathOp::Div_Seg_Const_Trunc(getNode(res)->getForward(), getNode(a)->getForward(), getNode(b)->getForward()));
}

void DTDAG::addOpMeanSquaredLoss(int res, int a, int b) {
    addedge(a, res);
    addedge(b, res);
    setOp(res, new MathOp::MeanSquaredLoss(getNode(res), getNode(a), getNode(b)));
}

void DTDAG::addOpSimilar(int res, int a, int b) {
    addedge(a, res);
    addedge(b, res);
    setOp(res, new MathOp::Similar(getNode(res), getNode(a), getNode(b)));
}

void DTDAG::addOpConcat(int res, int a, int b) {
    addedge(a, res);
    addedge(b, res);
    setOp(res, new MathOp::Concat(getNode(res), getNode(a), getNode(b)));
}

void DTDAG::addOpHstack(int res, int a, int b) {
    addedge(a, res);
    addedge(b, res);
    setOp(res, new MathOp::Hstack(getNode(res), getNode(a), getNode(b)));
}

void DTDAG::addOpVstack(int res, int a, int b) {
    addedge(a, res);
    addedge(b, res);
    setOp(res, new MathOp::Vstack(getNode(res), getNode(a), getNode(b)));
}

void DTDAG::addOpVia(int res, int a) {
    addedge(a, res);
    setOp(res, new MathOp::Via(getNode(res), getNode(a)));
}

void DTDAG::addOpTanh(int res, int a) {
    addedge(a, res);
    setOp(res, new MathOp::Tanh(getNode(res), getNode(a)));
}

void DTDAG::addOpHard_Tanh(int res, int a) {
    addedge(a, res);
    setOp(res, new MathOp::Hard_Tanh(getNode(res), getNode(a)));
}

void DTDAG::addOpTanh_ex(int res, int a) {
    addedge(a, res);
    setOp(res, new MathOp::Tanh_ex(getNode(res), getNode(a)));
}

void DTDAG::addOpSigmoid(int res, int a) {
    addedge(a, res);
    setOp(res, new MathOp::Sigmoid_Mat(getNode(res), getNode(a)));
}

void DTDAG::addOpLTZ(int res, int a) {
    addedge(a, res);
    setOp(res, new MathOp::LTZ(getNode(res)->getForward(), getNode(a)->getForward(), BIT_P_LEN));
}

void DTDAG::addOpArgmax(int res, int a) {
    addedge(a, res);
    setOp(res, new MathOp::Argmax(getNode(res), getNode(a)));
}

void DTDAG::addOpEqual(int res, int a, int b) {
    addedge(a, res);
    addedge(b, res);
    setOp(res, new MathOp::Equal(getNode(res), getNode(a), getNode(b)));
}

void DTDAG::toposort() {
    int l, r;
    vst = vector<bool>(MAX_NODE_NUM, 0);
    l = r = 0;
    for (int i = 1; i <= tot; i++) {
        if (!to[i]) {
            q[++r] = i;
            vst[i] = 1;
        }
    }
    while (l < r) {
        int u = q[++l];
        int len = adj[u].size();
        for (int i = 0; i < len; i++) {
            int j = adj[u][i];
            to[j]--;
            if (!to[j]) {
                vst[j] = 1;
                q[++r] = j;
            }
        }
    }
    for (int i = 1; i <= tot; i++) {
        DBGprint("%d ", q[i]);
    }
    DBGprint("\n");
}

void DTDAG::gradUpdate() {
    for (int i = 1; i <= tot; i++)
        nodes[i]->update();
}

bool DTDAG::forwardHasNext() {
//    cout << curForward << ": " << tot << endl;
    return curForward < tot || nodes[q[curForward]]->forwardHasNext();
}

// todo: overload, for the decision tree has different forward method
void DTDAG::forwardNext() {
    while (!nodes[q[curForward]]->forwardHasNext()) {
        curForward++;
        if (curForward > tot)
            return;
    }
    nodes[q[curForward]]->forward();
}

bool DTDAG::backHasNext() {
    return curGrad > 1 || nodes[q[curGrad]]->backHasNext();
}

void DTDAG::backNext() {
    if (!backHasNext())
        return;
    if (nodes[q[curGrad]]->backHasNext()) {
        nodes[q[curGrad]]->back();
    }
    else {
        nodes[q[--curGrad]]->back();
    }
}

bool DTDAG::updateHasNext() {
    for (int i = 1; i <= tot; i++) {
        if (nodes[i]->updateGradHasNext())
            return 1;
    }
    return 0;
}

void DTDAG::update() {
    for (int i = 1; i <= tot; i++) {
        nodes[i]->update_grad();
    }
}

bool DTDAG::revealHasNext() {
    for (int i = 1; i <= tot; i++) {
        if (nodes[i]->revealHasNext())
            return 1;
    }
    return 0;
}

void DTDAG::reveal() {
    for (int i = 1; i <= tot; i++) {
        nodes[i]->reveal();
    }
}

int DTDAG::getTot() {
    return tot;
}

// dt op
void DTDAG::addOpDT_FeatureNode(int res, int a, int feature_index, Mat* table_data) {
    addedge(a, res);
    setOp(res, new MathOp::FeatureNode(getNode(res)->getForward(), table_data, getNode(a)->getForward(), feature_index));
}

void DTDAG::addOpDT_InternalNode(int res, int feature, double feature_val, int sign, int last_internal, int sibling) {
    addedge(feature, res);
    if (sibling > 0) {
        addedge(sibling, res);
        setOp(res, new MathOp::Internal(getNode(res)->getForward(), getNode(feature)->getForward(), feature_val, sign, getNode(last_internal)->getForward(), sibling, getNode(sibling)->getForward()));
    } else {
        setOp(res, new MathOp::Internal(getNode(res)->getForward(), getNode(feature)->getForward(), feature_val, sign, getNode(last_internal)->getForward(), sibling, nullptr));
    }
}

void DTDAG::addOpDT_Opt_InternalNode(int res, int input, Mat *feature_vector, Mat *threshold, Mat *perm_bit) {
    addedge(input, res);
    setOp(res, new MathOp::OptInternalNode(getNode(res)->getForward(), getNode(input)->getForward(), feature_vector, threshold, perm_bit));
    
}

void DTDAG::addOpDT_PermTree(int res, int raw_tree) {
    // addedge(idx_input, res);
    // addedge(thresholds_input, res);
    // addedge(perm_bit_input, res);
    addedge(raw_tree, res);
    setOp(res, new MathOp::TreePerm(getNode(res), getNode(raw_tree)->get_idxs(), getNode(raw_tree)->get_thresholds()));
}

void DTDAG::addOpDT_LeafNode(int res, int a, int label) {
    addedge(a, res);
    setOp(res, new MathOp::Leaf(getNode(res)->getForward(), getNode(a)->getForward(), label));
}

void DTDAG::addOpDT_ResNode(int res, int a[], int size) {
    for (int i = 0; i < size; ++i) {
        addedge(a[i], res);
    }
    Mat **input = new Mat*[size];
    for (int j = 0; j < size; ++j) {
        input[j] = getNode(a[j])->getForward();
    }
    setOp(res, new MathOp::SigmaOutput(getNode(res)->getForward(), input, size));
}

// map op

void DTDAG::addOpMapGet(int res, int a, int b) {
    addedge(a, res);
    addedge(b, res);
    setOp(res, new MathOp::Map_Get(getNode(res)->getForward(), getNode(a)->getForward(), getNode(b)->getForward()));
}

void DTDAG::addOpMapSet(int res, int a, int b) {
    addedge(a, res);
    addedge(b, res);
    setOp(res, new MathOp::Map_Get(getNode(res)->getForward(), getNode(a)->getForward(), getNode(b)->getForward()));
}