#include "NodeMat.h"
NodeMat::NodeMat() {}

NodeMat::NodeMat(int r, int c, int k): NeuronMat(r, c, k) {
    this->table = nullptr;
    
    this->idxs = new Mat[NODE_NUM];
    this->thresholds = new Mat[NODE_NUM];
    this->perm_bits = new Mat[DEPTH-1];

    for (int i = 0; i < NODE_NUM; i++) {
        idxs[i].init(1, FEATURE_VECTOR_SIZE);
        // idxs[i].init(1, FEATURE_DIM);    // basic 
        thresholds[i].init(1, 1);
    }
    for (int i = 0; i < DEPTH-1; i++) {
        perm_bits[i].init(1, 1);
    }
}

void NodeMat::initTable() {
    table = new Mat(this->rows(), this->cols());
}

Mat * NodeMat::getTable() {
    return table;
}

vector<string> NodeMat::getAttrList() {
    return attributeList;
}

ll NodeMat::getTotalEnt() {

}

void NodeMat::fillForward(ll a) {
    for (int i = 0; i < getForward()->size(); ++i) {
        getForward()->setVal(i, a);
    }
}

Mat * NodeMat::getBitMap() {

}

Mat* NodeMat::get_idxs() {
    return idxs;
}

void NodeMat::set_idxs(Mat *input) {
    for (int i = 0; i < NODE_NUM; i++) {
        idxs[i] = input[i];
    }
}

Mat* NodeMat::get_thresholds() {
    return thresholds;
}

void NodeMat::set_thresholds(Mat *input) {
    for (int i = 0; i < NODE_NUM; i++) {
        thresholds[i] = input[i];
    }
}

Mat* NodeMat::get_perm_bits() {
    return perm_bits;
}

void NodeMat::set_perm_bits(Mat *input) {
    for (int i = 0; i < DEPTH-1; i++) {
        perm_bits[i] = input[i];
    }
}