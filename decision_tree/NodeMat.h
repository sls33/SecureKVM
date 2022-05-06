#ifndef SMMLF_NODEMAT_H
#define SMMLF_NODEMAT_H

#include "../NeuronMat.h"

class NodeMat: public NeuronMat{
    Mat *table;
    vector<string> attributeList;
    Mat *bitmap;
    Mat *cur_data;
    Mat *idxs, *thresholds, *perm_bits;
    public:
        NodeMat();
        NodeMat(int r, int c, int k);
        void initTable();
        Mat* getTable();
        void setTable(Mat *table);
        vector<string> getAttrList();
        ll getTotalEnt();
        Mat* getBitMap();
        void fillForward(ll a);
        Mat* get_idxs();
        Mat* get_thresholds();
        Mat* get_perm_bits();
        void set_idxs(Mat *idxs);
        void set_thresholds(Mat *thresholds);
        void set_perm_bits(Mat *perm_bits);
};


#endif //SMMLF_NODEMAT_H
