#ifndef SMMLF_NGRAMMODEL_H
#define SMMLF_NGRAMMODEL_H

#include <cassert>
#include "../Mat.h"
#include "../Constant.h"
extern int node_type;
class NgramModel {
public:
    static Mat train_data, train_label;
    static Mat test_data, test_label;
    static void load(ifstream& in, Mat& data, Mat& label, int size);
    static void secret_share(Mat& data, Mat& label, int size);
    static void load_ss(ifstream& in, Mat& data, Mat& label, int size);
    static void init();
};


#endif //SMMLF_NGRAMMODEL_H
