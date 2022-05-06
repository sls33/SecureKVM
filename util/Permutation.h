#ifndef UTIL_PERMUTATION_H
#define UTIL_PERMUTATION_H

#include <vector>
#include "../Mat.h"

class PermutationObj
{
private:
    vector<int> perm_mat; // save by row
    int r, c;

public:
    PermutationObj(int r, int c);
    // shuffle data by per_mat
    void PermMat(Mat *data);
    void PermIndex(Mat *index);
    void setPerm(vector<int> perm_mat);
    int getPerm(int mr,int mc);

    void printPermMat();

    int getSize();
    int rows();
    int cols();
private:
    void reversePermMat();
};

#endif // UTIL_PERMUTATION_H