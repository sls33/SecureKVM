#pragma once
#ifndef UTIL_RANDOMNESS_H
#define UTIL_RANDOMNESS_H

#include <cstring>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <vector>
#include "../Constant.h"
#include "../Mat.h"
#include "Permutation.h"

using namespace std;

class RandomnessObj
{
private:
    unsigned long long aes_key;
    std::mt19937 g;

public:
    RandomnessObj(string filename);

    std::vector<int> getRandomPermutation(int size);
    std::vector<ll128> getRandomPermutation(std::vector<ll128> data);
    void getRandomPermutation(Mat *data, Mat *res);
    void getRandomPermutationInPlace(std::vector<ll128> &data);
    void getRandomPermutationInPlace(Mat *data);
    std::mt19937 getg();
    std::pair<std::vector<int>, std::vector<int>> getTreePermutation(int depth);
    void initPermMat(PermutationObj *permObj);
    
};

#endif // UTIL_RANDOMNESS_H