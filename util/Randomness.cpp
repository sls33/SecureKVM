#include "Randomness.h"

using namespace std;

RandomnessObj::RandomnessObj(string filename)
{
    ifstream f(filename);
    string str{istreambuf_iterator<char>(f), istreambuf_iterator<char>()};
    f.close();

    int len = str.length();
    char common_aes_key[len + 1];
    memset(common_aes_key, '\0', len + 1);
    strcpy(common_aes_key, str.c_str());
    this->aes_key = stoull(common_aes_key, 0, 16);

    cout << "Key: " << common_aes_key << ", int: " << this->aes_key << endl;
    g.seed(this->aes_key);
}

std::vector<int> RandomnessObj::getRandomPermutation(int size)
{
    std::vector<int> a(size, 0);
    for (size_t i = 0; i < size; i++)
    {
        a[i] = i + 1;
    }
    std::shuffle(a.begin(), a.end(), g);
    return a;
}

std::vector<ll128> RandomnessObj::getRandomPermutation(std::vector<ll128> data)
{
    std::shuffle(data.begin(), data.end(), g);
    return data;
}

void RandomnessObj::getRandomPermutation(Mat *data, Mat *res)
{
    vector<ll128> value = data->to_vector();
    std::shuffle(value.begin(), value.end(), g);
    res->set_val(value);
}

void RandomnessObj::getRandomPermutationInPlace(std::vector<ll128> &data)
{
    std::shuffle(data.begin(), data.end(), g);
    // cout << g << endl;
}

// shuffle data
void RandomnessObj::getRandomPermutationInPlace(Mat *data)
{
    vector<ll128> value = data->to_vector();
    getRandomPermutationInPlace(value);
    *data = value;
}

std::mt19937 RandomnessObj::getg()
{
    return g;
}

std::pair<std::vector<int>, std::vector<int>> RandomnessObj::getTreePermutation(int depth)
{
    vector<int> tree_permutation;
    vector<int> perm_bits;

    std::pair<std::vector<int>, std::vector<int>> res;
    tree_permutation.push_back(1); // root node
    for (int dp = 2; dp <= depth; dp++)
    {
        int perm_bit = g() % 2;
        perm_bits.push_back(perm_bit);
        int parent_node_num_this_depth = 1 << (dp - 2);
        for (int i = 0; i < parent_node_num_this_depth; i++)
        {
            tree_permutation.push_back(tree_permutation[i + (1 << (dp - 2)) - 1] * 2 + perm_bit);     // left child node
            tree_permutation.push_back(tree_permutation[i + (1 << (dp - 2)) - 1] * 2 + 1 - perm_bit); // right child node
        }
    }
    if (LOG_MESSAGES)
    {
        for (int i : perm_bits)
            cout << i << endl;
        for (int i : tree_permutation)
            cout << i << endl;
    }

    res.first = perm_bits;
    res.second = tree_permutation;
    return res;
}

void RandomnessObj::initPermMat(PermutationObj *permObj)
{
    vector<int> perm_v;
    // perm_v.resize(permObj->getSize());
    int c = permObj->cols();
    int r = permObj->rows();
    for (size_t i = 0; i < c; i++)
    {
        vector<int> temp;
        for (size_t j = 0; j < r; j++)
        {
            temp.push_back(j);
        }
        std::shuffle(temp.begin(), temp.end(), g);
        perm_v.insert(perm_v.end(), temp.begin(), temp.end());
    }
    permObj->setPerm(perm_v); 
}