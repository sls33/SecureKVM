#include "Permutation.h"

PermutationObj::PermutationObj(int r, int c)
{
    this->r = r;
    this->c = c;
    this->perm_mat.resize(r * c);
}

void PermutationObj::setPerm(vector<int> perm_mat)
{
    this->perm_mat.assign(perm_mat.begin(), perm_mat.end());
}

void PermutationObj::reversePermMat()
{
    vector<int> newPerm(r * c);
    for (size_t i = 0; i < c; i++)
    {
        for (size_t j = 0; j < r; j++)
        {
            newPerm[i * r + getPerm(j, i)] = j;
        }
    }
    perm_mat.clear();
    perm_mat.assign(newPerm.begin(), newPerm.end());
}

void PermutationObj::PermIndex(Mat *index)
{
    if (index->cols() > c)
    {
        std::cout << "index col more than pemutation num" << std::endl;
        return;
    }
    else
    {
        reversePermMat();
        for (int i = 0; i < index->cols(); i++)
        {
            for (int j = 0; j < index->rows(); j++)
            {
                int temp = index->get(j, i);
                index->setVal(j, i, getPerm(temp, i));
            }
        }
    }
}

void PermutationObj::PermMat(Mat *data)
{
    if (data->rows() != r || data->cols() != c)
    {
        std::cout << "row or col is not match" << std::endl;
        return;
    }

    vector<ll128> new_val(r * c);
    for (size_t i = 0; i < c; i++)
    {
        int begin = i * r;
        for (size_t j = 0; j < r; j++)
        {
            // int new_r = perm_mat[r * i + j];
            new_val[begin + j] = data->get(perm_mat[begin + j], i);
        }
    }
    data->set_val(new_val);
}

int PermutationObj::getPerm(int mr, int mc)
{
    return perm_mat[mc * r + mr];
}

void PermutationObj::printPermMat()
{
    cout << "perm mat" << endl;
    for (size_t i = 0; i < r; i++)
    {
        for (size_t j = 0; j < c; j++)
        {
            cout << getPerm(r, c) << " ";
        }
    }
    cout << endl;
}

int PermutationObj::getSize()
{
    return r * c;
}

int PermutationObj::rows()
{
    return r;
}

int PermutationObj::cols()
{
    return c;
}