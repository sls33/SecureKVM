#include "DT_main.h"
int node_type;
SocketManager::SMMLF tel;
int globalRound;

int main(int argc, char *argv[])
{
    srand(time(NULL)); // random seed
    if (argc < 2)
    {
        DBGprint("Please enter node type:\n");
        scanf("%d", &node_type);
    }
    else
    {
        node_type = argv[1][0] - '0';
    }
    DBGprint("node type: %d\n", node_type);

    Player::init();

    int *arr = new int[2 * KEY_BATCH];
    for (int i = 0; i < 2 * KEY_BATCH; i++)
        arr[i] = rand() % (10 - 0 + 1) - 5;

    // load data
    InputReader trainInputReader("decision_tree/dataset/test.data");
    Table table = trainInputReader.getTable();

    // decide the number of attribute list size and label class number, todo: use PSI to improve this job
    long class_num = table.attrValueList.front().size();
    long attrSize = table.attrValueList.size() - 1;
    vector<long> attributeListNum(attrSize);
    for (int i = 0; i < attrSize - 1; ++i)
    {
        attributeListNum.push_back(table.attrValueList[i].size());
        // cout << attributeListNum.back() << ",";
    }
    cout << endl;
    cout << "attrSize: " << attrSize << ", class_num: " << class_num << endl;
    if (LOG)
    {
        for (int i = 0; i < table.attrName.size(); i++)
        {
            cout << table.attrName[i] << "\n";
        }
        for (int i = 0; i < table.data.size(); ++i)
        {
            for (int j = 0; j < table.data[i].size(); ++j)
            {
                cout << table.data[i][j] << ", ";
            }
            // cout << endl;
        }
    }

    cout << "data num: " << table.data.size() << endl;

    /**
     * encode table car evaluation
     * This is only used in car evaluation set, since we need to map text to int.
     * **/
    vector<map<string, ll>> encode_table = {
        {{"vhigh", 1}, {"high", 2}, {"med", 3}, {"low", 2}},
        {{"vhigh", 1}, {"high", 2}, {"med", 3}, {"low", 2}},
        {{"2", 1}, {"3", 2}, {"4", 3}, {"5more", 2}},
        {{"2", 1}, {"4", 2}, {"more", 3}},
        {{"small", 1}, {"med", 2}, {"big", 3}},
        {{"low", 1}, {"med", 2}, {"high", 3}},
    };

    /**
     * raw
     * **/
    // for (int i = 0; i < attrSize; ++i) {
    //     map<string, ll> tmp;
    //     ll cur = 1;
    //     for (int j = 0; j < table.attrValueList[i].size(); ++j) {
    //         tmp[table.attrValueList[i][j]] = cur++;
    //     }
    //     encode_table.push_back(tmp);
    // }

    // transform data car
    // Mat input_data(table.data.size(), attrSize);
    // for (int i = 0; i < input_data.rows(); ++i) {
    //     vector<string> data = table.data[i];
    //     for (int j = 0; j < input_data.cols(); ++j) {
    //         input_data(i, j) = encode_table[j][data[j]];
    //     }
    // }

    // transform data parkinsons
    // Mat input_data(table.data.size(), attrSize-1);
    // for (int i = 0; i < input_data.rows(); ++i) {
    //     vector<string> data = table.data[i];
    //     for (int j = 0; j < 16; ++j) {
    //         input_data(i, j) = (ll)(stod(data[j+1]) * IE);
    //     }
    //     for (int j = 18; j < data.size(); ++j) {
    //         input_data(i, j-2) = (ll)(stod(data[j]) * IE);
    //     }
    // }

    // transform data breast cancer
    Mat input_data(table.data.size(), attrSize - 1);
    for (int i = 0; i < input_data.rows(); ++i)
    {
        vector<string> data = table.data[i];
        for (int j = 0; j < attrSize - 1; ++j)
        {
            input_data(i, j) = stoi(data[j + 1]);
        }
    }

    // transform data spam collection
    // Mat input_data(table.data.size(), attrSize);
    // for (int i = 0; i < input_data.rows(); ++i) {
    //     vector<string> data = table.data[i];
    //     for (int j = 0; j < attrSize; ++j) {
    //         input_data(i, j) = (ll)(stod(data[j+1]) * IE);
    //     }
    // }

    input_data.residual();

    if (LOG)
        input_data.print();

    Mat *data = IOManager::secret_share_mat_data(input_data, input_data.size(), "");

    tel.init();
    DT_graph::DT *dt = new DT_graph::DT(data, encode_table);

    // dt->predict_graph();
    // dt->forward_predict();
    // return 0;

    Mat *idxs_perm = new Mat[NODE_NUM];
    Mat *thresholds_perm = new Mat[NODE_NUM];
    for (int i = 0; i < NODE_NUM; i++)
    {
        if (PERMUTE_USING_BASIC)
        { // Basic impl
            idxs_perm[i].init(1, FEATURE_VECTOR_SIZE);
            idxs_perm[i].setVal(i % FEATURE_DIM, IE);
        }
        else
        { // for fold-impl
            int fold_num = ceil(sqrt(FEATURE_DIM));
            idxs_perm[i].init(1, FEATURE_VECTOR_SIZE);
            idxs_perm[i].setVal(i % fold_num, IE);
            idxs_perm[i].setVal(i % fold_num + fold_num, IE);
        }

        thresholds_perm[i].init(1, 1);
        thresholds_perm[i].setVal(0, i + 1);
    }

    Mat *perm_bits = new Mat[DEPTH - 1];
    for (int i = 0; i < DEPTH - 1; i++)
    {
        perm_bits[i].init(1, 1);
        perm_bits[i].setVal(0, i % 2);
    }

    Mat input_data_test(TOTAL_SAMPLE_SIZE * PREDICTION_BATCH, FEATURE_DIM);
    for (int i = 0; i < input_data_test.rows(); ++i)
    {
        for (int j = 0; j < input_data_test.cols(); ++j)
        {
            input_data_test(i, j) = (ll)(i + j);
        }
    }

    NodeMat res(1, 1, 1);
    dt->permute_tree_graph(DEPTH);
    cout << "---- PERM ----" << endl;
    dt->forward_permute_tree(&res, idxs_perm, thresholds_perm);

    Mat *mat1 = res.get_idxs();
    Mat *mat2 = res.get_thresholds();
    Mat *mat3 = res.get_perm_bits();

    dt->optimized_predict_graph(DEPTH);
    dt->forward_optimized_graph(&input_data_test, mat1, mat2, mat3);
}