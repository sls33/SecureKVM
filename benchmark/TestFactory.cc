#include "TestFactory.h"
int node_type;
SocketManager::SMMLF tel;
int globalRound;

int main(int argc, char *argv[]) {
    srand(time(NULL)); // random seed
    if (argc < 2) {
        DBGprint("Please enter node type:\n");
        scanf("%d", &node_type);
    }
    else {
        node_type = argv[1][0] - '0';
    }
    DBGprint("node type: %d\n", node_type);

    Player::init();

    clock_t start, end;
    
    
    // init data
    Mat raw_data(FEATURE_DIM_BENCHMARK, 1);
    for (int i = 0; i < raw_data.rows(); ++i) {
        for (int j = 0; j < raw_data.cols(); ++j) {
            raw_data(i, j) = (i*raw_data.cols() + j) % FEATURE_DIM_BENCHMARK;
        }
    }
    cout << "RAW DATA \n";
    // raw_data.print();
    Mat *share_raw_data = IOManager::secret_share_mat_data(raw_data, raw_data.size(), "benchmark_secget_kv");
    Mat *enc_kv_data = new Mat(FEATURE_DIM_BENCHMARK, 1);
    Mat* kv_data = IOManager::secret_share_kv_data(raw_data, raw_data.size(), "kv", false);

    // TEST: reshare logic
    // Mat reveal_mat(FEATURE_DIM_BENCHMARK, 1);

    // reveal_mat = (share_raw_data[0] * player[0].lagrange + share_raw_data[1] * player[1].lagrange + share_raw_data[2] * player[2].lagrange) / IE;
    // reveal_mat.print();

    // reveal_mat = (share_raw_data[2] * player[2].reshare_key_next + share_raw_data[0] * player[0].reshare_key_prev) / IE;
    // reveal_mat.print();

    // reveal_mat = (share_raw_data[0] * player[0].reshare_key_next + share_raw_data[1] * player[1].reshare_key_prev) / IE;
    // reveal_mat.print();

    // reveal_mat = (share_raw_data[1] * player[1].reshare_key_next + share_raw_data[2] * player[2].reshare_key_prev) / IE;
    // reveal_mat.print();

    // player[node_type].rand_next->getTreePermutation(3);
    


    int total_sample_size = 1000;
    Mat queries_raw(total_sample_size, 1);
    for (int i = 0; i < queries_raw.rows(); ++i) {
        for (int j = 0; j < queries_raw.cols(); ++j) {
            queries_raw(i, j) = (i*queries_raw.cols() + 1 + j) % FEATURE_DIM_BENCHMARK;
        }
    }
    // queries_raw.print();


    Mat enc_perm_mat(FEATURE_DIM_BENCHMARK, FEATURE_DIM_BENCHMARK);
    Mat perm_mat(FEATURE_DIM_BENCHMARK, FEATURE_DIM_BENCHMARK), perm_mat_plain(FEATURE_DIM_BENCHMARK, FEATURE_DIM_BENCHMARK);

    tel.init();
    /**
     *  Tests for logarithm computation.
     **/
    {  
        // cout << "=======================================================" << endl;
        cout << "===========\tTests for logarithm\t===========" << endl;
        // cout << "=======================================================" << endl;
        int total_samples = 50;
        Mat test_log_data(total_samples*PREDICTION_BATCH_BENCHMARK, FEATURE_DIM_BENCHMARK);    // set FEATURE_DIM_BENCHMARK  to 1
        for (int i = 0; i < test_log_data.rows(); ++i) {
            for (int j = 0; j < test_log_data.cols(); ++j) {
                test_log_data(i, j) =  Constant::Util::randomlong();
                //test_log_data(i,j) = Constant::Util::randomlong();
            }
        }
        // test_log_data.print();

        GetBenchmarkGraph::GetBenchmark *kv = new GetBenchmarkGraph::GetBenchmark();
        kv->test_log_graph();
        // cout<<"log_graph newed"<<endl;
        kv->forward_log(&test_log_data);
        // cout << "=======================================================" << endl;
        cout << "===========\t End Tests for logarithm\t===========" << endl;
        //cout << "=======================================================" << endl;
        return 0;
    }
    
    {
        // if (node_type == 0) {
        //     enc_kv_data = &share_raw_data[0];
        //     for (int j = 1; j < M; ++j) {
        //         // share feature
        //         MathOp::broadcast_share(&share_raw_data[j], j);
        //     }
        // } else {
        //     MathOp::receive_share(enc_kv_data, 0);
        // }
        // cout << "init kv done \n";
        // Constant::Clock *clock_perm = new Constant::Clock(10);;
        // if (node_type == 0) {
        //     // SecGet Online Random Permutation Matrix
        //     Mat permutation_matrix = Mat::random_permutation_matrix(FEATURE_DIM_BENCHMARK, FEATURE_DIM_BENCHMARK);
        //     // permutation_matrix.print();
        //     Mat *share_perm_mat = IOManager::secret_share_mat_data(permutation_matrix, permutation_matrix.size(), "benchmark_secget_perm_mat");
        //     enc_perm_mat = share_perm_mat[0];
        //     perm_mat_plain = permutation_matrix;
        //     for (int j = 1; j < M; ++j) {
        //         // share feature
        //         MathOp::broadcast_share(&share_perm_mat[j], j);
        //     }
        // } else {
        //     MathOp::receive_share(&enc_perm_mat, 0);
        // }
        // DBGprint("send perm time: %.3f ", clock_perm->get());
        
        // perm_mat = enc_perm_mat;
        cout << "init perm done \n";
        GetBenchmarkGraph::GetBenchmark *model = new GetBenchmarkGraph::GetBenchmark(kv_data);
        model->secget_graph(&perm_mat, &perm_mat_plain);
        model->forward_secget(&queries_raw);
        // model->test_graph();
        // model->forward_test(enc_kv_data);
        return 0;
    }
    
}