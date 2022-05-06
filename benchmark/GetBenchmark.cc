#include "GetBenchmark.h"

GetBenchmarkGraph::GetBenchmark::GetBenchmark()
{
    nn = new NN();
    dim = FEATURE_DIM_BENCHMARK;
    cout << "init\n";
}

GetBenchmarkGraph::GetBenchmark::GetBenchmark(Mat *kv_data)
{
    this->kv_data = kv_data;
    this->size = kv_data->rows();
    dim = 1;
    nn = new NN();
    cout << "init\n";
}

void GetBenchmarkGraph::GetBenchmark::secget_graph(Mat *perm_mat, Mat *perm_mat_plain)
{
    kv_data_input = nn->addnode(FEATURE_DIM_BENCHMARK, 1, NeuronMat::NODE_INPUT);
    secget_input = nn->addnode(PREDICTION_BATCH_BENCHMARK, 1, NeuronMat::NODE_INPUT);
    secget_output = nn->addnode(PREDICTION_BATCH_BENCHMARK, 1, NeuronMat::NODE_OP);
    nn->global_variables_initializer();
    // nn->addOpSecGetBasic(secget_output, kv_data_input, secget_input);
    nn->addOpSecGetFold(secget_output, kv_data_input, secget_input);
    // nn->addOpSecGetPer(secget_output, kv_data_input, secget_input, 1);
    // nn->addOpSecGetOnline(secget_output, kv_data_input, secget_input, perm_mat, perm_mat_plain);

    cout << "edge added" << endl;

    nn->toposort();

    // nn->reveal_init(secget_input);
    // nn->reveal_init(secget_output);
}

void GetBenchmarkGraph::GetBenchmark::forward_secget(Mat *queries)
{
    Mat query_batch(PREDICTION_BATCH_BENCHMARK, 1);
    clock_train = new Constant::Clock(CLOCK_TRAIN);
    globalRound = 0;
    //    test();
    *nn->getNeuron(kv_data_input)->getForward() = *kv_data;
    assert(kv_data->rows() == FEATURE_DIM_BENCHMARK);
    int total_size = queries->size();
    for (int i = 0; i < queries->size() / PREDICTION_BATCH_BENCHMARK && i < 100; i++)
    {
        globalRound++;
        DBGprint("Epoch %d\n", i);
        next_batch(query_batch, i * PREDICTION_BATCH_BENCHMARK, queries, total_size);
        *nn->getNeuron(secget_input)->getForward() = query_batch;
        // query_batch.print();

        {
            nn->epoch_init();
            cout << "start forwarding……" << endl;
            while (nn->forwardHasNext())
            {
                nn->forwardNext();
            }

            nn->reveal_init();
            while (nn->revealHasNext())
            {
                nn->reveal();
            }
            DBGprint("-------------\n");
            // nn->getNeuron(secget_input)->getForward()->print();
            // nn->getNeuron(secget_output)->getForward()->print();
            // cout << nn->getNeuron(secget_output)->getForward()->size() << endl;
        }
        print_perd(i + 1);
    }

    cout << "-------END-----\n";
}

void GetBenchmarkGraph::GetBenchmark::test_graph()
{
    data = nn->addnode(FEATURE_DIM_BENCHMARK, 1, NeuronMat::NODE_INPUT);
    reshare = nn->addnode(PREDICTION_BATCH_BENCHMARK, 1, NeuronMat::NODE_OP);
    nn->global_variables_initializer();
    nn->addOpReshare(reshare, data, 0, 1);
    // nn->addOpSecGetOnline(secget_output, kv_data_input, secget_input, perm_mat, perm_mat_plain);

    cout << "edge added" << endl;

    nn->toposort();

    nn->reveal_init(data);
    nn->reveal_init(reshare);
}

void GetBenchmarkGraph::GetBenchmark::forward_test(Mat *input)
{
    clock_train = new Constant::Clock(CLOCK_TRAIN);
    globalRound = 0;
    for (int i = 0; i < input->size() / 3 && i < 1; i++)
    {
        globalRound++;
        DBGprint("Epoch %d\n", i);
        *nn->getNeuron(data)->getForward() = *input;
        {
            nn->epoch_init();
            cout << "start forwarding……" << endl;
            while (nn->forwardHasNext())
            {
                nn->forwardNext();
            }

            nn->reveal_init();
            while (nn->revealHasNext())
            {
                nn->reveal();
            }
            DBGprint("-------------\n");
            nn->getNeuron(data)->getForward()->print();
            nn->getNeuron(reshare)->getForward()->print();
        }
        print_perd(i + 1);
    }
}

void GetBenchmarkGraph::GetBenchmark::test_log_graph()
{
    test_log_input = nn->addnode(PREDICTION_BATCH_BENCHMARK, dim, NeuronMat::NODE_INPUT);
    test_log = nn->addnode(PREDICTION_BATCH_BENCHMARK, dim, NeuronMat::NODE_OP);
    nn->global_variables_initializer();

    nn->addOpLog_approximate(test_log, test_log_input, BIT_P_LEN, DECIMAL_PLACES);

    cout << "edge added" << endl;

    nn->toposort();

    // nn->reveal_init(test_log_input);
    // nn->reveal_init(test_log);
}

void GetBenchmarkGraph::GetBenchmark::forward_log(Mat *input_data)
{
    Mat batch(PREDICTION_BATCH_BENCHMARK, FEATURE_DIM_BENCHMARK);
    globalRound = 0;
    int nums = input_data->rows();
    cout << "nums " << nums <<  endl;
    cout << "batch " << PREDICTION_BATCH_BENCHMARK << endl;
    cout << "m" << M << endl;

    //    test();
    clock_train = new Constant::Clock(CLOCK_TRAIN);
    double sum_time = 0.0;

    for (int i = 0; i < nums / PREDICTION_BATCH_BENCHMARK; i++)
    {
        globalRound++;
        // DBGprint("Epoch %d\n", i);
        // cout << input_data->size() << endl;
        next_batch(batch, i * PREDICTION_BATCH_BENCHMARK, input_data, nums);
        *nn->getNeuron(test_log_input)->getForward() = batch;
        // batch.print();

        system_clock::time_point start = system_clock::now();
        nn->epoch_init();
        // cout << "start forwarding……" << endl;
        while (nn->forwardHasNext())
        {
            nn->forwardNext();
        }
        system_clock::time_point end = system_clock::now();
        sum_time += (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-3);

        // nn->reveal_init();
        // while (nn->revealHasNext())
        // {
        //     nn->reveal();
        // }
        // DBGprint("-------------\n");
        // nn->getNeuron(test_log_input)->getForward()->print();
        // nn->getNeuron(test_log)->getForward()->print();
        // end = system_clock::now();
        // cout << "log time spent: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()*1e-3 << endl;

        // DBGprint("-------------\n");
        // nn->getNeuron(test_log_input)->getForward()->print();
        // nn->getNeuron(test_log)->getForward()->print();

        // print_perd(i + 1);
    }
    cout << PREDICTION_BATCH_BENCHMARK << " " << OFFLINE_PHASE_ON << endl;
    cout << "avg time of single op: " << sum_time/nums << endl;

    cout << "-------END-----\n";
}

void GetBenchmarkGraph::GetBenchmark::next_batch(Mat &batch, int start, Mat *A, int mod)
{
    batch = A->row(start % mod, (start + PREDICTION_BATCH_BENCHMARK) % mod);
}

void GetBenchmarkGraph::GetBenchmark::print_perd(int round)
{
    ll tot_send = 0, tot_recv = 0;
    for (int i = 0; i < M; i++)
    {
        if (node_type != i)
        {
            tot_send += socket_io[node_type][i]->send_num;
            tot_recv += socket_io[node_type][i]->recv_num;
        }
    }
    // cout << clock_train->get() << endl;
    DBGprint("round: %d tot_time: %.3f \n", round, clock_train->get());
    DBGprint("tot_send: %lld tot_recv: %lld\n", tot_send, tot_recv);
}