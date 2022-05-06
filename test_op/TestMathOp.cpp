#include "TestMathOp.h"
#include "../MathOp.h"

TestMathOp::TestMathOp()
{
    nn = new NN();
}

void TestMathOp::test_secgetper(Mat *kv_data, Mat *index_m, int shuffle_type)
{
    int kv = nn->addnode(kv_data->rows(), kv_data->cols(), NeuronMat::NODE_INPUT);
    int index = nn->addnode(index_m->rows(), index_m->cols(), NeuronMat::NODE_INPUT);
    int output = nn->addnode(index_m->rows(), index_m->cols(), NeuronMat::NODE_OP);

    nn->global_variables_initializer();

    nn->addOpSecGetPer(output, kv, index, shuffle_type);

    nn->toposort();

    nn->reveal_init(kv);
    nn->reveal_init(index);
    nn->reveal_init(output);

    globalRound = 1;

    *nn->getNeuron(kv)->getForward() = *kv_data;
    *nn->getNeuron(index)->getForward() = *index_m;

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
    cout << "kv_data" << endl;
    nn->getNeuron(kv)->getForward()->print();
    cout << "index" << endl;
    nn->getNeuron(index)->getForward()->print();
    cout << "result" << endl;
    nn->getNeuron(output)->getForward()->print();

    cout << "-------END-----\n";
}

void TestMathOp::test_secgetper_graph(int r, int c, int index_num, int shuffle_type)
{
    test_kv = nn->addnode(r, c, NeuronMat::NODE_INPUT);
    test_index = nn->addnode(index_num, c, NeuronMat::NODE_INPUT);
    test_output = nn->addnode(index_num, c, NeuronMat::NODE_OP);

    nn->global_variables_initializer();

    nn->addOpSecGetPer(test_output, test_kv, test_index, shuffle_type);


    nn->toposort();

}

double TestMathOp::forward_secgetper(Mat *kv_data, Mat *index)
{

    *nn->getNeuron(test_kv)->getForward() = *kv_data;
    // get index batch
    *nn->getNeuron(test_index)->getForward() = *index;

    nn->epoch_init();

    system_clock::time_point start = system_clock::now();
    while (nn->forwardHasNext())
    {
        nn->forwardNext();
    }
    system_clock::time_point end = system_clock::now();
    double sum_time = (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-3);
    return sum_time;
}

void TestMathOp::test_genperm_graph(int r, int c)
{
    test_input = nn->addnode(r, c, NeuronMat::NODE_INPUT);
    test_output = nn->addnode(r, c, NeuronMat::NODE_OP);


    nn->global_variables_initializer();

    nn->addOpGenPerm(test_output, test_input);


    cout << "edge added" << endl;

    nn->toposort();

    nn->reveal_init(test_input);
    nn->reveal_init(test_output);
}

void TestMathOp::forward_genperm(Mat *input_data)
{
    globalRound = 0;

    *nn->getNeuron(test_input)->getForward() = *input_data;
    input_data->print();

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
    nn->getNeuron(test_input)->getForward()->print();
    nn->getNeuron(test_output)->getForward()->print();

    cout << "-------END-----\n";
}

void TestMathOp::test_permute_graph(int r, int c)
{
    test_input = nn->addnode(r, c, NeuronMat::NODE_INPUT);
    test_output = nn->addnode(r, c, NeuronMat::NODE_OP);

    nn->global_variables_initializer();

    cout << "edge added" << endl;

    nn->toposort();

    nn->reveal_init(test_input);
    nn->reveal_init(test_output);
}

void TestMathOp::forward_permute(Mat *input_data)
{
    globalRound = 0;

    *nn->getNeuron(test_input)->getForward() = *input_data;
    input_data->print();

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
    nn->getNeuron(test_input)->getForward()->print();
    nn->getNeuron(test_output)->getForward()->print();

    cout << "-------END-----\n";
}

void TestMathOp::test_shuffledata_graph(int r, int c)
{
    test_input = nn->addnode(r, c, NeuronMat::NODE_INPUT);
    test_output = nn->addnode(r, c, NeuronMat::NODE_OP);

    nn->global_variables_initializer();

    nn->addOpPShuffleData(test_output, test_input);

    cout << "edge added" << endl;

    nn->toposort();

    nn->reveal_init(test_input);
    nn->reveal_init(test_output);
}

void TestMathOp::forward_shuffledata(Mat *input_data)
{
    globalRound = 0;

    *nn->getNeuron(test_input)->getForward() = *input_data;
    input_data->print();

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
    nn->getNeuron(test_input)->getForward()->print();
    nn->getNeuron(test_output)->getForward()->print();

    cout << "-------END-----\n";
}

void TestMathOp::test_log_graph(int r, int c)
{
    test_input = nn->addnode(r, c, NeuronMat::NODE_INPUT);
    test_output = nn->addnode(r, c, NeuronMat::NODE_OP);

    nn->global_variables_initializer();

    nn->addOpLog_approximate(test_output, test_input, BIT_P_LEN, DECIMAL_PLACES);

    cout << "edge added" << endl;

    nn->toposort();
}

void TestMathOp::forward_log(Mat *input_data)
{
    globalRound = 0;

    *nn->getNeuron(test_input)->getForward() = *input_data;
    input_data->print();

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
    nn->getNeuron(test_input)->getForward()->print();
    nn->getNeuron(test_output)->getForward()->print();
    cout << "-------END-----\n";
}

double TestMathOp::forward_log_time(Mat *input_data)
{
    *nn->getNeuron(test_input)->getForward() = *input_data;
    system_clock::time_point start = system_clock::now();
    nn->epoch_init();
    while (nn->forwardHasNext())
    {
        nn->forwardNext();
    }
    system_clock::time_point end = system_clock::now();
    double sum_time = (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-3);
    return sum_time;

}

void TestMathOp::test_reshare_graph(int r, int c, int n0, int n1)
{
    test_input = nn->addnode(r, c, NeuronMat::NODE_INPUT);
    test_output = nn->addnode(r, c, NeuronMat::NODE_OP);


    nn->global_variables_initializer();

    nn->addOpReshare(test_output, test_input, n0, n1);

    cout << "edge added" << endl;

    nn->toposort();

    nn->reveal_init(test_input);
    nn->reveal_init(test_output);
}

void TestMathOp::forward_reshare(Mat *input_data)
{
    globalRound = 0;

    *nn->getNeuron(test_input)->getForward() = *input_data;
    cout << "input " << endl;
    nn->getNeuron(test_input)->getForward()->print();

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
    nn->getNeuron(test_input)->getForward()->print();
    nn->getNeuron(test_output)->getForward()->print();
    cout << "-------END-----\n";
}

void TestMathOp::test_graph_before(int r, int c)
{
    test_input = nn->addnode(r, c, NeuronMat::NODE_INPUT);
    test_output = nn->addnode(r, c, NeuronMat::NODE_OP);

    nn->global_variables_initializer();
}

NN *TestMathOp::getNN()
{
    return nn;
}

int TestMathOp::get_test_input()
{
    return test_input;
}

int TestMathOp::get_test_output()
{
    return test_output;
}

void TestMathOp::test_graph_after()
{
    nn->toposort();

    nn->reveal_init(test_input);
    nn->reveal_init(test_output);
}

void TestMathOp::forward(Mat *input_data)
{

    *nn->getNeuron(test_input)->getForward() = *input_data;

    nn->epoch_init();

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
    nn->getNeuron(test_input)->getForward()->print();
    nn->getNeuron(test_output)->getForward()->print();
}

double TestMathOp::forwardt(Mat *input_data)
{
    *nn->getNeuron(test_input)->getForward() = *input_data;

    nn->epoch_init();

    system_clock::time_point start = system_clock::now();
    while (nn->forwardHasNext())
    {
        nn->forwardNext();
    }

    system_clock::time_point end = system_clock::now();
    double spend_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-3; // ms
    return spend_time;
}

void TestMathOp::next_batch(Mat &batch, int start, Mat *A, int mod)
{
    batch = A->row(start % mod, (start + PREDICTION_BATCH_BENCHMARK) % mod);
}