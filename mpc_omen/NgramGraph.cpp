#include "NgramGraph.h"

NgramGraph::Ngram::Ngram() {
    DBGprint("Ngram constructor\n");
    nn = new NN();
}

NgramGraph::Ngram::Ngram(Mat *data) {
    DBGprint("Ngram constructor\n");
    nn = new NN();
    this->data = data;
    this->size = data->size();
}

NgramGraph::Ngram::Ngram(Mat *data_ip, Mat *data_cp) {
    DBGprint("Ngram constructor\n");
    nn = new NN();
    this->data_ip = data_ip;
    this->data_cp = data_cp;
    this->size = data_ip->cols();
}

void NgramGraph::Ngram::count_graph() {
    int key_num = 1;
    int output_num = 1;
    st_add[0] = nn->addnode(KEY_NUM, KEY_BATCH, NeuronMat::NODE_INPUT);
    for (int i = 0; i < M; ++i) {
        input[i] = nn->addnode(KEY_NUM, KEY_BATCH, NeuronMat::NODE_INPUT);
        st_add[i+1] = nn->addnode(KEY_NUM, KEY_BATCH, NeuronMat::NODE_OP);
    }
    tmp = nn->addnode(KEY_NUM, KEY_BATCH, NeuronMat::NODE_OP);
    total = nn->addnode(KEY_NUM, KEY_BATCH, NeuronMat::NODE_OP);
    output = nn->addnode(KEY_NUM, KEY_BATCH, NeuronMat::NODE_OP);
    ltz = nn->addnode(KEY_NUM, KEY_BATCH, NeuronMat::NODE_OP);
    pow = nn->addnode(KEY_NUM, KEY_BATCH, NeuronMat::NODE_OP);
    exp = nn->addnode(KEY_NUM, KEY_BATCH, NeuronMat::NODE_OP);
    seg_t = nn->addnode(KEY_NUM, 1000, NeuronMat::NODE_INPUT);
    nn->global_variables_initializer();

    for (int i = 0; i < M; ++i) {
        cout << "input: " << input[i] << endl;
        cout << "st_add: " << st_add[i] << endl;
        cout << "--> st_add: " << st_add[i+1] << endl;
        nn->addOpAdd_Mat(st_add[i+1], st_add[i], input[i]);
    }

    cout << "output: " << output << endl;
//    nn->addOpVia(tmp, st_add[M]);
    nn->addOpAdd_Mat(total, st_add[M], tmp);
    // nn->addOpMul_Const_Mat(output, st_add[M], 1);
    nn->addOpVia(output, st_add[M]);
    // nn->addOpLTZ(ltz, output);
    // nn->addOpPow_Log(pow, output, 6);
    // nn->addOpExp_approximate(exp, output, 4);
    nn->toposort();

//    nn->reveal_init(out_sig);
    /// if the output is secret-shared as well
    nn->reveal_init(st_add[M]);
//    nn->reveal_init(total);
    nn->reveal_init(output);
    // nn->reveal_init(ltz);
    // nn->reveal_init(pow);
    // nn->reveal_init(exp);
}

void NgramGraph::Ngram::forward_count(ll *total_count, Mat *count_data, Mat * seg_total) {
    cout << "Forward Count ...\n";
    Mat ip_batch(KEY_NUM, KEY_BATCH), cp_batch(KEY_NUM, KEY_BATCH), ep_batch(KEY_NUM, KEY_BATCH), ln_batch(KEY_NUM, KEY_BATCH);
    clock_train = new Constant::Clock(CLOCK_TRAIN);
    globalRound = 0;
//    test();
    for (int i = 0; i < count_data->size()/KEY_BATCH && i < TRAIN_ITE; i++) {
        globalRound++;
        DBGprint("Epoch %d\n", i);
        next_batch(ip_batch, KEY_BATCH, i * KEY_BATCH, &data[node_type], size);
        *nn->getNeuron(input[node_type])->getForward() = ip_batch;
//        ip_batch.print();
//        next_batch(cp_batch, i * KEY_BATCH, &cp_data[node_type], N);
//        next_batch(ep_batch, i * KEY_BATCH, &ep_data[node_type], N);
//        next_batch(ln_batch, i * KEY_BATCH, &ln_data[node_type], N);

        for (int j = 0; j < M; ++j) {
            if (j != node_type) {
                next_batch(ip_batch, KEY_BATCH, i * KEY_BATCH, &data[j], size);
//                ip_batch.print();
                MathOp::broadcast_share(&ip_batch, j);
            }
        }

        for (int l = 0; l < M; ++l) {
            if (l != node_type) {
                MathOp::receive_share(nn->getNeuron(input[l])->getForward(), l);
//                nn->getNeuron(input[l])->getForward()->print();
            }
        }
        {
            nn->epoch_init();
            while (nn->forwardHasNext()) {
                nn->forwardNext();
            }
            *nn->getNeuron(tmp)->getForward() = *nn->getNeuron(total)->getForward();
//            nn->getNeuron(output)->getForward()->print();
            nn->getNeuron(output)->getForward()->append(i * KEY_BATCH, i * KEY_BATCH + KEY_BATCH, count_data);

            nn->reveal_init();
            while (nn->revealHasNext()) {
                nn->reveal();
            }
            DBGprint("-------------\n");
            nn->getNeuron(st_add[M])->getForward()->print();
            nn->getNeuron(output)->getForward()->print();
            cout << nn->getNeuron(output)->getForward()->count_sum() << endl;
//            nn->getNeuron(total)->getForward()->print();
            // nn->getNeuron(ltz)->getForward()->print();
            // nn->getNeuron(pow)->getForward()->print();
            // nn->getNeuron(exp)->getForward()->print();
        }
        print_perd(i+1);
        if ((i+1)%PRINT_PRE_ITE == 0) {
//            test();
            print_perd(i+1);
        }
    }
    for (int k = 0; k < count_data->size(); k+=ALPHABET_SIZE) {
        ll cur = 0;
        for (int i = 0; i < ALPHABET_SIZE; ++i) {
            cur += count_data->getVal(k+i);
        }
        for (int i = 0; i < ALPHABET_SIZE; ++i) {
            seg_total->setVal(k+i, cur);
        }
    }
//    seg_total->print();
    cout << "-------END-----\n";
    *nn->getNeuron(seg_t)->getForward() = *seg_total;
    nn->reveal_init(seg_t);

    // plaintext total count sum
    nn->reveal_init(total);
    while (nn->revealHasNext()) {
        nn->reveal();
    }
    *seg_total = *nn->getNeuron(seg_t)->getForward();
    *total_count = nn->getNeuron(total)->getForward()->count_sum();
}

void NgramGraph::Ngram::smooth_level_graph(ll total_count, int level_adjust, bool isCp) {
    nn = new NN();
    count = nn->addnode(KEY_NUM, KEY_BATCH, NeuronMat::NODE_INPUT);
    delta = nn->addnode(KEY_NUM, KEY_BATCH, NeuronMat::NODE_INPUT);
    seg_t = nn->addnode(KEY_NUM, KEY_BATCH, NeuronMat::NODE_INPUT);
    ep = nn->addnode(KEY_NUM, KEY_BATCH, NeuronMat::NODE_INPUT);
    st_add_delta = nn->addnode(KEY_NUM, KEY_BATCH, NeuronMat::NODE_OP);
    st_div = nn->addnode(KEY_NUM, KEY_BATCH, NeuronMat::NODE_OP);
    st_mul = nn->addnode(KEY_NUM, KEY_BATCH, NeuronMat::NODE_OP);
    st_add_ep = nn->addnode(KEY_NUM, KEY_BATCH, NeuronMat::NODE_OP);
    output = nn->addnode(KEY_NUM, KEY_BATCH, NeuronMat::NODE_OP);
    nn->global_variables_initializer();

//    ll inverse = Constant::Util::inverse(total_count, MOD);
//    ll inverse = Constant::Util::get_residual(IE/total_count);
//    DBGprint("Inverse: %lld\n", inverse);

    nn->addOpAdd_Mat(st_add_delta, count, delta);
    if (isCp) {
        nn->addOpDiv_Seg_Const_Mat(st_div, st_add_delta, seg_t);
    } else {
        nn->addOpDiv_Const_Optimized_Mat(st_div, st_add_delta, total_count);
    }
    nn->addOpMul_Const_Mat(st_mul, st_div, level_adjust);
    nn->addOpAdd_Mat(st_add_ep, st_mul, ep);
    nn->addOpSmoothLevel(output, st_add_ep);
    nn->toposort();

    /// if the output is secret-shared as well
    nn->reveal_init(st_add_delta);
    nn->reveal_init(count);
    nn->reveal_init(st_div);
    nn->reveal_init(st_mul);
    nn->reveal_init(st_add_ep);
    nn->reveal_init(output);
}

void NgramGraph::Ngram::forward_smooth(Mat count_data, Mat seg_total, int delta_data, int ep_data, Mat *level_data) {
    cout << "Forward Smooth ...\n";
    Mat ip_batch(KEY_NUM, KEY_BATCH), cp_batch(KEY_NUM, KEY_BATCH), ep_batch(KEY_NUM, KEY_BATCH), ln_batch(KEY_NUM, KEY_BATCH);
    clock_train = new Constant::Clock(CLOCK_TRAIN);
    globalRound = 0;
    for (int j = 0; j < KEY_BATCH; ++j) {
        nn->getNeuron(delta)->getForward()->setVal(j, delta_data);
        nn->getNeuron(ep)->getForward()->setVal(j, ep_data);
    }
    for (int i = 0; i < level_data->size()/KEY_BATCH && i < TRAIN_ITE; i++) {
        globalRound++;
        DBGprint("Epoch %d\n", i);
        next_batch(ip_batch, KEY_BATCH, i * KEY_BATCH, &count_data, size);
        next_batch(cp_batch, KEY_BATCH, i * KEY_BATCH, &seg_total, size);
        *nn->getNeuron(count)->getForward() = ip_batch;
        *nn->getNeuron(seg_t)->getForward() = cp_batch;
//        nn->getNeuron(seg_t)->getForward()->print();
//        ip_batch.print();
//        next_batch(cp_batch, i * KEY_BATCH, &cp_data[node_type], N);
//        next_batch(ep_batch, i * KEY_BATCH, &ep_data[node_type], N);
//        next_batch(ln_batch, i * KEY_BATCH, &ln_data[node_type], N);
        {
            nn->epoch_init();
            while (nn->forwardHasNext()) {
                nn->forwardNext();
            }
//            *nn->getNeuron(tmp)->getForward() = *nn->getNeuron(total)->getForward();

//            nn->getNeuron(st_add_delta)->getForward()->print();
//            nn->getNeuron(st_mul)->getForward()->print();
//            nn->getNeuron(output)->getForward()->print();

            nn->reveal_init();
            while (nn->revealHasNext()) {
                nn->reveal();
            }
            DBGprint("-------------\n");
            nn->getNeuron(output)->getForward()->append(i * KEY_BATCH, i * KEY_BATCH + KEY_BATCH, level_data);
           nn->getNeuron(count)->getForward()->print();
            // nn->getNeuron(st_add_delta)->getForward()->print();
            // nn->getNeuron(st_div)->getForward()->print();
            // nn->getNeuron(st_mul)->getForward()->print();
            nn->getNeuron(st_add_ep)->getForward()->print();
            nn->getNeuron(output)->getForward()->print();
        }
        print_perd(i+1);
        if ((i+1)%PRINT_PRE_ITE == 0) {
//            test();
            print_perd(i+1);
        }
    }
}

void NgramGraph::Ngram::log_prob_graph() {

    inference_ip_and_cp_batch = nn->addnode(MARKOV_INFER_LOG_BATCH, 1, NeuronMat::NODE_INPUT);
    inference_ip_and_cp_batch_log = nn->addnode(MARKOV_INFER_LOG_BATCH, 1, NeuronMat::NODE_OP);
    // output = nn->addnode(KEY_DOMAIN_SIZE, 1, NeuronMat::NODE_OP);
    
    nn->global_variables_initializer();

    nn->addOpLog_approximate(inference_ip_and_cp_batch_log, inference_ip_and_cp_batch, BIT_P_LEN, DECIMAL_PLACES);

    nn->toposort();

    /// if the output is secret-shared as well
    // nn->reveal_init(output);
    // nn->reveal_init(inference_ip_and_cp_batch_log);
}

void NgramGraph::Ngram::forward_log_prob(Mat *ip_and_cp_data, Mat *log_prob_res) {
    cout << "Forward Log IP and CP data ...\n";
    Mat input_data_batch(MARKOV_INFER_LOG_BATCH, 1);
    globalRound = 0;

    int total_size = ip_and_cp_data->size();
    cout << "DIM: " << ip_and_cp_data->rows() << endl;
    size_t ite = ceil(ip_and_cp_data->rows()*1.0 / MARKOV_INFER_LOG_BATCH);
    clock_train = new Constant::Clock(CLOCK_TRAIN);
    for (int i = 0; i < ite && i < 1000; i++) {
        globalRound++;
        // DBGprint("Epoch %d\n", i);

        /**
         * Load Data
         * */
        next_batch_by_row(input_data_batch, i*MARKOV_INFER_LOG_BATCH, MARKOV_INFER_LOG_BATCH, ip_and_cp_data, total_size); // Batch * 10
        *nn->getNeuron(inference_ip_and_cp_batch)->getForward() = input_data_batch;


        /**
         * Inference Logic
         * */
        {
            nn->epoch_init();
            while (nn->forwardHasNext()) {
                nn->forwardNext();
            }
            // DBGprint("Forward done -------------\n");

            /**
             * Write the log value to res
             * */
            int write_len = MARKOV_INFER_LOG_BATCH;
            if ((i+1)*MARKOV_INFER_LOG_BATCH > ip_and_cp_data->rows()) write_len = ip_and_cp_data->rows() - i*MARKOV_INFER_LOG_BATCH;

            nn->getNeuron(inference_ip_and_cp_batch_log)->getForward()->append_with_size(i * MARKOV_INFER_LOG_BATCH, write_len, log_prob_res);

            // nn->reveal_init();
            // while (nn->revealHasNext()) {
            //     nn->reveal();
            // }
            // DBGprint("Reveal output -------------\n");
            // nn->getNeuron(st_add[M])->getForward()->print();
            // nn->getNeuron(output)->getForward()->print();
            // nn->getNeuron(inference_ip_and_cp_batch)->getForward()->print();
            // nn->getNeuron(inference_ip_and_cp_batch_log)->getForward()->print();
        }
        print_perd(i+1);
    }
    cout << "END----\n";
}

void NgramGraph::Ngram::inference_graph() {
    nn = new NN();
    // usingBasic = false;
    if (!USING_BASIC && !USING_FOLD) {
        inference_model = nn->addnode(KEY_DOMAIN_SIZE, INFER_BATCH*MAX_LEN, NeuronMat::NODE_INPUT);
    } else {
        inference_model = nn->addnode(KEY_DOMAIN_SIZE, 1, NeuronMat::NODE_INPUT);
    }

    if (usingBasic) {
        inference_input = nn->addnode(INFER_BATCH, KEY_DOMAIN_SIZE, NeuronMat::NODE_INPUT);
        inference_output = nn->addnode(INFER_BATCH, 1, NeuronMat::NODE_OP);
    } 
    else if (USING_FOLD) {
        inference_input = nn->addnode(INFER_BATCH, MAX_LEN, NeuronMat::NODE_INPUT);
        inference_output = nn->addnode(INFER_BATCH * MAX_LEN, 1, NeuronMat::NODE_OP);
    }
    else {
        inference_input = nn->addnode(INFER_BATCH*MAX_LEN, 1, NeuronMat::NODE_INPUT);
        inference_output = nn->addnode(INFER_BATCH * MAX_LEN, 1, NeuronMat::NODE_OP);
    }
    
    nn->global_variables_initializer();

    // for (int i = 0; i < MAX_LEN-NGRAM+1; ++i) {
    //     cout << "inference_nodes: " << inference_nodes[i] << endl;
    //     cout << "--> inference_medium: " << inference_medium[i] << endl;
    //     cout << "--> inference_nodes: " << inference_nodes[i+1] << endl;
    //     nn->addOpAdd_Mat(st_add[i+1], st_add[i], input[i]);
    // }

    // nn->addOpMul_Mat(output_ip, inference_ip, inference_input);
    // nn->addOpMul_Mat(output_cp, inference_cp, inference_input);
    // nn->addOpAdd_Mat(output, output_ip, output_cp);

    // nn->addOpLog_approximate(inference_ip_and_cp_log, inference_ip_and_cp, BIT_P_LEN, DECIMAL_PLACES);
    // basic protocol
    if (usingBasic) nn->addOpMul_Mat(inference_output, inference_input, inference_model);
    else if (USING_FOLD) nn->addOpSecGetFold(inference_output, inference_model, inference_input); // fold protocol
    else nn->addOpSecGetPer(inference_output, inference_model, inference_input, 1); // perm protocol
    
    
    // Online Perm Mat
    // Mat perm_mat(KEY_DOMAIN_SIZE, KEY_DOMAIN_SIZE), perm_mat_plain(KEY_DOMAIN_SIZE, KEY_DOMAIN_SIZE);
    // nn->addOpSecGetOnline(inference_output, inference_model, inference_input, &perm_mat, &perm_mat_plain);
    nn->toposort();

    /// if the output is secret-shared as well
    // nn->reveal_init(inference_input);
    //nn->reveal_init(inference_output);
    // nn->reveal_init(inference_model);
}

void NgramGraph::Ngram::calculate_graph() {

    inference_input = nn->addnode(KEY_DOMAIN_SIZE+ ALPHABET_SIZE * ALPHABET_SIZE, INFER_BATCH, NeuronMat::NODE_INPUT);
    inference_ip = nn->addnode(1, KEY_DOMAIN_SIZE, NeuronMat::NODE_INPUT);
    inference_cp = nn->addnode(1, KEY_DOMAIN_SIZE, NeuronMat::NODE_INPUT);
    inference_ip_and_cp = nn->addnode(1, KEY_DOMAIN_SIZE + ALPHABET_SIZE * ALPHABET_SIZE, NeuronMat::NODE_INPUT);
    // inference_nodes[0] = nn->addnode(KEY_DOMAIN_SIZE, INFER_BATCH, NeuronMat::NODE_OP);
    // for (int i = 1; i < MAX_LEN-NGRAM+1; ++i) {
    //     inference_nodes[i] = nn->addnode(KEY_NUM, INFER_BATCH, NeuronMat::NODE_OP);
    //     inference_medium[i] = nn->addnode(KEY_NUM, INFER_BATCH, NeuronMat::NODE_OP);
    // }
    output_ip = nn->addnode(1, INFER_BATCH, NeuronMat::NODE_OP);
    output_cp = nn->addnode(1, INFER_BATCH, NeuronMat::NODE_OP);
    output = nn->addnode(1, INFER_BATCH, NeuronMat::NODE_OP);
    // ltz = nn->addnode(KEY_NUM, KEY_BATCH, NeuronMat::NODE_OP);
    // pow = nn->addnode(KEY_NUM, KEY_BATCH, NeuronMat::NODE_OP);
    // exp = nn->addnode(KEY_NUM, KEY_BATCH, NeuronMat::NODE_OP);
    
    nn->global_variables_initializer();

    // for (int i = 0; i < MAX_LEN-NGRAM+1; ++i) {
    //     cout << "inference_nodes: " << inference_nodes[i] << endl;
    //     cout << "--> inference_medium: " << inference_medium[i] << endl;
    //     cout << "--> inference_nodes: " << inference_nodes[i+1] << endl;
    //     nn->addOpAdd_Mat(st_add[i+1], st_add[i], input[i]);
    // }

    cout << "output: " << output << endl;
    // nn->addOpMul_Mat(output_ip, inference_ip, inference_input);
    // nn->addOpMul_Mat(output_cp, inference_cp, inference_input);
    // nn->addOpAdd_Mat(output, output_ip, output_cp);

    nn->addOpMul_Mat(output, inference_ip_and_cp, inference_input);

    // nn->addOpLTZ(ltz, output);
    // nn->addOpPow_Log(pow, output, 6);
    // nn->addOpExp_approximate(exp, output, 4);
    nn->toposort();

    /// if the output is secret-shared as well
    // nn->reveal_init(output);
    // nn->reveal_init(inference_input);
    // nn->reveal_init(ltz);
    // nn->reveal_init(pow);
    // nn->reveal_init(exp);
}

void NgramGraph::Ngram::forward_inference(Mat *result, Mat* input_data, Mat *ip_and_cp_data) {
    cout << "Forward Calculate ...\n";
    // Mat ip_batch(KEY_NUM, KEY_BATCH), cp_batch(KEY_NUM, KEY_BATCH);
    // Mat input_data_batch(INFER_BATCH, KEY_DOMAIN_SIZE);
    Mat *input_data_batch;
    if (usingBasic)
        input_data_batch = new Mat(INFER_BATCH, KEY_DOMAIN_SIZE);
    else if (USING_FOLD) input_data_batch = new Mat(INFER_BATCH, MAX_LEN);
    else input_data_batch = new Mat(INFER_BATCH*MAX_LEN, 1);

    clock_train = new Constant::Clock(4);
    globalRound = 0;
    int total_size = input_data->rows();
    *nn->getNeuron(inference_model)->getForward() = *ip_and_cp_data;
    int i;
    for (i = 0; i < input_data->rows() / INFER_BATCH && i < 100; i++) {
        globalRound++;
        // DBGprint("Epoch %d\n", i);

        Mat dummy(1,1);
        // Basic
        if (usingBasic) {
            // next_batch_by_row(*input_data_batch, i * INFER_BATCH, INFER_BATCH, input_data, total_size);
            
            if (node_type == 0) {
                /** Load the secret shared input **/
                next_batch_by_row(*input_data_batch, i * INFER_BATCH, INFER_BATCH, input_data, total_size);
                Mat* sec_data = IOManager::secret_share_mat_data(*input_data_batch, input_data_batch->size(), "omen_eval");

                *nn->getNeuron(inference_input)->getForward() = sec_data[0];

                for (int j = 0; j < M; ++j) {
                    if (j != node_type) {
                        MathOp::broadcast_share(&sec_data[j], j);
                        MathOp::receive_share(&dummy, j);
                    }
                }
            } else {
                MathOp::broadcast_share(&dummy, 0);
                MathOp::receive_share(nn->getNeuron(inference_input)->getForward(), 0);
            }
        } else if (USING_FOLD){
            // Fold 
            next_batch_by_row(*input_data_batch, i*INFER_BATCH, INFER_BATCH, input_data, input_data->rows()); // Batch * 10          
            *nn->getNeuron(inference_input)->getForward() = *input_data_batch;
        } else {    // Permute
            Mat tmp(INFER_BATCH, MAX_LEN);
            next_batch_by_row(tmp, i*INFER_BATCH, INFER_BATCH, input_data, input_data->rows()); // Batch * 10
            for (int ii = 0; ii < INFER_BATCH; ii++) {
                for (int cur_l = 0; cur_l < MAX_LEN; cur_l++) {
                    input_data_batch->setVal(ii*MAX_LEN + cur_l, 0, tmp.get(ii, cur_l));
                }
            }
            *nn->getNeuron(inference_input)->getForward() = *input_data_batch;
            // input_data_batch->print();
        }
        
        // input_data_batch->print();
        {
            nn->epoch_init();
            while (nn->forwardHasNext()) {
                nn->forwardNext();
            }
            // DBGprint("Forward done -------------\n");
            // nn->reveal_init();
            // while (nn->revealHasNext()) {
            //     nn->reveal();
            // }
            // DBGprint("Reveal output -------------\n");
            // nn->getNeuron(st_add[M])->getForward()->print();
            // nn->getNeuron(inference_input)->getForward()->print();
            // nn->getNeuron(inference_output)->getForward()->print();
            if (usingBasic) {
                nn->getNeuron(inference_output)->getForward()->append(i * INFER_BATCH, i * INFER_BATCH + INFER_BATCH, result);
            } else {
                for(int idx = 0; idx < INFER_BATCH; idx++) {
                    ll128 res = 0;
                    for (int xx = 0; xx < MAX_LEN; xx++) {
                        res += nn->getNeuron(inference_output)->getForward()->get(idx*MAX_LEN+xx, 0);
                    }
                    result->setVal(i*INFER_BATCH + idx, 0, res);
                }
            }
            
            // nn->getNeuron(output)->getForward()->append(i * INFER_BATCH, i * INFER_BATCH + INFER_BATCH, result);

            // nn->getNeuron(inference_model)->getForward()->print();
        }
        print_perd(i+1);
//         if ((i+1)%PRINT_PRE_ITE == 0) {
// //            test();
//             print_perd(i+1);
//         }
    }
    print_perd(i);
    cout << "-------END-----\n";
}

void NgramGraph::Ngram::feed(NN* nn, Mat &x_batch, Mat &y_batch, int input, int output) {
    *nn->getNeuron(input)->getForward() = x_batch;
    *nn->getNeuron(output)->getForward() = y_batch;
}

void NgramGraph::Ngram::next_batch(Mat &batch, int batch_size, int start, Mat *A, int mod) {
    A->col(start%mod, start%mod + batch_size, batch);
}

void NgramGraph::Ngram::next_batch_by_row(Mat &batch, int start, int batch_size, Mat *A, int mod) {
    batch = A->row(start%mod, (start+batch_size) % mod);
}

void NgramGraph::Ngram::print_perd(int round) {
    ll tot_send = 0, tot_recv = 0;
    for (int i = 0; i < M; i++) {
        if (node_type != i) {
            tot_send += socket_io[node_type][i]->send_num;
            tot_recv += socket_io[node_type][i]->recv_num;
        }
    }
    DBGprint("round: %d tot_time: %.3f ",
             round, clock_train->get());
    DBGprint("tot_send: %lld tot_recv: %lld\n", tot_send, tot_recv);
}

Mat NgramGraph::Ngram::vector_to_mat(int *data, int r, int c) {
    Mat ret(1, r*c);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            ret(j, i) = data[j+i*r];
        }
    }
    return ret;
}