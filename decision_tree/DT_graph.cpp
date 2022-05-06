#include "DT_graph.h"
Mat * input_table_data;
DT_graph::DT::DT(Mat *data, vector<map<string, ll>> encode_table) {
    DBGprint("DT constructor\n");
    dag = new DTDAG();
    this->encode_table = encode_table;
    this->data = data;
    this->size = data->size();
}

void DT_graph::DT::train_graph() {
    for (int i = 0; i < M; ++i) {
        input[i] = dag->addnode(PREDICTION_BATCH, FEATURE_DIM, NeuronMat::NODE_INPUT);
    }
    input_table = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_INPUT);

    total_entropy = dag->addnode(1, 1, NeuronMat::NODE_OP);

    // calculate entropy gain for each feature
    for (int j = 0; j < encode_table.size(); ++j) {
        entropy_gain_feature[j] = dag->addnode(FEATURE_DIM, 1, NeuronMat::NODE_OP);
    }

    // compare to find the best split
    split_finding = dag->addnode(1, 1, NeuronMat::NODE_OP);

    dag->global_variables_initializer();

    // DT train graph construction
    dag->reveal_init(input_table);

}

void DT_graph::DT::forward_train() {
    clock_train = new Constant::Clock(CLOCK_TRAIN);
    globalRound = 0;

    // load data
    for (int j = 0; j < M; ++j) {
        if (j != node_type) {
            MathOp::broadcast_share(&data[j], j);
        }
    }
    int total_sample = 0;
    for (int l = 0; l < M; ++l) {
        if (l != node_type) {
            MathOp::receive_share(dag->getNode(input[l])->getForward(), l);
            total_sample += dag->getNode(input[l])->getForward()->rows();
        }
    }
    *dag->getNode(input[node_type])->getForward() = data[node_type];
    total_sample += dag->getNode(input[node_type])->getForward()->rows();
    cout << total_sample << endl;
    this->size = total_sample;

    int cur = 0;
    Mat *total_data = new Mat(total_sample, FEATURE_DIM);
    for (int l = 0; l < M; ++l) {
        for (int i = 0; i < dag->getNode(input[l])->getForward()->rows(); i++)
            for (int j = 0; j < dag->getNode(input[l])->getForward()->cols(); j++) {
//                dag->getNode(input_table)->getForward()->operator()(i + cur, j) = dag->getNode(input[l])->getForward()->operator()(i, j);
                total_data->operator()(i + cur, j) = dag->getNode(input[l])->getForward()->operator()(i, j);
            }
        cur += dag->getNode(input[l])->getForward()->rows();
    }

    /** log data info **/
//    total_data->print();
    cout << "total size: " << total_data->rows() << endl;

    for (int i = 0; i < M; ++i) {
//        cout << i << endl;
//        dag->getNode(input[i])->getForward()->print();
    }
    dag->getNode(input_table)->getTable()->init(PREDICTION_BATCH, FEATURE_DIM);
    input_table_data = dag->getNode(input_table)->getTable();

    for (int i = 0; i < TRAIN_ITE; i++) {
        globalRound++;
        DBGprint("Epoch %d\n", i);
        {
            dag->epoch_init();
            while (dag->forwardHasNext()) {
                dag->forwardNext();
            }
            *dag->getNode(tmp)->getForward() = *dag->getNode(total)->getForward();

            dag->reveal_init();
            while (dag->revealHasNext()) {
                dag->reveal();
            }
            DBGprint("-------------\n");
            dag->getNode(st_add[M])->getForward()->print();
            dag->getNode(output)->getForward()->print();
        }
        if ((i+1)%PRINT_PRE_ITE == 0) {
            print_perd(i+1);
        }
    }
    cout << "-------END-----\n";
    dag->reveal_init(seg_t);

    // plaintext total count sum
    dag->reveal_init(total);
    while (dag->revealHasNext()) {
        dag->reveal();
    }

}

void DT_graph::DT::predict_graph() {
    for (int i = 0; i < M; ++i) {
        input[i] = dag->addnode(PREDICTION_BATCH, FEATURE_DIM, NeuronMat::NODE_INPUT);
    }
    input_table = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_INPUT);

    feature1 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    feature2 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    feature3 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    feature4 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    feature5 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    feature6 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    feature7 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    feature8 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    feature9 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    feature10 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    feature11 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);

    internal1 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    internal2 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    internal3 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    internal4 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    internal5 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    internal6 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    internal7 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    internal8 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    internal9 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    internal10 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    internal11 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    internal12 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    internal13 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    internal14 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    internal15 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    internal16 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    internal17 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    internal18 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    internal19 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    internal20 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    internal21 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    internal22 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);

    leaf1 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    leaf2 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    leaf3 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    leaf4 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    leaf5 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    leaf6 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    leaf7 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    leaf8 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    leaf9 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    leaf10 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    leaf11 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);
    leaf12 = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);

    this->leafs.push_back(leaf1);
    this->leafs.push_back(leaf2);
    this->leafs.push_back(leaf3);
    this->leafs.push_back(leaf4);
    this->leafs.push_back(leaf5);
    this->leafs.push_back(leaf6);
    this->leafs.push_back(leaf7);
    this->leafs.push_back(leaf8);
    this->leafs.push_back(leaf9);
    this->leafs.push_back(leaf10);
    this->leafs.push_back(leaf11);
    this->leafs.push_back(leaf12);

    res = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP);

    dag->global_variables_initializer();
    input_data = dag->getNode(input_table)->getTable();

    /**
     * dt car evaluation
     * 3
     * 4
     * 6
     * **/
    {
        // dag->addOpDT_FeatureNode(feature1, input_table, 5, input_data);  
        // dag->addOpDT_InternalNode(internal1, feature1, 1.5, 1, input_table, 0);

        // dag->addOpDT_LeafNode(leaf1, internal1, 1);
        // dag->addOpDT_InternalNode(internal2, feature1, 1.5, 2, input_table, internal1);
        // dag->addOpDT_FeatureNode(feature2, internal2, 3, input_data);

        // dag->addOpDT_InternalNode(internal3, feature2, 1.5, 1, internal2, 0);
        // dag->addOpDT_LeafNode(leaf2, internal3, 1);
        // dag->addOpDT_InternalNode(internal4, feature2, 1.5, 2, internal2, internal3);
        // dag->addOpDT_FeatureNode(feature3, internal4, 0, input_data);

        // dag->addOpDT_InternalNode(internal5, feature3, 2.5, 1, internal4, 0);
        // dag->addOpDT_InternalNode(internal6, feature3, 2.5, 2, internal4, internal5);
        
        // dag->addOpDT_LeafNode(leaf3, internal5, 1);
        // dag->addOpDT_LeafNode(leaf4, internal6, 2);

        // depth 4
        // dag->addOpDT_FeatureNode(feature4, internal5, 1, input_data);
        // // dag->addOpDT_FeatureNode(feature5, internal6, 1, input_data);

        // dag->addOpDT_InternalNode(internal7, feature4, 1, 0, internal5, 0);
        // dag->addOpDT_LeafNode(leaf3, internal7, 1);

        // dag->addOpDT_InternalNode(internal8, feature4, 2, 0, internal5, 0);
        // dag->addOpDT_LeafNode(leaf4, internal8, 2);

        // dag->addOpDT_InternalNode(internal9, feature4, 2, 0, internal6, 0);
        // dag->addOpDT_LeafNode(leaf5, internal9, 2);

        // dag->addOpDT_InternalNode(internal10, feature4, 2.5, 2, internal6, 0);
        // dag->addOpDT_LeafNode(leaf6, internal10, 3);

        // depth 6
        // dag->addOpDT_FeatureNode(feature1, input_table, 3, input_data);  // persons
        // dag->addOpDT_InternalNode(internal1, feature1, 1.5, 1, input_table, 0);

        // dag->addOpDT_LeafNode(leaf1, internal1, 1);     // class 0
        // dag->addOpDT_InternalNode(internal2, feature1, 1.5, 2, input_table, internal1);
        // dag->addOpDT_FeatureNode(feature2, internal2, 5, input_data);   // safety

        // dag->addOpDT_InternalNode(internal3, feature2, 1.5, 1, internal2, 0);
        // dag->addOpDT_LeafNode(leaf2, internal3, 1);     // class 0
        // dag->addOpDT_InternalNode(internal4, feature2, 1.5, 2, internal2, internal3);
        // dag->addOpDT_FeatureNode(feature3, internal4, 0, input_data);

        // dag->addOpDT_InternalNode(internal5, feature3, 2.5, 1, internal4, 0);
        // dag->addOpDT_FeatureNode(feature4, internal5, 1, input_data);       // maint
        // dag->addOpDT_InternalNode(internal6, feature3, 2.5, 2, internal4, internal5);
        // // dag->addOpDT_FeatureNode(feature5, internal6, 0, input_data);

        // dag->addOpDT_InternalNode(internal7, feature4, 1.5, 1, internal5, 0);
        // dag->addOpDT_LeafNode(leaf3, internal7, 1);     // class 0
        // dag->addOpDT_InternalNode(internal8, feature4, 1.5, 2, internal5, internal7);
        // dag->addOpDT_FeatureNode(feature5, internal8, 4, input_data);       // lug_boot

        // dag->addOpDT_InternalNode(internal9, feature5, 1.5, 1, internal8, 0);
        // dag->addOpDT_FeatureNode(feature6, internal9, 5, input_data);       // safety
        // dag->addOpDT_InternalNode(internal10, feature5, 1.5, 2, internal8, internal9);
        // dag->addOpDT_FeatureNode(feature7, internal10, 1, input_data);      // maint

        // dag->addOpDT_InternalNode(internal11, feature6, 2.5, 1, internal9, 0);
        // dag->addOpDT_LeafNode(leaf4, internal11, 1);     // class 0
        // dag->addOpDT_InternalNode(internal12, feature6, 2.5, 2, internal9, internal11);
        // dag->addOpDT_LeafNode(leaf5, internal12, 2);     // class 1

        // dag->addOpDT_InternalNode(internal13, feature7, 2.5, 1, internal10, 0);
        // dag->addOpDT_LeafNode(leaf6, internal13, 1);     // class 0
        // dag->addOpDT_InternalNode(internal14, feature7, 2.5, 2, internal10, internal13);
        // dag->addOpDT_LeafNode(leaf7, internal14, 2);     // class 1

        // // right
        // dag->addOpDT_InternalNode(internal15, feature5, 2.5, 1, internal6, 0);
        // dag->addOpDT_LeafNode(leaf8, internal15, 2);     // class 1
        // dag->addOpDT_InternalNode(internal16, feature5, 2.5, 2, internal6, internal15);
        // dag->addOpDT_FeatureNode(feature9, internal16, 5, input_data);      // safety


        // dag->addOpDT_InternalNode(internal17, feature9, 2.5, 1, internal16, 0);
        // dag->addOpDT_FeatureNode(feature10, internal17, 4, input_data);     // lug_boot
        // dag->addOpDT_InternalNode(internal18, feature9, 2.5, 2, internal16, internal17);
        // dag->addOpDT_FeatureNode(feature11, internal18, 4, input_data);     // lug_boot

        // dag->addOpDT_InternalNode(internal19, feature10, 1.5, 1, internal17, 0);
        // dag->addOpDT_LeafNode(leaf9, internal19, 2);     // class 1
        // dag->addOpDT_InternalNode(internal20, feature10, 1.5, 2, internal17, internal19);
        // dag->addOpDT_LeafNode(leaf10, internal20, 3);     // class 2

        // dag->addOpDT_InternalNode(internal21, feature11, 1.5, 1, internal18, 0);
        // dag->addOpDT_LeafNode(leaf11, internal21, 3);     // class 2
        // dag->addOpDT_InternalNode(internal22, feature11, 1.5, 2, internal18, internal21);
        // dag->addOpDT_LeafNode(leaf12, internal22, 4);     // class 3
    }

    /**
     * parkinson
     * 3
     * 4
     * **/ 
    {
        // depth 3
        // dag->addOpDT_FeatureNode(feature1, input_table, 21, input_data);    // PPE
        // dag->addOpDT_InternalNode(internal1, feature1, 0.135, 1, input_table, 0);

        // dag->addOpDT_InternalNode(internal2, feature1, 0.135, 2, input_table, internal1);

        // dag->addOpDT_FeatureNode(feature2, internal1, 5, input_data);   // MDVP:RAP

        // dag->addOpDT_InternalNode(internal3, feature2, 0.001, 1, internal1, 0);
        // dag->addOpDT_LeafNode(leaf1, internal3, 2); // class 1
        // dag->addOpDT_InternalNode(internal4, feature2, 0.001, 2, internal1, internal3);
        // dag->addOpDT_LeafNode(leaf2, internal4, 1); // class 0

        // dag->addOpDT_FeatureNode(feature3, internal2, 19, input_data);  // spread2

        // dag->addOpDT_InternalNode(internal5, feature3, 0.22, 1, internal2, 0);
        // dag->addOpDT_FeatureNode(feature4, internal5, 0, input_data);   // MDVP:Fo(HZ)
        // dag->addOpDT_InternalNode(internal6, feature3, 0.22, 2, internal2, internal3);
        // dag->addOpDT_LeafNode(leaf3, internal6, 2); // class 1

        // dag->addOpDT_InternalNode(internal7, feature4, 117.548, 1, internal5, 0);
        // dag->addOpDT_InternalNode(internal8, feature4, 117.548, 2, internal5, internal7);
        
        // dag->addOpDT_LeafNode(leaf4, internal7, 1);
        // dag->addOpDT_LeafNode(leaf5, internal8, 2);

        // depth 4
        // dag->addOpDT_FeatureNode(feature1, input_table, 21, input_data);    // PPE
        // dag->addOpDT_InternalNode(internal1, feature1, 0.134, 1, input_table, 0);

        // dag->addOpDT_InternalNode(internal2, feature1, 0.134, 2, input_table, internal1);
        // dag->addOpDT_FeatureNode(feature2, internal1, 1, input_data);   // MDVP:Fhi(Hz)

        // dag->addOpDT_InternalNode(internal3, feature2, 201.716, 1, internal1, 0);
        // dag->addOpDT_FeatureNode(feature3, internal3, 16, input_data);  // RPDE
        // dag->addOpDT_InternalNode(internal4, feature2, 201.716, 2, internal1, internal3);
        // dag->addOpDT_LeafNode(leaf1, internal4, 1); // class 0


        // dag->addOpDT_InternalNode(internal5, feature3, 0.459, 1, internal3, 0);
        // dag->addOpDT_LeafNode(leaf2, internal5, 2); // class 1
        // dag->addOpDT_InternalNode(internal6, feature3, 0.459, 2, internal3, internal5);
        // dag->addOpDT_LeafNode(leaf3, internal6, 1); // class 0

        // dag->addOpDT_FeatureNode(feature4, internal2, 12, input_data);    // MDVP:APQ

        // dag->addOpDT_InternalNode(internal7, feature4, 0.02, 1, internal2, 0);
        // dag->addOpDT_FeatureNode(feature5, internal7, 19, input_data);    // spread2
        // dag->addOpDT_InternalNode(internal8, feature4, 0.02, 2, internal2, internal7);
        // dag->addOpDT_LeafNode(leaf4, internal8, 1); // class 1
        
        // dag->addOpDT_InternalNode(internal9, feature5, 0.194, 1, internal7, 0);
        // dag->addOpDT_FeatureNode(feature6, internal9, 17, input_data);  // DFA
        // dag->addOpDT_InternalNode(internal10, feature5, 0.194, 2, internal7, internal9);
        // dag->addOpDT_LeafNode(leaf5, internal10, 2); // class 1

        // dag->addOpDT_InternalNode(internal11, feature6, 0.671, 1, internal9, 0);
        // dag->addOpDT_LeafNode(leaf6, internal11, 1); // class 0
        // dag->addOpDT_InternalNode(internal12, feature6, 0.671, 2, internal9, internal11);
        // dag->addOpDT_LeafNode(leaf7, internal12, 2); // class 1
    }
    
    /**
     * breast cancer
     * 3
     * 4
     * **/
    {
        // depth 3
        dag->addOpDT_FeatureNode(feature1, input_table, 1, input_data);    // Uniformity of Cell Size

        dag->addOpDT_InternalNode(internal1, feature1, 2.5, 1, input_table, 0);
        dag->addOpDT_FeatureNode(feature2, internal1, 5, input_data);   // Bare Nuclei
        dag->addOpDT_InternalNode(internal2, feature1, 2.5, 2, input_table, internal1);
        // dag->addOpDT_FeatureNode(feature2, internal1, 5, input_data);   // Bare Nuclei
        
        dag->addOpDT_InternalNode(internal3, feature2, 3.5, 1, internal1, 0);
        dag->addOpDT_LeafNode(leaf1, internal3, 2); // class 2
        dag->addOpDT_InternalNode(internal4, feature2, 3.5, 2, internal1, internal3);
        dag->addOpDT_FeatureNode(feature3, internal4, 2, input_data); // Uniformity of Cell Shape

        dag->addOpDT_InternalNode(internal5, feature3, 2.5, 1, internal4, 0);
        dag->addOpDT_LeafNode(leaf2, internal5, 2); // class 2
        dag->addOpDT_InternalNode(internal6, feature3, 2.5, 2, internal4, internal5);
        dag->addOpDT_LeafNode(leaf3, internal6, 3); // class 3

        dag->addOpDT_InternalNode(internal7, feature2, 1.5, 1, internal2, 0);
        dag->addOpDT_FeatureNode(feature4, internal7, 0, input_data); // Uniformity of Cell Size
        dag->addOpDT_InternalNode(internal8, feature2, 1.5, 2, internal2, internal7);
        dag->addOpDT_LeafNode(leaf4, internal8, 3); // class 3

        dag->addOpDT_InternalNode(internal9, feature4, 3.5, 1, internal7, 0);
        dag->addOpDT_LeafNode(leaf5, internal9, 2); // class 2
        dag->addOpDT_InternalNode(internal10, feature4, 3.5, 2, internal7, internal9);
        dag->addOpDT_LeafNode(leaf6, internal10, 3); // class 3

        // depth 4
        // dag->addOpDT_FeatureNode(feature1, input_table, 1, input_data);    // Uniformity of Cell Size

        // dag->addOpDT_InternalNode(internal1, feature1, 2.5, 1, input_table, 0);
        // dag->addOpDT_FeatureNode(feature2, internal1, 5, input_data);   // Bare Nuclei
        // dag->addOpDT_InternalNode(internal2, feature1, 2.5, 2, input_table, internal1);
        // // dag->addOpDT_FeatureNode(feature2, internal1, 5, input_data);   // Bare Nuclei
        
        // dag->addOpDT_InternalNode(internal3, feature2, 6.1, 1, internal1, 0);
        // dag->addOpDT_FeatureNode(feature3, internal3, 2, input_data); // Clump Thickness
        // dag->addOpDT_InternalNode(internal4, feature2, 6.1, 2, internal1, internal3);
        // dag->addOpDT_FeatureNode(feature4, internal4, 2, input_data); // Uniformity of Cell Shape

        // dag->addOpDT_InternalNode(internal5, feature3, 6.5, 1, internal3, 0);
        // dag->addOpDT_LeafNode(leaf1, internal5, 2); // class 2
        // dag->addOpDT_InternalNode(internal6, feature3, 6.5, 2, internal3, internal5);
        // dag->addOpDT_FeatureNode(feature5, internal6, 2, input_data); // Uniformity of Cell Shape

        // dag->addOpDT_InternalNode(internal7, feature5, 3.1, 1, internal6, 0);
        // dag->addOpDT_LeafNode(leaf2, internal7, 2); // class 2
        // dag->addOpDT_InternalNode(internal8, feature5, 3.1, 2, internal6, internal7);
        // dag->addOpDT_LeafNode(leaf3, internal8, 3); // class 3

        // dag->addOpDT_InternalNode(internal9, feature4, 1.5, 1, internal4, 0);
        // dag->addOpDT_LeafNode(leaf4, internal9, 2); // class 2
        // dag->addOpDT_InternalNode(internal10, feature4, 1.5, 2, internal4, internal7);
        // dag->addOpDT_LeafNode(leaf5, internal10, 3); // class 3

        // dag->addOpDT_InternalNode(internal11, feature2, 2.5, 1, internal2, 0);
        // dag->addOpDT_FeatureNode(feature6, internal11, 2, input_data); // Uniformity of Cell Shape
        // dag->addOpDT_InternalNode(internal12, feature2, 2.5, 2, internal2, internal11);
        // dag->addOpDT_LeafNode(leaf6, internal12, 3); // class 3

        // dag->addOpDT_InternalNode(internal13, feature6, 3.5, 1, internal11, 0);
        // dag->addOpDT_FeatureNode(feature7, internal13, 3, input_data); // Marginal Adhesion
        // dag->addOpDT_InternalNode(internal14, feature6, 3.5, 2, internal11, internal13);
        // dag->addOpDT_LeafNode(leaf7, internal14, 3); // class 3

        // dag->addOpDT_InternalNode(internal15, feature7, 6.1, 1, internal13, 0);
        // dag->addOpDT_LeafNode(leaf8, internal15, 2); // class 2
        // dag->addOpDT_InternalNode(internal16, feature7, 6.1, 2, internal13, internal15);
        // dag->addOpDT_LeafNode(leaf9, internal16, 3); // class 3

    }

    /**
     * spam collection
     * 3
     * 4
     * **/
    {
        /**
         * depth 3, dimension = 10000
         * */
        // dag->addOpDT_FeatureNode(feature1, input_table, 6502, input_data);    // 6502
        // dag->addOpDT_InternalNode(internal1, feature1, 0.054, 1, input_table, 0);
        // dag->addOpDT_InternalNode(internal2, feature1, 0.054, 2, input_table, internal1);

        // dag->addOpDT_FeatureNode(feature2, internal1, 4248, input_data);   // 4248
        // dag->addOpDT_InternalNode(internal3, feature2, 0.065, 1, internal1, 0);
        // dag->addOpDT_InternalNode(internal4, feature2, 0.065, 2, internal1, internal3);

        // dag->addOpDT_FeatureNode(feature3, internal2, 7001, input_data);   // 7001
        // dag->addOpDT_InternalNode(internal5, feature2, 0.239, 1, internal2, 0);
        // dag->addOpDT_InternalNode(internal6, feature2, 0.239, 2, internal2, internal5);
        // dag->addOpDT_LeafNode(leaf1, internal6, 0); // class 0


        // dag->addOpDT_FeatureNode(feature4, internal3, 1820, input_data);  // 1820
        // dag->addOpDT_InternalNode(internal7, feature4, 0.068, 1, internal3, 0);
        // dag->addOpDT_InternalNode(internal8, feature4, 0.068, 2, internal3, internal7);
        // dag->addOpDT_LeafNode(leaf2, internal7, 0); // class 0
        // dag->addOpDT_LeafNode(leaf3, internal8, 1); // class 1

        // dag->addOpDT_FeatureNode(feature5, internal4, 3981, input_data);  // 3981
        // dag->addOpDT_InternalNode(internal9, feature5, 0.073, 1, internal4, 0);
        // dag->addOpDT_InternalNode(internal10, feature5, 0.073, 2, internal4, internal9);
        // dag->addOpDT_LeafNode(leaf4, internal9, 1); // class 1
        // dag->addOpDT_LeafNode(leaf5, internal10, 0); // class 0

        // dag->addOpDT_FeatureNode(feature6, internal5, 6502, input_data);  // 6502
        // dag->addOpDT_InternalNode(internal11, feature6, 0.389, 1, internal5, 0);
        // dag->addOpDT_InternalNode(internal12, feature6, 0.389, 2, internal5, internal11);
        // dag->addOpDT_LeafNode(leaf6, internal11, 1); // class 1
        // dag->addOpDT_LeafNode(leaf7, internal12, 0); // class 0

        /**
         * depth 4, dimension = 10000
         * */
        // dag->addOpDT_FeatureNode(feature1, input_table, 6502, input_data);    // 6502
        // dag->addOpDT_InternalNode(internal1, feature1, 0.054, 1, input_table, 0);
        // dag->addOpDT_InternalNode(internal2, feature1, 0.054, 2, input_table, internal1);

        // dag->addOpDT_FeatureNode(feature2, internal1, 4248, input_data);   // 4248
        // dag->addOpDT_InternalNode(internal3, feature2, 0.065, 1, internal1, 0);
        // dag->addOpDT_InternalNode(internal4, feature2, 0.065, 2, internal1, internal3);

        // dag->addOpDT_FeatureNode(feature3, internal2, 7001, input_data);   // 7001
        // dag->addOpDT_InternalNode(internal5, feature2, 0.239, 1, internal2, 0);
        // dag->addOpDT_InternalNode(internal6, feature2, 0.239, 2, internal2, internal5);
        // dag->addOpDT_LeafNode(leaf1, internal6, 0); // class 0


        // dag->addOpDT_FeatureNode(feature4, internal3, 1820, input_data);  // 1820
        // dag->addOpDT_InternalNode(internal7, feature4, 0.068, 1, internal3, 0);
        // dag->addOpDT_InternalNode(internal8, feature4, 0.068, 2, internal3, internal7);
        // dag->addOpDT_LeafNode(leaf2, internal8, 1); // class 1

        // dag->addOpDT_FeatureNode(feature5, internal4, 3981, input_data);  // 3981
        // dag->addOpDT_InternalNode(internal9, feature5, 0.073, 1, internal4, 0);
        // dag->addOpDT_InternalNode(internal10, feature5, 0.073, 2, internal4, internal9);
        // dag->addOpDT_LeafNode(leaf3, internal10, 0); // class 0

        // dag->addOpDT_FeatureNode(feature6, internal5, 6502, input_data);  // 6502
        // dag->addOpDT_InternalNode(internal11, feature6, 0.389, 1, internal5, 0);
        // dag->addOpDT_InternalNode(internal12, feature6, 0.389, 2, internal5, internal11);
        // dag->addOpDT_LeafNode(leaf4, internal12, 0); // class 0

        // dag->addOpDT_FeatureNode(feature7, internal7, 6238, input_data);  // 6238
        // dag->addOpDT_InternalNode(internal13, feature7, 0.058, 1, internal7, 0);
        // dag->addOpDT_InternalNode(internal14, feature7, 0.058, 2, internal7, internal13);
        // dag->addOpDT_LeafNode(leaf5, internal13, 0); // class 0
        // dag->addOpDT_LeafNode(leaf6, internal14, 1); // class 1

        // dag->addOpDT_FeatureNode(feature8, internal9, 3835, input_data);  // 3835
        // dag->addOpDT_InternalNode(internal15, feature8, 0.096, 1, internal9, 0);
        // dag->addOpDT_InternalNode(internal16, feature8, 0.096, 2, internal9, internal15);
        // dag->addOpDT_LeafNode(leaf7, internal15, 1); // class 1
        // dag->addOpDT_LeafNode(leaf8, internal16, 0); // class 0

        // dag->addOpDT_FeatureNode(feature9, internal11, 1258, input_data);  // 1258
        // dag->addOpDT_InternalNode(internal17, feature9, 0.188, 1, internal11, 0);
        // dag->addOpDT_InternalNode(internal18, feature9, 0.188, 2, internal11, internal17);
        // dag->addOpDT_LeafNode(leaf9, internal17, 1); // class 1
        // dag->addOpDT_LeafNode(leaf10, internal18, 0); // class 0

        // depth 3, dimension = 1000
        // dag->addOpDT_FeatureNode(feature1, input_table, 897, input_data);    // 897
        // dag->addOpDT_InternalNode(internal1, feature1, 0.061, 1, input_table, 0);
        // dag->addOpDT_InternalNode(internal2, feature1, 0.061, 2, input_table, internal1);

        // dag->addOpDT_FeatureNode(feature2, internal1, 557, input_data);   // 557
        // dag->addOpDT_InternalNode(internal3, feature2, 0.075, 1, internal1, 0);
        // dag->addOpDT_InternalNode(internal4, feature2, 0.075, 2, internal1, internal3);

        // dag->addOpDT_FeatureNode(feature3, internal2, 461, input_data);   // 461
        // dag->addOpDT_InternalNode(internal5, feature2, 0.173, 1, internal2, 0);
        // dag->addOpDT_InternalNode(internal6, feature2, 0.173, 2, internal2, internal5);
        // dag->addOpDT_LeafNode(leaf1, internal6, 0); // class 0


        // dag->addOpDT_FeatureNode(feature4, internal3, 182, input_data);  // 182
        // dag->addOpDT_InternalNode(internal7, feature4, 0.09, 1, internal3, 0);
        // dag->addOpDT_InternalNode(internal8, feature4, 0.09, 2, internal3, internal7);
        // dag->addOpDT_LeafNode(leaf2, internal7, 0); // class 0
        // dag->addOpDT_LeafNode(leaf3, internal8, 1); // class 1

        // dag->addOpDT_FeatureNode(feature5, internal4, 519, input_data);  // 519
        // dag->addOpDT_InternalNode(internal9, feature5, 0.087, 1, internal4, 0);
        // dag->addOpDT_InternalNode(internal10, feature5, 0.087, 2, internal4, internal9);
        // dag->addOpDT_LeafNode(leaf4, internal9, 1); // class 1
        // dag->addOpDT_LeafNode(leaf5, internal10, 0); // class 0

        // dag->addOpDT_FeatureNode(feature6, internal5, 858, input_data);  // 858
        // dag->addOpDT_InternalNode(internal11, feature6, 0.117, 1, internal5, 0);
        // dag->addOpDT_InternalNode(internal12, feature6, 0.117, 2, internal5, internal11);
        // dag->addOpDT_LeafNode(leaf6, internal11, 1); // class 1
        // dag->addOpDT_LeafNode(leaf7, internal12, 0); // class 0

        // depth 4, dimension = 1000
        // dag->addOpDT_FeatureNode(feature1, input_table, 897, input_data);    // 897
        // dag->addOpDT_InternalNode(internal1, feature1, 0.061, 1, input_table, 0);
        // dag->addOpDT_InternalNode(internal2, feature1, 0.061, 2, input_table, internal1);

        // dag->addOpDT_FeatureNode(feature2, internal1, 557, input_data);   // 557
        // dag->addOpDT_InternalNode(internal3, feature2, 0.075, 1, internal1, 0);
        // dag->addOpDT_InternalNode(internal4, feature2, 0.075, 2, internal1, internal3);

        // dag->addOpDT_FeatureNode(feature3, internal2, 109, input_data);   // 109
        // dag->addOpDT_InternalNode(internal5, feature2, 0.257, 1, internal2, 0);
        // dag->addOpDT_InternalNode(internal6, feature2, 0.257, 2, internal2, internal5);
        // dag->addOpDT_LeafNode(leaf1, internal6, 0); // class 0


        // dag->addOpDT_FeatureNode(feature4, internal3, 182, input_data);  // 182
        // dag->addOpDT_InternalNode(internal7, feature4, 0.09, 1, internal3, 0);
        // dag->addOpDT_InternalNode(internal8, feature4, 0.09, 2, internal3, internal7);
        // dag->addOpDT_LeafNode(leaf2, internal8, 1); // class 1

        // dag->addOpDT_FeatureNode(feature5, internal4, 382, input_data);  // 382
        // dag->addOpDT_InternalNode(internal9, feature5, 0.087, 1, internal4, 0);
        // dag->addOpDT_InternalNode(internal10, feature5, 0.087, 2, internal4, internal9);
        // dag->addOpDT_LeafNode(leaf3, internal10, 0); // class 0

        // dag->addOpDT_FeatureNode(feature6, internal5, 461, input_data);  // 461
        // dag->addOpDT_InternalNode(internal11, feature6, 0.173, 1, internal5, 0);
        // dag->addOpDT_InternalNode(internal12, feature6, 0.173, 2, internal5, internal11);
        // dag->addOpDT_LeafNode(leaf4, internal12, 0); // class 0

        // dag->addOpDT_FeatureNode(feature7, internal7, 852, input_data);  // 852
        // dag->addOpDT_InternalNode(internal13, feature7, 0.134, 1, internal7, 0);
        // dag->addOpDT_InternalNode(internal14, feature7, 0.134, 2, internal7, internal13);
        // dag->addOpDT_LeafNode(leaf5, internal13, 0); // class 0
        // dag->addOpDT_LeafNode(leaf6, internal14, 1); // class 1

        // dag->addOpDT_FeatureNode(feature8, internal9, 485, input_data);  // 485
        // dag->addOpDT_InternalNode(internal15, feature8, 0.128, 1, internal9, 0);
        // dag->addOpDT_InternalNode(internal16, feature8, 0.128, 2, internal9, internal15);
        // dag->addOpDT_LeafNode(leaf7, internal15, 1); // class 1
        // dag->addOpDT_LeafNode(leaf8, internal16, 0); // class 0

        // dag->addOpDT_FeatureNode(feature9, internal11, 858, input_data);  // 858
        // dag->addOpDT_InternalNode(internal17, feature9, 0.117, 1, internal11, 0);
        // dag->addOpDT_InternalNode(internal18, feature9, 0.117, 2, internal11, internal17);
        // dag->addOpDT_LeafNode(leaf9, internal17, 1); // class 1
        // dag->addOpDT_LeafNode(leaf10, internal18, 0); // class 0


        // depth 3, dimension = 5000
        // dag->addOpDT_FeatureNode(feature1, input_table, 4393, input_data);    // 4393
        // dag->addOpDT_InternalNode(internal1, feature1, 0.054, 1, input_table, 0);
        // dag->addOpDT_InternalNode(internal2, feature1, 0.054, 2, input_table, internal1);

        // dag->addOpDT_FeatureNode(feature2, internal1, 2406, input_data);   // 2406
        // dag->addOpDT_InternalNode(internal3, feature2, 0.137, 1, internal1, 0);
        // dag->addOpDT_InternalNode(internal4, feature2, 0.137, 2, internal1, internal3);

        // dag->addOpDT_FeatureNode(feature3, internal2, 4892, input_data);   // 4892
        // dag->addOpDT_InternalNode(internal5, feature2, 0.245, 1, internal2, 0);
        // dag->addOpDT_InternalNode(internal6, feature2, 0.245, 2, internal2, internal5);
        // dag->addOpDT_LeafNode(leaf1, internal6, 0); // class 0


        // dag->addOpDT_FeatureNode(feature4, internal3, 1144, input_data);  // 1144
        // dag->addOpDT_InternalNode(internal7, feature4, 0.073, 1, internal3, 0);
        // dag->addOpDT_InternalNode(internal8, feature4, 0.073, 2, internal3, internal7);
        // dag->addOpDT_LeafNode(leaf2, internal7, 0); // class 0
        // dag->addOpDT_LeafNode(leaf3, internal8, 1); // class 1

        // dag->addOpDT_FeatureNode(feature5, internal4, 1808, input_data);  // 1808
        // dag->addOpDT_InternalNode(internal9, feature5, 0.073, 1, internal4, 0);
        // dag->addOpDT_InternalNode(internal10, feature5, 0.073, 2, internal4, internal9);
        // dag->addOpDT_LeafNode(leaf4, internal9, 1); // class 1
        // dag->addOpDT_LeafNode(leaf5, internal10, 0); // class 0

        // dag->addOpDT_FeatureNode(feature6, internal5, 665, input_data);  // 665
        // dag->addOpDT_InternalNode(internal11, feature6, 0.211, 1, internal5, 0);
        // dag->addOpDT_InternalNode(internal12, feature6, 0.211, 2, internal5, internal11);
        // dag->addOpDT_LeafNode(leaf6, internal11, 1); // class 1
        // dag->addOpDT_LeafNode(leaf7, internal12, 0); // class 0

        

        // depth 4, dimension = 5000
        // dag->addOpDT_FeatureNode(feature1, input_table, 4393, input_data);    // 4393
        // dag->addOpDT_InternalNode(internal1, feature1, 0.054, 1, input_table, 0);
        // dag->addOpDT_InternalNode(internal2, feature1, 0.054, 2, input_table, internal1);

        // dag->addOpDT_FeatureNode(feature2, internal1, 2406, input_data);   // 2406
        // dag->addOpDT_InternalNode(internal3, feature2, 0.137, 1, internal1, 0);
        // dag->addOpDT_InternalNode(internal4, feature2, 0.137, 2, internal1, internal3);

        // dag->addOpDT_FeatureNode(feature3, internal2, 4892, input_data);   // 4892
        // dag->addOpDT_InternalNode(internal5, feature2, 0.245, 1, internal2, 0);
        // dag->addOpDT_InternalNode(internal6, feature2, 0.245, 2, internal2, internal5);
        // dag->addOpDT_LeafNode(leaf1, internal6, 0); // class 0


        // dag->addOpDT_FeatureNode(feature4, internal3, 1144, input_data);  // 1144
        // dag->addOpDT_InternalNode(internal7, feature4, 0.073, 1, internal3, 0);
        // dag->addOpDT_InternalNode(internal8, feature4, 0.073, 2, internal3, internal7);
        // dag->addOpDT_LeafNode(leaf2, internal8, 1); // class 1

        // dag->addOpDT_FeatureNode(feature5, internal4, 1808, input_data);  // 1808
        // dag->addOpDT_InternalNode(internal9, feature5, 0.073, 1, internal4, 0);
        // dag->addOpDT_InternalNode(internal10, feature5, 0.073, 2, internal4, internal9);
        // dag->addOpDT_LeafNode(leaf3, internal10, 0); // class 0

        // dag->addOpDT_FeatureNode(feature6, internal5, 665, input_data);  // 665
        // dag->addOpDT_InternalNode(internal11, feature6, 0.211, 1, internal5, 0);
        // dag->addOpDT_InternalNode(internal12, feature6, 0.211, 2, internal5, internal11);
        // dag->addOpDT_LeafNode(leaf4, internal12, 0); // class 0

        // dag->addOpDT_FeatureNode(feature7, internal7, 4129, input_data);  // 4129
        // dag->addOpDT_InternalNode(internal13, feature7, 0.058, 1, internal7, 0);
        // dag->addOpDT_InternalNode(internal14, feature7, 0.058, 2, internal7, internal13);
        // dag->addOpDT_LeafNode(leaf5, internal13, 0); // class 0
        // dag->addOpDT_LeafNode(leaf6, internal14, 1); // class 1

        // dag->addOpDT_FeatureNode(feature8, internal9, 2174, input_data);  // 2174
        // dag->addOpDT_InternalNode(internal15, feature8, 0.11, 1, internal9, 0);
        // dag->addOpDT_InternalNode(internal16, feature8, 0.11, 2, internal9, internal15);
        // dag->addOpDT_LeafNode(leaf7, internal15, 1); // class 1
        // dag->addOpDT_LeafNode(leaf8, internal16, 0); // class 0

        // dag->addOpDT_FeatureNode(feature9, internal11, 4393, input_data);  // 4393
        // dag->addOpDT_InternalNode(internal17, feature9, 0.483, 1, internal11, 0);
        // dag->addOpDT_InternalNode(internal18, feature9, 0.483, 2, internal11, internal17);
        // dag->addOpDT_LeafNode(leaf9, internal17, 1); // class 1
        // dag->addOpDT_LeafNode(leaf10, internal18, 0); // class 0
    }

    dag->addOpDT_ResNode(res, leafs.data(),leafs.size());
//    dag->addOpVia(tmp, st_add[M]);
//    dag->addOpAdd_Mat(total, st_add[M], tmp);
//    dag->addOpMul_Const_Mat(output, st_add[M], 1);
//    dag->addOpLTZ(ltz, output);
    dag->toposort();

//    dag->reveal_init(out_sig);
    /// if the output is secret-shared as well
    // dag->reveal_init(feature1);
    // dag->reveal_init(internal1);
    // dag->reveal_init(leaf1);
    // dag->reveal_init(internal2);
    // dag->reveal_init(internal3);
    // dag->reveal_init(leaf2);
    // dag->reveal_init(leaf3);
    // dag->reveal_init(leaf4);
    // dag->reveal_init(leaf5);
    // dag->reveal_init(leaf6);
    // dag->reveal_init(leaf7);
    // dag->reveal_init(leaf8);
    // dag->reveal_init(leaf9);
    dag->reveal_init(res);
//    dag->reveal_init(total);
//    dag->reveal_init(output);
//    dag->reveal_init(ltz);
}

void DT_graph::DT::forward_predict() {

    cout << "Begin Forward" << endl;
    // load data
    for (int j = 0; j < M; ++j) {
        if (j != node_type) {
            MathOp::broadcast_share(&data[j], j);
        }
    }
    int total_sample = 0;
    for (int l = 0; l < M; ++l) {
        if (l != node_type) {
            MathOp::receive_share(dag->getNode(input[l])->getForward(), l);
            total_sample += dag->getNode(input[l])->getForward()->rows();
        }
    }
    *dag->getNode(input[node_type])->getForward() = data[node_type];
    total_sample += dag->getNode(input[node_type])->getForward()->rows();
    cout << total_sample << endl;
    this->size = total_sample;

    int cur = 0;
    Mat *total_data = new Mat(total_sample, FEATURE_DIM);
    for (int l = 0; l < M; ++l) {
        for (int i = 0; i < dag->getNode(input[l])->getForward()->rows(); i++)
            for (int j = 0; j < dag->getNode(input[l])->getForward()->cols(); j++) {
//                dag->getNode(input_table)->getForward()->operator()(i + cur, j) = dag->getNode(input[l])->getForward()->operator()(i, j);
                total_data->operator()(i + cur, j) = dag->getNode(input[l])->getForward()->operator()(i, j);
            }
        cur += dag->getNode(input[l])->getForward()->rows();
    }

    /** log data info **/
//    total_data->print();
    cout << "total size: " << total_data->rows() << endl;

    dag->getNode(input_table)->getTable()->init(PREDICTION_BATCH, FEATURE_DIM);

    clock_train = new Constant::Clock(CLOCK_TRAIN);
    globalRound = 0;
    for (int i = 0; i < total_data->rows()/PREDICTION_BATCH && i < TRAIN_ITE; i++) {
        globalRound++;
        DBGprint("Epoch %d\n", i);
        next_batch(*dag->getNode(input_table)->getTable(), i * PREDICTION_BATCH, total_data, size);
        input_data = dag->getNode(input_table)->getTable();

        dag->getNode(input_table)->fillForward(IE);
        dag->getNode(res)->getForward()->clear();
        {
            dag->epoch_init();
            while (dag->forwardHasNext()) {
                dag->forwardNext();
            }
            dag->reveal_init();
            while (dag->revealHasNext()) {
                dag->reveal();
            }
            // DBGprint("-----Reveal--------\n");
            // cout << "leaf1\n";
            // dag->getNode(leaf1)->getForward()->print();
            // cout << "leaf2\n";
            // dag->getNode(leaf2)->getForward()->print();
            // cout << "leaf3\n";
            // dag->getNode(leaf3)->getForward()->print();
            // cout << "leaf4\n";
            // dag->getNode(leaf4)->getForward()->print();
            // cout << "leaf5\n";
            // dag->getNode(leaf5)->getForward()->print();
            // cout << "leaf6\n";
            // dag->getNode(leaf6)->getForward()->print();
            // cout << "leaf7\n";
            // dag->getNode(leaf7)->getForward()->print();
            // cout << "leaf8\n";
            // dag->getNode(leaf8)->getForward()->print();
            // cout << "leaf9\n";
            // dag->getNode(leaf9)->getForward()->print();

            // cout << "res\n";
            dag->getNode(res)->getForward()->print();
            // dag->getNode(feature7)->getForward()->print();
            // dag->getNode(input_table)->getForward()->print();
        }
        print_perd(i+1);
    }
    cout << "-------END-----\n";
}

void DT_graph::DT::permute_tree_graph(int depth) {
    dag = new DTDAG();
    this->depth = depth;

    // input tree model parameters
    raw_tree = dag->addnode(PREDICTION_BATCH, FEATURE_DIM, NeuronMat::NODE_INPUT); // raw tree

    shuffled_tree = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP); // stores the comparison result (XORed with random bit)

    dag->global_variables_initializer();

    dag->addOpDT_PermTree(shuffled_tree, raw_tree);
    
    dag->toposort();
}

void DT_graph::DT::forward_permute_tree(NodeMat *res, Mat *idxs_perm, Mat *thresholds_perm) {
    cout << "Begin Tree Permutation\n";

    int nodes_num = (1 << depth) - 1;

    clock_train = new Constant::Clock(CLOCK_TRAIN);
    globalRound = 0;
    dag->getNode(raw_tree)->set_idxs(idxs_perm);
    dag->getNode(raw_tree)->set_thresholds(thresholds_perm);

    {
        dag->epoch_init();
        while (dag->forwardHasNext()) {
            dag->forwardNext();
        }
        dag->reveal_init();
        while (dag->revealHasNext()) {
            dag->reveal();
        }
        res->set_idxs(dag->getNode(shuffled_tree)->get_idxs());
        res->set_thresholds(dag->getNode(shuffled_tree)->get_thresholds());
        res->set_perm_bits(dag->getNode(shuffled_tree)->get_perm_bits());
    }
    print_perd(1);
}

void DT_graph::DT::optimized_predict_graph(int depth) {
    dag = new DTDAG();
    this->depth = depth;
    input_table = dag->addnode(PREDICTION_BATCH, FEATURE_DIM, NeuronMat::NODE_INPUT); // stores the input data, in mat form

    internal_node = dag->addnode(PREDICTION_BATCH, 1, NeuronMat::NODE_OP); // stores the comparison result (XORed with random bit)

    dag->global_variables_initializer();

    Mat ph_feature(PREDICTION_BATCH, FEATURE_DIM), ph_threshold(PREDICTION_BATCH, 1), ph_perm_bit(PREDICTION_BATCH, 1); 
    dag->addOpDT_Opt_InternalNode(internal_node, input_table, &ph_feature, &ph_threshold, &ph_perm_bit);
    
    dag->toposort();

}

void DT_graph::DT::forward_optimized_graph(Mat *total_data, Mat *idxs_perm, Mat *thresholds_perm, Mat *perm_bits) {
    cout << "============== Begin Decision Tree Inference ==============" << endl;

    clock_train = new Constant::Clock(CLOCK_TRAIN);
    globalRound = 0;
    int total_size = total_data->rows();
    
    int idx;
    ll128 left_or_right;

    /**
     * Batch Version
     * */
    vector<int> idx_batch(PREDICTION_BATCH);
    vector<ll128> left_or_right_batch(PREDICTION_BATCH);

    Mat input_batch(PREDICTION_BATCH, FEATURE_DIM);
    Mat idx_vector_batch(PREDICTION_BATCH, FEATURE_VECTOR_SIZE);   

    Mat threshold_batch(PREDICTION_BATCH, 1);
    Mat perm_bits_batch(PREDICTION_BATCH, 1);
    // currently, this inference scheme does not support batch processing
    // TODO: Seems like batch processing can be enabled.
    int i;
    for (i = 0; i < total_data->rows() / PREDICTION_BATCH && i < TRAIN_ITE; i++) {
        globalRound++;
        DBGprint("Epoch %d\n", i);
        next_batch(input_batch, i * PREDICTION_BATCH, total_data, total_size);
        *dag->getNode(input_table)->getForward() = input_batch;

        idx = 1; // start from the root node
        left_or_right = 0;

        for (int j = 0; j < PREDICTION_BATCH; j++) {
            idx_batch[j] = 1;
            left_or_right_batch[j] = 0;
        }
        
        for (int dp = 0; dp < depth - 1; dp++) { // internal node

            {
                // set idxs and thresholds according to the comparision result
                for (int b_idx = 0; b_idx < PREDICTION_BATCH; b_idx++) {
                    for (int f_idx = 0; f_idx < FEATURE_VECTOR_SIZE; f_idx++) {
                        idx_vector_batch.setVal(f_idx*PREDICTION_BATCH+b_idx, idxs_perm[idx_batch[b_idx]-1].get(0, f_idx));
                    }
                    // idxs_perm[idx_batch[b_idx]-1].append(b_idx * FEATURE_DIM, (b_idx + 1) * FEATURE_DIM, &idx_vector_batch);
                    // thresholds_perm[idx_batch[b_idx]-1].append(b_idx, b_idx + 1, &threshold_batch);
                    threshold_batch.setVal(b_idx, 0, thresholds_perm[idx_batch[b_idx]].get(0, 0));
                    perm_bits_batch.setVal(b_idx, 0, perm_bits[dp].get(0, 0));
                }

                dag->setOp(internal_node, new MathOp::OptInternalNode(
                    dag->getNode(internal_node)->getForward(),      // res
                    dag->getNode(input_table)->getForward(),        // input kv data 
                    &idx_vector_batch, 
                    &threshold_batch, 
                    &perm_bits_batch
                ));
                dag->epoch_init();
                while (dag->forwardHasNext()) {
                    dag->forwardNext();
                }
                dag->reveal_init();
                while (dag->revealHasNext()) {
                    dag->reveal();
                }
            }

            for (int j = 0; j < PREDICTION_BATCH; j++) {
                left_or_right_batch[j] = dag->getNode(internal_node)->getForward()->get(j, 0);
                idx_batch[j] = 2 * idx_batch[j] + left_or_right_batch[j];
            }
        }

    }
    print_perd(i);
    cout << "-------END-----\n";
}

void DT_graph::DT::feed(DTDAG* dag, Mat &x_batch, Mat &y_batch, int input, int output) {
    *dag->getNode(input)->getForward() = x_batch;
    *dag->getNode(output)->getForward() = y_batch;
}

void DT_graph::DT::next_batch(Mat &batch, int start, Mat *A, int mod) {
    batch = A->row(start%mod, start%mod + PREDICTION_BATCH);
}

void DT_graph::DT::test() {
    Mat x_batch(1, PREDICTION_BATCH), y_batch(1, PREDICTION_BATCH);
    int total = 0;
    for (int i = 0; i < NM / PREDICTION_BATCH; i++) {
        globalRound++;
        next_batch(x_batch, i * B, data);
        next_batch(y_batch, i * PREDICTION_BATCH, data);
        feed(dag, x_batch, y_batch, input[0], output);
        dag->epoch_init();
        while (dag->forwardHasNext()) {
            dag->forwardNext();
        }
        dag->getNode(output)->getForward()->print();
        dag->reveal_init();
        while (dag->revealHasNext()) {
            dag->reveal();
        }
        DBGprint("-------------\n");
        dag->getNode(output)->getForward()->print();
    }
}

void DT_graph::DT::print_perd(int round) {
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

Mat DT_graph::DT::vector_to_mat(int *data, int r, int c) {
    Mat ret(1, r*c);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            ret(j, i) = data[j+i*r];
        }
    }
    return ret;
}