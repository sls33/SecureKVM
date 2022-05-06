#include "omen_eval.h"

extern "C"
{
#include "../comen/common.h"
#include "../comen/errorHandler.h"
#include "../comen/cmdlineEvalPW.h"
#include "../comen/evalPW.h"
#include "../comen/commonStructs.h"
#include "../comen/nGramReader.h"
    // == global variables ==
    // all global pointer should be freed in the exit_routine()

    extern struct filename_struct *glbl_filenamesIn;
    extern struct alphabet_struct *glbl_alphabet;
    extern struct nGram_struct *glbl_nGramLevel;

    extern char glbl_maxLevel;
    extern char *glbl_password;
    extern bool glbl_verboseMode;
    extern struct gengetopt_args_info glbl_args_info;

    // enum writeModes writeMode = writeMode_nonVerbose;
    //  === public functions ===

    /*
     * initializes all global parameters, setting them to their default value
     * !! this function must be called before any other operation !!
     */
    void initialize();

    /*
     *  prints all Error-Messages (if any), clears the allocated memory of the
     *  global variables and ends the application
     *  the char* exit_msg is printed out on the command line
     *  !! this function is set via atexit() !!
     */
    void exit_routine();

    /*
     * prints the by arguments selected mode as well as the output and input filenames
     */
    void print_settings();

    /*
     * Evaluates given command line arguments using the getopt-library
     * there has to be at least 1 argument: the input filename
     * additional arguments are evaluated in this method an the
     * corresponding parameters are set
     * returns TRUE, if the evaluation was successful
     */
    bool evaluate_arguments(struct gengetopt_args_info *args_info);

    /*
     * Reads any needed input file using the nGramIO-functions and
     * sets all needed variables accordingly.
     * Returns TRUE on success and FALSE if something went wrong.
     */
    bool apply_settings();

    /*
     * Evaluates the password given by command line argument and
     * prints the overall level.
     * Returns FALSE, if the password is to short.
     * Otherwise TRUE is returned.
     */
    bool run_evaluation();
}

int node_type;
SocketManager::SMMLF tel;
int globalRound;

// file pointer
char mpl_prefix[256] = {'\0'};

int test_train_NGram();
int test_eval_Pwds();
int test_eval_Pwds_online_get();
std::pair<vector<int>, double> tranform_to_position_pwd(string pwd);

int main(int argc, char *argv[])
{
    srand(time(NULL));
    // let's call our cmdline parser
    if (cmdline_parser(argc, argv, &glbl_args_info) != 0)
    {
        printf("failed parsing command line arguments\n");
        exit(EXIT_FAILURE);
    }
    // set exit_routine so thats automatically called
    atexit(exit_routine);

    initialize();

    if (!evaluate_arguments(&glbl_args_info))
        exit(1);

    if (!apply_settings())
        exit(1);

    if (glbl_verboseMode)
        print_settings();

    // if (!run_evaluation())
    //     exit(1);

    // exit(EXIT_SUCCESS);

    int length = strlen(glbl_password);
    double level = 0;
    int position = 0;
    int cur = 0;

    vector<int> positions(length - glbl_nGramLevel->sizeOf_N + 1);
    cout << glbl_password << ": " << length - glbl_nGramLevel->sizeOf_N << endl;
    if (length < glbl_nGramLevel->sizeOf_N - 1)
    {
        errorHandler_print(errorType_Error, "The password to be evaluated is to short.\n");
        return false;
    }
    printf("current len level: %d\n", glbl_nGramLevel->len[length]);
    get_positionFromNGram(&position, glbl_password, glbl_nGramLevel->sizeOf_N - 1, glbl_alphabet->sizeOf_alphabet,
                          glbl_alphabet->alphabet);
    positions[cur++] = position;
    printf("%f", glbl_nGramLevel->iP[position]);
    level -= glbl_nGramLevel->iP[position];

    for (size_t i = 1; i <= (length - glbl_nGramLevel->sizeOf_N); i++)
    {
        get_positionFromNGram(&position, glbl_password + i, glbl_nGramLevel->sizeOf_N, glbl_alphabet->sizeOf_alphabet,
                              glbl_alphabet->alphabet);
        positions[cur++] = position;
        printf(" + %f", -glbl_nGramLevel->cP[position]);
        level -= glbl_nGramLevel->cP[position];
    }
    printf(" = %f (overall level)\n", level);
    for (size_t i = 0; i < positions.size(); i++)
    {
        cout << "i: " << i << ", position: " << positions[i] << endl;
    }

    char *mpl_prefix = "--player";
    for (int i = 1; i < argc; i++)
    {
        if (strncmp(argv[i], mpl_prefix, strlen(mpl_prefix)) == 0)
        {
            i += 1;
            node_type = atoi(argv[i]);
            printf("Node type: %d\n", node_type);
        }
    }

    Player::init();

    // Mat test_a(5, 5, 1), test_b(5, 1, 0);
    // test_a.print();
    // test_b.print();
    // Mat a_b_hstack(5, 6);
    // Mat::hstack(&a_b_hstack, &test_a, &test_b);
    // a_b_hstack.print();

    // tranform_to_position_pwd("123456789");
    test_eval_Pwds_online_get();
} // main

int test_eval_Pwds()
{
    DBGprint("----------    Test Eval Pwds  ----------\n");

    // Mat input_data_ip;
    // Mat input_data_cp;
    Mat input_data;
    vector<double> levels;
    if (node_type == 0)
    {
        /** Load input password to be evalueated prob **/
        ifstream in("mpc_omen/pwd_data/test.txt");
        string line;
        vector<string> pwds;
        vector<vector<int>> pwds_t;
        int count = 0;
        if (in.is_open())
        {
            while (getline(in, line))
            {
                count++;
                cout << line << endl;
                pwds.push_back(line);
            }
        }

        /** Print the input sequence data (plaintext) **/
        for (size_t i = 0; i < count; i++)
        {
            cout << pwds[i] << endl;
            std::pair<vector<int>, double> parse_res = tranform_to_position_pwd(pwds[i]);
            pwds_t.push_back(parse_res.first);
            levels.push_back(parse_res.second);
        }

        for (size_t i = 0; i < pwds_t.size(); i++)
        {
            cout << "\ni: " << i << endl;
            for (size_t j = 0; j < pwds_t[i].size(); j++)
            {
                cout << pwds_t[i][j] << ", ";
            }
        }

        Mat pwd_data_ip(ALPHABET_SIZE * ALPHABET_SIZE, count, 0);

        for (size_t i = 0; i < count; i++)
        {
            pwd_data_ip.setVal(pwds_t[i][0], i, pwd_data_ip.get(pwds_t[i][0], i) + 1);
        }

        Mat pwd_data_cp(KEY_DOMAIN_SIZE, count, 0);

        for (size_t i = 0; i < count; i++)
        {
            for (size_t j = 1; j < pwds_t[i].size(); j++)
            {
                pwd_data_cp.setVal(pwds_t[i][j], i, pwd_data_cp.get(pwds_t[i][j], i) + 1);
            }
        }

        Mat data(KEY_DOMAIN_SIZE + ALPHABET_SIZE * ALPHABET_SIZE, count, 0);
        Mat::concat(&data, &pwd_data_ip, &pwd_data_cp);
        /** Load the secret shared input **/
        // data.print();
        // Mat* data_ip = IOManager::secret_share_mat_data(pwd_data_ip, pwd_data_ip.size(), "omen_eval_ip");
        // Mat* data_cp = IOManager::secret_share_mat_data(pwd_data_cp, pwd_data_cp.size(), "omen_eval_cp");
        // return 0;
        Mat *sec_data = IOManager::secret_share_mat_data(data, data.size(), "omen_eval");

        // input_data_ip = data_ip[0];
        // input_data_cp = data_cp[0];

        input_data = sec_data[0];

        for (int j = 0; j < M; ++j)
        {
            if (j != node_type)
            {
                //                ip_batch.print();
                // MathOp::broadcast_share(&data_ip[j], j);
                // MathOp::broadcast_share(&data_cp[j], j);
                MathOp::broadcast_share(&sec_data[j], j);
            }
        }
    }
    else
    {
        // MathOp::receive_share(&input_data_ip, 0);
        // MathOp::receive_share(&input_data_cp, 0);
        MathOp::receive_share(&input_data, 0);
    }
    // input_data.print();
    // cout << input_data_ip.rows() << " : " << input_data_ip.cols() << endl;
    // cout << input_data_cp.rows() << " : " << input_data_cp.cols() << endl;
    cout << input_data.rows() << " : " << input_data.cols() << endl;

    cout << "Load input done\n";
    return 0;

    /** Load the computed model parameter **/
    Mat ip_data = IOManager::load_secret_share_ngram(glbl_nGramLevel->iP, glbl_nGramLevel->sizeOf_iP);
    Mat cp_data = IOManager::load_secret_share_ngram(glbl_nGramLevel->cP, glbl_nGramLevel->sizeOf_cP);
    // Mat* ep_data = IOManager::load_secret_share_ngram(glbl_nGramLevel->eP, glbl_nGramLevel->sizeOf_eP);
    // Mat* ln_data = IOManager::load_secret_share_ngram(glbl_nGramLevel->len, glbl_nGramLevel->sizeOf_len);
    cout << "------ IP -----\n";
    ip_data.print();
    cout << "------ CP -----\n";
    cp_data.print();
    cout << "Load model done\n";

    /** Evaluate the give the input **/
    Mat result(1, input_data.cols());
    Mat ip_and_cp_data(1, input_data.rows());
    Mat::hstack(&ip_and_cp_data, &ip_data, &cp_data);
    NgramGraph::Ngram *ng = new NgramGraph::Ngram(&input_data);
    ng->calculate_graph();
    ng->forward_inference(&result, &input_data, &ip_and_cp_data);

    // Mat *input_pwds = IOManager::secret_share_ngram(glbl_nGramCount->iP, glbl_nGramCount->sizeOf_iP, "IP");
    return 0;
}

int test_eval_Pwds_online_get()
{
    DBGprint("----------    Test Eval Pwds  ----------\n");

    /** Load the computed model parameter **/
    Mat ip_data = IOManager::load_secret_share_ngram(glbl_nGramLevel->iP, glbl_nGramLevel->sizeOf_iP);
    Mat cp_data = IOManager::load_secret_share_ngram(glbl_nGramLevel->cP, glbl_nGramLevel->sizeOf_cP);
    // Mat* ep_data = IOManager::load_secret_share_ngram(glbl_nGramLevel->eP, glbl_nGramLevel->sizeOf_eP);
    // Mat* ln_data = IOManager::load_secret_share_ngram(glbl_nGramLevel->len, glbl_nGramLevel->sizeOf_len);
    cout << "------ IP -----\n";
    ip_data.print_shape();
    cout << "------ CP -----\n";
    cp_data.print_shape();
    Mat ip_and_cp_data(KEY_DOMAIN_SIZE, 1);
    Mat::hstack(&ip_and_cp_data, &ip_data, &cp_data);
    cout << "------ IP + CP -----\n";
    // ip_and_cp_data.print();
    ip_and_cp_data.print_shape();
    cout << "Load model done\n";

    int count = 0;
    int sample_size = TOTAL_SAMPLE_SIZE * INFER_BATCH;
    // Mat data(sample_size, KEY_DOMAIN_SIZE, 0);  // Basic
    Mat data(sample_size, MAX_LEN, 0); // Fold & Online
    vector<double> levels;
    if (node_type == 0)
    {
        /** Load input password to be evalueated prob **/
        ifstream in("mpc_omen/pwd_data/rockyou_all_test.txt");
        string line;
        vector<string> pwds;
        vector<vector<int>> pwds_t;

        if (in.is_open())
        {
            while (getline(in, line))
            {
                if (count == sample_size)
                    break;
                count++;
                // cout << line << endl;
                pwds.push_back(line);
            }
        }
        in.close();

        /** Print the input sequence data (plaintext) **/
        for (size_t i = 0; i < count; i++)
        {
            // cout << pwds[i] << endl;
            std::pair<vector<int>, double> parse_res = tranform_to_position_pwd(pwds[i]);
            pwds_t.push_back(parse_res.first);
            levels.push_back(parse_res.second);
        }

        // for (size_t i = 0; i < pwds_t.size(); i++)
        // {
        //     cout << "\ni: " << i << endl;
        //     for (size_t j = 0; j < pwds_t[i].size(); j++)
        //     {
        //         cout << pwds_t[i][j] << ", ";
        //     }
        // }

        if (USING_BASIC)
        {
            ////////////////////////////////////////// BASIC //////////////////////////
            Mat pwd_data_ip(count, IP_DOMAIN_SIZE, 0);
            pwd_data_ip.print_shape();

            for (size_t i = 0; i < count; i++)
            {
                pwd_data_ip.setVal(i, pwds_t[i][0], pwd_data_ip.get(i, pwds_t[i][0]) + 1);
            }

            Mat pwd_data_cp(count, CP_DOMAIN_SIZE, 0);

            for (size_t i = 0; i < count; i++)
            {
                for (size_t j = 1; j < pwds_t[i].size(); j++)
                {
                    pwd_data_cp.setVal(i, pwds_t[i][j], pwd_data_cp.get(i, pwds_t[i][j]) + 1);
                }
            }

            Mat tmp(count, KEY_DOMAIN_SIZE, 0);
            Mat::hstack(&tmp, &pwd_data_ip, &pwd_data_cp);
            data = tmp;
        }
        else
        {
            ////////////////////////////////////////// FOLD //////////////////////////
            Mat pwd_data_ip(count, 1, 0);
            pwd_data_ip.print_shape();

            for (size_t i = 0; i < count; i++)
            {
                pwd_data_ip.setVal(i, 0, pwds_t[i][0]);
            }

            Mat pwd_data_cp(count, MAX_LEN - 1, KEY_DOMAIN_SIZE - 1 - IP_DOMAIN_SIZE);

            for (size_t i = 0; i < count; i++)
            {
                for (size_t j = 1; j < pwds_t[i].size(); j++)
                {
                    pwd_data_cp.setVal(i, j - 1, pwds_t[i][j]);
                    // pwd_data_cp.setVal(i, pwds_t[i][j], pwd_data_cp.get(i, pwds_t[i][j]) + 1);
                }
            }
            pwd_data_cp = pwd_data_cp + IP_DOMAIN_SIZE;

            Mat tmp(count, MAX_LEN, 0);
            Mat::hstack(&tmp, &pwd_data_ip, &pwd_data_cp);
            data = tmp;
        }
        cout << "Load plain-text input data done\n";
    }

    tel.init();

    // Mat input_data_ip;
    // Mat input_data_cp;
    Mat input_data(count, KEY_DOMAIN_SIZE, 0);
    //     if (node_type == 0) {
    //         /** Load the secret shared input **/
    //         // Mat* data_ip = IOManager::secret_share_mat_data(pwd_data_ip, pwd_data_ip.size(), "omen_eval_ip");
    //         // Mat* data_cp = IOManager::secret_share_mat_data(pwd_data_cp, pwd_data_cp.size(), "omen_eval_cp");
    //         Mat* sec_data = IOManager::secret_share_mat_data(data, data.size(), "omen_eval");

    //         // input_data_ip = data_ip[0];
    //         // input_data_cp = data_cp[0];

    //         input_data = sec_data[0];

    //         for (int j = 0; j < M; ++j) {
    //             if (j != node_type) {
    // //                ip_batch.print();
    //                 // MathOp::broadcast_share(&data_ip[j], j);
    //                 // MathOp::broadcast_share(&data_cp[j], j);
    //                 MathOp::broadcast_share(&sec_data[j], j);
    //             }
    //         }
    //     } else {
    //         // MathOp::receive_share(&input_data_ip, 0);
    //         // MathOp::receive_share(&input_data_cp, 0);
    //         MathOp::receive_share(&input_data, 0);
    //     }

    cout << data.rows() << " : " << data.cols() << endl;

    cout << "Load input done\n";

    /** Evaluate the give the input **/
    Mat result(data.rows(), 1);

    NgramGraph::Ngram *ng = new NgramGraph::Ngram();
    Mat ip_and_cp_data_log(KEY_DOMAIN_SIZE, 1);

    /**
    cout << "========== BEGIN LOG PROBABILITY ==========" << endl;
    ng->log_prob_graph();
    ng->forward_log_prob(&ip_and_cp_data, &ip_and_cp_data_log);
    ip_and_cp_data_log.setVal(KEY_DOMAIN_SIZE-1, 0, 0);
    cout << "========== END LOG PROBABILITY ==========" << endl;
    **/

    // int ibs[2] = {5, 10};
    // bool onoffs[2] = {false, true};

    cout << "========== BEGIN INFERENCE ==========" << endl;
    ip_and_cp_data_log = ip_and_cp_data;

    ng->inference_graph();
    if (!USING_BASIC && !USING_FOLD)
    {
        int BB = INFER_BATCH * MAX_LEN;
        Mat duplicate_data(KEY_DOMAIN_SIZE, BB);
        int idx = 0;
        for (; idx < BB; idx++)
        {
            ip_and_cp_data_log.append(idx * KEY_DOMAIN_SIZE, (idx + 1) * KEY_DOMAIN_SIZE, &duplicate_data);
            // duplicate_data.insert_val(, ip_and_cp_data_log);
        }
        duplicate_data.print_shape();
        ng->forward_inference(&result, &data, &duplicate_data);
    }
    else
    {
        ng->forward_inference(&result, &data, &ip_and_cp_data_log);
    }
    cout << "========== END INFERENCE ==========" << endl;

    /**
    if (node_type == 0)
    {
        double mse = 0;
        double me = 0;
        double total = 0;
        for (int i = 0; i < result.rows(); i++)
        {
            double predict = ((result.get(i, 0) + MOD / 2) % MOD - MOD / 2) * 1.0 / IE;
            cout << predict << ", ";
            mse += (-predict - levels[i]) * (-predict - levels[i]);
            me += (-predict - levels[i]);
            total += levels[i];
        }
        cout << endl;
        cout << "=======================================" << endl;
        cout << "========== PRECISION METRICS ==========" << endl;
        cout << "=======================================" << endl;
        // cout << "Total MSE: " << mse / result.rows() << endl;
        // cout << "Total RMSE: " << sqrt(mse / result.rows()) << endl;
        // cout << "Total ME: " << me / result.rows() << endl;
        // cout << "Mean: " << total / result.rows() << endl;
    }
    **/

    // Mat *input_pwds = IOManager::secret_share_ngram(glbl_nGramCount->iP, glbl_nGramCount->sizeOf_iP, "IP");
    return 0;
}

std::pair<vector<int>, double> tranform_to_position_pwd(string pwd)
{
    int length = pwd.length();
    char *pwd_p = (char *)pwd.c_str();
    int position = 0;
    int cur = 0;
    double level = 0;

    cout << pwd << ": " << length - glbl_nGramLevel->sizeOf_N + 1 << endl;
    vector<int> positions(length - glbl_nGramLevel->sizeOf_N + 1);

    if (length < glbl_nGramLevel->sizeOf_N - 1)
    {
        errorHandler_print(errorType_Error, "The password to be evaluated is to short.\n");
        // return false;
    }
    // printf("current len level: %f\n", glbl_nGramLevel->len[length]);
    get_positionFromNGram(&position, pwd_p, glbl_nGramLevel->sizeOf_N - 1, glbl_alphabet->sizeOf_alphabet,
                          glbl_alphabet->alphabet);
    positions[cur++] = position;
    printf("%f", -glbl_nGramLevel->iP[position]);
    level -= log2(glbl_nGramLevel->iP[position]);

    for (size_t i = 1; i < (length - glbl_nGramLevel->sizeOf_N + 1); i++)
    {
        cout << "(" << pwd_p + i << ")";
        get_positionFromNGram(&position, pwd_p + i, glbl_nGramLevel->sizeOf_N, glbl_alphabet->sizeOf_alphabet,
                              glbl_alphabet->alphabet);
        positions[cur++] = position;
        printf(" + %f", -glbl_nGramLevel->cP[position]);
        level -= log2(glbl_nGramLevel->cP[position]);
    }
    printf(" = %f (overall level)\n", level);
    for (size_t i = 0; i < positions.size(); i++)
    {
        cout << "i: " << i << ", position: " << positions[i] << endl;
    }
    return std::pair<vector<int>, double>(positions, level);
}