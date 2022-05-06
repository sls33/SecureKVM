#include "omen_main.h"

extern "C" {
#include "../comen/createNG.h"
#include "../comen/common.h"
#include "../comen/cmdlineCreateNG.h"
#include "../comen/commonStructs.h"
#include "../comen/smoothing.h"
#include "../comen/errorHandler.h"

// == global variables ==
// all global pointer should be freed in the exit_routine()

// struct for nGram, initalProb and length array for counts
extern nGram_struct *glbl_nGramCount;
extern uint64_t glbl_countSums[ARRAY_TYPES_COUNT];  // access via the arrayType-enum (arrayType_nGram = 0, arrayType_initialProb, arrayType_endProb, arrayType_length)

// These global variables can be (or must be) set by command line arguments
// the default values are set in initializes() (if any)

// struct for filename of all output files (nGram, initialProb, length and config)
extern struct filename_struct *glbl_filenames;  // each should not be larger then [FILENAME_MAX - MAX_ATTACHMENT_LENGTH]

// Alphabet and size of Alphabet
extern struct alphabet_struct *glbl_alphabet;

extern char glbl_maxLevel;

// Modes:
extern bool glbl_verboseFileMode;  // write additional information to output files (the actual nGrams are written to the files as well)
extern bool glbl_verboseMode;  // print information to stdout during run time
extern bool glbl_countedPasswordList;  // if TRUE the password list read is interpreted as counted one
extern bool glbl_storeWarnings;  // if TRUE all occurring warnings are saved to HD

extern struct gengetopt_args_info glbl_args_info;
//enum writeModes writeMode = writeMode_nonVerbose;
// === public functions ===

/*
 *     initializes all global parameters, setting them to their default value
 *     !! this function must be called before any other operation !!
 */
void initialize();

/*
 *  prints all Error-Messages (if any), clears the allocated memory of the
 *  global variables and ends the application
 *  the char* exit_msg is printed out on the command line
 *  !! this function is set via atexit() and automatically called at the end of the application!!
 */
void exit_routine();

/*
 *     evaluates given command line arguments using the getopt-library
 *     there has to be at least 1 argument: the input filename
 *     additional arguments are evaluated in this method an the
 *     corresponding parameters are set
 *     returns TRUE, if the evaluation was successful
 */
bool evaluate_arguments(struct gengetopt_args_info *args_info);

/*
 *  main process: calls evaluate_InputFile and the Write-Methods
 *  returns TRUE, if the creation was successful (no Errors occurred)
 *  otherwise FALSE is returned and the occurred Errors can be viewed using
 *  the Error-Handler
 */
bool run_creation();

/*
 *     evaluates the input file, reading the 3grams, initial probabilities and the pwd lengths
 *     and storing them in the associated global variables
 *     1. counts the occurrence of any n-gram in the input file (stored in glbl_nGramCount->nG)
 *     2. counts the initial probability (as (n-1)-gram, stored in glbl_nGramCount->iP)
 *     3. counts passwords length (max length SIZE_LENGTH_FIELD, stored in glbl_nGramCount->len)
 */
bool evaluate_inputFile(const char *filenameIn);

/*
 *     Writes header and additional information into the given file
 *     the additional information are chosen filenames
 *     header syntax:
 *         # name value \n
 *     This functions uses the intern function write_headerToFile()
 */
bool write_config(const char *filenameConfig);

/*
 *  Opens the given files and writes data according to the given
 *  @arrayType. This may be:
 *   - arrayType_nGram
 *     - arrayType_initialProb
 *     - arrayType_length
 *     This functions uses the intern function write_headerToFile() and
 *     write_arrayToFile().
 */
bool write_array(const char *filename, enum arrayTypes arrayType);

/*
 *     Prints the by arguments selected mode as well as the output and input filenames
 *     to the given file pointer @fp.
 */
void print_settings_createNG(FILE *fp);

/*
 * Sets @alphabet according to file under the filename @filename
 * and adjust @sizeOf_alphabet to the size of the new alphabet.
 * If there are problems opening or reading the file, an error
 * is set using the given @errorHandler.
 * If the application runs out of memory, it will be aborted.
 */
bool alphabetFromFile(char **alphabet, // pointer to the (old) alphabet
                      int *sizeOf_alphabet,  // pointer to the size of the (old) alphabet
                      const char *filename); // new alphabet

/*
 *  Appends any given @prefix, @suffix or the current date (if @dateSuffix is TRUE)
 *  as suffix to all output filenames.
 *  The allocated memory for the char* is freed by this function.
 */
bool append_prefixSuffix(char **prefix,  // prefix (NULL if none prefix should be set)
                         char **suffix,  // suffix (NULL if none prefix should be set)
                         bool dateSuffix,  // append current date
                         filename_struct *filenames); // filenames the pre-/suffixes should be append to
void write_headerToFile (char *title, FILE * fp);
}



int node_type;
SocketManager::SMMLF tel;
int globalRound;

// file pointer
char glbl_resultsFolder[256] = { '\0' };
bool write_grammars (const char *filename, enum arrayTypes arrayType, Mat countArray, Mat levelArray);
bool write_grammarToFile (Mat countArray,  // containing the nGrams
                          Mat levelArray,    // containing the levels
                          int sizeOf_nGramArray,  // size of the given array
                          int sizeOf_N, // must be equal to the nGram-size of the nGrams stored in array
                          unsigned long long int totalSum,  // total sum of all counts in the given array
                          enum writeModes writeMode,  // write Mode - numeric, nGram or nonVerbose
                          FILE * fp_count,  // file pointer (must point to an opened file) for count
                          FILE * fp_level);  // file pointer (must point to an opened file) for level


inline double fastPow(double a, double b) {
    union {
        double d;
        int x[2];
    } u = { a };
    u.x[1] = (int)(b * (u.x[1] - 1072632447) + 1072632447);
    u.x[0] = 0;
    return u.d;
}

inline double fast_exp(double y) {
    double d;
    *(reinterpret_cast<int*>(&d) + 0) = 0;
    *(reinterpret_cast<int*>(&d) + 1) = static_cast<int>(1512775 * y + 1072632447);
    return d;
}
int test_train_NGram();
int test_eval_Pwds();

int main(int argc, char *argv[]) {
    srand(time(NULL));
    // let's call our cmdline parser
    if (cmdline_parser(argc, argv, &glbl_args_info) != 0) {
        printf("failed parsing command line arguments\n");
        exit (EXIT_FAILURE);
    }
    char *_glbl_result_folder = "--player";
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], _glbl_result_folder, strlen(_glbl_result_folder)) == 0) {
            i += 1;
            node_type = atoi(argv[i]);
            printf("Node type: %d\n", node_type);
        }
    }
    cout << "test fast pow\n";
    cout << fastPow(2*IE, 3) << endl;
    cout << fast_exp(2) <<endl;
    
    Player::init();

    ll128 inverse = Constant::Util::inverse(2, MOD);
    cout << "Inverse: " << inverse << endl;
    cout << "Resss: " << Constant::Util::get_residual(inverse * 4) << endl;

    tel.init();

    test_train_NGram();
}                               // main

int test_train_NGram() {
    // set exit_routine so thats automatically called
    atexit(exit_routine);

    // initialize global parameters
    initialize();

    // evaluate given arguments
    if (!evaluate_arguments(&glbl_args_info)) {
        exit (1);
    }

    // print selected mode and filenames
    if (glbl_verboseMode)
        print_settings_createNG(stdout);

    // run nGram creation
    if (!run_creation())
        exit (1);

    for (int i = 0; i < glbl_nGramCount->sizeOf_iP; ++i) {
        if (glbl_nGramCount->iP[i] > 0)
            printf("%d: %d\n", i, glbl_nGramCount->iP[i]);
    }

    // load corresponding shares
    Mat* ip_data = IOManager::secret_share_ngram(glbl_nGramCount->iP, glbl_nGramCount->sizeOf_iP, "IP");
    Mat* cp_data = IOManager::secret_share_ngram(glbl_nGramCount->cP, glbl_nGramCount->sizeOf_cP, "CP");
    Mat* ln_data = IOManager::secret_share_ngram(glbl_nGramCount->len, glbl_nGramCount->sizeOf_len, "LN");
    Mat* ep_data = IOManager::secret_share_ngram(glbl_nGramCount->eP, glbl_nGramCount->sizeOf_eP, "EP");
    for (int j = 0; j < M; ++j) {
//        ip_data[j].print();
    }

    ll total;
    Mat seg_total(KEY_NUM, glbl_nGramCount->sizeOf_cP);

    /**
     * IP grammar
     * **/
    Mat count_data(KEY_NUM, glbl_nGramCount->sizeOf_iP);
    count_data.clear();
    NgramGraph::Ngram *ng = new NgramGraph::Ngram(ip_data);
    ng->count_graph();
    ng->forward_count(&total, &count_data, &seg_total);
//    count_data.print();
//    seg_total = seg_total / IE + ALPHABET_SIZE * 1;
//    seg_total.print();
    // return 0; 
    DBGprint("Total: %lld\n", total/IE);
    total = total/IE+ALPHABET_SIZE*ALPHABET_SIZE;
    Mat level_data(KEY_NUM, glbl_nGramCount->sizeOf_iP);
    ng->smooth_level_graph(total, 250, false);
    ng->forward_smooth(count_data, seg_total,  IE, 1, &level_data);
    // write iP count and levels to disk
    level_data.print();
    // return 0;
    write_grammars(glbl_filenames->iP, arrayType_initialProb, count_data, level_data);

    /**
     * CP grammar
     * **/
    Mat count_data_cp(KEY_NUM, glbl_nGramCount->sizeOf_cP);
    count_data_cp.clear();
    NgramGraph::Ngram *ng_cp = new NgramGraph::Ngram(cp_data);
    ng_cp->count_graph();
    ng_cp->forward_count(&total, &count_data_cp, &seg_total);
    // count_data_cp.print();
    seg_total = seg_total / IE + ALPHABET_SIZE * 1;
    seg_total.print();
    total = total/IE+ALPHABET_SIZE*ALPHABET_SIZE*ALPHABET_SIZE;

    Mat level_data_cp(KEY_NUM, glbl_nGramCount->sizeOf_cP);
    ng_cp->smooth_level_graph(total, 2, true);
    ng_cp->forward_smooth(count_data_cp, seg_total,  IE, 1, &level_data_cp);
    level_data_cp.print();

    write_grammars(glbl_filenames->cP, arrayType_conditionalProb, count_data_cp, level_data_cp);

    /**
     * EP grammar
     * **/

    // total = 0;
    // Mat count_data_ep(KEY_NUM, glbl_nGramCount->sizeOf_eP);
    // count_data_ep.clear();
    // NgramGraph::Ngram *ng_ep = new NgramGraph::Ngram(ep_data);
    // ng_ep->count_graph();
    // ng_ep->forward_count(&total, &count_data_ep, &seg_total);
    // count_data_ep.print();
    // DBGprint("Total: %lld\n", total/IE);

    // Mat level_data_ep(KEY_NUM, glbl_nGramCount->sizeOf_eP);
    // ng_ep->smooth_level_graph(total/IE+100, 250, false);
    // ng_ep->forward_smooth(count_data_ep, seg_total,  IE, 1, &level_data_ep);
    // level_data_ep.print();

    // write_grammars(glbl_filenames->eP, arrayType_endProb, count_data_ep, level_data_ep);

    /**
     * LN grammar
     * **/
    // total = 0;
    // Mat count_data_len(KEY_NUM, glbl_nGramCount->sizeOf_len);
    // count_data_len.clear();
    // NgramGraph::Ngram *ng_len = new NgramGraph::Ngram(ln_data);
    // ng_len->count_graph();
    // ng_len->forward_count(&total, &count_data_len, &seg_total);
    // count_data_len.print();
    // DBGprint("Total: %lld\n", total/IE);

    // Mat level_data_len(KEY_NUM, glbl_nGramCount->sizeOf_len);
    // ng_len->smooth_level_graph(total/IE, 1, false);
    // ng_len->forward_smooth(count_data_len, seg_total,  0, 1, &level_data_len);
    // level_data_len.print();

    // write_grammars(glbl_filenames->len, arrayType_length, count_data_len, level_data_len);

    // persist grammar files
    if (!write_config ((glbl_filenames->cfg)))
        return 0;
    exit (EXIT_SUCCESS);
}

int test_eval_Pwds() {
    DBGprint("----------    Test Eval Pwds  ----------\n");
    ifstream in("test.txt");
    string line;

    if (in.is_open()) {
        while (getline(in, line)){
            
        }
        
    }
    Mat* input_pwds = IOManager::secret_share_ngram(glbl_nGramCount->iP, glbl_nGramCount->sizeOf_iP, "IP");
    return 1;
}

bool write_grammarToFile (Mat countArray,  // containing the nGrams
                        Mat levelArray,    // containing the levels
                        int sizeOf_nGramArray,  // size of the given array
                        int sizeOf_N, // must be equal to the nGram-size of the nGrams stored in array
                        unsigned long long int totalSum,  // total sum of all counts in the given array
                        enum writeModes writeMode,  // write Mode - numeric, nGram or nonVerbose
                        FILE * fp_count,  // file pointer (must point to an opened file) for count
                        FILE * fp_level)  // file pointer (must point to an opened file) for level
{
    char nGram[sizeOf_N];
    nGram[sizeOf_N] = '\0';

    // write according to write mode
    switch (writeMode)
    {
        case writeMode_nGram:        // write actual nGrams as well
            for (size_t i = 0; i < sizeOf_nGramArray; i++)
            {
                // get the actual nGram based on the current position to print to the file
                get_nGramFromPosition (nGram, i, sizeOf_N, (glbl_alphabet->sizeOf_alphabet), (glbl_alphabet->alphabet));
                // smooth the level using the current smoothing function
                fprintf (fp_level, "%i\t%s\n", (int)levelArray.getVal(i), nGram);
                fprintf (fp_count, "%i\t%s\n", (int)countArray.getVal(i), nGram);
            }
            break;
        case writeMode_numeric:      // write array index as well
            for (size_t i = 0; i < sizeOf_nGramArray; i++)
            {
                fprintf (fp_level, "%i\t%lu\n", (int)levelArray.getVal(i), i + 1);
                fprintf (fp_count, "%i\t%lu\n", (int)countArray.getVal(i), i + 1);
            }
            break;
        default:                     // writeMode_nonVerbose or any other, just write level
            for (size_t i = 0; i < sizeOf_nGramArray; i++)
            {
                fprintf (fp_level, "%i\n", (int)levelArray.getVal(i));
            }
            break;
    }

    if (fp_count != NULL && ferror (fp_count))
        return false;
    if (fp_level != NULL && ferror (fp_level))
        return false;

    return true;
}                               // (intern) write_arrayToFile
bool write_grammars (const char *filename, enum arrayTypes arrayType, Mat countArray, Mat levelArray)
{
    FILE *fp_count = NULL;
    FILE *fp_level = NULL;
    enum writeModes writeMode = writeMode_nonVerbose;

    // open files
    if (!(open_file (&fp_level, filename, DEFAULT_FILE_ATTACHMENT_LEVEL, "w")))
    {
        errorHandler_print (errorType_Error, "file not found %s\n", filename);
        return false;
    }
    if (glbl_verboseFileMode)
    {
        if (!(open_file (&fp_count, filename, DEFAULT_FILE_ATTACHMENT_COUNT, "w")))
        {
            errorHandler_print (errorType_Error, "file not found %s\n", filename);
            return false;
        }
    }

    switch (arrayType)
    {
        case arrayType_conditionalProb:
            // if verboseMode is active, set writeMode accordingly and write header to files
            if (glbl_verboseFileMode)
            {
                write_headerToFile ("CP-COUNTS", fp_count);
                write_headerToFile ("CP-LEVELS", fp_level);
                writeMode = writeMode_nGram;
            }
            write_grammarToFile(countArray, levelArray, (glbl_nGramCount->sizeOf_cP), (glbl_nGramCount->sizeOf_N), glbl_countSums[arrayType_conditionalProb], writeMode, fp_count, fp_level);

            break;
        case arrayType_initialProb:
            // if verboseMode is active, set writeMode accordingly and write header to files
            if (glbl_verboseFileMode)
            {
                write_headerToFile ("IP-COUNTS", fp_count);
                write_headerToFile ("IP-LEVELS", fp_level);
                writeMode = writeMode_nGram;
            }
            write_grammarToFile (countArray, levelArray, (glbl_nGramCount->sizeOf_iP), (glbl_nGramCount->sizeOf_N) - 1, glbl_countSums[arrayType_initialProb], writeMode, fp_count, fp_level);
            break;
        case arrayType_endProb:
            // if verboseMode is active, set writeMode accordingly and write header to files
            if (glbl_verboseFileMode)
            {
                write_headerToFile ("EP-COUNTS", fp_count);
                write_headerToFile ("EP-LEVELS", fp_level);
                writeMode = writeMode_nGram;
            }

            write_grammarToFile (countArray, levelArray, (glbl_nGramCount->sizeOf_eP), (glbl_nGramCount->sizeOf_N) - 1, glbl_countSums[arrayType_endProb], writeMode, fp_count, fp_level);
            break;
        case arrayType_length:
            // if verboseMode is active, set writeMode accordingly and write header to count file
            if (glbl_verboseFileMode)
            {
                write_headerToFile ("LN-COUNTS", fp_count);
                write_headerToFile ("LN-LEVELS", fp_level);
                writeMode = writeMode_numeric;
            }
            // write header and levels to file
            write_grammarToFile (countArray, levelArray, (glbl_nGramCount->sizeOf_len), 1, glbl_countSums[arrayType_length], writeMode, fp_count, fp_level);
            break;

        default:
            errorHandler_print (errorType_Error, "Unknown array type.\n");
            if (fp_count != NULL)
            {
                fclose (fp_count);
                fp_count = NULL;
            }
            if (fp_level != NULL)
            {
                fclose (fp_level);
                fp_level = NULL;
            }
            return false;
    }

    // clean up
    if (fp_count != NULL)
    {
        fclose (fp_count);
        fp_count = NULL;
    }
    if (fp_level != NULL)
    {
        fclose (fp_level);
        fp_level = NULL;
    }
    return true;
}                               // write_array