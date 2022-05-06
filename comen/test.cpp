#include "test.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <inttypes.h>
#include <time.h>
#include <getopt.h>
#include <string.h>
#include <math.h>

#include "cmdlineCreateNG.h"
#include "common.h"
#include "commonStructs.h"

#include "smoothing.h"

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
extern "C" {
    #include "createNG.h"
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
}

int main(int argc, char *argv[]) {
    // let's call our cmdline parser
    if (cmdline_parser(argc, argv, &glbl_args_info) != 0) {
        printf("failed parsing command line arguments\n");
        exit (EXIT_FAILURE);
    }
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

    exit (EXIT_SUCCESS);
}                               // main

