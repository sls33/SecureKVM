#include "omen_evaluate.h"


extern "C" {
#include "../comen/evalPW.h"
#include "../comen/common.h"
#include "../comen/evalPW.h"
#include "../comen/commonStructs.h"
#include "../comen/errorHandler.h"
#include "../comen/nGramReader.h"
#include "../comen/cmdlineEvalPW.h"
extern struct filename_struct *glbl_filenamesIn;
extern struct alphabet_struct *glbl_alphabet;
extern struct nGram_struct *glbl_nGramLevel;

extern char glbl_maxLevel;
extern char *glbl_password;
extern bool glbl_verboseMode;
extern struct gengetopt_args_info glbl_args_info;
}
double log_poly(ll x);
int main (int argc, char **argv)
{
    // let's call our cmdline parser
    if (cmdline_parser (argc, argv, &glbl_args_info) != 0)
    {
        printf ("failed parsing command line arguments\n");
        exit (EXIT_FAILURE);
    }
    // set exit_routine so thats automatically called
    atexit (exit_routine);

    initialize ();

    if (!evaluate_arguments (&glbl_args_info))
        exit (1);

    if (!apply_settings ())
        exit (1);

    if (glbl_verboseMode)
        print_settings ();
    for (int j = 0; j < glbl_nGramLevel->sizeOf_len; ++j) {
        printf("%d: %d\n", j, glbl_nGramLevel->len[j]);
    }
    printf("%s\n", glbl_password);
    if (!run_evaluation ())
        exit (1);
    printf("%f\n", log(10));
    printf("%f\n", log_poly(10));
    exit (EXIT_SUCCESS);
}
double log_poly(ll x) {
    double res = 0;
    res = x - pow(x, 2) /2 + pow(x, 3) /3 - pow(x, 4) /4;
    printf("Poly: %f", res);
}