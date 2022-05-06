#ifndef MPC_GENERIC_IOMANAGER_H
#define MPC_GENERIC_IOMANAGER_H
#include <cassert>
#include "../Mat.h"
#include "../Constant.h"
#include <sstream>
#include <ctime>
#include "MD5.h"
extern int node_type;
enum Handle_Case { DATE2LL, NUMBER2LL, STRING2LL };
class GenericIOManager
{
private:
    /* data */
public:
    GenericIOManager(/* args */) {

    }
    ~GenericIOManager(){

    }
    static void load_plaintext(ifstream& in, Mat& data, Handle_Case* cases);
    static Mat* secret_share(Mat& data, string store_path, string store_prefix);
    static void load_encrypted(ifstream& in, Mat& data);

    /* Util */
    static ll128 transfer2ll(string str, Handle_Case cur_case);
    static time_t date_str2timestamp(string date);
    static string timestamp2date_str(time_t date);
    static ll128 digit_str2ll(string number);
    static string ll2digit_str(ll128 number);
    static ll128 string2ll(string str);
    static string ll2string(ll128 str);
    static ll poly_eval(vector<ll>coefficients, ll x);
};

#endif //MPC_GENERIC_IOMANAGER_H