#include "GenericIOManager.h"

ll GenericIOManager::poly_eval(vector<ll>coefficients, ll x) {
    ll res = coefficients[TN-1];
    for (int l = TN-2; l >= 0; --l) {
        res = res * x + coefficients[l];
    }
    return res;
}

void GenericIOManager::load_plaintext(ifstream &in, Mat &data, Handle_Case* cases) {
    int count=0;
    vector<vector<string>> tmp_data;
    while(in){
        string line;
        if (!getline(in,line))
            break;
        istringstream sin(line);
        vector<string> fields;
        string field;
        while (getline(sin, field, ',')) {
            fields.push_back(field);
        }
        count++;
        tmp_data.push_back(fields);
    }
    int field_size = tmp_data[0].size();
    data.init(count, field_size);
    cout << 1 << endl;
    char* ch;
    for (int i = 0; i <count; i++) {
        for(int j = 0;j < field_size; j++){
            data(i, j) = transfer2ll(tmp_data[i][j], cases[j]);
        }
    }
    data.print();
}

Mat* GenericIOManager::secret_share(Mat &data, string store_path, string store_prefix) {
    vector<ll> coefficients(TN);
    vector<ofstream> out_files(M);
    Mat* res = new Mat[M];

    int r = data.rows();
    int c = data.cols();
    for (int i = 0; i < M; ++i) {
        res[i].init(r, c);
        out_files[i].open(store_path+store_prefix+"_"+to_string(i)+".csv", ios::out);
    }
    srand(time(NULL));

    cout << r << " : " << c << endl;
    for (int i = 0; i < r; ++i) {
        for (int j = 1; j < TN; ++j) {
            coefficients[j] = Constant::Util::randomlong();
        }
        for (int k = 0; k < M; ++k) {
            for (int j = 0; j < c; ++j) {
                res[k].setVal(j*r+i, poly_eval(coefficients, k + 2));
                if (j == c-1) {
                    coefficients[0] = data.get(i, j);
                    out_files[k] << poly_eval(coefficients, k + 2) << "\n";
                } else{
                    coefficients[0] = data.get(i, j);
                    out_files[k] << poly_eval(coefficients, k + 2) << ",";
                }
            }
        }
    }
    for (int k = 0; k < M; ++k) {
        out_files[k].close();
    }
    return res;
}

void GenericIOManager::load_encrypted(ifstream &in, Mat &data) {
    int count=0;
    vector<vector<string>> tmp_data;
    while(in){
        string line;
        if (!getline(in,line))
            break;
        istringstream sin(line);
        vector<string> fields;
        string field;
        while (getline(sin, field, ',')) {
            fields.push_back(field);
        }
        count++;
        tmp_data.push_back(fields);
    }
    int field_size = tmp_data[0].size();
    data.init(count, field_size);
    char* ch;
    for (int i = 0; i <count; i++) {
        for(int j = 0;j < field_size; j++){
            data(i, j) = transfer2ll(tmp_data[i][j], NUMBER2LL);
        }
    }
}

time_t GenericIOManager::date_str2timestamp(string date) {
    struct tm tm;  
    memset(&tm, 0, sizeof(tm));  
      
    sscanf(date.c_str(), "%d/%d/%d  %d:%d:%d",   
           &tm.tm_year, &tm.tm_mon, &tm.tm_mday,  
           &tm.tm_hour, &tm.tm_min, &tm.tm_sec);  
  
    tm.tm_year -= 1900;  
    tm.tm_mon--;
    return mktime(&tm);
}

string GenericIOManager::timestamp2date_str(time_t date) {
    struct tm p;
    p = *localtime(&date);
    char time[100];
    strftime(time, sizeof(time),"%Y-%m-%d %H:%M:%S", &p);
    return time;
}

ll128 GenericIOManager::digit_str2ll(string number) {
    return (ll128) stod(number);
}

string GenericIOManager::ll2digit_str(ll128 number) {
    char buffer[ 128 ];
    char* d = std::end( buffer );
    do{
        -- d;
        *d = "0123456789"[ number % 10 ];
        number /= 10;
    } while ( number != 0 );
    return d;
}

ll128 GenericIOManager::string2ll(string str) {
    MD5 md5;
    md5.update(str);
    string code = md5.toString();
    ll128 num = 0;
    char* p = const_cast<char *>(code.c_str());
    int bytes = 12;
    while (bytes > 0) {
        // cout << *p;
        if (isdigit(*p)){
            num = 16 * num + *p - '0';
        } else if (islower(*p)) {
            num = num * 16 + *p - 'a' + 10;
        }
        p++;
        bytes--;
    }
    return num;
}

ll128 GenericIOManager::transfer2ll(string str, Handle_Case cur_case) {
    ll128 res;
    switch (cur_case) {
        case DATE2LL: 
            res = date_str2timestamp(str);
            break;
        case NUMBER2LL:
            res = digit_str2ll(str);
            break;
        case STRING2LL:
            res = string2ll(str);
            break;
        default:
            cout << "No Type.\n";
            res = 0;
            break;
    }
    return res;
}
