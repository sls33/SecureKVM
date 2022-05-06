#include "IOManager.h"

//Mat train_data(N,D), train_label(N,1);
//Mat test_data(NM,D), test_label(NM,1);

Mat IOManager::train_data = Mat(D+1, N + B - 1);
Mat IOManager::train_label = Mat(1, N + B - 1);
Mat IOManager::test_data = Mat(D+1, NM + B - 1);
Mat IOManager::test_label = Mat(1, NM + B - 1);

ll poly_eval(vector<ll>coefficients, ll x) {
    ll res = coefficients[TN-1];
    for (int l = TN-2; l >= 0; --l) {
        res = ((res * x + coefficients[l]) % MOD + MOD) % MOD;
    }
    return res;
}
void IOManager::load(ifstream &in, Mat &data, Mat &label, int size) {
    int i=0;
    while(in){

        string s;
        if (!getline(in,s))
            break;
        char* ch;
        ch = const_cast<char *>(s.c_str());
        int temp;
        char c;

        temp = Constant::Util::getint(ch);
        if (temp > 1)
            temp = 1;
        label(0, i) = temp * IE;



        int nd = min(D, ND);
        data(nd, i) = IE;
        for(int j=0;j<nd;j++){
            temp = Constant::Util::getint(ch);
            data(j, i) = temp * IE / 256;
//            if (!i) {
//                printf("%d ", temp);
//                data(i, j).print();
//                DBGprint(" ");
//            }
        }
//        if (!i)
//            printf("\n");

        i++;
        if (i >= size)
            break;
//        if (i >= 5)
//            break;
//            printf("%d ", i);
//        DBGprint("%d ", i);
    }
//    cout<<"n= "<<i<<endl;
    for (i; i < size + B - 1; i++) {
        int tmp_r;
        tmp_r = data.rows();
        for (int j = 0; j < tmp_r; j++) {
            data(j, i) = data(j, i - size);
        }
        tmp_r = label.rows();
        for (int j = 0; j < tmp_r; j++) {
            label(j, i) = label(j, i - size);
        }
    }
    DBGprint("n=%d\n", i);
}

void IOManager::secret_share(Mat &data, Mat &label, string category) {
    vector<ll> coefficients(TN);
//    assert(size*D == data.size());
    vector<ofstream> out_files(M);
    for (int i = 0; i < M; ++i) {
//        ofstream out_file("mnist/mnist_train_"+to_string(M), ios::out);
//        out_files.push_back(out_file);
//        string s = "mnist/mnist_train_"+to_string(i)+".csv";
//        cout << s << endl;
        out_files[i].open("mnist/mnist_"+category+"_"+to_string(i)+".csv", ios::out);
    }
    srand(time(NULL));

    int r = data.rows();
    int c = data.cols();
    cout << r << " : " << c << endl;
    for (int i = 0; i < c; ++i) {
        for (int j = 1; j < TN; ++j) {
            coefficients[j] = Constant::Util::randomlong();
        }
        for (int k = 0; k < M; ++k) {
            for (int j = 0; j < r; ++j) {
                if (j == 0) {
                    coefficients[0] = label.getVal(i);
                    cout << i << " : " << poly_eval(coefficients, k + 2) << endl;
                    out_files[k] << poly_eval(coefficients, k + 2) << ",";
                }
                if (j == r-1) {
                    coefficients[0] = data.get(j, i);
                    out_files[k] << poly_eval(coefficients, k + 2) << "\n";
                } else{
                    coefficients[0] = data.get(j, i);
                    out_files[k] << poly_eval(coefficients, k + 2) << ",";
                }
            }
//            out_files[k] << "\n";
        }
    }
//    for (int i = 0; i < size; ++i) {
//        for (int j = 1; j < TN; ++j) {
//            coefficients[j] = Constant::Util::randomlong();
//        }
//
//        if ( i % data.rows() == 0) {
//            for (int k = 0; k < M; ++k) {
//                if (i != 0) {
//                    out_files[k] << "asd\n";
//                    cout << i << endl;
//                }
//                coefficients[0] = label.getVal(i / data.rows());
//                out_files[k] << poly_eval(coefficients, k + 2) << ",";
//            }
//        }
//        for (int k = 0; k < M; ++k) {
//            coefficients[0] = data.getVal(i);
//            if (i%data.rows() == 0) {
//
//            }
//            out_files[k] << poly_eval(coefficients, k+2) << ",";
//        }
//
//    }
    for (int k = 0; k < M; ++k) {
        out_files[k].close();
    }

}

void IOManager::load_ss(ifstream &in, Mat &data, Mat &label, int size) {
    int i=0;
    while(in){

        string s;
        if (!getline(in,s))
            break;
        char* ch;
        ch = const_cast<char *>(s.c_str());
        ll temp;
        char c;

        temp = Constant::Util::getll(ch);
        label(0, i) = temp;

        int nd = min(D, ND);
        data(nd, i) = IE;
        for(int j=0;j<nd;j++) {
            temp = Constant::Util::getll(ch);
            data(j, i) = temp;
        }
        i++;
        if (i >= size)
            break;
    }
    for (i; i < size + B - 1; i++) {
        int tmp_r;
        tmp_r = data.rows();
        for (int j = 0; j < tmp_r; j++) {
            data(j, i) = data(j, i - size);
        }
        tmp_r = label.rows();
        for (int j = 0; j < tmp_r; j++) {
            label(j, i) = label(j, i - size);
        }
    }
    DBGprint("n=%d\n", i);
}

/** For Markov model **/
Mat* IOManager::secret_share_ngram(int* data, int size, string prefix) {
    vector<ll> coefficients(TN);
    vector<ofstream> out_files(M);
    cout << prefix << " size: " << size << endl;

    Mat* res = new Mat[M];
    for (int i = 0; i < M; ++i) {
        res[i].init(1, size);
        string filename = "output/"+prefix+"_"+to_string(i)+".csv";
        cout << "File: " << filename << endl;
        out_files[i].open(filename, ios::out);
    }
    
    for (int j = 1; j < TN; ++j) {
        coefficients[j] = Constant::Util::randomlong();
    }

    for (int i = 0; i < size; ++i) {
        for (int j = 1; j < TN; ++j) {
            coefficients[j] = Constant::Util::randomlong();
        }
        coefficients[0] = data[i]*IE;
        for (int k = 0; k < M; ++k) {
            ll tmp = poly_eval(coefficients, k + 2);
            out_files[k] << tmp << ",";
            res[k].setVal(i, tmp);
        }
    }
    return res;
}

Mat IOManager::load_secret_share_ngram(int* data, int size) {
    Mat res(size, 1);
    for (int i = 0; i < size; ++i) {
        res.setVal(i, data[i]);
    }
    return res;
}

Mat IOManager::load_secret_share_ngram(double* data, int size) {
    Mat res(size, 1);
    for (int i = 0; i < size; ++i) {
        res.setVal(i, (ll)(data[i]*IE));
    }
    return res;
}

/** For Decision Tree Bitmap share **/
Mat* IOManager::secret_share_vector(double* data, int size) {
    vector<ll> coefficients(TN);

    Mat* res = new Mat[M];
    
    for (int j = 1; j < TN; ++j) {
        coefficients[j] = Constant::Util::randomlong();
    }

    for (int l = 0; l < M; ++l) {
        res[l].init(size, 1);
    }
    for (int i = 0; i < size; ++i) {
        for (int j = 1; j < TN; ++j) {
            coefficients[j] = Constant::Util::randomlong();
        }
        coefficients[0] = (ll)(data[i]*IE);
        for (int k = 0; k < M; ++k) {
            ll tmp = poly_eval(coefficients, k + 2);
            res[k].setVal(i, tmp);
        }
    }
    return res;
}

/** For K-V statics **/
Mat* IOManager::secret_share_kv_data(Mat &data, int size, string prefix, bool isFreq) {
    vector<ll> coefficients(TN);
    vector<ofstream> out_files(M);
    cout << prefix << " size: " << size << endl;

    int r = data.rows();
    int c = data.cols();

    Mat* res = new Mat[M];
    for (int i = 0; i < M; ++i) {
        res[i].init(r, c);
        string filename = "output/"+prefix+"_"+to_string(i)+".csv";
        cout << "File: " << filename << endl;
        out_files[i].open(filename, ios::out);
    }

    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; j++) {
            for (int l = 1; l < TN; ++l) {
                coefficients[l] = Constant::Util::randomlong();
            }
            if (isFreq) {
                if (data.get(i,j) > 0) {
                    coefficients[0] = IE;
                } else {
                    coefficients[0] = 0;
                }
            } else {
                coefficients[0] = data.get(i,j);
            }

            for (int k = 0; k < M; ++k) {
                ll tmp = poly_eval(coefficients, k + 2);

                res[k].setVal(j*r+i, tmp);
                if (j == c-1) {
                    out_files[k] << tmp << "\n";
                } else{
                    out_files[k] << tmp << ",";
                }
            }
        }
        
        
    }
    return res;
}

/** For Decision Tree & Markov Evaluation **/
Mat* IOManager::secret_share_mat_data(Mat &data, int size, string prefix) {
    vector<ll> coefficients(TN);
    int r = data.rows();
    int c = data.cols();

    // Note: Hack and shall be refactored. prefix is "" means the input does not require to multiply IE.
    bool write_to_file = true;
    if (prefix.compare("") == 0) { // not flush to files
        write_to_file = false;
    }

    vector<ofstream> out_files(M);
    Mat* res = new Mat[M];
    for (int i = 0; i < M; ++i) {
        res[i].init(r, c);
        if (write_to_file) out_files[i].open("output/"+prefix+"_"+to_string(i)+".csv", ios::out);
    }

    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            coefficients[0] = write_to_file ? data.get(i, j)*IE : data.get(i, j);

            for (int l = 1; l < TN; ++l) {
                coefficients[l] = Constant::Util::randomlong();
            }
            for (int k = 0; k < M; ++k) {
                ll tmp = poly_eval(coefficients, k + 2);
                res[k].setVal(j*r+i, tmp);
                if (write_to_file) {
                    if (j == c-1) {
                        out_files[k] << tmp << "\n";
                    } else{
                        out_files[k] << poly_eval(coefficients, k + 2) << ",";
                    }
                }
                
            }
        }

    }
    if (write_to_file) {
        for (int k = 0; k < M; ++k) {
            out_files[k].close();
        }
    }
    
    return res;
}

void IOManager::init() {
    DBGprint("load training data......\n");

    ifstream infile( "mnist/mnist_train.csv" );
    load(infile, train_data, train_label, N);

    infile.close();

    ifstream intest( "mnist/mnist_test.csv" );
    load(intest, test_data, test_label, NM);

    intest.close();
}