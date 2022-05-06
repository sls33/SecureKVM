#ifndef SMMLF_DT_MAIN_H
#define SMMLF_DT_MAIN_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <inttypes.h>
#include <time.h>
#include <getopt.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <map>
#include <sstream>

#include "../util/SocketManager.h"
#include "../Player.h"
#include "../util/IOManager.h"
#include "DT_graph.h"

#define LOG false

class DT_main {

};

class Table {
public:
    vector<string> attrName;
    vector<vector<string> > data;

    vector<vector<string> > attrValueList;
    double total_entropy = 0;
    void extractAttrValue() {
        // todo. modify to decresizenalized
        attrValueList.resize(attrName.size());
        for(int j=0; j<attrName.size(); j++) {
            map<string, int> value;
            for(int i=0; i<data.size(); i++) {
                value[data[i][j]]=1;
            }

            for(auto iter=value.begin(); iter != value.end(); iter++) {
                attrValueList[j].push_back(iter->first);
            }
        }
        for (int k = 0; k < attrValueList.size(); ++k) {
            for (int i = 0; i < attrValueList[k].size(); ++i) {
                if (LOG) cout << attrValueList[k][i] << ", ";
            }
            if (LOG) cout << endl;
        }
    }
    void getEntropy() {
        total_entropy = 0.0;

        int total_count = (int)data.size();
        map<string, int> labelCount;

        for(int i=0;i<data.size();i++) {
            labelCount[data[i].back()]++;
        }

        for(auto iter=labelCount.begin(); iter != labelCount.end(); iter++) {
            double p = (double)iter->second/total_count;
            total_entropy += -1.0 * p * log(p)/log(2);
        }
    }
};

class Node {
public:
    int criteriaAttrIndex;
    string attrValue;

    int treeIndex;
    bool isLeaf;
    string label;

    vector<int > children;

    Node() {
        isLeaf = false;
    }
};

class InputReader {
private:
    ifstream fin;
    Table table;
public:
    InputReader(string filename) {
        fin.open(filename);
        if(!fin) {
            cout << filename << " file could not be opened\n";
            exit(0);
        }
        parse();
    }
    void parse() {
        string str;
        bool isAttrName = true;
        while(getline(fin, str)){
            vector<string> row;
            istringstream sin(str);
            string col;
            while (getline(sin, col,',')) {
                row.push_back(col);
            }

            if(isAttrName) {
                table.attrName = row;
                isAttrName = false;
            } else {
                table.data.push_back(row);
            }
        }
        table.extractAttrValue();
        table.getEntropy();
    }
    Table getTable() {
        return table;
    }
};
#endif //SMMLF_DT_MAIN_H
