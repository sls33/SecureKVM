#ifndef SMMLF_BASICCART_H
#define SMMLF_BASICCART_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <iterator>
#include <cmath>

using namespace std;

class BasicCART {

};
class C4_5{
public:
    int getAttr(vector <int> *data,vector <int> index,vector<int> attr){
        double pos=0,neg=0;
        int len = index.size();
        for(int i=0;i<len;i++){ 
            int a = index[i];
            if(data[a][data[a].size()-1]==1) pos++;
            else neg++;
        }
        
        double H = -xlog2(pos/(len*1.0))-xlog2(neg/(len*1.0));
        
        double *gain = new double[attr.size()];
        double *splitinfo = new double[attr.size()];
        for(int k=0;k<attr.size();k++){
            int i = attr[k];
            double attr_pos[50]={0};
            double attr_neg[50]={0};
            for(int j=0;j<len;j++){ 
                int a = index[j];
                if(data[a][data[a].size()-1]==1) attr_pos[data[a][i]]++;
                else attr_neg[data[a][i]]++;
            }
            gain[k] = 0;
            splitinfo[k] = 0;
            for(int j=0;j<50;j++) { 
                if(attr_pos[j]!=0 || attr_neg[j]!=0){
                    double p_sum = attr_pos[j]+attr_neg[j];
                    splitinfo[k]+=-xlog2(p_sum/(len*1.0));
                    gain[k]+=p_sum/(len*1.0)*(-xlog2(attr_pos[j]/p_sum)
                                              -xlog2(attr_neg[j]/p_sum));
                }

            }
            
            gain[k]= (H - gain[k])/splitinfo[k];
        }
        double MAX = -99999;
        int position;
        for(int i=0;i<attr.size();i++){
            if(gain[i]>MAX)	{
                MAX = gain[i];
                position = attr[i];
            }
        }
        return position;
    }
private:
    double xlog2(double a){
        if(a==0) return 0;
        return a*log(a)/log(2);
    }
};

class CART{
public:
    int getAttr(vector <int> *data,vector <int> index,vector<int> attr){
        double pos=0,neg=0;
        int len = index.size();
        for(int i=0;i<len;i++){ 
            int a = index[i];
            if(data[a][data[a].size()-1]==1) pos++;
            else neg++;
        }
        
        double *gini = new double[attr.size()];
        for(int k=0;k<attr.size();k++){
            int i = attr[k];
            double attr_pos[50]={0};
            double attr_neg[50]={0};
            for(int j=0;j<len;j++){ 
                int a = index[j];
                if(data[a][data[a].size()-1]==1) attr_pos[data[a][i]]++;
                else attr_neg[data[a][i]]++;
            }
            gini[k] = 0;
            for(int j=0;j<50;j++) { 
                if(attr_pos[j]!=0 || attr_neg[j]!=0){
                    double p_sum = attr_pos[j]+attr_neg[j];
                    gini[k]=p_sum/(len*1.0)*(1-pow(attr_pos[j]/p_sum,2)-pow(attr_neg[j]/p_sum,2));
                }
            }
        }
        double MIN = 99999;
        int position;
        for(int i=0;i<attr.size();i++){
            if(gini[i]<MIN)	{
                MIN= gini[i];
                position = attr[i];
            }
        }
        return position;
    }
};

class ID3{
public:
    int getAttr(vector <int> *data,vector <int> index,vector<int> attr){
        double pos=0,neg=0;
        int len = index.size();
        for(int i=0;i<len;i++){ 
            int a = index[i];
            if(data[a][data[a].size()-1]==1) pos++;
            else neg++;
        }
        double H = -xlog2(pos/(len*1.0))-xlog2(neg/(len*1.0));
        
        double *gain = new double[attr.size()];
        for(int k=0;k<attr.size();k++){
            int i = attr[k];
            double attr_pos[50]={0};
            double attr_neg[50]={0};
            for(int j=0;j<len;j++){ 
                int a = index[j];
                if(data[a][data[a].size()-1]==1) attr_pos[data[a][i]]++;
                else attr_neg[data[a][i]]++;
            }
            gain[k] = 0;
            
            for(int j=0;j<50;j++) { 
                if(attr_pos[j]!=0 || attr_neg[j]!=0){
                    double p_sum = attr_pos[j]+attr_neg[j];
                    gain[k]+=p_sum/(len*1.0)*(-xlog2(attr_pos[j]/p_sum)
                                              -xlog2(attr_neg[j]/p_sum));
                }

            }
        }
        double MIN = 99999;
        int position;
        for(int i=0;i<attr.size();i++){ 
            cout<<H-gain[i]<<" ";
            if(gain[i]<MIN)	{
                MIN = gain[i];
                position = attr[i];
            }
        }
        cout<<endl;
        return position;
    }
private:
    double xlog2(double a){
        if(a==0) return 0;
        return a*log(a)/log(2);
    }
};
#endif //SMMLF_BASICCART_H
