#include "BasicCART.h"

using namespace std;

int USE_ID3 = 0;
int USE_C4_5 = 0;
int USE_CART = 0;
const int ATTR_NUM = 4;

typedef struct node {  
    int label = 0;
    int neg; 
    int pos; 
    int attr;  
    vector <int> attr_value;
    node* children[50] = {NULL}; 
}node;

set<int> *Attr_value = new set<int>[ATTR_NUM];  
node *root;	
vector<int> Data[1000];	//the dataset
int length; 
vector<int> attr; 
int current_pos = 0; 

void init() //initialize the root node and dataset and the attr set
{
    /* build data set */
    fstream fin("train.csv");
    string line;
    int m = 0;
    while(getline(fin,line)){
        istringstream sin(line);
        string field;
        int n = 0;
        while(getline(sin,field,',')){
            Data[m].push_back(atoi(field.c_str()));
            if(n!=ATTR_NUM) Attr_value[n].insert(atoi(field.c_str()));
            n++;
        }
        m++;
    }
    length = m;
    for(int i=0;i<ATTR_NUM;i++) attr.push_back(i);
    for (int k = 0; k < 10; ++k) {
        for (int i = 0; i < Data[k].size(); ++i) {
            cout << i << ": " << Data[k][i] << endl;
        }
    }
    for (int j = 0; j < attr.size(); ++j) {
        cout << j << ": " << attr[j] << endl;
    }
    set<int> :: iterator it;
}

bool meet_with_bound(vector<int> index)
{
    int len = index.size();
    int pos = 0;
    int neg = 0;
    if(len==0)	return true; 
    for(int i=0;i<len;i++){
        if(Data[index[i]][ATTR_NUM]==1)  pos++;
        else neg++;
    }
    if(neg==0 || pos==0)  return true; 
    if(attr.size()==0) return true; 
    for(int i=0;i<attr.size();i++){
        int a = Data[index[0]][attr[i]]; 
        for(int j=1;j<len;j++)	if(Data[index[j]][attr[i]]!=a) return false;
    }
    return true;
}

int choose_attr(vector<int> index)
{
    if(USE_ID3)
    {
        ID3 id3;
        return id3.getAttr(Data,index,attr);
    }
    else if(USE_C4_5)
    {
        C4_5 c4_5;
        return c4_5.getAttr(Data,index,attr);
    }
    else if(USE_CART)
    {
        CART cart;
        return cart.getAttr(Data,index,attr);
    }
}


vector<vector<int> > divide_data(int a,vector<int> index)
{
    int col = a;
    vector <vector<int>> v;
    vector <int> indexSet[50];
    vector <vector<int> > v2;
    set<int> :: iterator it;
    int m = 0;
    for(it=Attr_value[col].begin();it!=Attr_value[col].end();it++){
        for(int i=0;i<index.size();i++)	if(*it==Data[index[i]][col])  indexSet[m].push_back(index[i]);
        v.push_back(indexSet[m]);
        indexSet[m].clear(); 
        m++;
    }
    return v;
}

void recursive(node *p,vector <int> index)
{
    int attr_chosen = choose_attr(index);
    vector<vector<int> > subsets = divide_data(attr_chosen,index);
    p->attr = attr_chosen;
    for(int i=0;i<attr.size();i++) 
        if(attr[i]==attr_chosen)
            attr.erase(attr.begin()+i,attr.begin()+i+1);
    set<int> :: iterator it;
    
    for(it=Attr_value[attr_chosen].begin();it!=Attr_value[attr_chosen].end();it++)
        p->attr_value.push_back(*it);
    it = Attr_value[attr_chosen].begin();
    
    int pos=0,neg=0;
    for(int i=0;i<index.size();i++){
        if(Data[index[i]][ATTR_NUM]==1) pos++;
        else neg++;
    }
    p->pos=pos; p->neg = neg;
    
    for(int i=0;i<subsets.size();i++,it++)
    {
        node *subnode = new node;
        /* recursion */
        p->children[*it] = subnode;
        if(meet_with_bound(subsets[i])){
            if(subsets[i].size()==0){
                if(p->pos >= p->neg) subnode->label = 1;
                else 	subnode->label =  -1;
                subnode->pos = 0;
                subnode->neg = 0;
                continue;
            }
            else{	
                pos=0,neg = 0;
                for(int j=0;j<subsets[i].size();j++){
                    if(Data[subsets[i][j]][ATTR_NUM]==1) pos++;
                    else neg++;
                }
                if(pos >= neg)	subnode->label = 1;
                else	subnode->label =  -1;
                subnode->pos = pos;
                subnode->neg = neg;
                continue;
            }
        }
        recursive(subnode,subsets[i]);
    }
    attr.push_back(attr_chosen);
    subsets.clear();
    return;
}

void validation(vector<int> index){

    fstream fout("result.txt",ios::out);
    for(int i=0;i<length;i++){
        index.erase(index.begin(),index.begin()+1);	
        root = new node; 
        recursive(root,index);
        node* n = root;
        while(n->label==0){
            int attr_value = Data[i][n->attr];
            n = n->children[attr_value];
        }
        fout<<n->label<<"\n";
        index.push_back(i);
    }
}



void predict(){
    vector<int> index;
    for(int i=0;i<length;i++) index.push_back(i);
    root = new node;
    recursive(root,index);
    node* n = root;
    fstream fin("test.csv");
    fstream fout("test_result.txt",ios::out);
    string line;
    int m = 0;
    vector<int> Data_Test[1000];	//the test dataset
    while(getline(fin,line)){
        istringstream sin(line);
        string field;
        while(getline(sin,field,',')){
            Data_Test[m].push_back(atoi(field.c_str()));
        }
        node* n = root;
        int flag = 0;
        while(n->label==0){
            int attr_value = Data_Test[m][n->attr];
            if(n->children[attr_value]==NULL){
                flag = 1;
                if(n->pos >= n->neg) fout<<1<<"\n";
                else fout<<-1<<"\n";
                break;
            }
            n = n->children[attr_value];
        }
        if(flag==0) fout<<n->label<<"\n";
        m++;
    }
}

int main(){
    cout<<"ID3: 0"<<endl;
    cout<<"C4_5: 1"<<endl;
    cout<<"CART: 2"<<endl;
    cout<<"Select type: "<<endl;
    int a;
    cin >> a;
    if(a==0) USE_ID3 = 1;
    else if(a==1) USE_C4_5 = 1;
    else USE_CART = 1;
    init();
    vector<int> index;
    for(int i=0;i<length;i++) index.push_back(i);
    predict();
}