from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_iris
import csv
import pandas as pd
import graphviz
import sys
import lightgbm as lgb
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing

def load_data(data_file_name):
    """Loads data from module_path/data/data_file_name.

    Parameters
    ----------
    module_path : string
        The module path.

    data_file_name : string
        Name of csv file to be loaded from
        module_path/data/data_file_name. For example 'wine_data.csv'.

    Returns
    -------
    data : Numpy array
        A 2D array with each row representing one sample and each column
        representing the features of a given sample.

    target : Numpy array
        A 1D array holding target variables for all the samples in `data.
        For example target[0] is the target varible for data[0].

    target_names : Numpy array
        A 1D array containing the names of the classifications. For example
        target_names[0] is the name of the target[0] class.
    """
    with open(data_file_name) as input_file:
        attributes_line = input_file.readline()
        features = attributes_line.strip('\r\n').split(',')
        
        feature_names = []
        target_names = []
        print('index: ', index)
        if (index == 0):
            # breast cancer
            feature_names = features[1:-1]
            target_names = features[-1:]
        elif index == 1:
            # parkinsons
            features.remove('name')
            features.remove('status')
            feature_names = features
            target_names = ['status']
        elif index == 2:
            # car
            print('car')
            feature_names = features[0:-1]
            target_names = features[-1:]
        elif index == 3:
            # spambase
            print('spambase')
            feature_names = [str(i) for i in range(57)]
            target_names = ['class']
        elif index == 4:
            # wine
            print('wine')
            feature_names = [str(i) for i in range(13)]
            target_names = ['class']

        n_features = len(feature_names)
        print(feature_names)
        print(target_names)

        samples = input_file.readlines()
        n_samples = len(samples)
        print("feature: ", n_features, ", size: ", n_samples)

        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for i, line in enumerate(samples):
            line = line.replace('?', 'nan')
            segs = line.strip('\r\n').split(',')


            if index == 0:
                # breast cancer
                data[i] = np.asarray(segs[1:-1], dtype=np.float)
                target[i] = np.asarray(segs[-1], dtype=np.int)
            elif index == 1:
                # parkinsons
                cur = segs[1:-7]
                cur.extend(segs[-6:])
                data[i] = np.asarray(cur, dtype=np.float)
                target[i] = np.asarray(segs[-7], dtype=np.int)
            elif index == 2:
                # car, process the data to be categorical
                kvs = [
                    ['vhigh', 'high', 'med', 'low'],
                    ['vhigh', 'high', 'med', 'low'],
                    ['2', '3', '4', '5more'],
                    ['2', '4', 'more'],
                    ['small', 'med', 'big'],
                    ['low', 'med', 'high']
                ]
                target_kv = ['unacc', 'acc', 'good', 'vgood']
                tmp = np.zeros(len(segs)-1)
                for j in range(len(segs)-1):
                    tmp[j] = kvs[j].index(segs[j]) + 1
                data[i] = np.asarray(tmp, dtype=np.float)
                target[i] = np.asarray(target_kv.index(segs[-1]), dtype=np.int)
            elif index == 3:
                data[i] = np.asarray(segs[0:-1], dtype=np.float)
                target[i] = np.asarray(segs[-1], dtype=np.int)
            elif index == 4:
                data[i] = np.asarray(segs[1:], dtype=np.float)
                target[i] = np.asarray(segs[0], dtype=np.int)

    return data, target, feature_names, target_names

def convert_data_dataframe(caller_name, data, target,
                            feature_names, target_names):
    data_df = pd.DataFrame(data, columns=feature_names)
    data_df = data_df.fillna(1)
    target_df = pd.DataFrame(target, columns=target_names)
    combined_df = pd.concat([data_df, target_df], axis=1)
    print(1)
    X = combined_df[feature_names]
    y = combined_df[target_names]
    if y.shape[1] == 1:
        y = y.iloc[:, 0]
    return combined_df, X, y
index = 0
if __name__ == '__main__':
    base_path = "../datasets/"
    datasets = ['breast-cancer/breast-cancer-wisconsin.data', 'parkinsons/parkinsons.data', 'car/car.data', 'spambase/spambase.data', 'wine/wine.data']
    index = int(sys.argv[1])
    raw_data = load_data(base_path + datasets[index])
    frame, data, target = convert_data_dataframe('data', raw_data[0], raw_data[1], raw_data[2], raw_data[3])
    xtrain, xtest, ytrain, ytest = train_test_split(data, target, test_size=0.2)
    print(xtrain.shape)
    print(data.columns)
    print(ytrain.shape)
    
    # lgb
    # train_data = lgb.Dataset(data=xtrain, label=ytrain, feature_name=['buying','maint','doors','persons','lug_boot','safety'], categorical_feature=['buying','maint','doors','persons','lug_boot','safety'])
    # test_data = lgb.Dataset(data=xtest, label=ytest, feature_name=['buying','maint','doors','persons','lug_boot','safety'], categorical_feature=['buying','maint','doors','persons','lug_boot','safety'])
    # print(xtrain)
    # print(ytrain)
    # params = {'num_class': 4, 'max_depth':6, 'objective':'multiclass'}
    # bst = lgb.train(params, train_data, valid_sets=[train_data, test_data])

    # y_prob = bst.predict(xtest, num_iteration=bst.best_iteration)
    # y_pred = [list(x).index(max(x)) for x in y_prob]
    # print(y_pred)
    # print(ytest)
    # print('Accuracy: ', metrics.accuracy_score(y_pred, ytest))
    # graph = lgb.create_tree_digraph(bst, tree_index=1, name='dt_car_lgb_6')
    # graph.render(view=True)

    # sklearn model
    # vec = DictVectorizer()
    # dummyX = vec.fit(xtrain)
    # lb = preprocessing.LabelEncoder()
    # dummyY = lb.fit(ytrain)
    clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=7)
    clf = clf.fit(xtrain, ytrain)
    score = clf.score(xtest, ytest)
    print('Score: ', score)
    

    output_file = open("dt.dot", 'w')
    dot_data = tree.export_graphviz(clf
                    , feature_names=data.columns
                    #,class_names=['0', '1', '2', '3', '4', '5']
                    , filled=True
                    ,rounded=True
                    ,out_file=output_file)
    # graph = graphviz.Source(dot_data)
    # graph.view()