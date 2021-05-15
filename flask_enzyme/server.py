from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import json
import sys

app = Flask(__name__)


current_enzyme_global_data = {}

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/bert')
def bert():
    return render_template('bert.html')

@app.route('/visualization')
def visualization():
    return render_template('viz.html')

@app.route('/documentation')
def documentation():
    return render_template('doc.html')

@app.route('/enzyme/<enzyme_id>', methods=['GET'])
def enzyme(enzyme_id):
    #do your code here
    print('GET!!!!')
    print(enzyme_id)
    global current_enzyme_global_data
    print(current_enzyme_global_data)
    current_enzyme_classification=  current_enzyme_global_data['predict_class'][enzyme_id]
    print(current_enzyme_classification)
    return render_template("enzyme.html", current_enzyme_classification=current_enzyme_classification)

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        print('post!!')


    if request.form['model'] == 'ESM':
        class_file = 'enzyme_to_class_esm.json'
        npz_file = 'esm.npz'
    elif request.form['model'] == 'tape':
        class_file = 'enzyme_to_class_tape.json'
        npz_file = 'tape.npz'
    elif request.form['model'] == 'combined':
        class_file = 'combined.json'
        npz_file = 'combined.npz'

    
    embeddings_per_enzyme = {}
    enzyme_list = []

    with np.load(f'./input/{npz_file}', allow_pickle=True) as data:
        for a in data:
            x = data[a].item()
            embeddings_per_enzyme[a] = x['avg']
            enzyme_list.append(a)

    all_data = pd.DataFrame()



    all_data['Name'] = enzyme_list

    class_list = []

    with open(f'./input/{class_file}') as f:
        enzyme_to_class = json.load(f)


    for enzyme in enzyme_list:
        class_list.append(enzyme_to_class[enzyme])

    all_data['classes'] = class_list

    
    enzyme_non_enzyme = []

    for enzyme in enzyme_list:
        current_class = enzyme_to_class[enzyme]
        if current_class == 0:
            enzyme_non_enzyme.append(0)
        else:
            enzyme_non_enzyme.append(1)


    all_data['enzyme_non_enzyme'] = enzyme_non_enzyme
    embedings = []


    for enzyme in enzyme_list:
        word_embedidng = list(embeddings_per_enzyme[enzyme])
        embedings.append(word_embedidng)
    
    all_data['embeddings'] = embedings


    train, test = train_test_split(all_data, test_size=0.1)


    x_train = list(train['embeddings'])
    y_train_classes = train[['classes']]
    y_train_enzyme = train[['enzyme_non_enzyme']]
    x_test = list(test['embeddings'])
    y_test_class = test[['classes']]
    y_test_enzyme = test[['enzyme_non_enzyme']]

    test_enzyme_list = test['Name'].tolist()
    test_enzyme_list_non_enzyme = []
    test_enzyme_list_is_enzyme = []




    if request.form['down_stream_model'] == 'knn':
        neigh = KNeighborsClassifier(n_neighbors=5)

        neigh.fit(x_train, y_train_enzyme)
        y_pred_enzyme = neigh.predict(x_test)
        pred_enzyme = neigh.predict_proba(x_test)

        x_test_classes = []

        for i in range(len(y_pred_enzyme)):
            if y_pred_enzyme[i] == 0:
                test_enzyme_list_non_enzyme.append(test_enzyme_list[i])
            if y_pred_enzyme[i] == 1:
                test_enzyme_list_is_enzyme.append(test_enzyme_list[i])
                x_test_classes.append(x_test[i])

        neigh.fit(x_train, y_train_classes)
        y_pred_classes = neigh.predict(x_test_classes)
        pred_classes = neigh.predict_proba(x_test_classes)
    elif request.form['down_stream_model'] == 'svc':
        clf = SVC(C = 10, kernel = 'rbf', gamma='auto')
        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        pred = neigh.predict_proba(x_test)
    elif request.form['down_stream_model'] == 'deep_learning':
        clf = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes=(100,)).fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        pred = clf.predict_proba(x_test)
    elif request.form['down_stream_model'] == 'naive':
        gnb = GaussianNB()
        gnb.fit(x_train,y_train)
        y_pred = gnb.predict(x_test)
        pred = gnb.predict_proba(x_test)
    elif request.form['down_stream_model'] == 'dtree':
        clf = RandomForestClassifier(max_depth =  20, min_samples_leaf =  2, min_samples_split= 5, random_state=0)
        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        pred = clf.predict_proba(x_test)
    


    return_json = {}
    return_json['prob_class'] = {}
    return_json['prob_enzyme'] = {}
    return_json['predict_class']= {}


    pred_class = pred_classes.tolist()
    for i in range(len(test_enzyme_list_is_enzyme)):
        current_enzyme =  test_enzyme_list_is_enzyme[i]
        return_json['prob_class'][current_enzyme] = pred_classes[i]
        return_json['predict_class'][current_enzyme] = y_pred_classes[i]

    pred_enzyme = pred_enzyme.tolist()
    for i in range(len(test_enzyme_list_non_enzyme)):
        current_enzyme =  test_enzyme_list_non_enzyme[i]
        return_json['prob_enzyme'][current_enzyme] = pred_enzyme[i]
        return_json['predict_class'][current_enzyme] = y_pred_classes[i]

    global current_enzyme_global_data
    current_enzyme_global_data = return_json

    return render_template("result.html",result = return_json)

app.run(port='1090')



