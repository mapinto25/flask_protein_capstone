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

@app.route('/')
def home_page():
    return render_template('index.html')

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

    embedings = []


    for enzyme in enzyme_list:
        word_embedidng = list(embeddings_per_enzyme[enzyme])
        embedings.append(word_embedidng)
    
    all_data['embeddings'] = embedings


    train, test = train_test_split(all_data, test_size=0.1)


    x_train = list(train['embeddings'])
    y_train = train[['classes']]
    x_test = list(test['embeddings'])
    y_test = test[['classes']]


    if request.form['down_stream_model'] == 'knn':
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(x_train, y_train)
        y_pred = neigh.predict(x_test)
        pred = neigh.predict_proba(x_test)
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
    


    prob = list(pred[0])



    return_json = {}

    i = 1
    for probability in prob:
        return_json['class_' + str(i) + '_prob'] = probability
        i += 1
    return_json['y_pred'] = y_pred[0]
    return_json['prob'] = pred.tolist()
    print(pred)

    return render_template("result.html",result = return_json)

app.run(port='1090')



