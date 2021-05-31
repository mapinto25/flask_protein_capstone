from flask import Flask, render_template, request, flash, session, send_from_directory, send_file
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

import plotly.express as px
import plotly
import json
import sys
import os
import csv
import zipfile

from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './input'
ALLOWED_EXTENSIONS = set(['npz', 'NPZ', 'json', 'JSON'])

DECIMAL_POINTS = 3

#File extension checking
def allowed_filename(filename):
	return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS


current_enzyme_global_data = {}
enzyme_to_class = {}
enzyme_to_class_predicted = {}
matrix = []
f1Score = None
accuracy = None
enzyme_to_closest = {}
model = None

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/bert')
def bert():
    return render_template('bert.html')

@app.route('/visualization')
def visualization():
    return render_template('viz.html')

@app.route('/pca')
def pca():
    return render_template('pca.html')

@app.route('/documentation')
def documentation():
    return render_template('doc.html')

@app.route('/predictions')
def get_results():
    return render_template('predictions.html')

@app.route('/enzyme/<enzyme_id>', methods=['GET'])
def enzyme(enzyme_id):
    global current_enzyme_global_data
    known_enzyme = False
    known_enzyme_classification = ""

    current_enzyme_classification = current_enzyme_global_data['predict_class'][enzyme_id]
    enzyme_confidences = current_enzyme_global_data['prob_class'][enzyme_id]

    if enzyme_id in enzyme_to_class:
        known_enzyme = True
        known_enzyme_classification = enzyme_to_class[enzyme_id]

    return render_template("enzyme.html", 
                           probabilities = enzyme_confidences, 
                           current_enzyme_classification = current_enzyme_classification, 
                           matrix = matrix.tolist(),
                           f1Score = f1Score, 
                           accuracy = accuracy, 
                           model = model, 
                           enzyme_to_closest = enzyme_to_closest, 
                           known_enzyme = known_enzyme,
                           known_enzyme_classification = known_enzyme_classification,
                           enzyme_id = enzyme_id,
                           enzyme_to_class_predicted = enzyme_to_class_predicted,
                           pca = pca_div)

@app.route('/enzyme/pca.html')
def get_pca():
    return render_template('pca.html')


@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        print('post!!')
        if 'npzfile' in request.files:
            submitted_file = request.files['npzfile']
            if submitted_file and allowed_filename(submitted_file.filename):
                filename = secure_filename(submitted_file.filename)
                submitted_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        if 'jsonfile' in request.files:
            submitted_file = request.files['jsonfile']
            if submitted_file and allowed_filename(submitted_file.filename):
                json_file_name = secure_filename(submitted_file.filename)
                submitted_file.save(os.path.join(app.config['UPLOAD_FOLDER'], json_file_name))

        if 'test_npzfile' in request.files:
            submitted_file = request.files['test_npzfile']
            if submitted_file and allowed_filename(submitted_file.filename):
                test_filename = secure_filename(submitted_file.filename)
                submitted_file.save(os.path.join(app.config['UPLOAD_FOLDER'], test_filename))


    if request.form['embedding_model'] == 'ESM':
        class_file = 'enzyme_to_class_esm.json'
        npz_file = 'esm.npz'
    elif request.form['embedding_model'] == 'tape':
        class_file = 'enzyme_to_class_tape.json'
        npz_file = 'tape.npz'
    elif request.form['embedding_model'] == 'combined':
        class_file = 'combined.json'
        npz_file = 'combined.npz'
    elif request.form['embedding_model'] == 'custom':
        class_file = json_file_name
        npz_file = filename

    global embeddings_per_enzyme
    global enzyme_list

    embeddings_per_enzyme = {}
    enzyme_list = []

    with np.load(f'./input/{npz_file}', allow_pickle=True) as data:
        for a in data:
            x = data[a].item()
            embeddings_per_enzyme[a] = x['avg']
            enzyme_list.append(a)

    all_data = pd.DataFrame()

    all_data['Name'] = enzyme_list

    global enzyme_to_class

    with open(f'./input/{class_file}') as f:
        enzyme_to_class = json.load(f)

    enzyme_non_enzyme_list, class_list = process_enzyme_labels(enzyme_list, enzyme_to_class)
    
    all_data['classes'] = class_list

    all_data['enzyme_non_enzyme'] = enzyme_non_enzyme_list
    embedings = []


    for enzyme in enzyme_list:
        word_embedidng = list(embeddings_per_enzyme[enzyme])
        embedings.append(word_embedidng)
    
    all_data['embeddings'] = embedings

    test = pd.DataFrame()


    test_enzyme_list= []
    test_embeddings_per_enzyme = {}

    with np.load(f'./input/{test_filename}', allow_pickle=True) as data:
        for a in data:
            x = data[a].item()
            test_embeddings_per_enzyme[a] = x['avg']
            test_enzyme_list.append(a)

    test['Name'] = test_enzyme_list

    embedings = []
    for enzyme in test_enzyme_list:
        word_embedidng = list(test_embeddings_per_enzyme[enzyme])
        embedings.append(word_embedidng)
    
    test['embeddings'] = embedings

    print(test.shape[0])
    print(all_data.shape[0])

    train, validation = train_test_split(all_data)


    x_train = list(train['embeddings'])
    y_train_classes = train[['classes']]
    y_train_enzyme = train[['enzyme_non_enzyme']]

    x_val = list(validation['embeddings'])
    y_val_class = validation['classes']

    x_test = list(test['embeddings'])

    test_enzyme_list = test['Name'].tolist()
    test_enzyme_list_non_enzyme = []
    test_enzyme_list_is_enzyme = []

    model_name_formatted = ''
    global matrix
    global f1Score
    global accuracy
    global model
    global enzyme_to_closest

    if request.form['down_stream_model'] == 'knn':
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(x_train, y_train_enzyme)
        y_pred_enzyme = neigh.predict(x_test)
        pred_enzyme = neigh.predict_proba(x_test)

        print(y_pred_enzyme)

        x_train_classes, x_test_classes, test_enzyme_list_is_enzyme, test_enzyme_list_non_enzyme, pred_enzyme, y_pred_enzyme = reduce_test_train_classes(x_train, x_test, y_train_classes, y_pred_enzyme, pred_enzyme, test_enzyme_list)

        y_train_classes = y_train_classes[y_train_classes['classes'] != 'NA']


        nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(x_test_classes)
        distances, indices = nbrs.kneighbors(x_test_classes)

        model = 'knn'

        i = 0
        for index in indices:
            index_nearest = int(index[1])
            index_second_nearest = int(index[2])
            index_third_nearest = int(index[3])

            current_enzyme = test_enzyme_list_is_enzyme[i]
            enzyme_to_closest[current_enzyme] = [test_enzyme_list_is_enzyme[index_nearest], test_enzyme_list_is_enzyme[index_second_nearest], test_enzyme_list_is_enzyme[index_third_nearest]]
            i +=1

        neigh.fit(x_train_classes, y_train_classes)
        y_pred_classes = neigh.predict(x_test_classes)
        pred_classes = neigh.predict_proba(x_test_classes)
        model_name_formatted = 'KNN'

        y_val_classes = neigh.predict(x_val)

        matrix = confusion_matrix(y_val_class, y_val_classes)
        f1Score = round(f1_score(y_val_class, y_val_classes, average='macro'), DECIMAL_POINTS)
        accuracy = round(accuracy_score(y_val_class, y_val_classes), DECIMAL_POINTS)

    elif request.form['down_stream_model'] == 'svc':
        model = 'svc'
        model_name_formatted = 'SVC'


        clf = SVC(C = 10, kernel = 'rbf', gamma='auto', probability=True)
        clf.fit(x_train, y_train_enzyme)
        y_pred_enzyme = clf.predict(x_test)
        pred_enzyme = clf.predict_proba(x_test)


        x_train_classes, x_test_classes, test_enzyme_list_is_enzyme, test_enzyme_list_non_enzyme, pred_enzyme, y_pred_enzyme =reduce_test_train_classes(x_train, x_test, y_train_classes, y_pred_enzyme, pred_enzyme, test_enzyme_list)

        y_train_classes = y_train_classes[y_train_classes['classes'] != 'NA']


        nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(x_test_classes)
        distances, indices = nbrs.kneighbors(x_test_classes)

        i = 0
        for index in indices:
            index_nearest = int(index[1])
            index_second_nearest = int(index[2])
            index_third_nearest = int(index[3])

            current_enzyme = test_enzyme_list_is_enzyme[i]
            enzyme_to_closest[current_enzyme] = [test_enzyme_list_is_enzyme[index_nearest], test_enzyme_list_is_enzyme[index_second_nearest], test_enzyme_list_is_enzyme[index_third_nearest]]
            i +=1

        clf.fit(x_train_classes, y_train_classes)
        y_pred_classes = clf.predict(x_test_classes)
        pred_classes = clf.predict_proba(x_test_classes)

        y_val_classes = clf.predict(x_val)

        matrix = confusion_matrix(y_val_class, y_val_classes)
        f1Score = round(f1_score(y_val_class, y_val_classes, average='macro'), DECIMAL_POINTS)
        accuracy = round(accuracy_score(y_val_class, y_val_classes), DECIMAL_POINTS)

    elif request.form['down_stream_model'] == 'deep_learning':

        model = 'mlp'
        clf = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes=(100,))
        model_name_formatted = 'MLP'

        clf.fit(x_train, y_train_enzyme)
        y_pred_enzyme = clf.predict(x_test)
        pred_enzyme = clf.predict_proba(x_test)


        x_train_classes, x_test_classes, test_enzyme_list_is_enzyme, test_enzyme_list_non_enzyme, pred_enzyme, y_pred_enzyme =reduce_test_train_classes(x_train, x_test, y_train_classes, y_pred_enzyme, pred_enzyme, test_enzyme_list)

        y_train_classes = y_train_classes[y_train_classes['classes'] != 'NA']


        nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(x_test_classes)
        distances, indices = nbrs.kneighbors(x_test_classes)

        i = 0
        for index in indices:
            index_nearest = int(index[1])
            index_second_nearest = int(index[2])
            index_third_nearest = int(index[3])

            current_enzyme = test_enzyme_list_is_enzyme[i]
            enzyme_to_closest[current_enzyme] = [test_enzyme_list_is_enzyme[index_nearest], test_enzyme_list_is_enzyme[index_second_nearest], test_enzyme_list_is_enzyme[index_third_nearest]]
            i +=1

        clf.fit(x_train_classes, y_train_classes)
        y_pred_classes = clf.predict(x_test_classes)
        pred_classes = clf.predict_proba(x_test_classes)

        y_val_classes = clf.predict(x_val)

        matrix = confusion_matrix(y_val_class, y_val_classes)
        f1Score = round(f1_score(y_val_class, y_val_classes, average='macro'), DECIMAL_POINTS)
        accuracy = round(accuracy_score(y_val_class, y_val_classes), DECIMAL_POINTS)

    elif request.form['down_stream_model'] == 'naive':
        model = 'nvb'
        clf =  GaussianNB()
        model_name_formatted = 'Naive Bayes'

        clf.fit(x_train, y_train_enzyme)
        y_pred_enzyme = clf.predict(x_test)
        pred_enzyme = clf.predict_proba(x_test)


        x_train_classes, x_test_classes, test_enzyme_list_is_enzyme, test_enzyme_list_non_enzyme, pred_enzyme, y_pred_enzyme =reduce_test_train_classes(x_train, x_test, y_train_classes, y_pred_enzyme, pred_enzyme, test_enzyme_list)

        y_train_classes = y_train_classes[y_train_classes['classes'] != 'NA']


        nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(x_test_classes)
        distances, indices = nbrs.kneighbors(x_test_classes)


        i = 0
        for index in indices:
            index_nearest = int(index[1])
            index_second_nearest = int(index[2])
            index_third_nearest = int(index[3])

            current_enzyme = test_enzyme_list_is_enzyme[i]
            enzyme_to_closest[current_enzyme] = [test_enzyme_list_is_enzyme[index_nearest], test_enzyme_list_is_enzyme[index_second_nearest], test_enzyme_list_is_enzyme[index_third_nearest]]
            i +=1

        clf.fit(x_train_classes, y_train_classes)
        y_pred_classes = clf.predict(x_test_classes)
        pred_classes = clf.predict_proba(x_test_classes)

        y_val_classes = clf.predict(x_val)

        matrix = confusion_matrix(y_val_class, y_val_classes)
        f1Score = round(f1_score(y_val_class, y_val_classes, average='macro'), DECIMAL_POINTS)
        accuracy = round(accuracy_score(y_val_class, y_val_classes), DECIMAL_POINTS)

    elif request.form['down_stream_model'] == 'dtree':

        model = 'dtree'
        clf = RandomForestClassifier(max_depth =  20, min_samples_leaf =  2, min_samples_split= 5, random_state=0)
        model_name_formatted = 'Random Forest'


        clf.fit(x_train, y_train_enzyme)
        y_pred_enzyme = clf.predict(x_test)
        pred_enzyme = clf.predict_proba(x_test)


        x_train_classes, x_test_classes, test_enzyme_list_is_enzyme, test_enzyme_list_non_enzyme, pred_enzyme, y_pred_enzyme =reduce_test_train_classes(x_train, x_test, y_train_classes, y_pred_enzyme, pred_enzyme, test_enzyme_list)
        y_train_classes = y_train_classes[y_train_classes['classes'] != 'NA']


        nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(x_test_classes)
        distances, indices = nbrs.kneighbors(x_test_classes)


        i = 0
        for index in indices:
            index_nearest = int(index[1])
            index_second_nearest = int(index[2])
            index_third_nearest = int(index[3])

            current_enzyme = test_enzyme_list_is_enzyme[i]
            enzyme_to_closest[current_enzyme] = [test_enzyme_list_is_enzyme[index_nearest], test_enzyme_list_is_enzyme[index_second_nearest], test_enzyme_list_is_enzyme[index_third_nearest]]
            i +=1

        clf.fit(x_train_classes, y_train_classes)
        y_pred_classes = clf.predict(x_test_classes)
        pred_classes = clf.predict_proba(x_test_classes)

        y_val_classes = clf.predict(x_val)

        matrix = confusion_matrix(y_val_class, y_val_classes)
        f1Score = round(f1_score(y_val_class, y_val_classes, average='macro'), DECIMAL_POINTS)
        accuracy = round(accuracy_score(y_val_class, y_val_classes), DECIMAL_POINTS)

    global pca_div

    pca_div = pca_visualize_data(embeddings_per_enzyme,enzyme_to_class)
    return_json = {}
    return_json['prob_class'] = {}
    return_json['prob_enzyme'] = {}
    return_json['predict_class']= {}


    pred_class = pred_classes.tolist()
    global enzyme_to_class_predicted
    with open('./results/enzy_result_file.csv', 'w') as csv_file: 
        csv_file.write('EnzymeID, ClassPrediction,ClassProbs\n')
        for i in range(len(test_enzyme_list_is_enzyme)):
            current_enzyme =  test_enzyme_list_is_enzyme[i]
            return_json['prob_class'][current_enzyme] = np.around(pred_classes[i], decimals= DECIMAL_POINTS)
            return_json['predict_class'][current_enzyme] = y_pred_classes[i]
            enzyme_to_class_predicted[current_enzyme] =  y_pred_classes[i]

            csv_file.write(current_enzyme)
            csv_file.write(',')
            csv_file.write(str(y_pred_classes[i]))
            csv_file.write(',')
            csv_file.write(str(pred_classes[i]))
            csv_file.write('\n')

    with open('./results/nonenzy_result_file.csv', 'w') as csv_file: 
        csv_file.write("NonEnzymeID, ClassPrediction,ClassProbs\n")
        for i in range(len(test_enzyme_list_non_enzyme)):
            current_enzyme =  test_enzyme_list_non_enzyme[i]
            return_json['prob_enzyme'][current_enzyme] = pred_enzyme[i]
            return_json['predict_class'][current_enzyme] = y_pred_enzyme[i]
            csv_file.write(current_enzyme)
            csv_file.write(',')
            csv_file.write(str(y_pred_enzyme[i]))
            csv_file.write(',')
            csv_file.write(str(pred_enzyme[i]))
            csv_file.write('\n')
    
    return_json['model'] = model_name_formatted

    global current_enzyme_global_data
    current_enzyme_global_data = return_json

    output_from_parsed_template = render_template("result.html", result = return_json)
    with open("./templates/predictions.html", "w") as fh:
        fh.write(output_from_parsed_template)

    ### write results to csv file
   # write_to_csv(return_json)

    return output_from_parsed_template

def process_enzyme_labels(enzyme_list, enzyme_to_class):
    class_list = []
    enzyme_non_enzyme_list = []

    for enzyme in enzyme_list:
        current_class = enzyme_to_class[enzyme]
        if current_class == '0':
            enzyme_non_enzyme_list.append(0)
            class_list.append("NA")
        else:
            enzyme_non_enzyme_list.append(1)
            class_list.append(current_class)
    return enzyme_non_enzyme_list, class_list

def reduce_test_train_classes(x_train, x_test, y_train_classes, y_pred_enzyme, pred_enzyme, test_enzyme_list):
    x_test_classes, x_train_classes, y_test_true_classes = [], [], []
    test_enzyme_list_is_enzyme, test_enzyme_list_non_enzyme = [], []

    i = 0
    for index, row in y_train_classes.iterrows():
       if row['classes'] != 'NA':
           x_train_classes.append(x_train[i])
       i += 1

    pred_enzyme_filtered = []
    y_pred_enzyme_filtered = []

    for i in range(len(y_pred_enzyme)):
        if y_pred_enzyme[i] == 1:
            x_test_classes.append(x_test[i])
            test_enzyme_list_is_enzyme.append(test_enzyme_list[i])
        else:
            print(y_pred_enzyme[i])
            print(pred_enzyme[i])
            test_enzyme_list_non_enzyme.append(test_enzyme_list[i])
            pred_enzyme_filtered.append(np.around(pred_enzyme[i], decimals=DECIMAL_POINTS))
            y_pred_enzyme_filtered.append(y_pred_enzyme[i])

    return x_train_classes, x_test_classes, test_enzyme_list_is_enzyme, test_enzyme_list_non_enzyme, pred_enzyme_filtered, y_pred_enzyme_filtered
  
@app.route('/getEnzCsv' , methods = ['POST']) 
def getEnzCsv():
    return send_file('results/enzy_result_file.csv',
                     mimetype='text/csv',
                     attachment_filename='EnzymeResults.csv',
                     as_attachment=True)

@app.route('/getNonEnzCsv' , methods = ['POST']) 
def getNonEnzCsv():
    return send_file('results/nonenzy_result_file.csv',
                     mimetype='text/csv',
                     attachment_filename='NonEnzymeResults.csv',
                     as_attachment=True)

@app.route('/download_all')
def download_all():
    zipf = zipfile.ZipFile('Enz_NonEnz_Results.zip','w', zipfile.ZIP_DEFLATED)
    for root,dirs, files in os.walk('results/'):
        for file in files:
            zipf.write('results/'+file)
    zipf.close()
    return send_file('Enz_NonEnz_Results.zip',
            mimetype = 'zip',
            attachment_filename= 'Enz_NonEnz_Results.zip',
            as_attachment = True)


def write_to_csv(return_json):
    data_file = open('./results/result_file.csv', 'w')
  
    with open('./results/result_file.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in return_json.items():
            writer.writerow([key, value])

def gen_arr(embeddings, seq_id_to_label):
    """
    Iterate over all of the sequence IDs in the given subset of the dataset (embeddings),
    as a nested numpy array. Produce a numpy array of the average embeddings for each
    sequence, as will a list of the labels by looking up the sequence IDs in seq_id_to_label
    Args:
        embeddings (numpy.lib.npyio.NpzFile): Nested numpy array containing embeddings for each sequence ID
        seq_id_to_label (dict[str,str]): Map from sequence ID to classification label
    Returns:
        output (np.array): Average embeddings for each sequence
        labels (list[str])
    """
    # keys = embeddings.files
    output, labels, ids = [], [], []
    for key in embeddings:
        # print(embeddings[key])
        d = embeddings[key]
        labels.append(seq_id_to_label[key])
        output.append(d)
        ids.append(key)
    return np.array(output), labels, ids

def pca_visualize_data(npz_data,class_data):
    """
    Prepare and render an interactive plotly PCA visualization given the following:
        * n_components: Number of PCA components (must be 2 or 3)
        * targets: Labels file
        * input_data: gzipped npz file with sequence embeddings
    """
    
    n_components = 3

    #load labels file
    # lookup_d = json.load(open(f'./input/{class_file}'))

    # #load npz file
    # input_data = np.load(f'./input/{npz_file}', allow_pickle=True)

    # print(npz_data)
    # print(type(npz_data))
    print("generating dataframes")
    embed_arr, embed_labels, embed_ids = gen_arr(npz_data, class_data)
    print("generating PCA")
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(embed_arr)
    principal_df = pd.DataFrame(
        data=principal_components, columns=["pc1", "pc2", "pc3"]
    )
    principal_df["target"] = embed_labels
    principal_df["id"] = embed_ids
    principal_df["source"] = "Train"
    

    #########################################################################
    ##### FIX THIS
    ##setting the test datapoints to different symbols as determined by source
    ### for now subsetting to 1/10 of the datapoints.  Later pass in and use the test dataset ids
    test_ids = principal_df.id[:int(len(principal_df)/10)]
    principal_df['source'][principal_df['id'].isin(test_ids)] = 'Test'

    print("generating plot")

    # Adjust PCA according to the number of components
    # if n_components == 3:
    fig = px.scatter_3d(
        principal_df,
        x="pc1",
        y="pc2",
        z="pc3",
        color="target",
        title="PCA",
        hover_name="id",
        symbol = 'source',
        height=750,
        color_discrete_sequence=px.colors.qualitative.G10,
    )
    # if n_components == 2:
    #     fig = px.scatter(
    #         principal_df,
    #         x="pc1",
    #         y="pc2",
    #         color="target",
    #         hover_name="id",
    #         symbol = 'source',
    #         color_discrete_sequence=px.colors.qualitative.G10,
    #     )

    text = '''
    
    <a href="/predictions" target="_self"> Back </a>
             
   
    '''

    # print(fig.to_json())
    div = plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')
    # print(div)
    
    title = text +'\n' + 'PCA Enzyme Data'
    fig.update_layout(
    height=800,
    title_text= title
    )
    
    fig.write_html("templates/pca.html")

    file = open("templates/pca.html","a")
    file.write(text)
    file.close()
    return div
  

app.run(port='1090')



