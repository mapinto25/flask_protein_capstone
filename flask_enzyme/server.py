from flask import Flask, render_template, request, flash, session, send_from_directory, send_file
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
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
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './input'
ALLOWED_EXTENSIONS = set(['npz', 'NPZ', 'json', 'JSON'])

DECIMAL_POINTS = 3

#File extension checking
def allowed_filename(filename):
	return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS


current_enzyme_global_data = {}
enzyme_to_class = {}
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
                           enzyme_id = enzyme_id)

@app.route('/enzyme/pca.html')
def get_pca():
    return render_template('pca.html')


@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        print('post!!')
        custom_file = False
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


    if request.form['model'] == 'ESM':
        class_file = 'enzyme_to_class_esm.json'
        npz_file = 'esm.npz'
    elif request.form['model'] == 'tape':
        class_file = 'enzyme_to_class_tape.json'
        npz_file = 'tape.npz'
    elif request.form['model'] == 'combined':
        class_file = 'combined.json'
        npz_file = 'combined.npz'
    elif request.form['model'] == 'custom':
        class_file = json_file_name
        npz_file = filename


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


    train, test = train_test_split(all_data, test_size=0.05)


    x_train = list(train['embeddings'])
    y_train_classes = train[['classes']]
    y_train_enzyme = train[['enzyme_non_enzyme']]
    x_test = list(test['embeddings'])
    y_test_class =list(test['classes'])
    y_test_enzyme = test[['enzyme_non_enzyme']]

    test_enzyme_list = test['Name'].tolist()
    test_enzyme_list_non_enzyme = []
    test_enzyme_list_is_enzyme = []

    model_name_formatted = ''
    global matrix
    global f1Score
    global accuracy
    global model

    if request.form['down_stream_model'] == 'knn':
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(x_train, y_train_enzyme)
        y_pred_enzyme = neigh.predict(x_test)
        pred_enzyme = neigh.predict_proba(x_test)

        x_train_classes, x_test_classes, y_test_true_classes, test_enzyme_list_is_enzyme, test_enzyme_list_non_enzyme =reduce_test_train_classes(x_train, x_test, y_train_classes, y_test_class, y_pred_enzyme, test_enzyme_list)

        y_train_classes = y_train_classes[y_train_classes['classes'] != 'NA']


        nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(x_test_classes)
        distances, indices = nbrs.kneighbors(x_test_classes)

        model = 'knn'

        global enzyme_to_closest

        i = 0
        for index in indices:
            index_nearest = int(index[1])
            index_second_nearest = int(index[2])
            index_third_nearest = int(index[3])

            current_enzyme = test_enzyme_list_is_enzyme[i]
            enzyme_to_closest[current_enzyme] = [test_enzyme_list_is_enzyme[index_nearest], test_enzyme_list_is_enzyme[index_second_nearest], test_enzyme_list_is_enzyme[index_third_nearest]]
            i +=1


        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(x_train_classes, y_train_classes)
        y_pred_classes = neigh.predict(x_test_classes)
        pred_classes = neigh.predict_proba(x_test_classes)
        model_name_formatted = 'KNN'

        matrix = confusion_matrix(y_test_true_classes,y_pred_classes)
        f1Score = round(f1_score(y_test_true_classes, y_pred_classes, average='macro'), DECIMAL_POINTS)
        accuracy = round(accuracy_score(y_test_true_classes, y_pred_classes), DECIMAL_POINTS)

    elif request.form['down_stream_model'] == 'svc':
        model = 'svc'

        clf = SVC(C = 10, kernel = 'rbf', gamma='auto', probability=True)

        clf.fit(x_train, y_train_enzyme)
        y_pred_enzyme = clf.predict(x_test)
        pred_enzyme = clf.predict_proba(x_test)

        x_train_classes, x_test_classes, y_test_true_classes, test_enzyme_list_is_enzyme, test_enzyme_list_non_enzyme = reduce_test_train_classes(x_train, x_test, y_train_classes, y_test_class, y_pred_enzyme, test_enzyme_list)

        y_train_classes = y_train_classes[y_train_classes['classes'] != 'NA']
    
        clf.fit(x_train_classes, y_train_classes)
        y_pred_classes = clf.predict(x_test_classes)
        pred_classes = clf.predict_proba(x_test_classes)
        model_name_formatted = 'SVC'

        matrix = confusion_matrix(y_test_true_classes,y_pred_classes)
        f1Score = round(f1_score(y_test_true_classes, y_pred_classes, average='macro'), DECIMAL_POINTS)
        accuracy = round(accuracy_score(y_test_true_classes, y_pred_classes), DECIMAL_POINTS)

    elif request.form['down_stream_model'] == 'deep_learning':

        model = 'mlp'

        clf = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes=(100,))

        clf.fit(x_train, y_train_enzyme)
        y_pred_enzyme = clf.predict(x_test)
        pred_enzyme = clf.predict_proba(x_test)

        x_train_classes, x_test_classes, y_test_true_classes, test_enzyme_list_is_enzyme, test_enzyme_list_non_enzyme =reduce_test_train_classes(x_train, x_test, y_train_classes, y_test_class, y_pred_enzyme, test_enzyme_list)

        y_train_classes = y_train_classes[y_train_classes['classes'] != 'NA']
    
        clf.fit(x_train_classes, y_train_classes)
        y_pred_classes = clf.predict(x_test_classes)
        pred_classes = clf.predict_proba(x_test_classes)
        model_name_formatted = 'MLP'

        matrix = confusion_matrix(y_test_true_classes,y_pred_classes)
        f1Score = round(f1_score(y_test_true_classes, y_pred_classes, average='macro'), DECIMAL_POINTS)
        accuracy = round(accuracy_score(y_test_true_classes, y_pred_classes), DECIMAL_POINTS)

    elif request.form['down_stream_model'] == 'naive':
        model = 'nvb'

        clf =  GaussianNB()

        clf.fit(x_train, y_train_enzyme)
        y_pred_enzyme = clf.predict(x_test)
        pred_enzyme = clf.predict_proba(x_test)

        x_train_classes, x_test_classes, y_test_true_classes, test_enzyme_list_is_enzyme, test_enzyme_list_non_enzyme =reduce_test_train_classes(x_train, x_test, y_train_classes, y_test_class, y_pred_enzyme, test_enzyme_list)

        y_train_classes = y_train_classes[y_train_classes['classes'] != 'NA']
    
        clf.fit(x_train_classes, y_train_classes)
        y_pred_classes = clf.predict(x_test_classes)
        pred_classes = clf.predict_proba(x_test_classes)
        model_name_formatted = 'Naive Bayes'

        matrix = confusion_matrix(y_test_true_classes,y_pred_classes)
        f1Score = round(f1_score(y_test_true_classes, y_pred_classes, average='macro'), DECIMAL_POINTS)
        accuracy = round(accuracy_score(y_test_true_classes, y_pred_classes), DECIMAL_POINTS)

    elif request.form['down_stream_model'] == 'dtree':

        model = 'dtree'


        clf = RandomForestClassifier(max_depth =  20, min_samples_leaf =  2, min_samples_split= 5, random_state=0)

        clf.fit(x_train, y_train_enzyme)
        y_pred_enzyme = clf.predict(x_test)
        pred_enzyme = clf.predict_proba(x_test)

        x_train_classes, x_test_classes, y_test_true_classes, test_enzyme_list_is_enzyme, test_enzyme_list_non_enzyme =reduce_test_train_classes(x_train, x_test, y_train_classes, y_test_class, y_pred_enzyme, test_enzyme_list)

        y_train_classes = y_train_classes[y_train_classes['classes'] != 'NA']
    
        clf.fit(x_train_classes, y_train_classes)
        y_pred_classes = clf.predict(x_test_classes)
        pred_classes = clf.predict_proba(x_test_classes)
        model_name_formatted = 'Random Forest'

        matrix = confusion_matrix(y_test_true_classes,y_pred_classes)
        f1Score = round(f1_score(y_test_true_classes, y_pred_classes, average='macro'), DECIMAL_POINTS)
        accuracy = round(accuracy_score(y_test_true_classes, y_pred_classes), DECIMAL_POINTS)

    pca_visualize_data(npz_file,class_file)
    return_json = {}
    return_json['prob_class'] = {}
    return_json['prob_enzyme'] = {}
    return_json['predict_class']= {}


    pred_class = pred_classes.tolist()
    with open('./results/enzy_result_file.csv', 'w') as csv_file: 
        csv_file.write('EnzymeID, ClassPrediction,ClassProbs\n')
        for i in range(len(test_enzyme_list_is_enzyme)):
            current_enzyme =  test_enzyme_list_is_enzyme[i]
            return_json['prob_class'][current_enzyme] = pred_classes[i]
            return_json['predict_class'][current_enzyme] = y_pred_classes[i]
            csv_file.write(current_enzyme)
            csv_file.write(',')
            csv_file.write(str(y_pred_classes[i]))
            csv_file.write(',')
            csv_file.write(str(pred_classes[i]))
            csv_file.write('\n')

    pred_enzyme = pred_enzyme.tolist()
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

def reduce_test_train_classes(x_train, x_test, y_train_classes, y_test_class, y_pred_enzyme, test_enzyme_list):
    x_test_classes, x_train_classes, y_test_true_classes = [], [], []
    test_enzyme_list_is_enzyme, test_enzyme_list_non_enzyme = [], []

    i = 0
    for index, row in y_train_classes.iterrows():
       if row['classes'] != 'NA':
           x_train_classes.append(x_train[i])
       i += 1

    for i in range(len(y_pred_enzyme)):
        if y_pred_enzyme[i] == 1:
            x_test_classes.append(x_test[i])
            y_test_true_classes.append(y_test_class[i])
            test_enzyme_list_is_enzyme.append(test_enzyme_list[i])
        else:
            test_enzyme_list_non_enzyme.append(test_enzyme_list[i])

    return x_train_classes, x_test_classes, y_test_true_classes, test_enzyme_list_is_enzyme, test_enzyme_list_non_enzyme
  
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
    keys = embeddings.files
    output, labels = [], []
    for key in keys:
        d = embeddings[key].item()["avg"]
        labels.append(seq_id_to_label[key])
        output.append(d)
    return np.array(output), labels

def pca_visualize_data(npz_file,class_file):
    """
    Prepare and render an interactive plotly PCA visualization given the following:
        * n_components: Number of PCA components (must be 2 or 3)
        * targets: Labels file
        * input_data: gzipped npz file with sequence embeddings
    """
    
    n_components = 3

    #load labels file
    lookup_d = json.load(open(f'./input/{class_file}'))

    #load npz file
    input_data = np.load(f'./input/{npz_file}', allow_pickle=True)

    
    print("generating dataframes")
    embed_arr, embed_labels = gen_arr(input_data, lookup_d)
    print("generating PCA")
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(embed_arr)
    principal_df = pd.DataFrame(
        data=principal_components, columns=["pc1", "pc2", "pc3"]
    )
    principal_df["target"] = embed_labels
    print("generating plot")

    # Adjust PCA according to the number of components
    if n_components == 3:
        fig = px.scatter_3d(
            principal_df,
            x="pc1",
            y="pc2",
            z="pc3",
            color="target",
            color_discrete_sequence=px.colors.qualitative.G10,
        )
    if n_components == 2:
        fig = px.scatter(
            principal_df,
            x="pc1",
            y="pc2",
            color="target",
            color_discrete_sequence=px.colors.qualitative.G10,
        )
    
    text = '''
    
    <a href="/predictions">Back</a>
             
   
    '''
    
    title = text +'\n' + 'PCA Enzyme Data'
    fig.update_layout(
    height=800,
    title_text= title
    )
    
    fig.write_html("templates/pca.html")

    file = open("templates/pca.html","a")
    file.write(text)
    file.close()
    return
  

app.run(port='1090')



