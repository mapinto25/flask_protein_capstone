from flask import Flask, render_template, request, flash, session, send_from_directory, send_file

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import umap

import plotly.express as px
import plotly
import json
import sys
import os
import csv
import zipfile
import time
import pandas as pd
import numpy as np
import pickle

from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './input'
ALLOWED_EXTENSIONS = set(['npz', 'NPZ', 'json', 'JSON'])

DECIMAL_POINTS = 3
N = 2500

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
model_type = None

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
    # global current_enzyme_global_data
    known_enzyme = False
    known_enzyme_classification = ""

    current_enzyme_classification = current_enzyme_global_data['predict_class'][enzyme_id]
    enzyme_confidences = current_enzyme_global_data['prob_class'][enzyme_id]

    if enzyme_id in enzyme_to_class:
        known_enzyme = True
        known_enzyme_classification = enzyme_to_class[enzyme_id]

    if model_type == 'custom':
        pca_div = pca_visualize_data(embeddings_per_enzyme, enzyme_to_class, enzyme_id)
        tsne_div, t_pca_div, tsne_pca_div, umap_div, pca_v_ratio, pca50_v_ratio = compare_pca_to_tsne(embeddings_per_enzyme, enzyme_to_class, enzyme_id)
    else:
        pca_div  = None 
        tsne_div, t_pca_div, tsne_pca_div, umap_div, pca_v_ratio, pca50_v_ratio = 'N/A', 'N/A', 'N/A','N/A', 'N/A' ,'N/A'

    actual_closest_enzyme_class = {}
    for enzyme in enzyme_to_closest:
        actual_closest_enzyme_class[enzyme] = enzyme_to_class.get(enzyme, "N/A")

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
                           actual_closest_enzyme_class = actual_closest_enzyme_class,
                           pca = pca_div,
                           tsne = tsne_div,
                           t_pca = t_pca_div,
                           tsne_pca = tsne_pca_div,
                           umap = umap_div,
                           pca_v_ratio = pca_v_ratio,
                           pca50_v_ratio = pca50_v_ratio)

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



    classification_pickle = {
        "ESM" : {
            "svc": ('svc_enzyme_non_enzyme_esm.pkl', 'svc_enzyme_class_esm.pkl'),
            "deep_learning" : ('mlp_enzyme_non_enzyme_esm.pkl', 'mlp_enzyme_class_esm.pkl'),
            "knn" : ('knn_enzyme_non_enzyme_esm.pkl' , 'knn_enzyme_class_esm.pkl' ),
            "naive" : ('gb_enzyme_non_enzyme_esm.pkl', 'gb_enzyme_class_esm.pkl'),
            "dtree" : ('rf_enzyme_non_enzyme_esm.pkl', 'rf_enzyme_class_esm.pkl')
        }, 
        "tape" :  {
            "svc": ('svc_enzyme_non_enzyme_tape.pkl', 'svc_enzyme_class_tape.pkl'),
            "deep_learning" : ('mlp_enzyme_non_enzyme_tape.pkl','mlp_enzyme_class_tape.pkl'),
            "knn" : ('knn_enzyme_non_enzyme_tape.pkl', 'knn_enzyme_class_tape.pkl'),
            "naive" : ('gb_enzyme_non_enzyme_tape.pkl', 'gb_enzyme_class_tape.pkl'),
            "dtree" : ('rf_enzyme_non_enzyme_tape.pkl', 'rf_enzyme_class_tape.pkl')
        },
        
        "combined" :  {
            "svc": ('svc_enzyme_non__enzyme_combined_embeddings.pkl', 'svc_enzyme_class_combined_embeddings.pkl'),
            "deep_learning" : ('mlp_enzyme_non__enzyme_combined_embeddings.pkl', 'mlp_enzyme_class_combined_embeddings.pkl'),
            "knn" : ('knn_enzyme_non__enzyme_combined_embeddings.pkl', 'knn_enzyme_class_combined_embeddings.pkl'),
            "naive" : ('gb_enzyme_non__enzyme_combined_embeddings.pkl', 'gb_enzyme_class_combined_embeddings.pkl'),
            "dtree" : ('rf_enzyme_non__enzyme_combined_embeddings.pkl', 'rf_enzyme_class_combined_embeddings.pkl')
        }
    }



    accuracy_lookup = {
        "ESM" : {
            "svc": 'N/A',
            "deep_learning" : 99.2,
            "knn" : 98.6 ,
            "naive" : 62.1 ,
            "dtree" : 97.8
        }, 
        "tape" :  {
            "svc": 57.5,
            "deep_learning" : 66.1,
            "knn" : 64.8,
            "naive" : 44.6,
            "dtree" :46.3
        },
        
        "combined" :  {
            "svc": 96.5,
            "deep_learning" : 98.9,
            "knn" : 96.9,
            "naive" : 60.7,
            "dtree" : 96.8
        }
    }

    model_name_formatted_lookup = {
        "svc": "SVC",
        "deep_learning" : "MLP",
        "knn" : "KNN",
        "naive" : "Naive Bayes" ,
        "dtree" : "Random Forest"
    }

    model_name_lookup = {
        "svc": "svc",
        "deep_learning" : "mlp",
        "knn" : "knn",
        "naive" : 'nvb',
        "dtree" : "dtree"
    }

    if request.form['embedding_model'] == 'custom':
        class_file = json_file_name
        npz_file = filename

        global embeddings_per_enzyme

        embeddings_per_enzyme = {}
        enzyme_list = []

        with np.load(f'./input/{npz_file}', allow_pickle=True) as data:
            for a in data:
                if request.form['embedding_model'] != 'tape':
                    x = data[a]
                    embeddings_per_enzyme[a] = x
                else:
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
        print(test_filename)
        for a in data:
            if request.form['embedding_model'] != 'tape':
                x = data[a]
                test_embeddings_per_enzyme[a] = x
            else:
                x = data[a].item()
                test_embeddings_per_enzyme[a] = x['avg']
            test_enzyme_list.append(a)

    test['Name'] = test_enzyme_list

    embedings = []
    for enzyme in test_enzyme_list:
        word_embedidng = list(test_embeddings_per_enzyme[enzyme])
        embedings.append(word_embedidng)
    
    test['embeddings'] = embedings

    if request.form['embedding_model'] == 'custom':
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
    global model_type


    model_type = request.form['embedding_model']

    #general_predict_function

    model = model_name_lookup[request.form['down_stream_model']]
    model_name_formatted = model_name_formatted_lookup[request.form['down_stream_model']]


    if request.form['embedding_model'] != 'custom':
        print('using pickle')
        file_name = classification_pickle[request.form['embedding_model']][request.form['down_stream_model']][0]
        print(file_name)
        x_train = []
        y_train_classes = pd.DataFrame()
        with open(f'./input/{file_name}', 'rb') as fp:
            clf = pickle.load(fp)
    else:
        if model == 'svc':
            clf = SVC(C = 10, kernel = 'rbf', gamma='auto', probability=True)
        elif model == "mlp":
            clf = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes=(100,))
        elif model == "knn":
            clf = KNeighborsClassifier(n_neighbors=5)
        elif model == 'dtree':
            clf = RandomForestClassifier(max_depth =  20, min_samples_leaf =  2, min_samples_split= 5, random_state=0)
        elif model == 'nvb':
            clf = GaussianNB()
        clf.fit(x_train, y_train_enzyme)

    y_pred_enzyme = clf.predict(x_test)
    pred_enzyme = clf.predict_proba(x_test)


    x_train_classes, x_test_classes, test_enzyme_list_is_enzyme, test_enzyme_list_non_enzyme, pred_enzyme, y_pred_enzyme =reduce_test_train_classes(x_train, x_test, y_train_classes, y_pred_enzyme, pred_enzyme, test_enzyme_list)


    nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(x_test_classes)
    distances, indices = nbrs.kneighbors(x_test_classes)

    print('nearest neighbors obtain')


    i = 0
    for index in indices:
        index_nearest = int(index[1])
        index_second_nearest = int(index[2])
        index_third_nearest = int(index[3])

        current_enzyme = test_enzyme_list_is_enzyme[i]
        enzyme_to_closest[current_enzyme] = [test_enzyme_list_is_enzyme[index_nearest], test_enzyme_list_is_enzyme[index_second_nearest], test_enzyme_list_is_enzyme[index_third_nearest]]
        i +=1

    if request.form['embedding_model'] != 'custom':
        file_name = classification_pickle[request.form['embedding_model']][request.form['down_stream_model']][1]
        with open(f'./input/{file_name}', 'rb') as fp:
            clf = pickle.load(fp)
        matrix = np.array([])
        f1Score = 'NA'
        accuracy = accuracy_lookup[request.form['embedding_model']][request.form['down_stream_model']]
    else:
        if model == 'svc':
            clf = SVC(C = 10, kernel = 'rbf', gamma='auto', probability=True)
        elif model == "mlp":
            clf = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes=(100,))
        elif model == "knn":
            clf = KNeighborsClassifier(n_neighbors=5)
        elif model == 'dtree':
            clf = RandomForestClassifier(max_depth =  20, min_samples_leaf =  2, min_samples_split= 5, random_state=0)
        elif model == 'nvb':
            clf = GaussianNB()
        y_train_classes = y_train_classes[y_train_classes['classes'] != 'NA']
        clf.fit(x_train_classes, y_train_classes)


        y_val_classes = clf.predict(x_val)
        matrix = confusion_matrix(y_val_class, y_val_classes)
        f1Score = round(f1_score(y_val_class, y_val_classes, average='macro'), DECIMAL_POINTS)
        accuracy = round(accuracy_score(y_val_class, y_val_classes), DECIMAL_POINTS)

        get_random_subset(embeddings_per_enzyme)

            
    y_pred_classes = clf.predict(x_test_classes)
    pred_classes = clf.predict_proba(x_test_classes)


    # global pca_div

    # Calculate random permutation of the embeddings array
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

    print('returning html')
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

def get_random_subset(embeddings_array):
    # For reproducability of the results
    np.random.seed(42)

    global rndperm

    embeddings = list(embeddings_array.keys())
    rndperm = np.random.permutation(len(embeddings))
    print(len(rndperm))
    return rndperm

def create_features_df(embeddings_array, embeddings_labels, embedding_ids):
    feat_cols = [ 'feature'+str(i) for i in range(embeddings_array.shape[1]) ]

    df = pd.DataFrame(embeddings_array,columns=feat_cols)

    df['label'] = embeddings_labels
    df['id'] = embedding_ids

    return df, feat_cols

def subset_visualize_data(embeddings_array, embeddings_labels, embeddings_ids):
    features_df, feat_cols = create_features_df(embeddings_array, embeddings_labels, embeddings_ids)

    df_subset = features_df.loc[rndperm[:N],:].copy()
    data_subset = df_subset[feat_cols].values

    return df_subset, data_subset

def compare_pca_to_tsne(npz_data, class_data, enzyme_id):
    embed_arr, embed_labels, embed_ids = gen_arr(npz_data, class_data)

    df_subset, data_subset = subset_visualize_data(embed_arr, embed_labels, embed_ids)

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data_subset)
    df_subset['pca-one'] = pca_result[:,0]
    df_subset['pca-two'] = pca_result[:,1] 
    df_subset['pca-three'] = pca_result[:,2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    pca_variance_ratio = round(np.sum(pca.explained_variance_ratio_), DECIMAL_POINTS)

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(data_subset)
    print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))
    pca_50_variance_ratio = round(np.sum(pca_50.explained_variance_ratio_), DECIMAL_POINTS)

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform(pca_result_50)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    umap_2d = umap.UMAP(n_components=2, init='random', random_state=0)
    proj_2d = umap_2d.fit_transform(data_subset)

    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]

    df_subset['tsne-pca50-one'] = tsne_pca_results[:,0]
    df_subset['tsne-pca50-two'] = tsne_pca_results[:,1]

    df_subset['umap_one'] = proj_2d[:,0]
    df_subset['umap_two'] = proj_2d[:,1]

    df_subset.loc[df_subset["id"] == enzyme_id, "label"] = enzyme_id

    tsne_fig = px.scatter(
            df_subset, 
            x="tsne-2d-one", 
            y="tsne-2d-two",
            color=df_subset.label, 
            hover_name="id",
            labels={'color': 'label', 'source': 'source'}
        )
    pca_fig = px.scatter(
            df_subset,
            x="pca-one",
            y="pca-two",
            color=df_subset.label,
            hover_name="id",
            color_discrete_sequence=px.colors.qualitative.G10,
        )
    tsne_pca_fig = px.scatter(
            df_subset, 
            x="tsne-pca50-one", 
            y="tsne-pca50-two",
            color=df_subset.label, 
            hover_name="id",
            labels={'color': 'label', 'source': 'source'}
        )

    umap_fig = px.scatter(
            df_subset, 
            x="umap_one", 
            y="umap_two",
            color=df_subset.label, 
            hover_name="id",
            labels={'color': 'label', 'source': 'source'}
        )

    tsne_div = plotly.offline.plot(tsne_fig, include_plotlyjs=False, output_type='div')
    pca_div = plotly.offline.plot(pca_fig, include_plotlyjs=False, output_type='div')
    tsne_pca_div = plotly.offline.plot(tsne_pca_fig, include_plotlyjs=False, output_type='div')
    umap_div = plotly.offline.plot(umap_fig, include_plotlyjs=False, output_type='div')

    return tsne_div, pca_div, tsne_pca_div, umap_div, pca_variance_ratio, pca_50_variance_ratio,


def pca_visualize_data(npz_data,class_data, enzyme_id):
    """
    Prepare and render an interactive plotly PCA visualization given the following:
        * n_components: Number of PCA components (must be 2 or 3)
        * targets: Labels file
        * input_data: gzipped npz file with sequence embeddings
    """
    
    n_components = 3

    print("generating dataframes")
    embed_arr, embed_labels, embed_ids = gen_arr(npz_data, class_data)
    print("generating PCA")
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(embed_arr)
    principal_df = pd.DataFrame(
        data=principal_components, columns=["pc1", "pc2", "pc3"]
    )
    principal_df["target"] = embed_labels
    principal_df["id"] = embed_ids
    principal_df["source"] = "Train"
    
    col_labels = ["pc1", "pc2", "pc3", "target", "id", "source"]

    df_subset = principal_df.loc[rndperm[:N],:].copy()

    df_subset.loc[df_subset["id"] == enzyme_id, "source"] = enzyme_id

    print("generating plot")

    # Adjust PCA according to the number of components
    fig = px.scatter_3d(
        df_subset,
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

    div = plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')
    
    return div
  

app.run(port='1090')



