### all the import
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
import math
import numpy.random as rnd
import seaborn as sns
import librosa
from os import listdir
from os.path import isfile, join
import glob
import re
from scipy import signal
from scipy.fft import fft
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
import torch
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import xgboost as xgb


### data import, usefull because sometimes used in functions

data = [] 
label = [] 
genres = []
min_duration = None
words = ['avance','recule','tournegauche']
list_genres = ['M', 'F']

for file_name in glob.glob('FichierTest/*.wav'):
    record = librosa.load(file_name)[0]
    data.append(record)
    # Computation of the minimal size of recordings
    if min_duration is None or record.shape[0] < min_duration:
        min_duration = record.shape[0] 
    
    # Creation of the vector of label
    for i, word in enumerate(words):
      if re.search(word, file_name):
        label.append(i)

    # Creation of the vector of label
    for i, genre in enumerate(list_genres):
      if re.search(genre, file_name[12:]):# 12 is for ignoring "FichierTest/"
        genres.append(genre)

fs = librosa.load(file_name)[1] # Sampling frequency
genres = np.array(genres)

### Cross Validation function from scratch with logistic regression classifier
def CrossValidation(X, y, C, n = 3):
    #shuffle indexs
    idx = np.arange(X.shape[0])
    idx = shuffle(idx)
    Xs = X[idx]
    ys = y[idx]

    #we divide into 3 sets
    n_folds = n
    l = Xs.shape[0]
    X_test = []
    X_train = []
    y_test = []
    y_train = []
    for i in range(n_folds):
        idx_test = idx[round(l*i/n_folds):round(l*(i+1)/n_folds)]
        idX_train = idx.tolist()
        for j in idx_test:
            idX_train.remove(j)
        X_test.append(Xs[idx_test, :])
        X_train.append(Xs[idX_train, :])
        y_test.append(ys[idx_test])
        y_train.append(ys[idX_train])

    #model instanciation
    model = LogisticRegression(C = C, max_iter = 1000)

    #computing the score
    score_test = []
    score_train = []
    for i in range(n_folds):
        model.fit(X_train[i], y_train[i])
        y_pred_train = model.predict(X_train[i])
        y_pred_test = model.predict(X_test[i])
        fold_score_test = accuracy_score(y_test[i], y_pred_test)
        score_test.append(fold_score_test)
        fold_score_train = accuracy_score(y_train[i], y_pred_train)
        score_train.append(fold_score_train)
    mean_accuracy_test = np.mean(score_test)
    mean_accuracy_train = np.mean(score_train)
    return mean_accuracy_test, mean_accuracy_train, y_test, y_pred_test

### Function which returns an explanatory dataframe of the previous CrossValidation function. This calculates the accuracy for different C. For example, we have the C of logistic regression or the accucary score of tests and trains.
def df_C_variation(X, y, start, stop, step):
    tab_accuracy_test = []
    tab_accuracy_train = []
    tab_y_test = []
    tab_y_pred_test = []
    x_axis = []
    for C in np.arange(start, stop, step):
        x_axis.append(C)
        accuracy_test, accuracy_train, y_test, y_test_pred = CrossValidation(X, y, C)
        tab_accuracy_test.append(accuracy_test)
        tab_accuracy_train.append(accuracy_train)
        tab_y_test.append(y_test[2])
        tab_y_pred_test.append(y_test_pred)
        
    tab =  pd.DataFrame(
    {
        "C" : x_axis, 
        "Accuracy_test" : tab_accuracy_test,
        "Accuracy_train" : tab_accuracy_train,
        "Y_test" : tab_y_test,
        "Y_pred" : tab_y_pred_test,
        "Accuracy_test_max" : np.max(tab_accuracy_test),

    }
    )
    return tab


### Plot lines function for the previous CrossValidation function. This displays the test and train accuracy score of the different Cs. As well as the best score.
def plot_accuracy(df):
    fig = px.line(df, x="C", y=['Accuracy_test', 'Accuracy_train', 'Accuracy_test_max'], title='<b>Accuracy results in function of C </b>')
    fig.add_hline(y=df.Accuracy_test.max(), line_width=1, line_dash="dash", line_color="green")
    fig.update_layout(yaxis_range=[0,1.1], yaxis_title = 'Score')
    fig.show()
    print("The best test score is : "+str(df.Accuracy_test.max()))


### Displaying a custom confusion matrix that works with the CrossValidation function dataframe.
def confusion_matrix_from_df(df):
    cm = confusion_matrix(df.Y_test[df.Accuracy_test.idxmax()],df.Y_pred[df.Accuracy_test.idxmax()], normalize = 'true')
    z = np.round(cm, 2)

    # creation of labels
    x = ['Avance', 'Recule', 'Tournegauche']
    y =  ['Avance', 'Recule', 'Tournegauche']
    
    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]
    
    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text)
    
    # add title
    fig.update_layout(title_text='<b>Confusion matrix</b>',
                     )
    
    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))
    
    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=100))
    
    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.1,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))
    # put the x_axis annotation on the bottom
    fig.update_layout(
        xaxis=dict(side='bottom')
    )
    
    # add colorbar
    fig['data'][0]['showscale'] = True
    fig.show()

### definition of FFT class, necessary to be able to use it in a pipeline 
class FFT(BaseEstimator, TransformerMixin):
    def __init__(self, idx_frequence_max=None):
        self.idx_frequence_max = idx_frequence_max
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return np.absolute(fft(X)[:self.idx_frequence_max])
        
### definition of STFT class, necessary to be able to use it in a pipeline 
class STFT():
    def __init__(self, idx_frequence_max = None, stat = None):
        self.idx_frequence_max = idx_frequence_max    
        self.stat = stat
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        nperseg = 253
        f, t, Zxxm = signal.stft(X, fs=fs, window='hann', nperseg=nperseg, noverlap=None)
        Zxxm_trans = np.absolute(Zxxm[:, :self.idx_frequence_max, :])
        X_trans = np.zeros((Zxxm_trans.shape[0],Zxxm_trans.shape[1]))
        for s in range(Zxxm_trans.shape[0]):
            for i in range(Zxxm_trans.shape[1]):
                X_trans[s, i] = self.stat(Zxxm_trans[s,i,:])
        return X_trans
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

### Function which defines all the pipelines that we are going to use, i.e. the 4 transformations & classifier
def set_pipelines(X, y, method, method_name, params):
    np.random.seed(42)
    loo = LeaveOneOut()
    pca = PCA()
    scaler = StandardScaler()
    fft_class = FFT()
    stft_class = STFT()
    pg1, pg2, pg3, pg4 = params
    
    pipe_1 = Pipeline(steps=[("fft", fft_class), ("pca", pca), (method_name, method)])
    pipe_2 = Pipeline(steps=[("fft", fft_class), ("scaler", scaler), ("pca", pca), (method_name, method)])
    pipe_3 = Pipeline(steps=[("stft", stft_class), (method_name, method)])
    pipe_4 = Pipeline(steps=[("stft", stft_class), ("scaler", scaler), (method_name, method)])
    
    clf1 = GridSearchCV(pipe_1, pg1, cv=loo,return_train_score=True)
    clf1.fit(X, y)
    clf2 = GridSearchCV(pipe_2, pg2, cv=loo,return_train_score=True)
    clf2.fit(X, y)
    clf3 = GridSearchCV(pipe_3, pg3, cv=loo,return_train_score=True)
    clf3.fit(X, y)
    clf4 = GridSearchCV(pipe_4, pg4, cv=loo,return_train_score=True)
    clf4.fit(X, y)

    return clf1, clf2, clf3, clf4


### Function that displays barplot of best classifiers of each pipeline  
def get_best_classifier(clfs):
    clf1, clf2, clf3, clf4 = clfs

    classifiers = {
        "Classifier1": clf1,
        "Classifier2": clf2,
        "Classifier3": clf3,
        "Classifier4": clf4,
    }
    
    max_results = {
        "Classifier1": max(classifiers["Classifier1"].cv_results_["mean_test_score"]),
        "Classifier2": max(classifiers["Classifier2"].cv_results_["mean_test_score"]),
        "Classifier3": max(classifiers["Classifier3"].cv_results_["mean_test_score"]),
        "Classifier4": max(classifiers["Classifier4"].cv_results_["mean_test_score"]),
    }
    
    best_params = {
        "Classifier1": {k: str(v) for k, v in clf1.best_params_.items()},
        "Classifier2": {k: str(v) for k, v in clf2.best_params_.items()},
        "Classifier3": {k: str(v) for k, v in clf3.best_params_.items()},
        "Classifier4": {k: str(v) for k, v in clf4.best_params_.items()},
    }

    models = list(max_results.keys())
    accuracies = list(max_results.values())
    
    best_clf_name = max(max_results, key=max_results.get)
    best_score = max_results[best_clf_name]

    colors = ['red' if model == best_clf_name else 'blue' for model in models]

    fig = go.Figure(data=[
        go.Bar(name='Accuracy', x=models, y=accuracies, marker_color=colors)
    ])

    fig.update_layout(
        title="<b>Comparison of Best Test Accuracy for Each Pipeline's Best Classifier with Cross Validation<b>",
        xaxis_title='Pipeline',
        yaxis_title='Accuracy',
        barmode='group',
        template='plotly_white'
    )

    fig.add_annotation(
        x=best_clf_name,
        y=best_score,
        text=f"Best: {best_score:.4f}",
        showarrow=True,
        arrowhead=2,
        ax=20,
        ay=-30
    )

    fig.show()

    best_clf = classifiers[best_clf_name]
    return best_clf, best_score, best_params

### Function that transforms a pipeline output into a dataframe. The dataframe contains the model, the test score, the train score and the execution time. This function also returns the best model in the pipeline and the best score.
def clf_transform(clf, pipe):
    scores = clf.cv_results_
    params_list = scores["params"]

    ### transform the pipeline output into string --> can be optimize
    if pipe == "reg_1":
        params_strings = [f"C: {param['logistic__C']}, N: {param['pca__n_components']}" for param in params_list]
    if pipe == "reg_2":
        params_strings = [f"C: {param['logistic__C']}, stat: {param['stft__stat'].__name__}, idx_max: {param['stft__idx_frequence_max']}" for param in params_list]
    if pipe == 'svm_l1':
        params_strings = [f"C: {param['SVM__C']}, kern: {param['SVM__kernel']}, N: {param['pca__n_components']}" for param in params_list]
    if pipe == 'svm_l2':
        params_strings = [f"C: {param['SVM__C']}, kern: {param['SVM__kernel']}, N: {param['pca__n_components']}" for param in params_list]
    if pipe == 'svm_l3':
        params_strings = [f"C: {param['SVM__C']}, kern: {param['SVM__kernel']}, stat: {param['stft__stat'].__name__}, idx_max: {param['stft__idx_frequence_max']}" for param in params_list]
    if pipe == 'svm_l4':
        params_strings = [f"C: {param['SVM__C']}, kern: {param['SVM__kernel']}, stat: {param['stft__stat'].__name__}, idx_max: {param['stft__idx_frequence_max']}" for param in params_list]
    if pipe == 'svm_g1':
        params_strings = [f"C: {param['SVM__C']}, kern: {param['SVM__kernel']}, G: {param['SVM__gamma']}, N: {param['pca__n_components']}" for param in params_list]
    if pipe == 'svm_g2':
        params_strings = [f"C: {param['SVM__C']}, kern: {param['SVM__kernel']}, G: {param['SVM__gamma']}, N: {param['pca__n_components']}" for param in params_list]
    if pipe == 'svm_g3':
        params_strings = [f"C: {param['SVM__C']}, kern: {param['SVM__kernel']}, G: {param['SVM__gamma']}, stat: {param['stft__stat'].__name__}, idx_max: {param['stft__idx_frequence_max']}" for param in params_list]
    if pipe == 'svm_g4':
        params_strings = [f"C: {param['SVM__C']}, kern: {param['SVM__kernel']}, G: {param['SVM__gamma']}, stat: {param['stft__stat'].__name__}, idx_max: {param['stft__idx_frequence_max']}" for param in params_list]
    if pipe == 'svm_p1':
        params_strings = [f"C: {param['SVM__C']}, kern: {param['SVM__kernel']}, G: {param['SVM__gamma']}, D: {param['SVM__degree']}, N: {param['pca__n_components']}" for param in params_list]
    if pipe == 'svm_p2':
        params_strings = [f"C: {param['SVM__C']}, kern: {param['SVM__kernel']}, G: {param['SVM__gamma']}, D: {param['SVM__degree']}, N: {param['pca__n_components']}" for param in params_list]
    if pipe == 'svm_p3':
        params_strings = [f"C: {param['SVM__C']}, kern: {param['SVM__kernel']}, G: {param['SVM__gamma']}, D: {param['SVM__degree']}, stat: {param['stft__stat'].__name__}, idx_max: {param['stft__idx_frequence_max']}" for param in params_list]
    if pipe == 'svm_p4':
        params_strings = [f"C: {param['SVM__C']}, kern: {param['SVM__kernel']}, G: {param['SVM__gamma']}, D: {param['SVM__degree']}, stat: {param['stft__stat'].__name__}, idx_max: {param['stft__idx_frequence_max']}" for param in params_list]

    tab_clf =  pd.DataFrame(
    {
        "Model" : params_strings, 
        "Mean_test_score" : scores["mean_test_score"],
        "Mean_train_score" : scores["mean_train_score"],
        "Mean_score_time" : scores["mean_score_time"],
        "Mean_fit_time" : scores["mean_fit_time"],
        "Total_time" : scores["mean_score_time"] + scores["mean_fit_time"],
    }).sort_values(['Mean_test_score', 'Total_time'], ascending = [False, True])
    
    return tab_clf, clf.best_params_, clf.best_score_

### Plot line for the previous dataframe. Plot the train & test accuracy for each model and the execution time. Also print the best model & score.
def plot_score_time_clf(tuple):
    df, best_params_, best_score_ = tuple
    
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.05)

    fig.append_trace(go.Scatter(
        x=df.Model,
        y=df.Mean_test_score,
        name = "Mean test score",
        mode='lines+markers',
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        x=df.Model,
        y=df.Mean_train_score,
        name = "Mean train score",
        mode='lines+markers',
    ), row=1, col=1)
    
    fig.append_trace(go.Scatter(
        x=df.Model,
        y=df.Total_time,
        name = "Execution time",
        mode='lines+markers',
    ), row=2, col=1)
    
    fig.update_layout(height=600, width=1000, title_text="<b>Score and execution time of each model</b>")
    fig.update_xaxes(
        showticklabels=False,
        row=1, col=1
    )
    fig.update_yaxes(
        title='Score',
        row=1, col=1,
        range = [0,1.1],
    )
    fig.update_xaxes(
        tickangle=-45,
        row=2, col=1,
        title = "Model",
    )
    fig.update_yaxes(
        title='Time <i>(s)</i>',
        row=2, col=1
    )
    fig.show()
    
    print("Best parameters:", best_params_)
    print("Best score:", best_score_)

### A custom confusion matrix that works with a pipeline output
def confusion_matrix_clf(clf, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)
    np.random.seed(seed=42)
    clf__ = clf.best_estimator_
    clf__.fit(X_train, y_train)
    cm = confusion_matrix(y_test, clf__.predict(X_test))
    z = np.round(cm, 2)
    
    x = ['Avance', 'Recule', 'Tournegauche']
    y =  ['Avance', 'Recule', 'Tournegauche']
    
    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]
    
    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text)
    
    # add title
    fig.update_layout(title_text='<b>Confusion matrix</b>',
                     )
    
    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))
    
    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=100))
    
    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.1,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))
    
    # put the x_axis annotation on the bottom
    fig.update_layout(
        xaxis=dict(side='bottom')
    )
    
    # add colorbar
    fig['data'][0]['showscale'] = True
    fig.show()


### Returns the name as a string of a function object
def get_function_name(func):
    return func.__name__ if callable(func) else func


### Returns the steps of a pipeline as a string
def get_grid_search_pipeline_string(clf):
    
    best_estimator = clf.best_estimator_
    
    pipeline_steps = best_estimator.named_steps
    
    pipeline_string = ""
    
    for step_name, step_obj in pipeline_steps.items():
        pipeline_string += f"{step_name} "
    
    return pipeline_string

### Presents pipeline results as a dataframe. We have as information the name of the model and all the parameters of the classifier
def dataframe_model_summary(list_clf):
    param_list = []
    model_list = []
    best_score_list = []

    for i in list_clf:
        param_list.append(i.best_params_)
        model_list.append(get_grid_search_pipeline_string(i))
        best_score_list.append(i.best_score_)

    
    if not param_list:
        return pd.DataFrame()
    df_models = pd.DataFrame(
    {
        "Pipeline" : model_list, 
        "Score" : best_score_list,
    }
    )
    df_params = pd.DataFrame(param_list)
    df_combined = pd.concat([df_models, df_params], axis=1)
    df_combined['stft__stat'] = df_combined['stft__stat'].apply(get_function_name)
    return df_combined


### Split data in train test and plot accuracies
def train_test_best_model(X,y,best_clf, best_params,test_size=0.4):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)
    np.random.seed(seed=42)
    best_estim = best_clf.best_estimator_
    best_estim.fit(X_train, y_train)
    y_pred_train = best_estim.predict(X_train)
    y_pred_test = best_estim.predict(X_test)
    train_score = accuracy_score(y_train, y_pred_train)
    test_score = accuracy_score(y_test, y_pred_test)

    fig = go.Figure(data=[
    go.Bar(name='Train Accuracy', x=['Train'], y=[train_score]),
    go.Bar(name='Test Accuracy with Train Test Split', x=['Test'], y=[test_score]),
    go.Bar(name='Test Accuracy with Cross validation', x=['Test CV'], y=[best_clf.best_score_])
    ])
    
    fig.update_layout(
        title='<b>Comparison of Train vs Test Accuracy and Test Accuracy Methods</b>',
        xaxis_title='Dataset',
        yaxis_title='Accuracy',
        barmode='group'
    )
    
    fig.show()
    
    return X_train, X_test, y_train, y_test, train_score, test_score

### Function that displays the decision areas of an SVM
def visu_border_svm(estim, X, y):

    best_estim = estim.best_estimator_
    ### We are taking the first steps in our pipeline
    X_stft = best_estim.named_steps['stft'].transform(X)
    X_scale = best_estim.named_steps['scaler'].transform(X_stft)
    X_reduced = best_estim.named_steps['pca'].transform(X_scale)

    ### We save our classifier separately
    classifier = best_estim.named_steps['SVM']
    classifier.fit(X_reduced, y)

    ### We create the grid for the display
    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # Prédiction sur la grille de points
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)


    ### Mapping dictionnary
    class_labels = {0: 'Avance', 1: 'Recule', 2: 'Tournegauche'}

    ### We make our display in plotly
    fig = go.Figure()
    
    ### Addition of decision zones
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, 100),
        y=np.linspace(y_min, y_max, 100),
        z=Z,
        # colorscale='coolwarm',
        showscale=False,
        opacity=0.8,

    ))

    ### Adding data points
    fig.add_trace(go.Scatter(
        x=X_reduced[:, 0],
        y=X_reduced[:, 1],
        mode='markers',
        marker=dict(
            color=y, 
            showscale=True,
            colorbar=dict(
                title='Class',
                tickvals=list(class_labels.keys()),
                ticktext=list(class_labels.values())
            ),
            line=dict(width=0.8, color='black')
        ),
        showlegend=False,
    ))
    
    ### Formatting the figure
    fig.update_layout(
        title="<b>Decision boundary with PCA-reduced data and " + best_estim.named_steps['SVM'].kernel + " kernel </b>",
        xaxis_title="PCA component 1",
        yaxis_title="PCA component 2",
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max])
    )

    print("Best parameters:", estim.best_params_)
    print("Best score:", estim.best_score_)

    return fig

### Function which calculates SVM for the stft & scaler & pca transformations with fixed values.
def clf_visu(X,y):
    
    loo = LeaveOneOut()
    pca_visu = PCA(n_components=2)
    scaler = StandardScaler()
    stft_class = STFT(idx_frequence_max = 90, stat = np.max)
    SVM = SVC(random_state = 42)

    pipe_visu_coeff = Pipeline(steps=[("stft", stft_class), ("scaler", scaler), ("pca", pca_visu), ("SVM", SVM)])

    ### for linear kernel
    param_grid_svm_visu_lin = {
        "SVM__kernel": ['linear'],
        "SVM__C": [1],
    }
    
    ### for gaussian kernel
    param_grid_svm_visu_gaus = {
        "SVM__kernel": ['rbf'],
        "SVM__C": [20],
        "SVM__gamma": ['scale'] ,
    }
    
    ### for polynomial kernel
    param_grid_svm_visu_poly = {
        "SVM__kernel": ['poly'],
        "SVM__C": [0.1],
        "SVM__gamma": ['auto'] ,
    }

    clf_visu_lin = GridSearchCV(pipe_visu_coeff, param_grid_svm_visu_lin, cv=loo, return_train_score=True)
    clf_visu_gaus = GridSearchCV(pipe_visu_coeff, param_grid_svm_visu_gaus, cv=loo, return_train_score=True)
    clf_visu_poly = GridSearchCV(pipe_visu_coeff, param_grid_svm_visu_poly, cv=loo, return_train_score=True)

    clf_visu_lin.fit(X, y)
    clf_visu_gaus.fit(X, y)
    clf_visu_poly.fit(X, y)

    return clf_visu_lin, clf_visu_gaus, clf_visu_poly


### Return the BIC score to pass to GridSearchCV
def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)

### Function that performs GridSearch for Gaussian mixtures used with fft & scale and stft & scale transformations
def gaussian_mixture_grid_search(X):
    pca_test = PCA(n_components=15)
    fft_class = FFT()
    stft_class = STFT(idx_frequence_max = 90, stat = np.mean)
    scaler = StandardScaler()
    
    ### We define a param grid for 2 & 3 cluster using different covariance type
    param_grid_gm = {
        "n_components": range(2,4),
        "covariance_type": ["spherical", "tied", "diag", "full"],
    
    }
    
    grid_search_fft = GridSearchCV(
        GaussianMixture(), param_grid=param_grid_gm, scoring=gmm_bic_score
    )
    
    grid_search_stft = GridSearchCV(
        GaussianMixture(), param_grid=param_grid_gm, scoring=gmm_bic_score
    )

    X_fft = fft_class.transform(X)
    X_stft = stft_class.transform(X)
    X_fft_scale = scaler.fit_transform(X_fft)
    X_stft_scale = scaler.fit_transform(X_stft)
    X_fft_pca = pca_test.fit_transform(X_fft_scale)

    grid_search_fft.fit(X_fft_pca)
    grid_search_stft.fit(X_stft_scale)

    return grid_search_fft, grid_search_stft

### Function that returns a dataframe with information from a Gaussian mixture grid search. Such as type of tranform, number of cluster, type of covariance, parameters and BIC score
def df_grid_search(grid_search_fft, grid_search_stft):
    df_fft = (
    pd.DataFrame(grid_search_fft.cv_results_)
    [["param_n_components", "param_covariance_type",  "mean_test_score"]]
    .assign(mean_test_score = lambda d: -d.mean_test_score)
    .rename(
    columns={
        "param_n_components": "Number of components",
        "param_covariance_type": "Type of covariance",
        "mean_test_score": "BIC score",
    }
    )   
    .assign(Transform = 'fft & scale & pca')
    .assign(Params = lambda d: d['Type of covariance'] + ' ' + d['Number of components'].astype('str'))
    [['Transform', 'Number of components', 'Type of covariance', 'Params', 'BIC score']]
    .sort_values(by = ['BIC score'])

    )
    df_stft = (
    pd.DataFrame(grid_search_stft.cv_results_)
    [["param_n_components", "param_covariance_type",  "mean_test_score"]]
    .assign(mean_test_score = lambda d: -d.mean_test_score)
    .rename(
    columns={
        "param_n_components": "Number of components",
        "param_covariance_type": "Type of covariance",
        "mean_test_score": "BIC score",
    }
    )   
    .assign(Transform = 'stft & scale')
    .assign(Params = lambda d: d['Type of covariance'] + ' ' + d['Number of components'].astype('str'))
    [['Transform', 'Number of components', 'Type of covariance', 'Params', 'BIC score']]
    .sort_values(by = ['BIC score'])
    )
    return df_fft, df_stft

### Function that plot the BIC score of the explanatory dataframe
def plot_BIC_score(df, transform):
    if transform == 'fft':
        title='<b>BIC score for fft & scale & pca transform</b>' 
    if transform == 'stft':
        title='<b>BIC score for stft & scale transform</b>' 

    (
        px
        .line(df, x='Number of components', y='BIC score', title=title, facet_col = 'Type of covariance')
        .show()
    )

### Custom confusion matrix that works for a Gaussian mixture grid search
def confusion_matrix_clust(grid_search, X):
    y_predict = grid_search.predict(X)
    
    if len(np.unique(y_predict)) == 2:
        cm = confusion_matrix(y_predict, np.where(genres == 'F', 1, 0))
        z = np.round(cm, 2)
        x_ = ['Class 1', 'Class 2']
        y_ =  ['Homme', 'Femme']
    if len(np.unique(y_predict)) == 3:
        cm = confusion_matrix(y_predict, label)
        z = np.round(cm, 2)
        x_ = ['Class 1', 'Class 2', 'Class 3']
        y_ =  ['Avance', 'Recule', 'Tournegauche']
        
    # change each element of z to type string for annotations
    z_text = [[str(y_) for y_ in x_] for x_ in z]
        
    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=x_, y=y_, annotation_text=z_text)
        
    # add title
    fig.update_layout(title_text='<b>Confusion matrix</b>',
                         )
        
    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                                x=0.5,
                                y=-0.15,
                                showarrow=False,
                                text="Predicted value",
                                xref="paper",
                                yref="paper"))
        
    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=100))
        
    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                                x=-0.1,
                                y=0.5,
                                showarrow=False,
                                text="Real value",
                                textangle=-90,
                                xref="paper",
                                yref="paper"))
        
    # put the x_axis annotation on the bottom
    fig.update_layout(
            xaxis=dict(side='bottom')
        )
        
    
    fig.show()

### Display preprocessing results
def display_best_params_table(best_params):
    params = list(best_params.keys())
    values = [str(value) for value in best_params.values()] 
    
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Parameter', 'Value']),
        cells=dict(values=[params, values]))
    ])
    
    fig.update_layout(
        title='<b>Best Parameters of the Classifier<b>'
    )
    
    fig.show()


### Function that returns a dataframe with the results per epoch (train/test) 
def compute_nn(X_train, X_test, y_train, y_test, model, lr, num_epochs, loss, wc=0):
    
    model_nn_1 = model
    optimizer = torch.optim.Adam(model_nn_1.parameters(),lr,weight_decay=wc)
    
    result_test_loss = []
    result_train_loss= []
    result_accuracy_train = []
    result_accuracy_test = []
        
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_acc = 0
        for i in range(X_train.shape[0]):
            y_pred_train = model(X_train[i])
            loss_value = loss(y_pred_train,  y_train[i])
            loss_value.backward()
            optimizer.step()
            train_losses.append(loss_value)
            if y_pred_train.argmax() == y_train[i].argmax() :
                train_acc += 1
                    
        model.eval()
        test_losses = []
        test_acc = 0
        for i in range(X_test.shape[0]):
            y_pred_test = model(X_test[i])
            loss_value_test = loss(y_pred_test,  y_test[i])
            test_losses.append(loss_value_test)  
            if y_pred_test.argmax() == y_test[i].argmax() :
                test_acc += 1
            
        result_train_loss.append(torch.stack(train_losses).mean().item())
        result_test_loss.append(torch.stack(test_losses).mean().item())
        result_accuracy_train.append(train_acc/X_train.shape[0])
        result_accuracy_test.append(test_acc/X_test.shape[0])
    
    tab =  pd.DataFrame(
        {
            "epoch" : np.arange(0,num_epochs,1),
            "train_loss" : result_train_loss, 
            "test_loss" : result_test_loss,
            "accuracy_train" : result_accuracy_train,
            "accuracy_test" : result_accuracy_test,
        }
        )
    return tab

### Plot accuracies et losses
def nn_plot(df):
    fig = make_subplots(rows=2, cols=1)
    
    fig.add_trace(
        go.Scatter(x=df.epoch, y=df.train_loss, mode='lines', name='Train loss'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.epoch, y=df.test_loss, mode='lines', name='Test loss'),
        row=1, col=1,
    )
    
    fig.add_trace(
        go.Scatter(x=df.epoch, y=df.accuracy_train, mode='lines', name='Train accuracy'),
        row=2, col=1,
    )
    
    fig.add_trace(
        go.Scatter(x=df.epoch, y=df.accuracy_test, mode='lines', name='Test accuracy'),
        row=2, col=1
    )
    
    fig.update_layout(title_text='<b>Neural network accuracy & loss plot</b>', width=1200, height=800)
    
    fig.update_yaxes(title_text='Loss value', row=1, col=1)
    fig.update_xaxes(title_text='# epoch', row=2, col=1)
    fig.update_xaxes(title_text='# epoch', row=1, col=1)
    fig.update_yaxes(title_text='Accuracy value', row=2, col=1)

    fig.show()