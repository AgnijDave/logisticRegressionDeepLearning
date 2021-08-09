# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 23:00:52 2021
@author: Agnij
"""

import pandas as pd
import numpy as np
import pickle as pk
import sys
import traceback
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score

from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

df_test = pd.read_csv('test.csv')

df_train_features = pd.read_csv('train.csv')
df_train_labels = pd.read_csv('trainLabels.csv')

classes = list(df_train_labels.columns)
classes.pop(0)

## Total 33 classes
## 1. Check if any of the classes are the same to avoid redundancy
## 2. 

'''
## 1.
identical=[]
for col in classes:
    for col_ in classes:
        if col != col_:
            if df_train_labels[col].equals(df_train_labels[col_]):
                identical.append([str(col)+'=='+str(col_)])
                print('\n',str(col),'==',str(col_))
                
'''
             
df_train_labels_split = df_train_labels.head(9999)
#df_train_features.describe()

## Dealing with null values
train_set = df_train_features.isnull().sum()
test_set = df_test.isnull().sum()

''' 
same set of columns across train set and test set have null values ~ unreliable featuress
 1. drop these columns as already large number of features available
 2. drop samples/rows having nan values [train-set reduces by 26% i.e. 1/4th and test-set(reduced by 27%)]

 axis = 0 || remove rows having nan/empty values in any column
 axis =1 || remove columns containing null/nan values
 
*labels/classes - A lot of classes have a class imbalance in the dataset, which is throwing off the model
'''
df_train = df_train_features.dropna(how="any", axis=1)
df_test = df_test.dropna(how="any", axis=1)

df_labels = df_train_labels_split[df_train_labels_split.index.isin(df_train.index)]

## Convert all YES/NO column values to an integer representation
df_train = df_train.replace({'YES':1, 'NO':0})
df_test = df_test.replace({'YES':1, 'NO':0})

df_train.reset_index(inplace =True)
df_labels.reset_index(inplace=True)
df_test.reset_index(inplace =True)

assert df_train['id'].equals(df_labels['id'])

df_train.drop(['id','index'], axis=1, inplace=True)
df_labels.drop(['id','index'], axis=1, inplace=True)
test_ids = df_test['id'].to_list()
df_test.drop(['id','index'], axis=1, inplace=True)

# separate numerical and categorical features
df_train_numerical = df_train.loc[:,df_train.dtypes!=np.object]
df_train_categorical = df_train.loc[:,df_train.dtypes==np.object]

df_test_numerical = df_test.loc[:,df_test.dtypes!=np.object]
df_test_categorical = df_test.loc[:,df_test.dtypes==np.object]

def explore_dataset(df_train_numerical, df_train_categorical):
    
    ## get correlation of numerical features with diferent labels
    
    df_n1 = df_train_numerical.copy(deep=True)
    df_n1['y1'] = df_labels['y1']
    corr1 = df_n1.corr()
    
    ## testing some of the categorical features
    
    df_train_categorical.x61.value_counts()
    df_c1 = df_train_categorical.copy(deep=True)
    df_c1['y1'] = df_labels['y1']
    
    # feature value x61[value 316] has all labels mapped to 0
    np.unique(df_c1[df_c1['x61'].str.contains('MZZbXga8gvaCBqWpzrh2iKdOkcsz/bG/z4BVjUnqWT0=')]['y1'], return_counts=True)
    
    df_train_categorical.x91.value_counts()
    df_c1['y33'] = df_labels['y33']
    # feature values x64[max value 351], 246 - 0, 105 - 1
    np.unique(df_c1[df_c1['x91'].str.contains('WV5vAHFyqkeuyFB5KVNGFOBuwjkUGKYc8wh9QfpVzAA=')]['y33'], return_counts=True)
    
    return corr1

#num_corr = explore_dataset(df_train_numerical, df_train_categorical)

def check_args(original_func):
    def wrap_func(df, dim=None, df_=pd.DataFrame, n_components=2):
        print('\nDecorator - validity check of passed arguments\n')
        #print('---->>',dim, '\t',df_)
        
        if dim == 2 and df_.empty:
            sys.exit('''For Dimensionality reduction, you have selected Linear Discrimant Analysis
            which is a supervised Algorithm and thus
            requires another df having the respective labels''')
        elif dim==2 and n_components>=len(pd.unique(df_[list(df_.columns)[-1]])):
            sys.exit('''For LDA, the n-components value needs to be max(classes)-1''')
        else:
            print('validity check - passed\n')
            return original_func(df, dim, df_, n_components)
    return wrap_func

@check_args
def preprocessing(df, dim=None, df_=pd.DataFrame, n_components=2):
    '''
    input params-
    dim - default None, 1-PCA, 2-LDA, 3-Kernel PCA
    df - training data
    df_ - default None,labels
    
    transforms dataset using feature scaling and applies specified dimensionality reduction technique
    *note - LDA requires labels along with dataset
    '''
    #print('*******************************')
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    df = sc.fit_transform(df)
    
    
    if dim == 1:
        # unsupervised
        from sklearn.decomposition import PCA
        pca = PCA(0.99)
        #pca = PCA(n_components=n_components)
        df = pca.fit_transform(df)
    
        return df, sc, pca
    
    elif dim == 2:
        # supervised
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        lda = LDA(n_components = n_components)
        df = lda.fit_transform(df, np.ravel(df_))
        
        return df, sc, lda
    
    elif dim == 3:
        #print(n_components)
        from sklearn.decomposition import KernelPCA
        kpca = KernelPCA(n_components = n_components, kernel='rbf')
        df = kpca.fit_transform(df)
                
        return df, sc, kpca
    
    else:
        #print(df)
        return df, sc, None
    
## plain logistic regression model defn.
def create_model(learning_rate, feature_layer, metrics, regularizer):
    model = tf.keras.models.Sequential()
    model.add(feature_layer)
    
    model.add(tf.keras.layers.Dense(units=1,input_shape=(1,),
                                    activation=tf.sigmoid, kernel_regularizer = regularizer))
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate), #Adagrad
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=metrics)
    
    return model
    
def train_model(model, dataset, epochs, label_name,
                batch_size=None, shuffle=True):
    features={name:np.array(value) for name, value in dataset.items()}
    label=np.array(features.pop(label_name))
    
    #print(label)
    #print(np.unique(label, return_counts=True))
    
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=shuffle)
    
    return history

def plot(epochs_, hist, plot_metrics):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    
    for m in plot_metrics:
        x = hist[m]
        plt.plot(epochs_[1:], x[1:], label=m)
    
    plt.legend()


'''
my_data = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])
my_column_names = ['rain', 'activity']
my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)

z = preprocessing(my_dataframe, 3, n_components=1)

my_data_1 = np.array([[14 ,5], [19,8], [25, 15], [31, 18], [40, 24]])
my_dataframe_1 = pd.DataFrame(data=my_data_1, columns=my_column_names)
my_dataframe_1 = z[1].transform(my_dataframe_1)
my_dataframe_1 = z[2].transform(my_dataframe_1)

'''
# PCA(0.99) -> 88% test-set f1, label y33
preprocessed_data = preprocessing(df_train_numerical, dim=1)

# LDA -> 87% test-set f1, label y33
#preprocessed_data = preprocessing(df_train_numerical, dim=2, df_=pd.DataFrame(data=df_labels['y33'], columns=['y33']), n_components=1)

# Kernel PCA -> 84% test-set f1, label y33
#preprocessed_data = preprocessing(df_train_numerical, dim =3, n_components=5)

df_train_numerical = pd.DataFrame(data=preprocessed_data[0], columns=[str(i)+'_' for i in range(len(preprocessed_data[0][0]))])
df_train_preprocessed = pd.concat([df_train_numerical, df_train_categorical], axis=1)

df_test_numerical = preprocessed_data[1].transform(df_test_numerical)
df_test_numerical = preprocessed_data[2].transform(df_test_numerical)
df_test_numerical = pd.DataFrame(data=df_test_numerical, columns =[str(i)+'_' for i in range(len(preprocessed_data[0][0]))])
df_test_preprocessed = pd.concat([df_test_numerical, df_test_categorical], axis=1)

'''
# Save the pickled Files
pk.dump(preprocessed_data[1], open("sc.pkl", "wb"))
pk.dump(preprocessed_data[2], open("pca.pkl","wb"))

# Load the pickled files
pca = pk.load(open("pca.pkl",'rb'))
sc = pk.load(open("sc.pkl", 'rb'))
'''

import tensorflow as tf
from tensorflow.keras import layers

feature_columns=[]
for col in list(df_train_numerical.columns):
    col = tf.feature_column.numeric_column(col)
    feature_columns.append(col)
    
for col in list(df_train_categorical.columns):
    col = tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_hash_bucket(col, hash_bucket_size=8000), dimension=8)
    feature_columns.append(col)
    
feature_layer = layers.DenseFeatures(feature_columns)
#feature_layer(dict(df_train_preprocessed))

## Hyper-parameters
learning_rate = 0.01
# classification_threshold = 0.52

l1=0
l2=0.03
regularizer = tf.keras.regularizers.l1_l2(l1=l1, l2=l2)

epochs = 10
batch_size = 200
metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
        ]
plot_metrics = ['loss', 'precision', 'recall', 'accuracy', 'auc']
models = []
myformat = lambda x: "%.2f" % x

def test_f1(df_train_preprocessed, df_labels, estimator = None):
    f1_list, accuracy_list= [], []
    for col in list(df_labels.columns):
        if col == 'y33': # y1 y2, y5
            print('Training for label -> ',col,'\n\n')
            label_name=col
            X_train, X_test, y_train, y_test = train_test_split(df_train_preprocessed, df_labels, test_size = 0.2, random_state = 0)
            
            if not estimator:
                # Check for class imbalance -> use SMOTE upsampling + downsampling
                counter = Counter(y_train[label_name])
                ## %more in majority class
                imbalance_percentage = abs(counter[0] - counter[1])/(counter[0]+counter[1])
                
                if imbalance_percentage>0.3:
                    kn = min([counter[0], counter[1]])
                    if kn==1:
                        kn=1
                    elif kn<6:
                        kn-=1
                    else:
                        kn=5
                    print('++++++',kn,'++++++')
                    over = SMOTENC(categorical_features=[56, 57], sampling_strategy=0.3, k_neighbors=kn)
                    under = RandomUnderSampler(sampling_strategy=0.5)
                    steps = [('o', over), ('u', under)]
                    pipeline = Pipeline(steps=steps)
                    print(counter,'imb% = ',imbalance_percentage,'\n')
                    
                    try:
                        X_train, y_train_ = pipeline.fit_resample(X_train, y_train[label_name])
                    except:
                        print('*************',counter,'  for label -> ',label_name,'\n')
                        print(traceback.format_exc())
                        y_train_=y_train[label_name]
                        pass
                    
                    print(Counter(y_train_))
                    df = X_train.copy(deep=True)
                    df[label_name] = y_train_
                    
                else:
                    print('No sampling required ->',Counter(y_train[label_name]))
                    df = X_train.copy(deep=True)
                    df[label_name] = y_train[label_name]
                
                model = None
                model = create_model(learning_rate, feature_layer, metrics, regularizer)
                history = train_model(model, df, epochs=epochs, label_name=label_name,
                                      batch_size=batch_size)
                
                hist = pd.DataFrame(history.history)
                epochs_ = hist.index
                
                plot(epochs_, hist, plot_metrics)
                
                ## Testing on the Test set
                X_test[label_name] = y_test[label_name]
                features={name:np.array(value) for name, value in X_test.items()}
                label=np.array(features.pop(label_name))
                
                prediction_y33_list = model.predict(features).tolist()
                prediction_y33_list = [float(myformat(x[0])) for x in prediction_y33_list]
                prediction_y33_list = [1 if x > 0.5 else 0 for x in prediction_y33_list]
                
                cm = confusion_matrix(label, prediction_y33_list)
                print(cm)
                print(accuracy_score(label, prediction_y33_list))
                report = classification_report(label, prediction_y33_list, output_dict=True)
                print(classification_report(label, prediction_y33_list))
                
                f1_list.append(report['macro avg']['f1-score'])
                accuracy_list.append(accuracy_score(label, prediction_y33_list))
                
            else:
                from sklearn.ensemble import RandomForestClassifier
                '''
                from sklearn.preprocessing import OneHotEncoder
                
                onehotencoder = OneHotEncoder(handle_unknown='ignore')
                X_train = onehotencoder.fit_transform(X_train)
                X_test = onehotencoder.transform(X_test)
                '''
                #print('\n',len(list(X_train.columns)), '+++','\n')
                
                X_train.drop(['x61','x91'], axis=1, inplace=True)
                X_test.drop(['x61','x91'], axis=1, inplace=True)
                
                model = None
                model = RandomForestClassifier()
                
                model.fit(X_train, y_train[label_name])
                label = y_test[label_name]
                prediction = model.predict(X_test)
                
                cm = confusion_matrix(label, prediction)
                print(cm)
                print(accuracy_score(label, prediction))
                report = classification_report(label, prediction, output_dict=True)
                print(classification_report(label, prediction))
                np.unique(prediction, return_counts=True)
                
                f1_list.append(report['macro avg']['f1-score'])
                accuracy_list.append(accuracy_score(label, prediction_y33_list))
        
        #break
    return f1_list, accuracy_list
        
## input estimator value for RandomForrestClassifier
# f1_list, accuracy_list =  test_f1(df_train_preprocessed, df_labels, estimator=1)
    
#f1_list, accuracy_list = test_f1(df_train_preprocessed, df_labels)

def get_results(df_train_preprocessed, df_labels, df_test_preprocessed, test_ids):
    
    for col in list(df_labels.columns):
        if col != '_':
            print('Training for label -> ',col,'\n\n')
            label_name=col
            
            # Check for class imbalance -> use SMOTE upsampling + downsampling
            counter = Counter(df_labels[label_name])
            ## %more in majority class
            imbalance_percentage = abs(counter[0] - counter[1])/(counter[0]+counter[1])
            if imbalance_percentage>0.3:
                kn = min([counter[0], counter[1]])
                if kn==1:
                    kn=1
                elif kn<6:
                    kn-=1     
                else:
                    kn=5
                over = SMOTENC(categorical_features=[56, 57], sampling_strategy=0.3, k_neighbors=kn)
                under = RandomUnderSampler(sampling_strategy=0.5)
                steps = [('o', over), ('u', under)]
                pipeline = Pipeline(steps=steps)
                print(counter,'imb% = ',imbalance_percentage,'\n')
                
                try:
                    df_train_preprocessed_, y_train_ = pipeline.fit_resample(df_train_preprocessed, df_labels[label_name])
                except:
                    print('*************',counter,'  for label -> ',label_name,'\n')
                    print(traceback.format_exc())
                    y_train_=df_labels[label_name]
                    pass
                
                print(Counter(y_train_))
                df = df_train_preprocessed_.copy(deep=True)
                df[label_name] = y_train_
                
            else:
                print('No sampling required ->',Counter(df_labels[label_name]))
                df = df_train_preprocessed.copy(deep=True)
                df[label_name] = df_labels[label_name]
            
            model = None
            model = create_model(learning_rate, feature_layer, metrics, regularizer)
            history = train_model(model, df, epochs=epochs, label_name=label_name,
                                  batch_size=batch_size)
            
            '''
            #plot metrics
            hist = pd.DataFrame(history.history)
            epochs_ = hist.index
            
            plot(epochs_, hist, plot_metrics)
            '''
            ## Predictions on the Test set
            features={name:np.array(value) for name, value in df_test_preprocessed.items()}
            
            prediction_list = []
            prediction_list = model.predict(features).tolist()
            prediction_list = [float(myformat(x[0])) for x in prediction_list]
            prediction_list = [1 if x > 0.5 else 0 for x in prediction_list]
            
            for i, p in enumerate(prediction_list):
                k = list(test_ids[i].keys())
                test_ids[i][k[0]].append(p)
            
            #break
    return test_ids
    
test_ids = [{_:[]} for _ in test_ids]
preds = get_results(df_train_preprocessed, df_labels, df_test_preprocessed, test_ids)

columns_ = ['id_label', 'pred']
a = []
for d in preds:
    k_ = list(d.keys())
    #print(k_)
    #print(d[k_[0]])
    for i, val in enumerate(d[k_[0]]):
        b = []
        #print(str(k_[0])+'_y'+str(i+1),'----',val,'\n')
        b.append(str(k_[0])+'_y'+str(i+1))
        b.append(val)
        
        a.append(b)

df_submission = pd.DataFrame(data=a, columns=columns_)
df_submission.to_csv('finalSubmissionFile.csv', index=False)

'''
from collections import Counter
Counter(prediction_y33_list).keys()
Counter(prediction_y33_list).values()

# y1 dict_values([1477, 4])
# y33 dict_values([513, 968])

for col in list(df_labels.columns):
    print(col, '\t', np.unique(df_labels[col], return_counts=True))
'''