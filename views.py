from django.shortcuts import render
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm



def home(request):
    return render(request,  'home.html')


def predict(request):
    return render(request,  'predict.html')


def result(request):
    df = pd.read_csv(r'DiabetesPrediction/diabetes.csv')
    X = df.loc[:, df.columns != 'Outcome']
    y = df['Outcome']

    scaler = StandardScaler()
    scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pred = model.predict([[val1,val2,val3,val4,val5,val6,val7,val8]])

    result2 = ' '
    if pred == [1]:
        result2 = 'positive'
    else:
        result2 = 'negative'




    return render(request,  'predict.html', {'result2':result2})

