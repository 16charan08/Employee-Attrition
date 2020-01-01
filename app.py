import csv
import os
from distutils.util import execute

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from sklearn.externals import joblib
from werkzeug.utils import secure_filename
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib import  pyplot as pp
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import metrics
from sklearn.model_selection import train_test_split


app = Flask(__name__)

def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'

# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','csv','rtf'])

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/')
def EDA():
    return render_template('index.html')

@app.route('/')
def Findings():
    return render_template('index.html')

@app.route('/EDA', methods=['GET', 'POST'])
def index_func():
    if request.method == 'POST':
        return redirect(url_for('EDA'))
    return render_template('EDA.html')

@app.route('/Findings', methods=['GET', 'POST'])
def index_fun():
    if request.method == 'POST':
        return redirect(url_for('findings'))
    return render_template('findings.html')

@app.route('/upload', methods=['POST'])
def upload():
    uploaded_files = request.files.getlist("file[]")

    filenames = []

    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
  #  path = os.getcwd()+"/"+app.config['UPLOAD_FOLDER']
  #  print(path)
    files = []

    def prob(mean, sd, x):
        exponent = np.exp(-((x - mean) ** 2 / (2 * sd ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * sd)) * exponent

    def predictRecord(record, posSum, negSum):
        posProb = 1
        negProb = 1
        for index in range(len(record)):
            posProb *= prob(posSum[index][0], posSum[index][1], record[1][index])
            negProb *= prob(negSum[index][0], negSum[index][1], record[1][index])
        if posProb > negProb:
            return 1
        else:
            return 0

    def predict(test, posSum, negSum):
        result = []
        for record in test.iterrows():
            result.append(predictRecord(record, posSum, negSum))
        return result

    def summary(trainData):
        mean = trainData.mean()
        stdev = trainData.std()
        summary = []
        for m, s in zip(mean, stdev):
            summary.append((m, s))
        return summary
    import pandas as pd
    import numpy as np

    Singledata = pd.read_csv('uploads/Singleemployee.csv')
    Singledata = Singledata.drop(columns='StandardHours')
    Singledata = Singledata.drop(columns='EmployeeCount')
    Singledata = Singledata.drop(columns='EmployeeNumber')
    Singledata = Singledata.drop(columns='StockOptionLevel')

    cat_col = Singledata.select_dtypes(exclude=np.number)

    numerical_col = Singledata.select_dtypes(include=np.number)

    Singledata.BusinessTravel.value_counts()

    Singledata.columns.shape
    one_hot_categorical_variables = pd.get_dummies(cat_col)
    one_hot_categorical_variables.head()
    Singledata = pd.concat([numerical_col, one_hot_categorical_variables], sort=False, axis=1)

    trainDatasingle = pd.DataFrame(Singledata)

    data = pd.read_csv("uploads/IBM.csv")
    data = data.drop(columns='StandardHours')
    data = data.drop(columns='EmployeeCount')
    data = data.drop(columns='EmployeeNumber')
    data = data.drop(columns='StockOptionLevel')
    data['Attrition'] = data['Attrition'].map(lambda x: 1 if x == 'Yes' else 0)

    cat_col = data.select_dtypes(exclude=np.number)

    numerical_col = data.select_dtypes(include=np.number)

    data.BusinessTravel.value_counts()

    data.columns.shape
    one_hot_categorical_variables = pd.get_dummies(cat_col)
    one_hot_categorical_variables.head()
    data = pd.concat([numerical_col, one_hot_categorical_variables], sort=False, axis=1)

    trainData = pd.DataFrame(data)

    dataNegative = trainData[trainData.Attrition == 0]
    dataPositive = trainData[trainData.Attrition == 1]

    dataNegative = dataNegative.drop(columns='Attrition')
    dataPositive = dataPositive.drop(columns='Attrition')

    summaryPositive = summary(dataPositive)

    summaryNegative = summary(dataNegative)
    predicted_single = predict(trainDatasingle, summaryPositive, summaryNegative)

    if predicted_single == [0]:
        predicted_single = "No"
    elif predicted_single == [1]:
        predicted_single = "Yes"


#------------------Imple of Decision tree for single prediction----------------------------------------------


    df = pd.read_csv('uploads/IBM.csv')
    df = df.drop(["EmployeeNumber", "Over18", "EmployeeCount", "StandardHours"], axis=1)
    df = pd.concat([df.loc[:, df.columns != 'Attrition'], df.Attrition], axis=1, sort=False)

    df['Education'] = df['Education'].astype(object)
    df['EnvironmentSatisfaction'] = df['EnvironmentSatisfaction'].astype(object)
    df['JobInvolvement'] = df['JobInvolvement'].astype(object)
    df['JobLevel'] = df['JobLevel'].astype(object)
    df['JobSatisfaction'] = df['JobSatisfaction'].astype(object)
    df['PerformanceRating'] = df['PerformanceRating'].astype(object)
    df['RelationshipSatisfaction'] = df['RelationshipSatisfaction'].astype(object)
    df['StockOptionLevel'] = df['StockOptionLevel'].astype(object)
    df['TrainingTimesLastYear'] = df['TrainingTimesLastYear'].astype(object)
    df['WorkLifeBalance'] = df['WorkLifeBalance'].astype(object)
    df = df.sample(frac=1)

    X = df.loc[:, df.columns != 'Attrition']
    y = df.Attrition
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=1)  # 70% training and 30% test
    train_data = pd.concat([X_train, y_train], axis=1, sort=False)
    test_data = pd.concat([X_test, y_test], axis=1, sort=False)

    def is_numeric(value):
        """Test if a value is numeric."""
        return isinstance(value, int) or isinstance(value, float)

    header = list(df)

    class Question:

        def __init__(self, column, value):
            self.column = column
            self.value = value

        def match(self, example):
            # Compare the feature value in an example to the
            # feature value in this question.
            val = example[self.column]
            if is_numeric(val):
                return val >= self.value
            else:
                return val == self.value

        def __repr__(self):
            # This is just a helper method to print
            # the question in a readable format.
            condition = "=="
            if is_numeric(self.value):
                condition = ">="
            return "Is %s %s %s?" % (
                header[self.column], condition, str(self.value))

    def partition(rows, question):

        true_rows, false_rows = [], []
        for row in rows:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows

    def class_counts(rows):
        counts = {}  # a dictionary of label -> count.
        for row in rows:
            # in our dataset format, the label is always the last column
            label = row[-1]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts

    def gini(rows):

        counts = class_counts(rows)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(rows))
            impurity -= prob_of_lbl ** 2
        return impurity

    def info_gain(left, right, current_uncertainty):

        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

    def find_best_split(rows):

        best_gain = 0  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        current_uncertainty = gini(rows)
        n_features = len(rows[0]) - 1  # number of columns

        for col in range(n_features):  # for each feature

            values = set([row[col] for row in rows])  # unique values in the column

            for val in values:  # for each value

                question = Question(col, val)

                # try splitting the dataset
                true_rows, false_rows = partition(rows, question)

                # Skip this split if it doesn't divide the
                # dataset.
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue

                # Calculate the information gain from this split
                gain = info_gain(true_rows, false_rows, current_uncertainty)

                # You actually can use '>' instead of '>=' here
                # but I wanted the tree to look a certain way for our
                # toy dataset.
                if gain >= best_gain:
                    best_gain, best_question = gain, question

        return best_gain, best_question

    class Leaf:

        def __init__(self, rows):
            self.predictions = class_counts(rows)

    class Decision_Node:

        def __init__(self,
                     question,
                     true_branch,
                     false_branch):
            self.question = question
            self.true_branch = true_branch
            self.false_branch = false_branch

    def build_tree(rows):

        gain, question = find_best_split(rows)

        if gain == 0:
            return Leaf(rows)

        true_rows, false_rows = partition(rows, question)

        true_branch = build_tree(true_rows)

        false_branch = build_tree(false_rows)

        return Decision_Node(question, true_branch, false_branch)

    def classify(row, node):

        if isinstance(node, Leaf):
            return node.predictions
        if node.question.match(row):
            return classify(row, node.true_branch)
        else:
            return classify(row, node.false_branch)

    def print_leaf(counts):
        total = sum(counts.values()) * 1.0
        probs = {}
        for lbl in counts.keys():
            probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
        return probs

    my_tree = build_tree(train_data.values)  # without cv
    actual = list(y_test)
    pred = []
    for row in test_data.values:
        pred.append(list(print_leaf(classify(row, my_tree)).keys())[0])
    from sklearn import metrics
    print("Accuracy with out CV with random splitting: ", metrics.accuracy_score((actual), pred))





    import pandas as pd
    df = pd.read_csv('uploads/Singleemployee.csv')
    df = df.drop(["EmployeeNumber", "Over18", "EmployeeCount", "StandardHours"], axis=1)
    df['Education'] = df['Education'].astype(object)
    df['EnvironmentSatisfaction'] = df['EnvironmentSatisfaction'].astype(object)
    df['JobInvolvement'] = df['JobInvolvement'].astype(object)
    df['JobLevel'] = df['JobLevel'].astype(object)
    df['JobSatisfaction'] = df['JobSatisfaction'].astype(object)
    df['PerformanceRating'] = df['PerformanceRating'].astype(object)
    df['RelationshipSatisfaction'] = df['RelationshipSatisfaction'].astype(object)
    df['StockOptionLevel'] = df['StockOptionLevel'].astype(object)
    df['TrainingTimesLastYear'] = df['TrainingTimesLastYear'].astype(object)
    df['WorkLifeBalance'] = df['WorkLifeBalance'].astype(object)

    row = list(df.values[0])

    print(list(print_leaf(classify(row, my_tree)).keys())[0])



#-----------------------NNET FOR SINGLE ROW-----------------
    import pandas as pd
    import numpy as np
    import math

    def sigmoid(data_array):
        for i in range(0, len(data_array)):
            for j in range(0, len(data_array[i])):
                data_array[i, j] = 1 / (1 + math.exp(-data_array[i, j]))
        return data_array

    def prediction(file):

        original_data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv").drop(["EmployeeCount", "EmployeeNumber",
                                                                                   "Over18", "StandardHours",
                                                                                   "Attrition"], axis=1)
        columns = pd.get_dummies(original_data, drop_first=True).columns

        data = pd.read_csv(file)

        # Deleting Columns EmployeeCount, EmployeeNumber, Over18 and StandardHours
        data = data.drop(["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"], axis=1)

        if len(data) > 1:
            y_temp = pd.get_dummies(data.Attrition, drop_first=True)
            y = y_temp.iloc[:].values
            data = data.drop(["Attrition"], axis=1)

        # Separating Data
        data = pd.get_dummies(data)
        uploaded_file_columns = list(data.columns)
        data = data.to_numpy()
        x = []
        for i in range(0, len(data)):
            to_append = []
            for j in range(0, len(columns)):
                if columns[j] in uploaded_file_columns:
                    to_append_data = data[i, uploaded_file_columns.index(columns[j])]
                    to_append.append(to_append_data)
                else:
                    to_append.append(0)
            x.append(to_append)
        w1_file = np.genfromtxt('Epoch 99 W1', delimiter=',')
        w2_file = (np.genfromtxt('Epoch 99 W2', delimiter=','))
        w2_file = np.reshape(w2_file, [len(w2_file), 1])
        accuracy = 0
        for i in range(0, len(x)):
            X_testi_reshaped = np.reshape(x[i], [1, len(x[i])])
            if len(x) > 1:
                Y_testi_reshaped = np.reshape(y[i], [1, len(y[i])])
            dot1 = np.dot(X_testi_reshaped, w1_file)
            sigmoid1 = sigmoid(dot1)
            dot2 = np.array(np.dot(sigmoid1, w2_file))
            sigmoid2 = sigmoid(dot2)
            if sigmoid2[0, 0] >= 0.5:
                compareArray = np.array([1])
            else:
                compareArray = np.array([0])

            if len(x) > 1:
                if np.array_equal(compareArray, Y_testi_reshaped[0]):
                    accuracy += 1
            else:
                if np.array_equal(compareArray, np.array([1])):
                    return "Yes"
                else:
                    return "No"
        return accuracy / len(x)

    print(prediction("uploads/Singleemployee.csv"))

    return render_template("output1.html", out= "ATTRITION:" ,output= predicted_single, out1= list(print_leaf(classify(row, my_tree)).keys())[0], out2= prediction("uploads/Singleemployee.csv"))

@app.route('/upload2', methods=['POST'])
def upload2():
    uploaded_files = request.files.getlist("file[]")

    filenames = []

    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)

 #   path = os.getcwd()+"/"+app.config['UPLOAD_FOLDER']
  #  print(path)

    def prob(mean, sd, x):
        exponent = np.exp(-((x - mean) ** 2 / (2 * sd ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * sd)) * exponent

    def predictRecord(record, posSum, negSum):
        posProb = 1
        negProb = 1
        for index in range(len(record)):
            posProb *= prob(posSum[index][0], posSum[index][1], record[1][index])
            negProb *= prob(negSum[index][0], negSum[index][1], record[1][index])
        if posProb > negProb:
            return 1
        else:
            return 0

    def predict(test, posSum, negSum):
        result = []
        for record in test.iterrows():
            result.append(predictRecord(record, posSum, negSum))
        return result

    def summary(trainData):
        mean = trainData.mean()
        stdev = trainData.std()
        summary = []
        for m, s in zip(mean, stdev):
            summary.append((m, s))
        return summary
    import pandas as pd
    import numpy as np

    data = pd.read_csv("uploads/IBM.csv")
    data = data.drop(columns='StandardHours')
    data = data.drop(columns='EmployeeCount')
    data = data.drop(columns='EmployeeNumber')
    data = data.drop(columns='StockOptionLevel')
    data['Attrition'] = data['Attrition'].map(lambda x: 1 if x == 'Yes' else 0)

    cat_col = data.select_dtypes(exclude=np.number)

    numerical_col = data.select_dtypes(include=np.number)

    data.BusinessTravel.value_counts()

    data.columns.shape
    one_hot_categorical_variables = pd.get_dummies(cat_col)
    one_hot_categorical_variables.head()
    data = pd.concat([numerical_col, one_hot_categorical_variables], sort=False, axis=1)
    from sklearn.model_selection import train_test_split
    trainwcv, testwcv = train_test_split(data, test_size=0.30, random_state=15, stratify=data['Attrition'])
    trainDatawcv = pd.DataFrame(trainwcv)
    testDatawcv = pd.DataFrame(testwcv)
    # print(trainData)
    # print(testData)

    dataNegativewcv = trainDatawcv[trainDatawcv.Attrition == 0]
    dataPositivewcv = trainDatawcv[trainDatawcv.Attrition == 1]

    dataNegativewcv = dataNegativewcv.drop(columns='Attrition')
    dataPositivewcv = dataPositivewcv.drop(columns='Attrition')

    summaryPositivewcv = summary(dataPositivewcv)
    # print(summaryPositive)

    summaryNegativewcv = summary(dataNegativewcv)
    # print(summaryNegative)

    y1trainwcv = trainDatawcv['Attrition']
    x1trainwcv = trainDatawcv.drop('Attrition', axis=1)
    predicted_train = predict(x1trainwcv, summaryPositivewcv, summaryNegativewcv)

    yTestwcv = testDatawcv['Attrition']
    xTestwcv = testDatawcv.drop('Attrition', axis=1)

    predicted_resultwcv = predict(xTestwcv, summaryPositivewcv, summaryNegativewcv)
   # print(accuracy_score(yTestwcv, predicted_resultwcv))
   # print(classification_report(yTestwcv, predicted_resultwcv))

#---------------------------Imple of Decision tree for accuracy ------------------------------------------------------------------------------
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn import metrics

    df = pd.read_csv('uploads/IBM.csv')
    df = df.drop(["EmployeeNumber", "Over18", "EmployeeCount", "StandardHours"], axis=1)
    df = pd.concat([df.loc[:, df.columns != 'Attrition'], df.Attrition], axis=1, sort=False)

    df['Education'] = df['Education'].astype(object)
    df['EnvironmentSatisfaction'] = df['EnvironmentSatisfaction'].astype(object)
    df['JobInvolvement'] = df['JobInvolvement'].astype(object)
    df['JobLevel'] = df['JobLevel'].astype(object)
    df['JobSatisfaction'] = df['JobSatisfaction'].astype(object)
    df['PerformanceRating'] = df['PerformanceRating'].astype(object)
    df['RelationshipSatisfaction'] = df['RelationshipSatisfaction'].astype(object)
    df['StockOptionLevel'] = df['StockOptionLevel'].astype(object)
    df['TrainingTimesLastYear'] = df['TrainingTimesLastYear'].astype(object)
    df['WorkLifeBalance'] = df['WorkLifeBalance'].astype(object)
    df = df.sample(frac=1)

    X = df.loc[:, df.columns != 'Attrition']
    y = df.Attrition
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=1)  # 70% training and 30% test
    train_data = pd.concat([X_train, y_train], axis=1, sort=False)
    test_data = pd.concat([X_test, y_test], axis=1, sort=False)

    def is_numeric(value):
        """Test if a value is numeric."""
        return isinstance(value, int) or isinstance(value, float)

    header = list(df)

    class Question:
        def __init__(self, column, value):
            self.column = column
            self.value = value

        def match(self, example):
            val = example[self.column]
            if is_numeric(val):
                return val >= self.value
            else:
                return val == self.value

        def __repr__(self):
            condition = "=="
            if is_numeric(self.value):
                condition = ">="
            return "Is %s %s %s?" % (
                header[self.column], condition, str(self.value))

    def partition(rows, question):
        true_rows, false_rows = [], []
        for row in rows:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows

    def class_counts(rows):
        counts = {}  # a dictionary of label -> count.
        for row in rows:
            label = row[-1]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts

    def gini(rows):
        counts = class_counts(rows)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(rows))
            impurity -= prob_of_lbl ** 2
        return impurity

    def info_gain(left, right, current_uncertainty):
        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

    def find_best_split(rows):
        best_gain = 0  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        current_uncertainty = gini(rows)
        n_features = len(rows[0]) - 1  # number of columns

        for col in range(n_features):  # for each feature

            values = set([row[col] for row in rows])  # unique values in the column

            for val in values:  # for each value

                question = Question(col, val)

                # try splitting the dataset
                true_rows, false_rows = partition(rows, question)

                # Skip this split if it doesn't divide the
                # dataset.
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue

                # Calculate the information gain from this split
                gain = info_gain(true_rows, false_rows, current_uncertainty)

                # You actually can use '>' instead of '>=' here
                # but I wanted the tree to look a certain way for our
                # toy dataset.
                if gain >= best_gain:
                    best_gain, best_question = gain, question

        return best_gain, best_question

    class Leaf:
        def __init__(self, rows):
            self.predictions = class_counts(rows)

    class Decision_Node:
        def __init__(self,
                     question,
                     true_branch,
                     false_branch):
            self.question = question
            self.true_branch = true_branch
            self.false_branch = false_branch

    def build_tree(rows):
        gain, question = find_best_split(rows)
        if gain == 0:
            return Leaf(rows)
        true_rows, false_rows = partition(rows, question)
        true_branch = build_tree(true_rows)
        false_branch = build_tree(false_rows)
        return Decision_Node(question, true_branch, false_branch)

    def classify(row, node):
        if isinstance(node, Leaf):
            return node.predictions
        if node.question.match(row):
            return classify(row, node.true_branch)
        else:
            return classify(row, node.false_branch)

    def print_leaf(counts):
        total = sum(counts.values()) * 1.0
        probs = {}
        for lbl in counts.keys():
            probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
        return probs

    my_tree = build_tree(train_data.values)  # without cv

    actual = list(y_test)
    pred = []
    for row in test_data.values:
        pred.append(list(print_leaf(classify(row, my_tree)).keys())[0])
#-------------------------
        # ----------------------------Logestic Reression-----------------------------------------

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import metrics

    dfl = pd.read_csv('uploads/IBM.csv')
    dfl = pd.concat([dfl.loc[:, dfl.columns != 'Attrition'], dfl.Attrition], axis=1, sort=False)
    dfl = dfl.drop(["EmployeeNumber", "Over18", "EmployeeCount", "StandardHours"], axis=1)
    dfl['Education'] = dfl['Education'].astype(object)
    dfl['EnvironmentSatisfaction'] = dfl['EnvironmentSatisfaction'].astype(object)
    dfl['JobInvolvement'] = dfl['JobInvolvement'].astype(object)
    dfl['JobLevel'] = dfl['JobLevel'].astype(object)
    dfl['JobSatisfaction'] = dfl['JobSatisfaction'].astype(object)
    dfl['PerformanceRating'] = dfl['PerformanceRating'].astype(object)
    dfl['RelationshipSatisfaction'] = dfl['RelationshipSatisfaction'].astype(object)
    dfl['StockOptionLevel'] = dfl['StockOptionLevel'].astype(object)
    dfl['TrainingTimesLastYear'] = dfl['TrainingTimesLastYear'].astype(object)
    dfl['WorkLifeBalance'] = dfl['WorkLifeBalance'].astype(object)
    dfl = dfl.sample(frac=1)
    target = {'Yes': 1, 'No': 0}
    Xl = dfl.loc[:, dfl.columns != 'Attrition']
    yl = [target[x] for x in list(dfl.Attrition)]

    categorical = []
    for col, value in Xl.iteritems():
        if value.dtype == 'object':
            categorical.append(col)
    numerical = Xl.columns.difference(categorical)
    attrition_numl = dfl[numerical]
    attrition_catl = dfl[categorical]
    attrition_catl = pd.get_dummies(attrition_catl)
    Xl = pd.concat([attrition_catl, attrition_numl], axis=1, sort=False)

    Xl_train, Xl_test, yl_train, yl_test = train_test_split(Xl, yl, test_size=0.3,
                                                            random_state=1)  # 70% training and 30% test
    import numpy as np

    class LogisticRegression:

        def __init__(self, learning_rate=0.01, n_iters=200):
            self.lr = learning_rate
            self.n_iters = n_iters
            self.weights = None
            self.bias = None

        def fit(self, X, y):
            n_samples, n_features = X.shape

            # init parameters
            self.weights = np.zeros(n_features)
            self.bias = 0

            # gradient descent
            for _ in range(self.n_iters):
                # approximate y with linear combination of weights and x, plus bias
                linear_model = np.dot(X, self.weights) + self.bias
                # apply sigmoid function
                y_predicted = self._sigmoid(linear_model)

                # compute gradients
                dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
                db = (1 / n_samples) * np.sum(y_predicted - y)
                # update parameters
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

        def predict(self, X):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
            return np.array(y_predicted_cls)

        def _sigmoid(self, x):
            return 1 / (1 + np.exp(-x))

    reg = LogisticRegression()
    reg.fit(Xl_train, yl_train)
    predl = reg.predict(Xl_test)

# -----------------------NNET FOR SINGLE ROW-----------------
    import pandas as pd
    import numpy as np
    import math

    def sigmoid(data_array):
        for i in range(0, len(data_array)):
            for j in range(0, len(data_array[i])):
                data_array[i, j] = 1 / (1 + math.exp(-data_array[i, j]))
        return data_array

    def prediction(file):

        original_data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv").drop(["EmployeeCount", "EmployeeNumber",
                                                                                   "Over18", "StandardHours",
                                                                                   "Attrition"], axis=1)
        columns = pd.get_dummies(original_data, drop_first=True).columns

        data = pd.read_csv(file)

        # Deleting Columns EmployeeCount, EmployeeNumber, Over18 and StandardHours
        data = data.drop(["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"], axis=1)

        if len(data) > 1:
            y_temp = pd.get_dummies(data.Attrition, drop_first=True)
            y = y_temp.iloc[:].values
            data = data.drop(["Attrition"], axis=1)

        # Separating Data
        data = pd.get_dummies(data)
        uploaded_file_columns = list(data.columns)
        data = data.to_numpy()
        x = []
        for i in range(0, len(data)):
            to_append = []
            for j in range(0, len(columns)):
                if columns[j] in uploaded_file_columns:
                    to_append_data = data[i, uploaded_file_columns.index(columns[j])]
                    to_append.append(to_append_data)
                else:
                    to_append.append(0)
            x.append(to_append)
        w1_file = np.genfromtxt('Epoch 99 W1', delimiter=',')
        w2_file = (np.genfromtxt('Epoch 99 W2', delimiter=','))
        w2_file = np.reshape(w2_file, [len(w2_file), 1])
        accuracy = 0
        for i in range(0, len(x)):
            X_testi_reshaped = np.reshape(x[i], [1, len(x[i])])
            if len(x) > 1:
                Y_testi_reshaped = np.reshape(y[i], [1, len(y[i])])
            dot1 = np.dot(X_testi_reshaped, w1_file)
            sigmoid1 = sigmoid(dot1)
            dot2 = np.array(np.dot(sigmoid1, w2_file))
            sigmoid2 = sigmoid(dot2)
            if sigmoid2[0, 0] >= 0.5:
                compareArray = np.array([1])
            else:
                compareArray = np.array([0])

            if len(x) > 1:
                if np.array_equal(compareArray, Y_testi_reshaped[0]):
                    accuracy += 1
            else:
                if np.array_equal(compareArray, np.array([1])):
                    return "Yes"
                else:
                    return "No"
        return accuracy / len(x)

    x= prediction("uploads/IBM.CSV")

    return render_template("output2.html", output2= accuracy_score(yTestwcv, predicted_resultwcv) ,
                           output3 = metrics.accuracy_score((actual),pred),
                           output4 = metrics.accuracy_score(yl_test, predl), output5 = x )


@app.route('/upload1', methods=['POST'])
def upload1():
    uploaded_files = request.files.getlist("file[]")

    filenames = []

    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)

    #path = os.getcwd()+"/"+app.config['UPLOAD_FOLDER']
    data = pd.read_csv("uploads/IBM.csv")
    pp.hist(data['Attrition'])
    pp.savefig("uploads/eda1.png")
    pp.close()
    pp.hist(data['Age'])
    pp.savefig("uploads/eda2.png")
    pp.close()
    fig, ax = pp.subplots(3, 3, figsize=(10, 10))  # 'ax' has references to all the four axes
    sns.distplot(data['TotalWorkingYears'], ax=ax[0, 0])
    sns.distplot(data['YearsAtCompany'], ax=ax[0, 1])
    sns.distplot(data['DistanceFromHome'], ax=ax[0, 2])
    sns.distplot(data['YearsInCurrentRole'], ax=ax[1, 0])
    sns.distplot(data['YearsWithCurrManager'], ax=ax[1, 1])
    sns.distplot(data['YearsSinceLastPromotion'], ax=ax[1, 2])
    sns.distplot(data['PercentSalaryHike'], ax=ax[2, 0])
    sns.distplot(data['YearsSinceLastPromotion'], ax=ax[2, 1])
    sns.distplot(data['TrainingTimesLastYear'], ax=ax[2, 2])
    pp.savefig("uploads/eda3.png")
    pp.close()
    total_records = len(data)
    columns = ["Gender", "MaritalStatus", "WorkLifeBalance", "EnvironmentSatisfaction", "JobSatisfaction",
               "JobLevel", "BusinessTravel", "Department"]
    pp.figure(figsize=(20, 20))
    j = 0
    for i in columns:
        j += 1
        pp.subplot(4, 2, j)
        ax1 = sns.countplot(data=data, x=i, hue="Attrition")
        if (j == 8 or j == 7):
            pp.xticks(rotation=90)
        for p in ax1.patches:
            height = p.get_height()
            ax1.text(p.get_x() + p.get_width() / 2.,
                     height + 3,
                     '{:1.2f}'.format(height / total_records, 0),
                     ha="center", rotation=0)

    pp.savefig("uploads/eda4.png")
    pp.close()
    sns.factorplot(x='Department',  # Categorical
                   y='MonthlyIncome',  # Continuous
                   hue='Attrition',  # Categorical
                   col='JobLevel',
                   col_wrap=2,  # Wrap facet after two axes
                   kind='swarm',
                   data=data)
    pp.xticks(rotation=90)
    pp.savefig("uploads/eda5.png")
    pp.close()
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'eda1.png')

    full_filename1 = os.path.join(app.config['UPLOAD_FOLDER'], 'eda2.png')

    full_filename2 = os.path.join(app.config['UPLOAD_FOLDER'], 'eda3.png')

    full_filename3 = os.path.join(app.config['UPLOAD_FOLDER'], 'eda4.png')

    full_filename4 = os.path.join(app.config['UPLOAD_FOLDER'], 'eda5.png')
    return render_template("EDA.html", user_image = full_filename,user_image1 = full_filename1,user_image2 = full_filename2,user_image3 = full_filename3,user_image4 = full_filename4,
                           output = "INTERPRETATION: Attrition Distribution Plot." ,output1= "INTERPRETATION: Age Distribution Plot.",output2= "INTERPRETATION: Multiple Distribution Plots.",
                           output3 ="INTERPRETATION: 1)Single attrition rate is 50% in marital status.     2)Job Level -1 attrition rate is also high comapre to other job levels.       3)EnvironmentSatisfaction Level 1 has high attrition rate.       4)Attrition raltes are high in these attribute Sales Deparment, Male,Jobsatisfaction 1",
                           output4 ="INTERPRETATION: 1)Attrition rate is high in JobLevel 1 at low level salary(Between -10% and +10 % of 2500) after that in JobLevel-2 and LobLvel-3 at salary range between 7500 to 10000).     2)Attrition rate is high in Sales and Research & Development Departments. especially in JobLevel-1 both the departments.       CONCLUSION : High Attrition rates are in Sales Representive(JobLevel- 1 & Who are single ), Laboratory Technician (JobLevel - 1 ) , Sales Executive (JobLevel-3 ,JobLevel 2 and who has salary range of 7500 and 10000)")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == '__main__':
    app.run(host="127.0.0.1:5000",
        port=int("80"),
        debug=True)
