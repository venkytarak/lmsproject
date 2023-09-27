# from google.colab import files



import pandas as pd
import io

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("updateddata.csv")
print(data)

data.shape

print("Number of Rows",data.shape[0])
print("Number of Columns",data.shape[1])
# data[(data['degree_t']=="Sci&Tech") & (data['status']=="Placed")].sort_values(by="salary",ascending=False).head()
print(data.info)


data = data.drop(['sl_no','salary'],axis=1)


data.head(1)

data['ssc_b'].unique()

data['ssc_b'] = data['ssc_b'].map({'Central':1,'state':0})

data.head(2)

data['Branch'].unique()

data['Branch'] = data['Branch'].map({'Electronics and Communication Engineering':3,'information technology':2,'computer science specialization':1,'Computer scinece':0})

data.head(2)

data['workexperience'].unique()

data['workexperience'] = data['workexperience'].map({'Yes':1,'No':0})
data.head(2)

data['status'].unique()

data['status'] = data['status'].map({'Placed':1,'Not Placed':0})
data.head()

data.columns

X = data.drop('status',axis=1)
y= data['status']
y

data['hsc_b'].unique()

data['hsc_b'] = data['hsc_b'].map({'Central':1,'state':0})

data.head

categorical_cols = ['ssc_b', 'Branch', 'workexperience', 'hsc_b']

# Apply one-hot encoding using pandas get_dummies
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)


X = data_encoded.drop('status', axis=1)
y = data_encoded['status']

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Model Training
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
svm = SVC()
svm.fit(X_train, y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred2 = svm.predict(X_test)
y_pred3 = knn.predict(X_test)

from sklearn.metrics import accuracy_score

score2=accuracy_score(y_test,y_pred2)
score3=accuracy_score(y_test,y_pred3)
print(score2,score3)

new_data = pd.DataFrame({
    'gender': 0,
    'ssc_p': 65.0,
    'ssc_b': 0,
    'hsc_p': 91.0,
    'hsc_b': 0,
    'Btech_cgpa':18.0,
    'Branch': 3,
    'workexperience': 1,
    'etest_p': 55.0,
}, index=[0])

# ... (previous preprocessing code up to model training) ...

# Assuming these are your categorical columns
categorical_cols = ['ssc_b', 'hsc_b', 'Branch', 'workexperience']

# Convert categorical columns into one-hot encoding
new_data_encoded = pd.get_dummies(new_data, columns=categorical_cols, drop_first=True)

# Make sure all columns match those used during training
missing_cols = set(X.columns) - set(new_data_encoded.columns)
for col in missing_cols:
    new_data_encoded[col] = 0  # Add missing columns with default values of 0

# Ensure columns are in the same order as in the training data
new_data_encoded = new_data_encoded[X.columns]

# Make predictions using the trained model
p = lr.predict(new_data_encoded)
prob = lr.predict_proba(new_data_encoded)

if p == 1:
    print('Placed')
    print(f"You will be placed with a probability of {prob[0][1]:.2f}")
else:
    print("Not-placed")


import joblib
joblib.dump(lr,'model_campus_placement')