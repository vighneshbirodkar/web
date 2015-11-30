from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import os
import numpy as np
from nltk.corpus import stopwords


label_fn = []
for root, subdirs, files in os.walk('data'):
    if root != 'data':
        label = root.split('/')[-1]
    for fn in files:
        label_fn.append((label, root + '/' + fn))

labels = [t[0] for t in label_fn]
filenames = [t[1] for t in label_fn]

tf = TfidfVectorizer(input='filename', stop_words=stopwords.words('english'),
                      decode_error='ignore')
X = tf.fit_transform(filenames)
print('Vectorization Done')
print('Number of features = %d' % X.shape[1])

le = LabelEncoder()

y_str = labels
y = le.fit_transform(y_str)
print('Label Encoding Done')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

clf = SVC()
clf.fit(X_train, y_train)
print('Learning Complete')

y_pred = clf.predict(X_test)
print('Testing Samples = %d' % len(y_test))
print('Correctly classified Samples = %d' % np.sum(y_pred == y_test))

    #exit(0)
    #if root != 'data':
    #    exit(0)