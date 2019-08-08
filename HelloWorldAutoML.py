# https://www.ml4aad.org/automl/
  
# https://github.com/hibayesian/awesome-automl-papers

os.environ["MAX_TEXT_LENGTH"] = "8000000"
!pip3 install auto-sklearn

!ulimit -a

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
X, y = sklearn.datasets.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = \
  sklearn.model_selection.train_test_split(X, y, random_state=1)
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=120,per_run_time_limit=60)
automl.fit(X_train, y_train)
y_hat = automl.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))
