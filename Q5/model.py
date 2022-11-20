import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from joblib import dump
from sklearn import svm, tree
import pdb
from sklearn import datasets, svm, metrics, tree
from joblib import dump, load


train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0

def tune_and_save(
    clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path
):
    best_model, best_metric, best_h_params = h_param_tuning(
        h_param_comb, clf, x_train, y_train, x_dev, y_dev, metric
    )
    best_param_config = "_".join(
        [h + "=" + str(best_h_params[h]) for h in best_h_params]
    )

    if type(clf) == svm.SVC:
        model_type = "svm"

    if type(clf) == tree.DecisionTreeClassifier:
        model_type = "tree"

    best_model_name = model_type + "_" + best_param_config + ".joblib"
    if model_path == None:
        model_path = best_model_name
    dump(best_model, model_path)

    print("Best hyperparameters were:" + str(best_h_params))

    print("Best Metric on Dev was:{}".format(best_metric))

    return model_path


def macro_f1(y_true, y_pred, pos_label=1):
    return f1_score(y_true, y_pred, pos_label=pos_label, average='macro', zero_division='warn')
def get_all_combs(param_vals, param_name, combs_so_far):
    new_combs_so_far = []        
    for c in combs_so_far:        
        for v in param_vals:
            cc = c.copy()
            cc[param_name] = v
            new_combs_so_far.append(cc)
    return new_combs_so_far

def get_all_h_param_comb(params):
    h_param_comb = [{}]
    for p_name in params:
        h_param_comb = get_all_combs(
            param_vals=params[p_name], param_name=p_name, combs_so_far=h_param_comb
        )

    return h_param_comb

def preprocess_digits(dataset):
    n_samples = len(dataset.images)
    data = dataset.images.reshape((n_samples, -1))
    label = dataset.target
    return data, label

def data_viz(dataset):
    
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, dataset.images, dataset.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)
def pred_image_viz(x_test, predictions):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, x_test, predictions):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
        
def train_dev_test_split(data, label, train_frac, dev_frac):

    dev_test_frac = 1 - train_frac
    x_train, x_dev_test, y_train, y_dev_test = train_test_split(
        data, label, test_size=dev_test_frac, shuffle=True
    )
    x_test, x_dev, y_test, y_dev = train_test_split(
        x_dev_test, y_dev_test, test_size=(dev_frac) / dev_test_frac, shuffle=True
    )

    return x_train, y_train, x_dev, y_dev, x_test, y_test
def h_param_tuning(h_param_comb, clf, x_train, y_train, x_dev, y_dev, metric, verbose=False):
    best_metric = -1.0
    best_model = None
    best_h_params = None
    for cur_h_params in h_param_comb:

       
        hyper_params = cur_h_params
        clf.set_params(**hyper_params)
        clf.fit(x_train, y_train)

        predicted_dev = clf.predict(x_dev)

        
        cur_metric = metric(y_pred=predicted_dev, y_true=y_dev)

      
        if cur_metric > best_metric:
            best_metric = cur_metric
            best_model = clf
            best_h_params = cur_h_params
            if verbose:
                print("Found new best metric with :" + str(cur_h_params))
                print("New best val metric:" + str(cur_metric))
    return best_model, best_metric, best_h_params
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]
svm_params = {}
svm_params["gamma"] = gamma_list
svm_params["C"] = c_list
svm_h_param_comb = get_all_h_param_comb(svm_params)
max_depth_list = [2, 10, 20, 50, 100]

dec_params = {}
dec_params["max_depth"] = max_depth_list
dec_h_param_comb = get_all_h_param_comb(dec_params)

h_param_comb = {"svm": svm_h_param_comb, "tree": dec_h_param_comb}


digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)

del digits
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-clf_name', type=str, choices=['svm', 'tree'], help='specify classifier name')
parser.add_argument('-random_state', type=int, help='specify random seed/state value')
args = parser.parse_args()

print(args.clf_name)


metric_list = [metrics.accuracy_score, macro_f1]
h_metric = metrics.accuracy_score
random_s = args.random_state
n_cv = 1
results = {}
for n in range(n_cv):
    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        data, label, train_frac, dev_frac
    )
    models_of_choice = {
        "svm": svm.SVC(),
        "tree": tree.DecisionTreeClassifier(),
    }
    clf_name = args.clf_name
    clf = models_of_choice[clf_name]
    print("[{}] Running hyper param tuning for {}".format(n,clf_name))
    actual_model_path = tune_and_save(
        clf, x_train, y_train, x_dev, y_dev, h_metric, h_param_comb[clf_name], model_path=None
    )

      
    best_model = load(actual_model_path)

       
    predicted = best_model.predict(x_test)
    if not clf_name in results:
        results[clf_name]=[]    

    results[clf_name].append({m.__name__:m(y_pred=predicted, y_true=y_test) for m in metric_list})
        
    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )
m1 = macro_f1(y_test,predicted)
file = open("results/"+clf_name+"_"+str(random_s)+".txt", "a") 
  
file.write("test accuracy"+str(metrics.accuracy_score(y_test,predicted))+"\ntest micro f1:"+str(m1)+"\nmodel saved at:"+"./model/"+actual_model_path) 
  
file.close()

print(results)
