import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
#from sklearn.utils.fixes import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
import re
import random
from os import listdir
"""
TODO 0: Find out the image shape as a tuple and include it in your report. (Complete)
"""
folder = 'lfw_crop/'
file_names = []
for file in listdir(folder):
    file_names.append(file)
path = 'lfw_crop/'+file_names[0]
image = cv2.imread(path)
IMG_SHAPE = image.shape
print(IMG_SHAPE)

def load_data(data_dir, top_n=10):
    print("Start Load Data")
    """
    Load the data and return a list of images and their labels.
    :param data_dir: The directory where the data is located
    :param top_n: The number of people with the most images to use
    Suggested return values, feel free to change as you see fit
    :return data_top_n: A list of images of only people with top n number of images
    :return target_top_n: Corresponding labels of the images 
    :return target_names: A list of all labels
    :return target_count: A dictionary of the number of images per person 
    """
    # read and randomize list of file names
    file_list = [fname for fname in listdir(data_dir) if fname.endswith('.pgm')]
    random.shuffle(file_list)
    name_list = [re.sub(r'_\d{4}.pgm', '', name).replace('_', ' ') for name in file_list]
    # get a list of all labels
    target_names = sorted(list(set(name_list)))
    # get labels for each image
    target = np.array([target_names.index(name) for name in name_list])
    # read in all images
    print("target")
    print(target)
    data = np.array([cv2.imread(data_dir + fname, 0) for fname in file_list])
    data = data.flatten().reshape(13233,64*64)
    #print("data after flatten faces")
    #print(data)
    """
    TODO 1: Only preserve images of 10 people with the highest occurence, (Complete) 
            then plot a histogram of the number of images per person in the preserved dataset. (Complete)
            Include the histogram in your report.
    """
    # YOUR CODE HERE
    data_top_n = [] #list of images of top people
    target_top_n = [] #list of names that match image locations
    target_count = {}
    for name in target_names:
        target_count[str(name)] = name_list.count(str(name))
    TopValues = list(target_count.values())
    TopValues = sorted(TopValues, reverse=True)
    target_count_top_n = {}
    top_name_list = []
    for key in target_count:
        if target_count.get(key) >= TopValues[top_n-1]:
            target_count_top_n[str(key)] = target_count.get(key)
            if key not in top_name_list:
                top_name_list.append(key)
        else:
            pass
    print(top_name_list)
    i = 0
    names = []
    for name in file_list:
        first_last = name.split("_")
        ii = 0
        fullname = ""
        for word in first_last:
            if '.pgm' not in word:
                fullname = fullname + first_last[ii]+" "
            else:
                pass
            ii = ii + 1
        fullname = fullname.rstrip()
        if fullname in top_name_list:
            image = data[i]
            data_top_n.append(image)
            label = name_list[i] 
            target_top_n.append(label)
            names.append(name)
        else:
            pass
        i = i + 1
    target_count_top_n = dict(sorted(target_count_top_n.items(), key=lambda item: item[1], reverse=True))
    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot(111)
    ax.bar(target_count_top_n.keys(), target_count_top_n.values())
    ax.set_xlabel('face names')
    ax.set_ylabel('number of faces')
    ax.set_title('Histogram of Face Data')
    plt.show()
    data_top_n = np.array(data_top_n)
    target_top_n = np.array(target_top_n)
    print("target_top_n")
    print(target_top_n)
    #print("data_top_n")
    #print(data_top_n)
    #print("target_names")
    #print(target_names)
    #print("target_count")
    #print(target_count)
    return data_top_n, target_top_n, target_names, target_count


def load_data_nonface(data_dir):
    print("Load Non-face Data")
    """
    Your can write your functin comments here.
    """
    
    """
    TODO 2: Load the nonface data and return a list of images. (Complete)
    """
    # YOUR CODE HERE
    file_list = [fname for fname in listdir(data_dir) if fname.endswith('.png')]
    random.shuffle(file_list)
    data = np.array([cv2.imread(data_dir + fname, 0) for fname in file_list])
    data = data.flatten().reshape(1000,64*64)
    print("data after flatten nonfaces")
    print(data.shape)
    print("Completed Non-face Data")
    return data


def perform_pca(data_train, data_test, data_noneface, n_components, plot_PCA=True):
    """
    Your can write your functin comments here.
    """
    """
    TODO 3: Perform PCA on the training data, then transform the training, testing, 
            and nonface data. Return the transformed data. This includes:
            a) Flatten the images if you haven't done so already (Complete)
            b) Standardize the data (0 mean, unit variance) (Complete)
            c) Perform PCA on the standardized training data (Complete)
            d) Transform the standardized training, testing, and nonface data (Complete)
            e) Plot the transformed training and nonface data using the first three 
               principal components if plot_PCA is True. Include the plots in your report. (Plot Created Seperate Face from Nonface)
            f) Return the principal components and transformed training, testing, and nonface data (Complete)
    """
    # YOUR CODE HERE
    scaler_train = StandardScaler()
    scaler_train.fit(data_train)
    data_train_centered = scaler_train.transform(data_train)
    scaler_test = StandardScaler()
    scaler_test.fit(data_train)
    data_test_centered = scaler_test.transform(data_test)
    scaler_nonface = StandardScaler()
    scaler_nonface.fit(data_train)
    data_noneface_centered = scaler_nonface.transform(data_noneface)
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(data_train_centered)
    data_train_pca = pca.transform(data_train_centered)
    data_test_pca = pca.transform(data_test_centered)
    data_noneface_pca = pca.transform(data_noneface_centered)

    if plot_PCA == True:
        fig = plt.figure(1, figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)
        scatter = ax.scatter(data_train_pca[:, 0],data_train_pca[:, 1],data_train_pca[:, 2],c ='b')
        scatter = ax.scatter(data_noneface_pca[:, 0],data_noneface_pca[:, 1],data_noneface_pca[:, 2],c='r')
        ax.set(
            title="First three PCA dimensions",
            xlabel="PC1",
            ylabel="PC2",
            zlabel="PC3",
        )
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        target_names = ["train","test","nonface"]    
        legend1 = ax.legend(
            ["face","nonface"],  
            loc="upper right",
            title="Classes",
        )
        ax.add_artist(legend1)
        print("plotting completed")
        plt.show()
    
    return pca, data_train_pca, data_test_pca, data_noneface_pca

def plot_eigenfaces(pca):

    """
    TODO 4: Plot the first 8 eigenfaces. Include the plot in your report.
    """
    n_row = 2
    n_col = 4
    #image = pca.components_[1].reshape(64,64)
    #plt.imshow(image)
    fig, axes = plt.subplots(n_row, n_col, figsize=(12, 6))
    for i in range(n_row * n_col):
        
        # YOUR CODE HERE
        # The eigenfaces are the principal components of the training data
        # Since we have flattened the images, you can use reshape() to reshape to the original image shape
        n_components = pca.components_.shape[0]
        image = pca.components_.reshape((n_components,64,64)) #Only plots 2
        axes = fig.add_subplot(n_row, n_col, i+1)
        axes.imshow(image[i], cmap="gray")

    plt.show()


def train_classifier(data_train_pca, target_train):
    """
    TODO 5: OPTIONAL: Train a classifier on the training data.
            SVM is recommended, but feel free to use any classifier you want.
            Also try using the RandomizedSearchCV to find the best hyperparameters.
            Include the classifier you used as well as the parameters in your report.
            Feel free to look up sklearn documentation and examples on usage of classifiers.
    """
    # YOUR CODE HERE
    # You can read the documents from sklearn to learn about the classifiers provided by sklearn
    # https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    # If you are using SVM, you can also check the example below
    # https://scikit-learn.org/stable/modules/svm.html
    # Also, you can use the RandomizedSearchCV to find the best hyperparameters
    #change the classifier being used
    classifier = svm.SVC(kernel='rbf')
    classifier = classifier.fit(data_train_pca, target_train)
    print("classifier fitting complete")
    clf = classifier
    print(clf)
    return clf


if __name__ == '__main__':
    """
    Load the data
    Face Dataset from https://conradsanderson.id.au/lfwcrop/
    Modified from original dataset http://vis-www.cs.umass.edu/lfw/
    Noneface Dataset modified from http://image-net.org/download-images
    All modified datasets are available in the Box folder
    """
    data, target, target_names, target_count = load_data('lfw_crop/', top_n=10)
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.25, random_state=42)
    data_noneface = load_data_nonface('imagenet_val1000_downsampled/')
    print("Total dataset size:", data.shape[0])
    print("Training dataset size:", data_train.shape[0])
    print("Test dataset size:", data_test.shape[0])
    print("Nonface dataset size:", data_noneface.shape[0])
    # Perform PCA, you can change the number of components as you wish
    pca, data_train_pca, data_test_pca, data_noneface_pca = perform_pca(
        data_train, data_test, data_noneface, n_components=8, plot_PCA=True
    )
    # Plot the first 8 eigenfaces. To do this, make sure n_components is at least 8
    plot_eigenfaces(pca)
    """
    #Start of PS 8-2
    #This part is optional. You will get extra credits if you complete this part.
    """
    # Train a classifier on the transformed training data
    classifier = train_classifier(data_train_pca, target_train)
    # Evaluate the classifier
    pred = classifier.predict(data_test_pca)
    # Use a simple percentage of correct predictions as the metric
    accuracy = np.count_nonzero(np.where(pred == target_test)) / pred.shape[0]
    print("Accuracy:", accuracy)
    """
    #TODO 6: OPTIONAL: Plot the confusion matrix of the classifier.
    #        Include the plot and accuracy in your report.
    #        You can use the sklearn.metrics.ConfusionMatrixDisplay function.
    """
    
    # YOUR CODE HERE
    cm = confusion_matrix(target_test, pred, labels=classifier.classes_)
    confusion = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classifier.classes_)
    confusion.plot(xticks_rotation='vertical')
    
    plt.show()
    """
    #TODO 7: OPTIONAL: Plot the accuracy with different number of principal components.
    #        This might take a while to run. Feel free to decrease training iterations if
    #        you want to speed up the process. We won't set a hard threshold on the accuracy.
    #        Include the plot in your report.
    
    """
    
    n_components_list = [3, 5, 10, 20, 40, 60, 80, 100, 120, 130]
    # YOUR CODE HERE
    accuracy_list = []
    for n_components in n_components_list:
        print("training classifier")
        pca, data_train_pca, data_test_pca, data_noneface_pca = perform_pca(data_train, data_test, data_noneface, n_components=n_components, plot_PCA=False)
        classifier = train_classifier(data_train_pca, target_train)
        # Evaluate the classifier
        print("checking classifier")
        pred = classifier.predict(data_test_pca)
        # Use a simple percentage of correct predictions as the metric
        print("check accuracy")
        accuracy = np.count_nonzero(np.where(pred == target_test)) / pred.shape[0]
        print(accuracy)
        accuracy_list.append(accuracy)
    
    X = n_components_list
    Y = accuracy_list
    plt.plot(X,Y)
    plt.xlabel('n_components')
    plt.ylabel('accuracy')
    plt.title('n_components vs accuracy')
    plt.show()
    
    
