import pickle
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self,n_weakers_limit,weak_classifier=DecisionTreeClassifier(max_depth=1)):
        '''Initialize AdaBoostClassifier
        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier=weak_classifier
        self.n_weakers_limit=n_weakers_limit
        self.w=None
        self.alpha=[]
        self.classifier=[]
    def is_good_enough(self,err_rate):
        '''Optional'''
        if err_rate<0.05:
            return True
        else:
            return False

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).
        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        self.w=np.ones(len(X)) / len(X)

        for i in range(self.n_weakers_limit):
            classifier_t=self.weak_classifier.fit(X,y,sample_weight=self.w)
            self.classifier.append(classifier_t)
            predits=self.weak_classifier.predict(X)
            #indicator function
            y=np.squeeze(y)
            miss = [int(x) for x in (predits != y)]#H(x_i)==y→0
            #error_rate
            error_rate=np.dot(self.w, miss)
            #alpha
            alpha_t = 0.5 * np.log((1 - error_rate) / float(error_rate))
            self.alpha.append(alpha_t)#把本次基分类器的权值alpha加入list，留待后续预测使用
            #update weight of each sample
            miss2=[x if x == 1 else -1 for x in miss]# -1 * y_i * H(x_i): 1 / -1
            e=np.exp([float(x) * alpha_t for x in miss2])
            w=np.multiply(self.w,e)
            self.w=w/np.sum(w)
            print('第{0}次Error_rata:{1}'.format(i,error_rate))
            if self.is_good_enough(error_rate):
                break


    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        return self.w

    def Predict(self, X ,threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        #classifer=self.weak_classifier.fit(X, y)
        alpha=np.array(self.alpha)
        result=np.zeros(len(X))
        count=0
        for i,classifier_i in zip(alpha,self.classifier):
            count+=1
            result = i*classifier_i.predict(X)
        print(count)
        result=np.sign(result)
        return result

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
