from Work.NLP import text_parser
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
p = text_parser.Parser


class Bag_classifier():
    '''
    Algorithm:
    1 - set data(already after parser)
    2 - prepare data for training
    3 - train classifier (using Naive Bayes)
    4 - show results
    '''

    def __init__(self):
        self.vectorized_data = None
        self.rating_data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.clf = None

    def manual_set_data(self, vectorized_data, rating_data):
        '''
        Set data manually
        '''
        self.vectorized_data = vectorized_data
        self.rating_data = rating_data

    def set_data_from_parser(self,path):
        '''
        Slow for big documents and useless for debugging(because you will wait too long)
        '''
        parser = p(path)
        self.vectorized_data = parser.get_fitted_data_bag()
        self.rating_data = list(map(float, parser.c_data['rating']))

    def fast_set_data(self, path):
        '''
        Get data from .pickle file, very useful for work
        '''
        parser = p(path)
        self.vectorized_data, self.rating_data = parser.fast_pull_data('bag')
        self.rating_data = list(map(float, self.rating_data.values))

    def prepare_data_for_train(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.vectorized_data, self.rating_data,
                                                                                test_size=0.3, random_state=1)

    def train(self):
        self.clf = MultinomialNB().fit(self.X_train, self.y_train)

    def show_results(self):
        predicted = self.clf.predict(self.X_test)
        print("MultinomialNB Accuracy:", metrics.accuracy_score(self.y_test, predicted))
        conf_matr = metrics.confusion_matrix(self.y_test, predicted)
        print('Confusion matrix: \n', conf_matr)
        print(metrics.classification_report(self.y_test, predicted))

    def run(self):
        self.prepare_data_for_train()
        self.train()
        self.show_results()




class Tfidf_classifier(Bag_classifier):
    def set_data_from_parser(self,path):
        parser = p(path)
        self.vectorized_data = parser.get_fitted_data_tfidf()
        self.rating_data = list(map(float, parser.c_data['rating']))

    def fast_set_data(self,path):
        parser = p(path)
        self.vectorized_data, self.rating_data = parser.fast_pull_data('tfidf')
        self.rating_data = list(map(float, self.rating_data.values))




if __name__ == '__main__':
    b_clf = Bag_classifier()
    b_clf.fast_set_data('c_amazon_baby.csv')
    b_clf.run()


    tf = Tfidf_classifier()
    tf.fast_set_data('c_amazon_baby.csv')
    tf.run()




