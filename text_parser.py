import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
ps = nltk.PorterStemmer()


class Parser():
    def __init__(self, path, language='english'):
        self.path = path
        self.language = language
        self.data = self.get_data_from_csv(path)
        self.stop_words = nltk.corpus.stopwords.words(self.language)
        self.c_data = self.data

    def __str__(self):
        print(self.data.head())


    @staticmethod
    def get_data_from_csv(path):
        res = pd.read_csv(path, sep=',', names=['name', 'review', 'rating'])       #, comment='#'
        res = pd.DataFrame(res)
        res = res.dropna()
        res = res.drop(columns='name')  # deleting unnecessary column of names
        return res


    def clean_data(self):
        '''
        Cleaning(removing stop words) and stemming words.
        :return:
        '''
        for index, row in enumerate(self.data['review']):
            cleaned_row = []
            for word in word_tokenize(row, language=self.language):
                if word not in self.stop_words:
                    cleaned_row.append(ps.stem(word))
            self.c_data.at[index, 'review'] = ' '.join(cleaned_row)
        self.c_data.dropna()
        self.c_data.to_csv('c_'+self.path)


    def get_fitted_data_bag(self):
        '''
        Preparing data using Bag-of-Words method
        :return:
        '''
        token = RegexpTokenizer(r'[a-zA-Z0-9]+')
        cv = CountVectorizer(lowercase=True, ngram_range=(1, 1), tokenizer=token.tokenize)
        data_vectorized = cv.fit_transform(self.c_data['review'])
        return data_vectorized


    def get_fitted_data_tfidf(self):
        '''
        Preparing data using Bag-of-Words method
        :return:
        '''
        tf = TfidfVectorizer()
        data_vectorized = tf.fit_transform(self.c_data['review'])
        return data_vectorized


    def fast_push_data(self, mode='bag'):
        '''
        Dumping data to .pickle file for future extracting
        :param mode: bag / tfidf
        '''
        if mode == 'bag':
            f = open('bag_data','wb')
            pickle.dump([self.get_fitted_data_bag(), self.c_data['rating']],f)

        if mode == 'tfidf':
            f = open('tfidf_data','wb')
            pickle.dump([self.get_fitted_data_tfidf(), self.c_data['rating']], f)


    @staticmethod
    def fast_pull_data(mode='bag'):
        '''
        Extracting data from .pickle file
        :param mode: bag / tfidf
        '''
        if mode == 'bag':
            f = open('bag_data', 'rb')
            return pickle.load(f)

        if mode == 'tfidf':
            f = open('tfidf_data', 'rb')
            return pickle.load(f)



if __name__ == '__main__':
    p = Parser('c_amazon_baby.csv')
    p.fast_push_data('bag')
    p.fast_push_data('tfidf')




