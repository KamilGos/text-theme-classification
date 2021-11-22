import os  # Miscellaneous operating system interfaces
from collections import defaultdict  # helps to create dataframe from files
import pandas as pd  # Working with data
import random  # Random number generation
import seaborn as sns  # Statistical data visualization
import matplotlib.pyplot as plt  # Ploting
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords  # list of the most commonly used stopwords ('i', 'me' etc.) in English
import re  # regular expression operations
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # text vectorize
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # text vectorize
from sklearn.model_selection import train_test_split    # spliting data to train and test part
from tqdm import tqdm   # Very nice loading bar
from time import time   # time measurements
from scipy.sparse import hstack  # horizontal stacking of arrays
from prettytable import PrettyTable
import pickle

# Read files with text from given directory and save them to dataframe
# in_dir_name - name of directory with data (divided into categories using directories)
#
class FileReader:
    def __init__(self, in_dir_name):
        self.dir_name = in_dir_name

    def showFiles(self):
        for dirname, _, filenames in os.walk(self.dir_name):
            print(dirname)
            count = 0
            for filename in filenames:
                if count > 5:
                    break
                print(os.path.join(dirname, filename))
                count += 1
            print("Number of elements: ", len(os.listdir(dirname)))

    def showDirectories(self):
        print(os.listdir(self.dir_name))

    def readFileToFrame(self, LEARN):
        print(self.dir_name)
        frame = defaultdict(list)
        for dirname, _, filenames in os.walk(self.dir_name):
            for filename in filenames:
                if LEARN:
                    frame['category'].append(os.path.basename(dirname))
                frame['file_id'].append(os.path.splitext(filename)[0])

                with open(os.path.join(dirname, filename), encoding='unicode_escape') as file:
                    frame['text'].append(file.read())
        df = pd.DataFrame.from_dict(frame)
        text = df['text'].str.split("\n", n=1, expand=True)
        df['title'] = text[0]
        df['story'] = text[1]
        return df


# Data modifications. Helps to prepare data to other algorithms
# in_dir_name - name of directory with data (divided into categories using directories)
#
class DataAssistant:
    def __init__(self, in_dir_name, LEARN):
        self.FileRd = FileReader(in_dir_name)
        self.data = self.FileRd.readFileToFrame(LEARN)
        self.pTitles = []
        self.pStories = []

    def showDataInfo(self):
        print("*** Data summary ***")
        print(self.data.info())
        print("Columns: ", self.data.shape[1], "Rows: ", self.data.shape[0])
        # print("**** Data head ****")
        # print(self.data.head(3))
        print("*** Data distribution ***")
        print(self.data['category'].value_counts())

    def showRandomText(self, nsamples):
        sample = random.sample(range(self.data['text'].shape[0]), nsamples)
        print(sample)
        for idx in sample:
            print('*' * 30)
            values = self.data.iloc[idx]
            print('Document ID : ', values['file_id'])
            print('Category : ', values['category'])
            print('Title : \n' + '-' * 9)
            print(values['title'])
            print('\nStory : \n' + '-' * 9)
            print(values['story'])
            print('=' * 36)

    def showDataHistogram(self):
        plt.figure(figsize=(7, 6))
        sns.countplot(self.data['category'])
        plt.title('Number of files in each category')
        plt.ylabel("Number of files")
        plt.xlabel("Category name")

    def showTitleWordsDistribution(self):
        words_counter = defaultdict(list)
        for category in self.data['category'].unique():
            val = self.data[self.data['category'] == category]['title'].str.split().apply(len).values
            words_counter[category] = val
        plt.figure(figsize=(7, 6))
        plt.boxplot(words_counter.values(), notch=True)
        keys = words_counter.keys()
        plt.xticks([i + 1 for i in range(len(keys))], keys)
        plt.ylabel('Number of words in title')
        plt.xlabel('Category name')
        plt.title('Distribution of titles across categories')
        plt.grid()

    def showTextWordsDistribution(self):
        words_counter = defaultdict(list)
        for category in self.data['category'].unique():
            val = self.data[self.data['category'] == category]['story'].str.split().apply(len).values
            words_counter[category] = val
        plt.figure(figsize=(7, 6))
        plt.boxplot(words_counter.values(), notch=True)
        keys = words_counter.keys()
        plt.xticks([i + 1 for i in range(len(keys))], keys)
        plt.ylabel('Number of words in story')
        plt.xlabel('Category name')
        plt.title('Distribution of stories across categories')
        plt.grid()

    @staticmethod
    def showPlots():
        plt.show()

    @staticmethod
    def cleanText(text):
        stop_words = stopwords.words('english')
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        text = text.replace('\\r', ' ')
        text = text.replace('\\"', ' ')
        text = text.replace('\\t', ' ')
        text = text.replace('\\n', ' ')
        text = ' '.join(word for word in text.split() if word not in stop_words)
        text = re.sub('[^A-Za-z0-9]+', ' ', text)
        text = text.lower()
        return text

    def processTitlesAndStories(self):
        print("*** Text preprocessing ***\n")
        for title in tqdm(self.data['title'].values):
            tmp = self.cleanText(title)
            self.pTitles.append(tmp)
        for story in tqdm(self.data['story'].values):
            tmp = self.cleanText(story)
            self.pStories.append(tmp)
        print("*** DONE ***")

    def showSomeProcessedTitle(self):
        samples = random.sample(range(len(self.data['title'])), 10)
        for idx in samples:
            raw = self.data.iloc[idx]
            print("Before: ", raw['title'])
            clean = self.pTitles[idx]
            print("After:  ", clean)

    def vectorizeTitles(self, df, out_model_dir):
        print("*** Vectorizing titles ***")
        start = time()

        vectorizer = CountVectorizer(min_df=df)
        vectorizer.fit(self.pTitles)
        pickle.dump(vectorizer, open((out_model_dir+'titles_cv.sav'), 'wb'))
        title_bow = vectorizer.transform(self.pTitles)

        vectorizer = TfidfVectorizer(min_df=df)
        vectorizer.fit(self.pTitles)
        pickle.dump(vectorizer, open((out_model_dir+'titles_tfidf.sav'), 'wb'))
        title_tfidf = vectorizer.transform(self.pTitles)

        vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=750, min_df=df)
        vectorizer.fit(self.pTitles)
        pickle.dump(vectorizer, open((out_model_dir+'titles_ngram.sav'), 'wb'))
        title_tfidf_ngram = vectorizer.transform(self.pTitles)

        print("*** DONE *** Duration: ", round(time() - start, 2), "s")
        return title_bow, title_tfidf, title_tfidf_ngram

    def vectorizeStories(self, df, out_model_dir):
        start = time()
        print("*** Vectorizing stories ***")

        vectorizer = CountVectorizer(min_df=df)
        vectorizer.fit(self.pStories)
        pickle.dump(vectorizer, open((out_model_dir+'stories_cv.sav'), 'wb'))
        story_bow = vectorizer.transform(self.pStories)

        vectorizer = TfidfVectorizer(min_df=df)
        vectorizer.fit(self.pStories)
        pickle.dump(vectorizer, open((out_model_dir+'stories_tfidf.sav'), 'wb'))
        story_tfidf = vectorizer.transform(self.pStories)

        vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 4), max_features=7500, min_df=df)
        vectorizer.fit(self.pStories)
        pickle.dump(vectorizer, open((out_model_dir+'stories_ngram.sav'), 'wb'))
        story_tfidf_ngram = vectorizer.transform(self.pStories)

        print("*** DONE *** Duration: ", round(time() - start, 2), "s")
        return story_bow, story_tfidf, story_tfidf_ngram

    def vectorizeCategories(self, out_model_dir):
        print("*** Vectorizing categories (target) ***")
        start = time()
        vectorizer = LabelEncoder()
        vectorizer.fit(self.data['category'].values)
        pickle.dump(vectorizer, open((out_model_dir+'categories.sav'), 'wb'))
        category_enc = vectorizer.transform(self.data['category'].values)
        print("*** DONE *** Duration: ", round(time() - start, 2), "s")
        return category_enc

    def returnModelInput(self, out_models_dir):
        self.processTitlesAndStories()
        title_bow, title_tfidf, title_tfidf_ngram = self.vectorizeTitles(5, out_models_dir)
        story_bow, story_tfidf, story_tfidf_ngram = self.vectorizeStories(15, out_models_dir)
        category_enc = self.vectorizeCategories(out_models_dir)

        in_bow = hstack((title_bow, story_bow))
        in_tfidf = hstack((title_tfidf, story_tfidf))
        in_ngram = hstack((title_tfidf_ngram, story_tfidf_ngram))
        print("\nSize of vectors: ")
        x = PrettyTable()
        x.field_names = ['BOW', 'TFIDF', 'n-gram']
        x.add_row([in_bow.shape, in_tfidf.shape, in_ngram.shape])
        print(x)
        return in_bow, in_tfidf, in_ngram, category_enc

    def returnNeuralNetworkInput(self, data_x, data_y):
        onehotencoder = OneHotEncoder()
        onehotencoder.fit(data_y.reshape(-1, 1))
        labels = onehotencoder.transform(data_y.reshape(-1, 1)).toarray()
        return train_test_split(data_x, labels, test_size=0.15)

    def returnClassificationInput(self, models_dir):
        self.processTitlesAndStories()

        dict_titles_bow = pickle.load(open((models_dir + 'titles_cv.sav'), 'rb'))
        dict_titles_tfidf = pickle.load(open((models_dir + 'titles_tfidf.sav'), 'rb'))
        dict_titles_ngram = pickle.load(open((models_dir + 'titles_ngram.sav'), 'rb'))
        dict_stories_bow = pickle.load(open((models_dir + 'stories_cv.sav'), 'rb'))
        dict_stories_tfidf = pickle.load(open((models_dir + 'stories_tfidf.sav'), 'rb'))
        dict_stories_ngram = pickle.load(open((models_dir + 'stories_ngram.sav'), 'rb'))

        titles_bow = dict_titles_bow.transform(self.pTitles)
        titles_tfidf = dict_titles_tfidf.transform(self.pTitles)
        titles_ngram = dict_titles_ngram.transform(self.pTitles)
        stories_bow = dict_stories_bow.transform(self.pStories)
        stories_tfidf = dict_stories_tfidf.transform(self.pStories)
        stories_ngram = dict_stories_ngram.transform(self.pStories)

        in_bow = hstack((titles_bow, stories_bow))
        in_tfidf = hstack((titles_tfidf, stories_tfidf))
        in_ngram = hstack((titles_ngram, stories_ngram))
        return in_bow, in_tfidf, in_ngram


if __name__ == "__main__":
    dir_name = './raw_text/'
    out_modules_dir = './models_from_DA/'
    DA = DataAssistant(dir_name)
    DA.showDataInfo()
    DA.showRandomText(1)
    DA.showDataHistogram()
    DA.showTitleWordsDistribution()
    DA.showTextWordsDistribution()
    DA.processTitlesAndStories()
    DA.showSomeProcessedTitle()
    DA.returnModelInput(out_modules_dir)
    DA.showPlots()
