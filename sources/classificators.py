from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import linear_model, naive_bayes, svm, ensemble
from matplotlib import pyplot as plt
import seaborn as sns
from time import time
from sklearn.model_selection import cross_val_score
from tensorflow import keras
from prettytable import PrettyTable
import pickle

from sources.data_manipulations import DataAssistant


class Models:
    def __init__(self, in_data_dir, out_models_dir, in_models_dir, LEARN):
        self.DA = DataAssistant(in_data_dir, LEARN)
        if LEARN:
            in_bow, in_tfidf, in_ngram, in_cat = self.DA.returnModelInput(out_models_dir)
            self.in_bow = in_bow
            self.in_tfidf = in_tfidf
            self.in_ngram = in_ngram
            self.in_cat = in_cat
        else:
            in_bow, in_tfidf, in_ngram = self.DA.returnClassificationInput(in_models_dir)
            self.in_bow = in_bow
            self.in_tfidf = in_tfidf
            self.in_ngram = in_ngram

    @staticmethod
    def showConfusionMatrix(prediction, y_test, alg_name):
        labels = ['tech', 'sport', 'business', 'entertainment', 'politics']
        cm = confusion_matrix(y_test, prediction)
        print(cm)
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.yaxis.set_ticklabels(labels[::-1], rotation=90)
        plt.title(('Confusion matrix of ' + alg_name))

    def learnWithBOW(self, save_dir):
        data_x = self.in_bow.toarray()
        data_y = self.in_cat
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=9)
        x = PrettyTable()
        x.field_names = ['Algorithm', 'Prediction', 'Building time [s]', '5-C-V', '5-C-V time [s]']
        print('\n\n ===  BOW === \n')

        start = time()
        model = linear_model.LogisticRegression(max_iter=200)
        model.fit(x_train, y_train)
        pickle.dump(model, open((save_dir+'lr_bow.sav'), 'wb'))
        LR_prediction = model.predict(x_test)
        pred_time = time() - start
        print(f'### Logistic regression ### Duration %0.2f' % pred_time)
        print(f'  Test accuracy: %0.3f' % (accuracy_score(LR_prediction, y_test)))
        start = time()
        scores = cross_val_score(model, x_train, y_train, cv=5)
        cv_time = time() - start
        cv_acc = '%0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)
        print("  5-C-V Accuracy: " + cv_acc + " Duration: %0.2f" % cv_time)
        x.add_row(['Log.Reg', round(accuracy_score(LR_prediction, y_test), 3), round(pred_time, 2), cv_acc,
                   round(cv_time, 2)])

        start = time()
        model = naive_bayes.MultinomialNB()
        model.fit(x_train, y_train)
        pickle.dump(model, open((save_dir+'nb_bow.sav'), 'wb'))
        NB_prediction = model.predict(x_test)
        pred_time = time() - start
        print(f'### Naive Bayes ### Duration %0.2f' % pred_time)
        print(f'  Test accuracy: %0.3f' % (accuracy_score(NB_prediction, y_test)))
        start = time()
        scores = cross_val_score(model, x_train, y_train, cv=5)
        cv_time = time() - start
        cv_acc = '%0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)
        print("  5-C-V Accuracy: " + cv_acc + " Duration: %0.2f" % cv_time)
        x.add_row(
            ['Bayes', round(accuracy_score(NB_prediction, y_test), 3), round(pred_time, 2), cv_acc, round(cv_time, 2)])

        start = time()
        model = svm.SVC()
        model.fit(x_train, y_train)
        pickle.dump(model, open((save_dir+'svc_bow.sav'), 'wb'))
        SVM_prediction = model.predict(x_test)
        pred_time = time() - start
        print(f'Support Vector Classification ### Duration %0.2f' % pred_time)
        print(f'  Test accuracy: %0.3f' % (accuracy_score(SVM_prediction, y_test)))
        start = time()
        scores = cross_val_score(model, x_train, y_train, cv=5)
        cv_time = time() - start
        cv_acc = '%0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)
        print("  5-C-V Accuracy: " + cv_acc + " Duration: %0.2f" % cv_time)
        x.add_row(['Sup.Vec.', round(accuracy_score(SVM_prediction, y_test), 3), round(pred_time, 2), cv_acc,
                   round(cv_time, 2)])

        start = time()
        model = svm.LinearSVC(max_iter=400)
        model.fit(x_train, y_train)
        pickle.dump(model, open((save_dir+'lsvc_bow.sav'), 'wb'))
        LSVM_prediction = model.predict(x_test)
        pred_time = time() - start
        print(f'### Linear Support Vector Classification ###  Duration %0.2f' % pred_time)
        print(f'  Test accuracy: %0.3f' % (accuracy_score(LSVM_prediction, y_test)))
        start = time()
        scores = cross_val_score(model, x_train, y_train, cv=5)
        cv_time = time() - start
        cv_acc = '%0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)
        print("  5-C-V Accuracy: " + cv_acc + " Duration: %0.2f" % cv_time)
        x.add_row(['Lin.Sup.Vec', round(accuracy_score(LSVM_prediction, y_test), 3), round(pred_time, 2), cv_acc,
                   round(cv_time, 2)])

        start = time()
        model = ensemble.RandomForestClassifier()
        model.fit(x_train, y_train)
        pickle.dump(model, open((save_dir+'rf_bow.sav'), 'wb'))
        RandomForest_prediction = model.predict(x_test)
        pred_time = time() - start
        print(f'### Random Forest ### Duration %0.2f' % pred_time)
        print(f'  Test accuracy: %0.3f' % (accuracy_score(RandomForest_prediction, y_test)))
        start = time()
        scores = cross_val_score(model, x_train, y_train, cv=5)
        cv_time = time() - start
        cv_acc = '%0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)
        print("  5-C-V Accuracy: " + cv_acc + " Duration: %0.2f" % cv_time)
        x.add_row(['Rand.For', round(accuracy_score(RandomForest_prediction, y_test), 3), round(pred_time, 2), cv_acc,
                   round(cv_time, 2)])
        print("\n Summary")
        print(x)

    def learnWithTFIDF(self, save_dir):
        data_x = self.in_tfidf.toarray()
        data_y = self.in_cat
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=9)
        x = PrettyTable()
        x.field_names = ['Algorithm', 'Prediction', 'Building time [s]', '5-C-V', '5-C-V time [s]']
        print('\n\n === TFIDF ===\n')

        start = time()
        model = linear_model.LogisticRegression(max_iter=200)
        model.fit(x_train, y_train)
        pickle.dump(model, open((save_dir + 'lr_tfidf.sav'), 'wb'))
        LR_prediction = model.predict(x_test)
        pred_time = time() - start
        print(f'### Logistic regression ### Duration %0.2f' % pred_time)
        print(f'  Test accuracy: %0.3f' % (accuracy_score(LR_prediction, y_test)))
        start = time()
        scores = cross_val_score(model, x_train, y_train, cv=5)
        cv_time = time() - start
        cv_acc = '%0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)
        print("  5-C-V Accuracy: " + cv_acc + " Duration: %0.2f" % cv_time)
        x.add_row(['Log.Reg', round(accuracy_score(LR_prediction, y_test), 3), round(pred_time, 2), cv_acc,
                   round(cv_time, 2)])

        start = time()
        model = naive_bayes.MultinomialNB()
        model.fit(x_train, y_train)
        pickle.dump(model, open((save_dir + 'nb_tfidf.sav'), 'wb'))
        NB_prediction = model.predict(x_test)
        pred_time = time() - start
        print(f'### Naive Bayes ### Duration %0.2f' % pred_time)
        print(f'  Test accuracy: %0.3f' % (accuracy_score(NB_prediction, y_test)))
        start = time()
        scores = cross_val_score(model, x_train, y_train, cv=5)
        cv_time = time() - start
        cv_acc = '%0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)
        print("  5-C-V Accuracy: " + cv_acc + " Duration: %0.2f" % cv_time)
        x.add_row(
            ['Bayes', round(accuracy_score(NB_prediction, y_test), 3), round(pred_time, 2), cv_acc, round(cv_time, 2)])

        start = time()
        model = svm.SVC()
        model.fit(x_train, y_train)
        pickle.dump(model, open((save_dir + 'svc_tfidf.sav'), 'wb'))
        SVM_prediction = model.predict(x_test)
        pred_time = time() - start
        print(f'Support Vector Classification ### Duration %0.2f' % pred_time)
        print(f'  Test accuracy: %0.3f' % (accuracy_score(SVM_prediction, y_test)))
        start = time()
        scores = cross_val_score(model, x_train, y_train, cv=5)
        cv_time = time() - start
        cv_acc = '%0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)
        print("  5-C-V Accuracy: " + cv_acc + " Duration: %0.2f" % cv_time)
        x.add_row(['Sup.Vec.', round(accuracy_score(SVM_prediction, y_test), 3), round(pred_time, 2), cv_acc,
                   round(cv_time, 2)])

        start = time()
        model = svm.LinearSVC(max_iter=200)
        model.fit(x_train, y_train)
        pickle.dump(model, open((save_dir + 'lsvc_tfidf.sav'), 'wb'))
        LSVM_prediction = model.predict(x_test)
        pred_time = time() - start
        print(f'### Linear Support Vector Classification ###  Duration %0.2f' % pred_time)
        print(f'  Test accuracy: %0.3f' % (accuracy_score(LSVM_prediction, y_test)))
        start = time()
        scores = cross_val_score(model, x_train, y_train, cv=5)
        cv_time = time() - start
        cv_acc = '%0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)
        print("  5-C-V Accuracy: " + cv_acc + " Duration: %0.2f" % cv_time)
        x.add_row(['Lin.Sup.Vec', round(accuracy_score(LSVM_prediction, y_test), 3), round(pred_time, 2), cv_acc,
                   round(cv_time, 2)])

        start = time()
        model = ensemble.RandomForestClassifier()
        model.fit(x_train, y_train)
        pickle.dump(model, open((save_dir + 'rf_tfidf.sav'), 'wb'))
        RandomForest_prediction = model.predict(x_test)
        pred_time = time() - start
        print(f'### Random Forest ### Duration %0.2f' % pred_time)
        print(f'  Test accuracy: %0.3f' % (accuracy_score(RandomForest_prediction, y_test)))
        start = time()
        scores = cross_val_score(model, x_train, y_train, cv=5)
        cv_time = time() - start
        cv_acc = '%0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)
        print("  5-C-V Accuracy: " + cv_acc + " Duration: %0.2f" % cv_time)
        x.add_row(['Rand.For', round(accuracy_score(RandomForest_prediction, y_test), 3), round(pred_time, 2), cv_acc,
                   round(cv_time, 2)])
        print("\n Summary")
        print(x)

    def learnWithNgram(self, save_dir):
        data_x = self.in_ngram.toarray()
        data_y = self.in_cat
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=9)
        x = PrettyTable()
        x.field_names = ['Algorithm', 'Prediction', 'Building time [s]', '5-C-V', '5-C-V time [s]']
        print('\n\n === TFIDF n-gram ===\n')

        start = time()
        model = linear_model.LogisticRegression(max_iter=200)
        model.fit(x_train, y_train)
        pickle.dump(model, open((save_dir + 'lr_ngram.sav'), 'wb'))
        LR_prediction = model.predict(x_test)
        pred_time = time() - start
        print(f'### Logistic regression ### Duration %0.2f' % pred_time)
        print(f'  Test accuracy: %0.3f' % (accuracy_score(LR_prediction, y_test)))
        start = time()
        scores = cross_val_score(model, x_train, y_train, cv=5)
        cv_time = time() - start
        cv_acc = '%0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)
        print("  5-C-V Accuracy: " + cv_acc + " Duration: %0.2f" % cv_time)
        x.add_row(['Log.Reg', round(accuracy_score(LR_prediction, y_test), 3), round(pred_time, 2), cv_acc,
                   round(cv_time, 2)])

        start = time()
        model = naive_bayes.MultinomialNB()
        model.fit(x_train, y_train)
        pickle.dump(model, open((save_dir + 'nb_ngram.sav'), 'wb'))
        NB_prediction = model.predict(x_test)
        pred_time = time() - start
        print(f'### Naive Bayes ### Duration %0.2f' % pred_time)
        print(f'  Test accuracy: %0.3f' % (accuracy_score(NB_prediction, y_test)))
        start = time()
        scores = cross_val_score(model, x_train, y_train, cv=5)
        cv_time = time() - start
        cv_acc = '%0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)
        print("  5-C-V Accuracy: " + cv_acc + " Duration: %0.2f" % cv_time)
        x.add_row(
            ['Bayes', round(accuracy_score(NB_prediction, y_test), 3), round(pred_time, 2), cv_acc, round(cv_time, 2)])

        start = time()
        model = svm.SVC()
        model.fit(x_train, y_train)
        pickle.dump(model, open((save_dir + 'svc_ngram.sav'), 'wb'))
        SVM_prediction = model.predict(x_test)
        pred_time = time() - start
        print(f'Support Vector Classification ### Duration %0.2f' % pred_time)
        print(f'  Test accuracy: %0.3f' % (accuracy_score(SVM_prediction, y_test)))
        start = time()
        scores = cross_val_score(model, x_train, y_train, cv=5)
        cv_time = time() - start
        cv_acc = '%0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)
        print("  5-C-V Accuracy: " + cv_acc + " Duration: %0.2f" % cv_time)
        x.add_row(['Sup.Vec.', round(accuracy_score(SVM_prediction, y_test), 3), round(pred_time, 2), cv_acc,
                   round(cv_time, 2)])

        start = time()
        model = svm.LinearSVC(max_iter=200)
        model.fit(x_train, y_train)
        pickle.dump(model, open((save_dir + 'lsvc_ngram.sav'), 'wb'))
        LSVM_prediction = model.predict(x_test)
        pred_time = time() - start
        print(f'### Linear Support Vector Classification ###  Duration %0.2f' % pred_time)
        print(f'  Test accuracy: %0.3f' % (accuracy_score(LSVM_prediction, y_test)))
        start = time()
        scores = cross_val_score(model, x_train, y_train, cv=5)
        cv_time = time() - start
        cv_acc = '%0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)
        print("  5-C-V Accuracy: " + cv_acc + " Duration: %0.2f" % cv_time)
        x.add_row(['Lin.Sup.Vec', round(accuracy_score(LSVM_prediction, y_test), 3), round(pred_time, 2), cv_acc,
                   round(cv_time, 2)])

        start = time()
        model = ensemble.RandomForestClassifier()
        model.fit(x_train, y_train)
        pickle.dump(model, open((save_dir + 'rf_ngram.sav'), 'wb'))
        RandomForest_prediction = model.predict(x_test)
        pred_time = time() - start
        print(f'### Random Forest ### Duration %0.2f' % pred_time)
        print(f'  Test accuracy: %0.3f' % (accuracy_score(RandomForest_prediction, y_test)))
        start = time()
        scores = cross_val_score(model, x_train, y_train, cv=5)
        cv_time = time() - start
        cv_acc = '%0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)
        print("  5-C-V Accuracy: " + cv_acc + " Duration: %0.2f" % cv_time)
        x.add_row(['Rand.For', round(accuracy_score(RandomForest_prediction, y_test), 3), round(pred_time, 2), cv_acc,
                   round(cv_time, 2)])
        print("\n Summary")
        print(x)


    def classifyUsingExistingModels(self, models_dir):
        l_dict = {0:'business', 1:'entertainment', 2:'politics', 3:'sport', 4:'tech'}
        data_x = self.in_bow.toarray()

        # USING BOW
        x = PrettyTable()
        x.field_names = ['File','BOW_LR', 'BOW_NBC', 'BOW_SVC', 'BOW_LSVC', "BOW_RF"]
        BOW_LR = pickle.load(open((models_dir + 'lr_bow.sav'), 'rb'))
        BOW_NBC = pickle.load(open((models_dir + 'nb_bow.sav'), 'rb'))
        BOW_SVC = pickle.load(open((models_dir + 'svc_bow.sav'), 'rb'))
        BOW_LSVC = pickle.load(open((models_dir + 'lsvc_bow.sav'), 'rb'))
        BOW_RF = pickle.load(open((models_dir + 'rf_bow.sav'), 'rb'))
        for i in range(len(data_x)):
            BOW_LR_pre = BOW_LR.predict(data_x[i].reshape(1, -1))
            BOW_NBC_pre = BOW_NBC.predict(data_x[i].reshape(1, -1))
            BOW_SVC_pre = BOW_SVC.predict(data_x[i].reshape(1, -1))
            BOW_LSVC_pre = BOW_LSVC.predict(data_x[i].reshape(1, -1))
            BOW_RF_pre = BOW_RF.predict(data_x[i].reshape(1, -1))
            data = self.DA.data.iloc[i]
            x.add_row([data['file_id'], l_dict[BOW_LR_pre[0]], l_dict[BOW_NBC_pre[0]], l_dict[BOW_SVC_pre[0]],
                       l_dict[BOW_LSVC_pre[0]], l_dict[BOW_RF_pre[0]]])
        print("\nPredictions using BOW model")
        print(x)

        ## USING TFIDF
        x = PrettyTable()
        x.field_names = ['File','TFIDF_LR', 'TFIDF_NBC', 'TFIDF_SVC', 'TFIDF_LSVC', "TFIDF_RF"]
        TFIDF_LR = pickle.load(open((models_dir + 'lr_tfidf.sav'), 'rb'))
        TFIDF_NBC = pickle.load(open((models_dir + 'nb_tfidf.sav'), 'rb'))
        TFIDF_SVC = pickle.load(open((models_dir + 'lsvc_tfidf.sav'), 'rb'))
        TFIDF_LSVC = pickle.load(open((models_dir + 'lsvc_tfidf.sav'), 'rb'))
        TFIDF_RF = pickle.load(open((models_dir + 'rf_tfidf.sav'), 'rb'))
        for i in range(len(data_x)):
            TFIDF_LR_pre = TFIDF_LR.predict(data_x[i].reshape(1, -1))
            TFIDF_NBC_pre = TFIDF_NBC.predict(data_x[i].reshape(1, -1))
            TFIDF_SVC_pre = TFIDF_SVC.predict(data_x[i].reshape(1, -1))
            TFIDF_LSVC_pre = TFIDF_LSVC.predict(data_x[i].reshape(1, -1))
            TFIDF_RF_pre = TFIDF_RF.predict(data_x[i].reshape(1, -1))
            data = self.DA.data.iloc[i]
            x.add_row([data['file_id'], l_dict[TFIDF_LR_pre[0]], l_dict[TFIDF_NBC_pre[0]], l_dict[TFIDF_SVC_pre[0]],
                       l_dict[TFIDF_LSVC_pre[0]], l_dict[TFIDF_RF_pre[0]]])
        print("\nPredictions using TFIDF model")
        print(x)

        x = PrettyTable()
        x.field_names = ['File','NGRAM_LR', 'NGRAM_NBC', 'NGRAM_SVC', 'NGRAM_LSVC', "NGRAM_RF"]
        NGRAM_LR = pickle.load(open((models_dir + 'lr_bow.sav'), 'rb'))
        NGRAM_NBC = pickle.load(open((models_dir + 'nb_bow.sav'), 'rb'))
        NGRAM_SVC = pickle.load(open((models_dir + 'svc_bow.sav'), 'rb'))
        NGRAM_LSVC = pickle.load(open((models_dir + 'lsvc_bow.sav'), 'rb'))
        NGRAM_RF = pickle.load(open((models_dir + 'rf_bow.sav'), 'rb'))
        for i in range(len(data_x)):
            NGRAM_LR_pre = NGRAM_LR.predict(data_x[i].reshape(1, -1))
            NGRAM_NBC_pre = NGRAM_NBC.predict(data_x[i].reshape(1, -1))
            NGRAM_SVC_pre = NGRAM_SVC.predict(data_x[i].reshape(1, -1))
            NGRAM_LSVC_pre = NGRAM_LSVC.predict(data_x[i].reshape(1, -1))
            NGRAM_RF_pre = NGRAM_RF.predict(data_x[i].reshape(1, -1))
            data = self.DA.data.iloc[i]
            x.add_row([data['file_id'], l_dict[NGRAM_LR_pre[0]], l_dict[NGRAM_NBC_pre[0]], l_dict[NGRAM_SVC_pre[0]],
                       l_dict[NGRAM_LSVC_pre[0]], l_dict[NGRAM_RF_pre[0]]])
        print("\nPredictions using NGRAM model")
        print(x)

if __name__ == "__main__":
    learn_data_dir = './raw_text_v2/'
    save_dir = './models/'
    Models = Models(learn_data_dir, save_dir, None, True)
    Models.learnWithBOW(save_dir)
    Models.learnWithTFIDF(save_dir)
    Models.learnWithNgram(save_dir)
