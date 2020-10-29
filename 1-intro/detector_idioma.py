from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
import re
from nltk.stem.porter import PorterStemmer


class Modelo:
    def __init__(self):
        pass

    def treinar(self, entrada, saida):
        print("Treinando...")
        
        X_train = entrada#
        y_train = saida

        def clean_text(text):
            text = re.sub("\n", '', text.lower()) # Remove the \n tag
            #text = re.sub('[\W]+', ' ', text.lower()) # Remove all non-word characters
            return text

        def tokenizer(text):
            return text.split()

        porter = PorterStemmer()
            
        def tokenizer_porter(text):
            return [porter.stem(word) for word in text.split()]

        tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
        param_grid = [{'vect__ngram_range': [(1,1)], 
                    'vect__stop_words': [None],
                    'vect__tokenizer': [tokenizer,
                                        tokenizer_porter],
                    'clf__penalty': ['l1', 'l2'],
                    'clf__C': [1.0, 10.0, 100.0]},
                    {'vect__ngram_range': [(1,1)],
                    'vect__stop_words': [ None],
                    'vect__tokenizer': [tokenizer,
                                        tokenizer_porter],
                    'vect__use_idf':[False],
                    'vect__norm':[None],
                    'clf__penalty': ['l1', 'l2'],
                    'clf__C': [1.0, 10.0, 100.0]
                    }]
            
        lr_tfidf = Pipeline([('vect', tfidf), 
                    ('clf', LogisticRegression(random_state=42, solver='liblinear'))])
        gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=0, n_jobs=-1, iid=False)
        gs_lr_tfidf.fit(X_train, y_train)
        self.clf = gs_lr_tfidf.best_estimator_
        print("Treinamento Concluído!")

    def prever(self, texto):
        texto = [texto]
        result = self.clf.predict(texto)
        confianca = self.clf.predict_proba(texto)

        print("O texto digitado está em {}".format("Português" if result==1 else "Inglês"))
        print("Confiança: {:.2f}%".format(float(confianca[0][result])*100))
