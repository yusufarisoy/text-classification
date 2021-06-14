import re
import codecs
import pickle
from nltk.corpus import stopwords
from googletrans import Translator
from sklearn.datasets import load_files
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def generate(base_sentense):
    resEn = translator.translate(base_sentense, src='tr', dest='en')
    return translator.translate(resEn.text, src='en', dest='tr').text

def pre_proccess(word):
    word = word.lower()
    word = re.sub(r'\d', ' ', word)
    word = re.sub(r'[^\w\s]', '', word)
    word = re.sub(r'\s+', ' ', word)
    return word

def writeToFolder(category, index, sentence):
    if(category == 'Alışveriş kredisi'):
        categoryFolder = 'alisveris_kredisi'
    elif(category == 'İade ve değişim'):
        categoryFolder = 'iade_degisim'
    else:
        categoryFolder = 'urun_bilgisi'
    
    file = codecs.open('txt_sentoken/' + categoryFolder + '/text' + str(index) + '.txt', 'w', 'utf-8-sig')
    file.write(pre_proccess(sentence))
    file.close()

examples = [['İade ve değişim', 'İade ve değişim', 'İade ve değişim', 'Alışveriş kredisi', 'İade ve değişim',
             'İade ve değişim', 'İade ve değişim', 'İade ve değişim', 'İade ve değişim', 'İade ve değişim',
             'İade ve değişim', 'İade ve değişim', 'Alışveriş kredisi', 'Alışveriş kredisi', 'Alışveriş kredisi',
             'Alışveriş kredisi', 'Alışveriş kredisi', 'Alışveriş kredisi', 'Alışveriş kredisi', 'Alışveriş kredisi', 
             'Alışveriş kredisi', 'Ürün bilgisi', 'Ürün bilgisi', 'Ürün bilgisi', 'Ürün bilgisi', 'Ürün bilgisi',
             'Ürün bilgisi', 'Ürün bilgisi', 'Ürün bilgisi', 'Ürün bilgisi', 'Ürün bilgisi', 'İade ve değişim',
             'Alışveriş kredisi', 'Ürün bilgisi', 'İade ve değişim', 'Ürün bilgisi', 'Alışveriş kredisi'],
            
            ['Ürünüm arızalı geldi ne yapmalıyım?', 'Son aldığım ürünü nasıl iade edebilirim?',
             'Gömleğimin bedeni küçük geldi, değiştirmek istiyorum.', 'Nasıl kredi çekebilirim?',
             'Ürünümün garanti süresi devam ediyor, ne yapmam gerekiyor?', 'İade talebim neden kabul edilmedi?',
             'Siparişim siyah renk geldi ancak ben beyaz renk sipariş etmiştim. Değişim yapabilir miyiz?',
             'Gelen ürünü kullandım ve memnun kalmadım, açıklamda bahsedildiği gibi değil. İade etmek istiyorum.',
             'Servis raporum var, iade işlemini nasıl gerçekleştireceğim?', 'Aldığım bütün ürünleri iade etmek istiyorum.',
             'Siparişim belirttiğimden farklı renk geldi, değişim istiyorum.', 'Ürünü alalı 15 günü geçti, yine de iade edebilir miyim?',
             'Alışveriş kredisi çekebilmek için minimum limit nedir?', 'Kredi başvurusu kabul şartlarını öğrenebilir miyim?',
             'Kredi için peşinat vermem gerekiyor mu?', 'Kredi için peşinat versem indirim oluyor mu?', 'Kredi ile alacağım ürünü iade edebiliyor muyum?',
             'Kredi çekebilmek için yaş sınırı var mı?', 'Bir krediyi öderken başka kredi çekebilir miyim?', 
             'Kredi başvurum onaylanmamış, sebebi nedir?', 'Başvurumun onaylanıp onaylanmadığını ne zaman öğrenebilirim?',
             'Ürünler orjinal midir?', 'Ürünlerin garantisi var mı?', 'Ürün özellikleriyle ilgili detaylı bilgiye nasıl ulaşabilirim?',
             'İstediğim ürünleri nasıl karşılaştırabilirim?', 'Bir ürünle ilgili verilen bilgilerin yanıltıcı olduğunu düşünüyorum.',
             'Ürünlerin fiyatları neden farklı?', 'Ürünlerin satıcıları neden farklı?', 'Ürünü hangi satıcıdan aldığım önemli mi?',
             'Ürünleri kendiniz mi üretiyorsunuz?', 'Aradığım bazı ürünleri bulamadım, elinizde yok mu?', 'Memnun kalmazsam değiştirebilir miyim?',
             'Kredi çekiminde vade farkı alıyor musunuz?', 'Ürünlerin fiyatları yanlış yazıyor.', 'Gelen ürünü denedikten sonra değişebilir miyim?',
             'Ürünlerim ne zaman elime ulaşır?', 'Önceki kredimin son 2 taksidi, yeni kredi çekebilir miyim?']]
translator = Translator()

j = 0
for i in range(len(examples[0])):
    writeToFolder(examples[0][i], j, examples[1][i])
    base = examples[1][i]
    j += 1
    while True:
        result = generate(base)
        if(result != base):
            writeToFolder(examples[0][i], j, result)
            base = result
            j += 1
        else:
            break

sentences = load_files('txt_sentoken/', encoding='utf-8')
X, y = sentences.data, sentences.target

with open('X.pickle', 'wb') as file:
    pickle.dump(X, file)
    
with open('y.pickle', 'wb') as file:
    pickle.dump(y, file)

X_1 = open('X.pickle', 'rb')
y_1 = open('y.pickle', 'rb')
X = pickle.load(X_1)
y = pickle.load(y_1)

corpus = []
for i in range(0, j):
    corpus.append(str(X[i]))

vectorizer = CountVectorizer(max_features=2000, min_df=3, max_df=0.6, stop_words=stopwords.words('turkish'))
X = vectorizer.fit_transform(corpus).toarray()

transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()

vectorizer = TfidfVectorizer(max_features=2000, min_df=3, max_df=0.6, stop_words=stopwords.words('turkish'))
X = vectorizer.fit_transform(corpus).toarray()

text_train, text_test, category_train, category_test = train_test_split(X, y, test_size=0.1, random_state=2)

classifier = LogisticRegression()
classifier.fit(text_train, category_train)

category_predictions = classifier.predict(text_test)

conf_matr = confusion_matrix(category_test, category_predictions)