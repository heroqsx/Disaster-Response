# 导入相关的库
import sys
import pandas as pd
from sqlalchemy import create_engine

# Scikit Learn相关的库
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# nltk相关的库
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import nltk

# 对象序列化库
import pickle

#加载必要的nltk数据
nltk.download(["punkt", "wordnet", "averaged_perceptron_tagger"])

def load_data(database_filepath):
    """
    数据加载函数，从SQLite数据库中读取消息数据，并且拆分成特征列和目标值列，同时返回目标值标题

    INPUT:
        database_filepath - SQLite数据库文件名
    OUTPUT:
        X - 特征值data frame
        y - 目标值data frame
        category_names - 目标值标题
    """

    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table("Message",engine)

    X = df["message"]
    y = df.iloc[:,4:]
    category_names = y.columns

    return X, y, category_names

def tokenize(text):
    """
    自定义的分词函数，完成文本标准化，分词和词形还原

    INPUT:
        text - 消息文本
    OUTPUT:
        clean_words - 处理完后的单词列表
    """
    text = text.lower().strip() # 统一转换为小写字母并去除头尾空格
    # text = re.sub(r"[^a-z0-9]", " ", text) 不能这么简单粗暴，有时候标点符号是有意义的
    lemmatizer = WordNetLemmatizer()
    
    clean_words = []
    for word, tag in pos_tag(word_tokenize(text)):
        # 对每个单词，根据词性，进行词形还原
        if word == ".": # 单独的标点符号没有意义
            continue
        
        # 尽量减少单词的各种变化形式
        if tag.startswith("NN"):
            clean_words.append(lemmatizer.lemmatize(word, pos="n"))
        elif tag.startswith("VB"):
            clean_words.append(lemmatizer.lemmatize(word, pos="v"))
        elif tag.startswith("JJ"):
            clean_words.append(lemmatizer.lemmatize(word, pos="a"))
        elif tag.startswith("R"):
            clean_words.append(lemmatizer.lemmatize(word, pos="r"))
        else:
            clean_words.append(word)
    return clean_words


def build_model():
    """
    生成一个pipeline对象

    INPUT:
        无
    OUTPUT:
        pipeline对象
    """

    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return model


def evaluate_model(model, X_test, y_test, category_names):
    """
    模型评估函数，打印模型的评估结果

    INPUT:
        model - 需要评估的模型
        X_test - 测试特征数据
        y_test - 测试目标数据
        category_name - 目标类别名称
    OUTPUT:
    """

    # 用模型进行预测
    y_pred = model.predict(X_test)

    # 循环每一个预测列
    for i, col in enumerate(category_names):
        print("--------category: {}--------".format(col))
        # 按照提示，使用的时classification_report函数
        print(classification_report(y_test[col], y_pred[:, i]))

def save_model(model, model_filepath):
    """
    将模型序列化成本地文件，方便后续使用

    INPUT:
        model - 训练好的模型
    OUTPUT:
        model_filepath - 模型文件名称
    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        
        # 训练集和测试集拆分，习惯了用大写的X，小写的y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/disaster_response.db classifier.pkl')


if __name__ == '__main__':
    main()
   