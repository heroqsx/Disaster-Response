import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib # 用pickle保存的，为什么不用pickle加载？
import pickle
from sqlalchemy import create_engine


app = Flask(__name__)

# def tokenize(text):
#    tokens = word_tokenize(text)
#    lemmatizer = WordNetLemmatizer()
#
#    clean_tokens = []
#    for tok in tokens:
#        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#        clean_tokens.append(clean_tok)
#
#    return clean_tokens

# 这个函数放在一个独立的文件里头，供model和app使用，会更高
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



# load data
print("Reading SQLite database file: ../data/disaster_response.db") # 我用了自己的命名规则，请留意
engine = create_engine('sqlite:///../data/disaster_response.db')
df = pd.read_sql_table('Message', engine)

# load model
# model = joblib.load("../models/classifier.pkl")
model = pickle.load(open("../models/classifier.pkl", "rb"))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # 看了一下数据集，还能够进行分类统计的是那36个类别及其对应的消息数字
    # 36个类别有点多，选Top 10吧
    category_counts = (df.iloc[:, 4:] != 0).sum().sort_values(ascending=False).head(10)
    category_names = list(category_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Top 10 Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }

    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()