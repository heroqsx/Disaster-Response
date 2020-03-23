# 加载相关的包
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    数据加载函数，将消息和消息类别两个CSV文件加载到data frame数据结构中

    INPUT:
        messages_filepath - 消息文件路径
        categories_filepath - 消息类别文件路径

    OUTPUT:
        df - 合并后的data frame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on="id")

    return df

def clean_data(df):
    """
    数据清洗函数，去除重复数据并重整data frame的数据结构

    INPUT:
        df - data frame对象，包含所有的消息数据

    OUTPUT: 
        df - data frame对象，包含重整后的消息数据
    """

    # 用categories列的str属性，对列数据进行拆分，获得一个data frame对象
    categories = df["categories"].str.split(";",expand=True)

    # 这个data frame对象任意一行，都包含了新的列名，取第一行处理
    row = categories.iloc[0,:]

    # 列名可以用数据项的内容中去掉最后两个字符来产生
    category_colnames = [x[:-2] for x in row]

    # 对数据列进行重新命名
    categories.columns = category_colnames

    # 遍历每一列，用str的slice函数，截取单元格内容
    for column in category_colnames:
        categories[column] = categories[column].str.slice(start=-1) # 单元格保留最后一个字符
        categories[column] = categories[column].astype("int32") # 转换成整型

    
    # 生成36个category列，并去除重复行
    del df["categories"]
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)

    return df # 数据整理完毕，返回data frame


def save_data(df, database_filename):
    """
    将data frame保存成SQLite数据库，方便后续实用

    INPUT:
        df - data frame数据结构，包含消息数据
        database_filename - SQLite数据库文件名字，表名则默认为"Message"

    OUTPUT:
        None
    """

    engine = create_engine("sqlite:///" + database_filename)
    df.to_sql('Message', engine, index=False, if_exists="replace")  


def main():
    """
    数据处理主函数

    INPUT
        从命令行读取三个字符串
        1 - 消息文件路径
        2 - 消息类别文件路径
        3 - 生成的数据库文件名称
    OUTPUT

    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'disaster_response.db')


# 当前脚本为主脚本时，运行main函数
if __name__ == '__main__':
    main()
