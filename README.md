# Disaster Response Pipeline Project

### 说明:

此仓库存储的是我在优达学城数据科学家项目的项目文件，可以按照如下步骤进行数据准备，监督学习模型训练，并启动一个本地网站。

1. 在项目的根目录运行以下两个脚本完成数据整理和模型训练

    - process_data.py脚本将读取两个CSV文件中的内容，清理后合并成一个数据表，存在一个本地SQLite数据库中
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`

        命令行参数解释：
        data/disaster_messages.csv：(输入)消息数据文件路径
        data/disaster_categories.csv: (输入)消息类别数据文件路径
        data/disaster_response.db: (输出)SQLite数据库文件路径


    - train_classifier.py脚本将读取disaster_response.db数据库中的Message表，利用里头的数据训练一个多类别分类模型，模型经过优化后序列化到文件中，方便web程序使用
        `python models/train_classifier.py data/disaster_response.db models/classifier.pkl`

        命令行参数解释：
        data/disaster_response.db: (输入)SQLite数据库文件路径
        models/classifier.pkl: (输出)训练后的分类模型

2. 进入项目的app子目录，运行以下脚本启动web服务
    `python run.py`

3. 通过网址：http://localhost:3001/ 访问主页

