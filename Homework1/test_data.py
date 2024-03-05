# import pandas as pd

# origin_data = pd.read_csv('./data/train_data.csv')
# origin_columns = origin_data.columns.to_list()
# origin_columns.sort() 

# total_data = pd.read_csv('./kumarajarshi-life-expectancy-who/Life Expectancy Data.csv')
# total_columns = total_data.columns.to_list()
# total_columns.sort()

# # 保留相同的列顺序
# origin_data = origin_data[origin_columns]  
# total_data = total_data[origin_columns]

# # 对两份数据按列排序 
# for col in origin_data.columns:
#     origin_data = origin_data.sort_values(by=col)
# for col in total_data.columns:
#     total_data = total_data.sort_values(by=col)
    
# # 将total_data中不在origin_data的数据保存为test.csv
# test_data = total_data[~total_data.isin(origin_data)]
# test_data.to_csv('test.csv', index=False)

# print('已将测试数据保存为 test.csv')

import pandas as pd
from tqdm import tqdm

origin_data = pd.read_csv('./data/train_data.csv')
total_data = pd.read_csv('./kumarajarshi-life-expectancy-who/Life Expectancy Data.csv')

# 保存列名
origin_columns = origin_data.columns.to_list()
total_data = total_data[origin_columns]
# 初始化保存testData的DataFrame
test_data = pd.DataFrame(columns=origin_columns) 

# 逐行比较两份数据
for i, row in tqdm(total_data.iterrows(), total=len(total_data)):
    for j, o_row in origin_data.iterrows():
        # 逐列比较两行数据
        for col in origin_columns:
            if row[col] != o_row[col]:
                test_data = test_data.append(row)
                break

# 删除重复行        
# test_data.drop_duplicates(inplace=True)

# 保存结果
test_data.to_csv('test.csv', index=False)