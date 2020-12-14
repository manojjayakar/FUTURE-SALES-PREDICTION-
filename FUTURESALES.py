import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from keras.models import load_model, Model


try:
    items = pd.read_csv('items.csv')
    item_cats = pd.read_csv('item_categories.csv')
    shops = pd.read_csv('shops.csv')
    sales = pd.read_csv('sales_train.csv')
    test = pd.read_csv('test.csv')
    submission = pd.read_csv('sample_submission.csv')

    print("Data set loaded successfully.")
except:
    print(" data not loaded ")

sales['year'] = pd.to_datetime(sales['date']).dt.strftime('%Y')
sales['month'] = sales.date.apply(lambda x: datetime.strptime(x,'%d.%m.%Y').strftime('%m')) 
sales.head(2)

grouped = pd.DataFrame(sales.groupby(['year','month'])['item_cnt_day'].sum().reset_index())
sns.pointplot(x='month', y='item_cnt_day', hue='year', data=grouped)


#Price
grouped_price = pd.DataFrame(sales.groupby(['year','month'])['item_price'].mean().reset_index())
sns.pointplot(x='month', y='item_price', hue='year', data=grouped_price)


ts = sales.groupby(['date_block_num'])['item_cnt_day'].sum()
ts.astype('float')
plt.figure(figsize=(16,8))
plt.title('Total Sales of the whole time period',color='yellow')
plt.xlabel('Time',color='yellow')
plt.ylabel('Sales',color='yellow')
plt.plot(ts, color='green');


sales.item_cnt_day.hist(bins=100)
sales.item_cnt_day.describe()


print('Data set size before remove item price 0 cleaning:', sales.shape)
sales = sales.query('item_price > 0')
print('Data set size after remove item price 0 cleaning:', sales.shape)


print('Data set size before filter valid:', sales.shape)
# Only shops that exist in test set.
sales = sales[sales['shop_id'].isin(test['shop_id'].unique())]
# Only items that exist in test set.
sales = sales[sales['item_id'].isin(test['item_id'].unique())]
print('Data set size after filter valid:', sales.shape)



print('Data set size before remove outliers:', sales.shape)
sales = sales.query('item_cnt_day >= 0 and item_cnt_day <= 125 and item_price < 75000')
print('Data set size after remove outliers:', sales.shape)


# Aggregate to monthly level the sales
monthly_sales=sales.groupby(["date_block_num","shop_id","item_id"])[
    "date_block_num","date","item_price","item_cnt_day"].agg({"date_block_num":'mean',"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})

monthly_sales.head(5)


sales_data_flat = monthly_sales.item_cnt_day.apply(list).reset_index()
#Keep only the test data of valid
sales_data_flat = pd.merge(test,sales_data_flat,on = ['item_id','shop_id'],how = 'left')
#fill na with 0
sales_data_flat.fillna(0,inplace = True)
sales_data_flat.drop(['shop_id','item_id'],inplace = True, axis = 1)
sales_data_flat.head(20)


#We will create pivot table.
# Rows = each shop+item code
# Columns will be out time sequence
pivoted_sales = sales_data_flat.pivot_table(index='ID', columns='date_block_num',fill_value = 0,aggfunc='sum' )
pivoted_sales.head(20)

# X we will keep all columns execpt the last one 
X_train = np.expand_dims(pivoted_sales.values[:,:-1],axis = 2)
# the last column is our prediction
y_train = pivoted_sales.values[:,-1:]

# for test we keep all the columns execpt the first one
X_test = np.expand_dims(pivoted_sales.values[:,1:],axis = 2)

# lets have a look on the shape 
print(X_train.shape,y_train.shape,X_test.shape)

# our defining sales model 
sales_model = Sequential()
sales_model.add(LSTM(units = 64,input_shape = (33,1)))
sales_model.add(Dropout(0.5))
sales_model.add(Dense(1))

sales_model.compile(loss = 'mse',optimizer = 'adam', metrics = ['mean_squared_error'])
sales_model.summary()

sales_model.fit(X_train,y_train,batch_size = 4096,epochs = 20)


submission_output = sales_model.predict(X_test)
submission = pd.DataFrame({'ID':test['ID'],'item_cnt_month':submission_output.ravel()})

submission.to_csv('submission_stacked.csv',index = False)
submission.head()











    


