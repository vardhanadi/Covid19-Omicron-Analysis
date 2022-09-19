import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *

Omicron = pd.read_csv("D:/Omicron Datset/covid-variants.csv")

Omicron['date'] = Omicron['date'].astype("datetime64")

countries = Omicron.groupby(['location'])['num_sequences_total'].sum().sort_values(ascending = False).reset_index()

figure = px.choropleth(countries,locations='location', locationmode= 'country names', color= 'num_sequences_total', 
                       hover_name='location', color_continuous_scale='tealgrn', 
                       range_color=[1,1000000],title='Countries with Number of cases')
figure.show()

daily_cases = Omicron.groupby(['date'])['num_sequences_total'].sum().sort_values(ascending = False).reset_index() 
fig = px.bar(daily_cases, x='date', y='num_sequences_total')
fig.show()

class CovidPredictor:
    
    
    def __init__(self,length=80,batch_size=1):
        self.length = length
        self.batch_size = batch_size
    
    def model(self,length,batch_size,scaled_train,scaled_test):
        
        generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=batch_size)
        model = Sequential()
        model.add(LSTM(150,input_shape=(length,scaled_train.shape[1])))
        model.add(Dense(scaled_train.shape[1]))

        model.compile(optimizer='rmsprop', loss='mean_squared_error')
        model.summary()
        early_stop = EarlyStopping(monitor='val_loss',patience=25)
        validation_generator = TimeseriesGenerator(scaled_test,scaled_test, 
                                           length=length, batch_size=batch_size)
        model.fit_generator(generator,epochs=100,
                    validation_data=validation_generator,
                   callbacks=[early_stop])
        
        print(model.history.history.keys())
        plt.figure(figsize=(15,5))
        losses = pd.DataFrame(model.history.history)
        losses.plot()
        plt.show()
        return model
    
    def performance(self,model):
        
        model.history.history.keys()
        losses = pd.DataFrame(model.history.history)
        return losses
    
    def predict(self,model,scaled_train,scaled_test,test_lenght):
        
        features = scaled_train.shape[1]
        pred = []

        first_batch = scaled_train[-test_lenght:]
        batch = first_batch.reshape((1, test_lenght, features))

        for i in range(test_lenght):
            pred_frs = model.predict(batch)[0]
            pred.append(pred_frs) 
            batch = np.append(batch[:,1:,:],[[pred_frs]],axis=1)
        return pred
    
    def scale(self,train,test):
        
        scaler = MinMaxScaler()
        scaler.fit(train)
        scaled_train = scaler.transform(train)
        scaled_test = scaler.transform(test)
        return scaler,scaled_train,scaled_test,train,test
    
    def inverse(self,data,scaler):
        
        test= scaler.inverse_transform(data)
        return test
    
    def data_preb(self,data,index_size):
        try:
            cols = ['location','variant','perc_sequences','num_sequences']
            data.drop(cols,axis=1,inplace=True)
        except:
            pass
        data.sort_index(ascending=True,inplace=True)
        train = data.iloc[:index_size]
        test = data.iloc[index_size:]
        scaler,scaled_train,scaled_test,train,test = self.scale(train,test)
        return scaler,scaled_train,scaled_test,train,test
    
    def measure_error(self,true,test):
        
        print(mean_absolute_error(true,test))
        print(mean_squared_error(true,test))
        print(r2_score(true,test))



covid = CovidPredictor()
# help(covid)

Omicron['Time'] = Omicron.index
data = pd.DataFrame(Omicron.groupby('Time')['num_sequences_total'].sum())
scaler,scaled_train,scaled_test,train,test = covid.data_preb(data,40)

model = covid.model(length=4,batch_size=1,scaled_train=scaled_train,scaled_test=scaled_test)

prediction = covid.predict(model,scaled_train,scaled_test,5)
preds = covid.inverse(prediction,scaler)
preds = pd.DataFrame(preds,columns=['Pred'],index=test.index)
plt.figure(figsize=(15,5))
plt.plot(test.values, linewidth=2, markersize=1,label="True")
plt.plot(preds, linewidth=2, markersize=1,label="Predictions")
plt.title("Forcasting",loc='center', fontdict={'fontsize': 30, 'fontweight': 'bold'})
plt.legend()