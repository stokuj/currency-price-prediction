import os
from locale import currency
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler

# Hide TensorFlow INFO logs in console output.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, RNN, LSTMCell
from tensorflow.keras.models import Sequential



class Model():
    """ Model class usef for program logic
    """
    def __init__(self,controller):
        self.controller = controller
    
    ### Variable definition
    ### 
    ready_to_do_plot    = False
    crypto_currency     = 'BTC'
    against_currency    = 'USD'
    data_source         = 'yahoo'
    use_online_db = True
    start = dt.datetime(2016,1,1)
    end = dt.datetime.now()

    data = yf.download(
        f'{crypto_currency}-{against_currency}',
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )
    scaler = MinMaxScaler(feature_range=(0,1))

    prediction_days = 30
    future_day = 1

    x_train, y_train = [], []
    
    model_id = 1
    model = Sequential()

    test_start  = dt.datetime(2022,1,23)

    def train(self):
        """ Main fuction for training
        """
        self.x_train, self.y_train = [], []
        self.model = Sequential()

        if(self.use_online_db):

            #self.data = web.DataReader(f'{self.crypto_currency}-{self.against_currency}', self.data_source, self.start, self.end)]
            self.data = yf.download(
                f'{self.crypto_currency}-{self.against_currency}',
                start=self.start,
                end=self.end,
                auto_adjust=False,
                progress=False,
            )

        else:

            filename = "test.csv"
            data = pd.read_csv(filename)


        scaled_data = self.scaler.fit_transform(self.data['Close'].values.reshape(-1,1))

        for x in range(self.prediction_days, len(scaled_data)-self.future_day):

            self.x_train.append(scaled_data[x - self.prediction_days:x, 0])
            self.y_train.append(scaled_data[x + self.future_day, 0])

        self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))

        ### Create Neural Network

        if self.model_id == 1:
            self.model.add(LSTM(units=50, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(units=50, return_sequences=True))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(units=50))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(units=1))

        elif self.model_id == 2: 
            self.model.add(GRU(units=50, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
            self.model.add(Dropout(0.2))
            self.model.add(GRU(units=50, return_sequences=True))
            self.model.add(Dropout(0.2))
            self.model.add(GRU(units=50))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(units=1))
        else:
            self.model.add(RNN(cell = LSTMCell(50), return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
            self.model.add(Dropout(0.25))
            self.model.add(GRU(units=50, return_sequences=True))
            self.model.add(Dropout(0.10))
            self.model.add(GRU(units=50))
            self.model.add(Dropout(0.10))
            self.model.add(Dense(units=1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(self.x_train, self.y_train, epochs=100, batch_size=32)
        
        self.ready_to_do_plot = True
        
    def plot(self):
        """ Main function for making plot
        """
        test_end    = dt.datetime.now() + dt.timedelta(days=self.future_day)
        #test_data = web.DataReader(f'{self.crypto_currency}-{self.against_currency}', 'yahoo', self.test_start, test_end)
        test_data = yf.download(
            f'{self.crypto_currency}-{self.against_currency}',
            start=self.test_start,
            end=test_end,
            auto_adjust=False,
            progress=False,
        )


        actual_prices = test_data['Close'].values

        total_dataset = pd.concat((self.data['Close'], test_data['Close']), axis=0)

        model_inputs = total_dataset[len(total_dataset) - len(test_data) - self.prediction_days:].values
        
        ########
        ### Support array used to fix problem of hight/low starting values of predition on day0
        array_supp2 = total_dataset[len(total_dataset) - len(test_data)-self.prediction_days- len(test_data):len(total_dataset) - len(test_data)- len(test_data)]
        array_supp2 = np.array(array_supp2)
        
        model_inputs_temp= model_inputs[self.prediction_days:]
        array_supp2 = np.concatenate([array_supp2, model_inputs_temp])
    
        model_inputs = array_supp2
        ####
        
        model_inputs = model_inputs.reshape(-1, 1)
        model_inputs = self.scaler.fit_transform(model_inputs)

        x_test = []

        for x in range(self.prediction_days , len(model_inputs)):
            x_test.append(model_inputs[x - self.prediction_days :x, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        prediction_prices = self.model.predict(x_test)
        prediction_prices = self.scaler.inverse_transform(prediction_prices)

        plt.plot(actual_prices, color='cyan', label='Actual Prices')
        plt.plot(prediction_prices, color='indigo', label='Predicted Prices')
        plt.title(f'CryptoCurrency Price Prediction')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend(loc='upper right')
        plt.show()
        
    def gain(self):
        """Main fuction for calulating gain

        Returns:
            numpyInt: It returns gain
        """
        start_date_gain = dt.datetime.now() - dt.timedelta(days=self.prediction_days)
        #df = web.DataReader(f'{self.crypto_currency}-{self.against_currency}', self.data_source, start_date_gain , dt.datetime.now())
        df = yf.download(
            f'{self.crypto_currency}-{self.against_currency}',
            start=start_date_gain,
            end=dt.datetime.now(),
            auto_adjust=False,
            progress=False,
        )
        x =df['Close'].iloc[:1].values
        y =df['Close'].iloc[-1:].values

        return (y/x * 100)
    
    def set_currency(self,i):
        """ Basic setter for currency

        Args:
            i (int): Value to be set
        """
        if i == 1:
            self.crypto_currency = 'BTC'
        if i == 2:
            self.crypto_currency = 'ETH'
        if i == 3:
            self.crypto_currency = 'DOGE'
        if i == 4:
            self.crypto_currency = 'LTC'

            
    def set_data_source(self,i):
        """ Basic setter for data source id

        Args:
            i (int): Value to be set
        """
        if i == 1:
            self.data_source    = 'yahoo'
        if i == 2:
            self.data_source    = 'stooq'
        if i == 3:
            self.data_source    = 'naver'

    def set_model_id(self,i):
        """ Basic setter for model id

        Args:
            i (int): Value to be set
        """
        if i == 1:
            self.model_id   = 1
        if i == 2:
            self.model_id   = 2
        if i == 3:
            self.model_id   = 3
    

    def set_file_path(self, p):
        """ Basic setter for filepath

        Args:
            i (str): String to be set
        """
        self.filename = p


    def set_switch_state(self, i):
        """ Basic setter for switch state

        Args:
            i (bool): Bool to be set
        """
        if i == 1:
            self.use_online_db = True
        else:
            self.use_online_db = False


    def set_predition_days(self, i):
        """ Basic setter for prediction days

        Args:
            i (int): Value to be set
        """
        self.prediction_days = i
        
    def set_future_days(self, i):
        """ Basic setter for future days

        Args:
            i (int): Value to be set
        """
        self.future_day = i
        
    def set_test_start_date(self, i):
        """ Basic setter for start date, used in gain

        Args:
            i (int): Number of days used to calculate day to set
        """
        self.test_start = dt.datetime.now() - dt.timedelta(days=i) 
