import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from collections import deque
import matplotlib.pyplot as plt

def print_confusion_matrix(Y_2019, confusion_matrix_df):
    '''
    Y_2019: input vector for the confusion matrix
    confusion_matrix_df: the input confusion df
    '''
    total_data_points = len(Y_2019)
    true_positive_number = confusion_matrix_df['Predicted: GREEN']['Actual: GREEN']
    true_positive_rate = np.round(np.multiply(np.divide(true_positive_number, total_data_points), 100), 2)
    true_negative_number = confusion_matrix_df['Predicted: RED']['Actual: RED']
    true_negative_rate = np.round(np.multiply(np.divide(true_negative_number, total_data_points), 100), 2)
    print("True positive rate: {}%".format(true_positive_rate))
    print("True negative rate: {}%".format(true_negative_rate))

def transform_trading_days_to_trading_weeks(df, drop=True):
    '''
    df: dataframe of relevant data
    returns: dataframe with processed data, only keeping weeks, their open and close for said week
    '''
    trading_list = deque()
    # Iterate through each trading week
    for trading_week, df_trading_week in df.groupby(['Year','Week_Number']):
        classification =  df_trading_week.iloc[0][['Classification']].values[0]
        opening_day_of_week = df_trading_week.iloc[0][['Open']].values[0]
        closing_day_of_week = df_trading_week.iloc[-1][['Close']].values[0]
        trading_list.append([trading_week[0], trading_week[1], opening_day_of_week, closing_day_of_week, classification])
    trading_list_df = pd.DataFrame(np.array(trading_list), columns=['Year', 'Trading Week', 'Week Open', 'Week Close', 'Classification'])
    if drop:
        trading_list_df.drop(columns=['Year', 'Trading Week', 'Week Open'], inplace=True)
    return trading_list_df

def make_trade(cash, open, close):
    '''
    cash: float of cash on hand
    open: float of open price
    close: float of close price
    returns: The cash made from a long position from open to close
    '''
    shares = np.divide(cash, open)
    return np.multiply(shares, close)

def trading_strategy(trading_df, prediction_label, weekly_balance=100):
    '''
    trading_df: dataframe of relevant weekly data
    prediction_label: the label for which we're going to trade off of
    returns: A df of trades made based on Predicted Labels
    '''
    # The weekly balance we will be using
    weekly_balance_acc = weekly_balance
    trading_history = deque()
    index = 0
    
    while(index < len(trading_df.index) - 1):
        trading_week_index = index
        if weekly_balance_acc != 0:
            # Find the next consecutive green set of weeks and trade on them
            while(trading_week_index < len(trading_df.index) - 1 and trading_df.iloc[trading_week_index][[prediction_label]].values[0] == 'GREEN'):
                trading_week_index += 1
            green_weeks = trading_df.iloc[index:trading_week_index][['Week Open', 'Week Close']]
            # Check if there are green weeks, and if there are not, we add a row for trading history
            if len(green_weeks.index) > 0:
                # Buy shares at open and sell shares at close of week
                green_weeks_open = float(green_weeks.iloc[0][['Week Open']].values[0])
                green_weeks_close = float(green_weeks.iloc[-1][['Week Close']].values[0])
                # We append the money after we make the trade
                weekly_balance_acc = make_trade(weekly_balance_acc, green_weeks_open, green_weeks_close)
            # Regardless of whether we made a trade or not, we append the weekly cash and week over
            trading_history.append([trading_df.iloc[trading_week_index][['Year']].values[0],
                trading_df.iloc[trading_week_index][['Trading Week']].values[0],
                weekly_balance_acc])
        else:
            # If we have no money we will not be able to trade
            trading_history.append([trading_df.iloc[trading_week_index][['Year']].values[0],
                    trading_df.iloc[trading_week_index][['Trading Week']].values[0],
                    weekly_balance_acc])
        index = trading_week_index+1
    trading_hist_df = pd.DataFrame(np.array(trading_history), columns=['Year', 'Trading Week', 'Balance'])
    trading_hist_df['Balance'] = np.round(trading_hist_df[['Balance']].astype(float), 2)

    return trading_hist_df

def get_windowed_slice_and_fit(df, df_predict, d=1, W=5):
    '''
    df: the dataframe where we are splitting into time series
    df_predict: the dataframe we are using to predict values
    return: dataframe with prediction indices
    '''
    all_predictions = deque()
    df_values = df['Week Close']
    df_predict_values = df_predict['Week Close']
    df_predict_classifications = df_predict['Classification']

    for i in range(W-1, len(df_values)-2):
        # Calculate coefficients given a W and a d
        coeff = np.polyfit(np.arange(i-W+1, i+1), df_values.loc[i-W+1:i].values.T.astype(float), d)
        # Calculate the W+1 week, which will be i + 2
        max_power = len(coeff) - 1
        # X value is the the predicted X value, eg taking a window of 5, X = 6
        X = i + 2
        predict = 0
        for index, value in enumerate(coeff):
            # The power to multiply i by
            power_value = max_power - index
            predict += np.multiply(value, np.power(X, power_value))
        # Classify using rules stated in the doc
        if predict > float(df_predict_values.loc[i]):
            all_predictions.append('GREEN')
        elif predict < float(df_predict_values.loc[i]):
            all_predictions.append('RED')
        else:
            all_predictions.append(df_predict_classifications.loc[i])
    df_prediction = pd.DataFrame(np.array(all_predictions))
    df_prediction.index += W+1
    return df_prediction

def main():
    ticker='WMT'
    file_name = '{}_weekly_return_volatility.csv'.format(ticker)
    file_name_self_labels = 'WMT_Labeled_Weeks_Self.csv'
    iris_dataset = 'iris.data'

    # Read from that file for answering our questions
    df = pd.read_csv(file_name_self_labels, encoding='ISO-8859-1')
    df_2018 = df[df['Year'] == 2018]
    df_2019 = df[df['Year'] == 2019]
    df_trading_weeks_2018 = transform_trading_days_to_trading_weeks(df_2018)
    df_trading_weeks_2019 = transform_trading_days_to_trading_weeks(df_2019)

    print('Question 1')
    print('Using year 1 data to predict year 1 data')
    range_w = range(5, 13)
    range_d = range(1, 4)
    df_table_overall_accuracy = pd.DataFrame(index=range_w, columns=range_d)

    for W in range_w:
        for d in range_d:
            # Take the predicted classifications and compare them to 2018 classifications
            predicted_df_2018 = get_windowed_slice_and_fit(df_trading_weeks_2018, df_trading_weeks_2018, W=W, d=d)
            predicted_df_2018.columns = ['Classification']
            df_2018_classifications = df_trading_weeks_2018.iloc[W+1: len(df_trading_weeks_2018.index)]['Classification'].to_frame()
            df_2018_classifications.columns = ['Classification']
            accuracy = np.round(np.multiply(np.mean(df_2018_classifications == predicted_df_2018), 100)).loc['Classification']
            df_table_overall_accuracy[d][W] = accuracy
    print('Overall Accuracy Table')
    print(df_table_overall_accuracy)
    plt.title('Accuracy vs Sliding Window Range')
    plt.xlabel('W (Sliding window range)')
    plt.ylabel('Accuracy (%)')
    for n in range_d:
        plt.plot(df_table_overall_accuracy.index, df_table_overall_accuracy[n], label='d={}'.format(n))
    plt.legend()
    plt.savefig(fname='./Question_1_Window_Trading.png')
    plt.show()
    plt.close()
    print('\nQuestion 2')
    print('d = 1 and W = 5 gives the best accuracy')
    predicted_df_2019 = get_windowed_slice_and_fit(df_trading_weeks_2018, df_trading_weeks_2019, W=5, d=1)
    predicted_df_2019.columns = ['Classification']
    df_2019_classifications = df_trading_weeks_2019.iloc[6: len(df_trading_weeks_2019.index)]['Classification'].to_frame()
    df_2019_classifications.columns = ['Classification']
    accuracy = np.round(np.multiply(np.mean(df_2019_classifications == predicted_df_2019), 100)).loc['Classification']
    print("Accuracy is {}%".format(accuracy))
    print('d = 2 and W = 6 gives the best accuracy')
    predicted_df_2019 = get_windowed_slice_and_fit(df_trading_weeks_2018, df_trading_weeks_2019, W=6, d=2)
    predicted_df_2019.columns = ['Classification']
    df_2019_classifications = df_trading_weeks_2019.iloc[7: len(df_trading_weeks_2019.index)]['Classification'].to_frame()
    df_2019_classifications.columns = ['Classification']
    accuracy = np.round(np.multiply(np.mean(df_2019_classifications == predicted_df_2019), 100)).loc['Classification']
    print("Accuracy is {}%".format(accuracy))
    print('d = 3 and W = 10 gives the best accuracy')
    predicted_df_2019 = get_windowed_slice_and_fit(df_trading_weeks_2018, df_trading_weeks_2019, W=10, d=3)
    predicted_df_2019.columns = ['Classification']
    df_2019_classifications = df_trading_weeks_2019.iloc[11: len(df_trading_weeks_2019.index)]['Classification'].to_frame()
    df_2019_classifications.columns = ['Classification']
    accuracy = np.round(np.multiply(np.mean(df_2019_classifications == predicted_df_2019), 100)).loc['Classification']
    print("Accuracy is {}%".format(accuracy))
    print('\nQuestion 3')
    confusion_matrix_array = confusion_matrix(df_2019_classifications, predicted_df_2019)
    confusion_matrix_df = pd.DataFrame(confusion_matrix_array, columns= ['Predicted: GREEN', 'Predicted: RED'], index=['Actual: GREEN', 'Actual: RED'])
    print(confusion_matrix_df)
    print('\nQuestion 4')
    df_trading_weeks_2019_no_drop = transform_trading_days_to_trading_weeks(df_2019, drop=False)
    df_trading_weeks_2019_no_drop = df_trading_weeks_2019_no_drop.iloc[6: len(df_trading_weeks_2019_no_drop.index)]
    print('Best values for each d:')
    print('d=1 : W=5')
    print('Trading Strategy value: ')
    predicted_df_2019 = get_windowed_slice_and_fit(df_trading_weeks_2018, df_trading_weeks_2019, W=5, d=1)
    df_trading_weeks_2019_no_drop['Predicted Labels'] = predicted_df_2019
    predicted_trading_df = trading_strategy(df_trading_weeks_2019_no_drop, 'Predicted Labels')
    print('${}'.format(predicted_trading_df[['Balance']].iloc[-1].values[0]))
    print('d=2 : W=6')
    predicted_df_2019 = get_windowed_slice_and_fit(df_trading_weeks_2018, df_trading_weeks_2019, W=6, d=2)
    df_trading_weeks_2019_no_drop['Predicted Labels'] = predicted_df_2019
    predicted_trading_df = trading_strategy(df_trading_weeks_2019_no_drop, 'Predicted Labels')
    print('${}'.format(predicted_trading_df[['Balance']].iloc[-1].values[0]))
    print('d=3 : W=10')
    predicted_df_2019 = get_windowed_slice_and_fit(df_trading_weeks_2018, df_trading_weeks_2019, W=10, d=3)
    df_trading_weeks_2019_no_drop['Predicted Labels'] = predicted_df_2019
    predicted_trading_df = trading_strategy(df_trading_weeks_2019_no_drop, 'Predicted Labels')
    print('${}'.format(predicted_trading_df[['Balance']].iloc[-1].values[0]))

if __name__ == "__main__":
    main()