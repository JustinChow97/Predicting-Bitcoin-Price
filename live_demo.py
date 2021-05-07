import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
all_bitcoin_data = pd.read_excel (r'Orginal_data.xlsx', engine="openpyxl")
all_bitcoin_data = all_bitcoin_data[["Close", "Volume"]]
bitcoin_data_ML = pd.read_excel (r'FINAL_BITCOIN_DATA.xlsx', engine="openpyxl")

X_columns = bitcoin_data_ML.drop(["Pos or Neg Change in Close Price"], axis=1)
y_close_price = bitcoin_data_ML[['Pos or Neg Change in Close Price']]

KNN = KNeighborsClassifier(n_neighbors=30, weights="distance", algorithm="auto")
KNN.fit(X_columns, y_close_price.values.ravel())

logreg = LogisticRegression(solver='lbfgs', max_iter=3000)
logreg.fit(X_columns, y_close_price.values.ravel())


mlp = MLPClassifier(solver='adam', random_state = 11, hidden_layer_sizes = 3)
print(type(mlp))
mlp.fit(X_columns, y_close_price.values.ravel())


def machine_learning_algos(all_parameters):
    KNN_predicted_value = KNN.predict([all_parameters])
    logreg_predicted_value = logreg.predict([all_parameters])
    mlp_predicted_value = mlp.predict([all_parameters])
    return mlp_predicted_value, logreg_predicted_value, KNN_predicted_value


def get_percentile(current_price, max_price, min_price):
    if (max_price == min_price): return 0
    percentile = (current_price - min_price) / (max_price -  min_price)
    percentile = math.floor(percentile * 10.0) / 10
    return float(percentile)

def get_time_quadarent(time):
    quadarent = 0
    am_pm = 0
    if(quadarent >= 0 and quadarent<6):
        quadarent = 1
        am_pm = 0
    if(quadarent>=6 and quadarent < 12):
        quadarent = 2
        am_pm = 0
    if(quadarent>=12 and quadarent<18):
        quadarent = 3
        am_pm = 1
    if(quadarent>=18 and quadarent<24):
        quadarent = 4
        am_pm = 1

    return int(am_pm), int(quadarent)

def significant_high_low(current_price, feature, choice, time):
    bitcoin_time_df = all_bitcoin_data.tail(time)
    bitcoin_time_featur_df = bitcoin_time_df[feature].tolist()
    if (choice == "High"):
        comapre_too =  max(bitcoin_time_featur_df, default=0)
        if (current_price == comapre_too): return 1
        else: return 0
    else:
        comapre_too =  min(bitcoin_time_featur_df, default=0)
        if (current_price == comapre_too): return 1
        else: return 0

def ratio_moving_averge(close_price, feature, time_list):
    bitcoin_time_df = all_bitcoin_data.tail(time_list)
    bitcoin_time_df = bitcoin_time_df[feature].tolist()
    average = sum(bitcoin_time_df) / len(bitcoin_time_df)
    if (average == 0): return 0
    is_negtive = False
    ratio_from_avergae = (close_price - average) / average
    if (ratio_from_avergae > 100): return 1
    if (ratio_from_avergae < 0):
        is_negtive = True
        ratio_from_avergae = ratio_from_avergae * -1
    ratio_from_avergae = math.floor(ratio_from_avergae * 100.0) / 100
    if(is_negtive == True and ratio_from_avergae != 0):
        ratio_from_avergae = ratio_from_avergae * -1
    return float(ratio_from_avergae)

def find_high_low(time, feature, price):
    bitcoin_time_df = all_bitcoin_data.tail(time)
    df_feature_list = bitcoin_time_df[feature].tolist()
    bitcoin_time_max = max(df_feature_list)
    bitcoin_time_min = min(df_feature_list)
    return float(bitcoin_time_max), float(bitcoin_time_min)


def find_features(month, time, close_price, volume):
    feature_list = ["Close", "Volume"]
    time_length_hours = [24, 168, 672, 10000000]
    high_low = ["High", "Low"]
    list_features = []

    list_features.append(month)
    am_pm, quadarent = get_time_quadarent(time)
    list_features.append(am_pm)
    list_features.append(quadarent)
    percentile_coff = [0.4, 0.1, 0.2, 0.3]

    for feature in feature_list:
        calc_percentile = 0
        index = 0
        for time_list in time_length_hours:
            if (feature == "Volume"): pass_parameter = volume
            else: pass_parameter = close_price
            high_price, low_price = find_high_low(time_list, feature, pass_parameter)
            percentile = get_percentile(pass_parameter, high_price, low_price)
            calc_percentile = calc_percentile + percentile * percentile_coff[index]
            index = index + 1
        list_features.append(calc_percentile)

    is_high_low_found = False
    for choice in high_low:
        for feature in feature_list:
            for time in time_length_hours:
                if (feature == "Volume"): pass_parameter = volume
                else: pass_parameter = close_price
                if(significant_high_low(pass_parameter, feature, choice, time) == 1):
                    is_high_low_found = True
            if(is_high_low_found):list_features.append(1)
            else: list_features.append(0)
            is_high_low_found == False

    for feature in feature_list:
        calc_moving_average = 0
        index = 0
        for time_list in time_length_hours:
            moving_average = ratio_moving_averge(close_price, feature, time_list)
            calc_moving_average = calc_moving_average + moving_average * percentile_coff[index]
            index = index + 1
        list_features.append(calc_moving_average)
    return list_features

print("======")
print("Welcome to the Worlds Greatest Bitcoin Predictor ")
print("======")
while(1):
    print(" ")
    print("--------")
    print("Enter The Following Information To know if the Current Bitcoin Price Will Rise or Fall.")
    current_date = input("What is the current Month (1 - 12): ")
    current_time = input("The Current Hour (24 hours): ")
    current_close_price = input("Closing price of the hour(USD): ")
    current_volume = input("Volume during the hour(MM): ")

    new_row = {'Close':float(current_close_price), "Volume":float(current_volume)}
    all_bitcoin_data = all_bitcoin_data.append(new_row, ignore_index=True)

    feature_for_price = find_features(int(current_date), int(current_time), float(current_close_price), float(current_volume))

    mlp1, log, knn = machine_learning_algos(feature_for_price)
    if (log[0] == -1): log_prediction = "Fall"
    else: log_prediction = "Rise"

    if (knn[0] == -1): knn_prediction = "Fall"
    else: knn_prediction = "Rise"

    if (mlp1[0] == -1): mlp_prediction = "Fall"
    else: mlp_prediction = "Rise"

    total = mlp1[0] + knn[0] + log[0]
    print("total")
    print(total)
    if (total >= 2): total_prediction = "Rise"
    else: total_prediction  = "Fall"
    print(" ")
    print("We got results ")
    print("The Logistic Regression Algorithm Prediced Bitcoin prices will: " +str(log_prediction))
    print("The K-Nearest Neighbors Algorithm Prediced Bitcoin prices will: " +str(knn_prediction))
    print("The Neural Network Algorithm Prediced Bitcoin prices will: " +str(mlp_prediction))
    print("")
    print("We Predict Bitcoin Close Price will " +str(total_prediction) + " In the Next Hour.")
    print("")
    print("*****************************")
    print("Thank you. We are not financial advisors. Trade at your own risk")
    print("*****************************")
    print(" ")
