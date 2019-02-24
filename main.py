import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score, \
    r2_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# prepare min max scaler
min_max_scaler = preprocessing.MinMaxScaler()
# text feature column
# currently this feature is not used to prediction
text_features = ['genres', 'plot_keywords', 'movie_title']
# category feature column
catagory_features = ['actor_1_name', 'actor_2_name', 'actor_3_name', 'director_name', 'country', 'content_rating',
                     'language']
# numeric feature column
number_features = ['actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes',
                   'director_facebook_likes', 'cast_total_facebook_likes', 'budget', 'gross']
# all features
all_selected_features = text_features + catagory_features + number_features + ["imdb_score"]
# list of columns whose values are empty must be removed
eliminate_if_empty_list = ['actor_1_name', 'actor_2_name', 'director_name', 'country', 'actor_1_facebook_likes',
                           'actor_2_facebook_likes', 'director_facebook_likes', 'cast_total_facebook_likes', 'gross',
                           "imdb_score"]


def data_clean(path):
    """
    pre-processing
    :param path: string
    :return:
    """
    # read data csv using pandas
    read_data = pd.read_csv(path)
    # select columns in data
    select_data = read_data[all_selected_features]
    # delete data when value is empty
    data = select_data.dropna(axis=0, how='any', subset=eliminate_if_empty_list)
    # reset index
    data = data.reset_index(drop=True)
    for x in catagory_features:
        # fill NA/NaN values with None
        data[x] = data[x].fillna('None').astype('category')
    for y in number_features:
        # fill NA/NaN values with 0.0
        data[y] = data[y].fillna(0.0).astype(np.float)
    return data


def preprocessing_numerical_minmax(data):
    """
    min-max normalization
    :param data: numpy array of shape [n_samples, n_features]
    :return:
    """
    scaled_data = min_max_scaler.fit_transform(data)
    return scaled_data


def preprocessing_categorical(data):
    """
    labeling
    :param data: array-like of shape [n_samples]
    :return:
    """
    label_encoder = LabelEncoder()
    label_encoded_data = label_encoder.fit_transform(data)
    label_binarizer = preprocessing.LabelBinarizer()
    label_binarized_data = label_binarizer.fit_transform(label_encoded_data)
    return label_binarized_data


def regression_with_cross_validation(model, data, target, n_fold, model_name, pred_type):
    print(pred_type, " (Regression Model: ", model_name)
    cross_val_score_mean_abs_err = cross_val_score(model, data, np.ravel(target), scoring='neg_mean_absolute_error',
                                                   cv=n_fold)
    print("\nCross Validation Score (Mean Absolute Error)        : \n", -cross_val_score_mean_abs_err)
    print("\nCross Validation Score (Mean Absolute Error) (Mean) : \n", -cross_val_score_mean_abs_err.mean())
    cross_val_score_mean_sqr_err = cross_val_score(model, data, np.ravel(target), scoring='neg_mean_squared_error',
                                                   cv=n_fold)
    print("\nCross Validation Score (Mean Squared Error)         : \n", -cross_val_score_mean_sqr_err)
    print("\nCross Validation Score (Mean Squared Error)  (Mean) : \n", -cross_val_score_mean_sqr_err.mean())


def regression_scores(original_val, predicted_val, model_name):
    """
    calculate regression score
    :param original_val:
    :param predicted_val:
    :param model_name:
    :return:
    """
    print("Regression Model Name: ", model_name)
    mean_abs_error = mean_absolute_error(original_val, predicted_val)
    mean_sqr_error = mean_squared_error(original_val, predicted_val)
    median_abs_error = median_absolute_error(original_val, predicted_val)
    explained_var_score = explained_variance_score(original_val, predicted_val)
    r2__score = r2_score(original_val, predicted_val)

    print("\n")
    print("\nRegression Scores(train_test_split):\n")
    print("Mean Absolute Error    :", mean_abs_error)
    print("Mean Squared Error     :", mean_sqr_error)
    print("Median Absolute Error  :", median_abs_error)
    print("Explained Var Score    :", explained_var_score)
    print("R^2 Score              :", r2__score)
    print("\n\n")


def inverse_scaling(scaled_val):
    """
    undo the scaling of X according to feature_range
    :param scaled_val:
    :return:
    """
    unscaled_val = min_max_scaler.inverse_transform(scaled_val.reshape(-1, 1))
    return unscaled_val


def to_millions(value):
    """
    convert numeric to millions
    :param value: numeric
    :return:
    """
    return value / 10000000


def prediction_performance_plot(original_val, predicted_val, model_name, start, end, n, plot_type, prediction_type):
    """
    plotting actual vs predicted for all data
    :param original_val:
    :param predicted_val:
    :param model_name:
    :param start:
    :param end:
    :param n:
    :param plot_type:
    :param prediction_type:
    :return:
    """
    # inverse transform and convert to millions
    original_val = to_millions(inverse_scaling(original_val))
    predicted_val = to_millions(inverse_scaling(predicted_val))
    print("\n")
    plt.title(
        "\n" + prediction_type + " Prediction Performance using " + model_name + "(Actual VS Predicted)" + plot_type + "\n")
    if plot_type == "all":
        plt.plot(original_val, c='g', label="Actual")
        plt.plot(predicted_val, c='b', label="Prediction")
    if plot_type == "seq":
        plt.plot(original_val[start: end + 1], c='g', label="Actual")
        plt.plot(predicted_val[start: end + 1], c='b', label="Prediction")
    if plot_type == "random":
        original_val_list = []
        predicted_val_list = []
        for k in range(n):
            i = random.randint(0, len(predicted_val) - 1)
            original_val_list.append(original_val[i])
            predicted_val_list.append(predicted_val[i])
        plt.plot(original_val_list, c='g', label="Actual")
        plt.plot(predicted_val_list, c='b', label="Prediction")
    plt.legend(["Actual", "Predicted"], loc='center left', bbox_to_anchor=(1, 0.8))
    plt.ylabel('Prediction (In Millions)', fontsize=14)
    plt.grid()
    plt.show()


def print_original_vs_predicted(original_val, predicted_val, i, j, n, print_type, prediction_type):
    """
    printing actual vs predicted in a range
    :param original_val:
    :param predicted_val:
    :param i:
    :param j:
    :param n:
    :param print_type:
    :param prediction_type:
    :return:
    """
    # inverse transform and convert to millions
    original_val = to_millions(inverse_scaling(original_val))
    predicted_val = to_millions(inverse_scaling(predicted_val))

    print("\n" + prediction_type + " Comparision of Actual VS Predicted" + print_type + "\n")
    if print_type == "seq":
        if j < len(predicted_val):
            for k in range(i, j + 1):
                print("Actual" + prediction_type + " : ", original_val[k], ",   Predicted " + prediction_type, " : ",
                      predicted_val[k])
    if print_type == "random":
        for k in range(n):
            i = random.randint(0, len(predicted_val) - 1)
            print("Actual ", prediction_type, " : ", original_val[i], ",   Predicted " + prediction_type + " : ",
                  predicted_val[i])


def bar_plot_original_vs_predicted_rand(original_val, predicted_val, n, model_name, pred_type):
    """
    plotting actual vs predicted in a randomly using a bar chart
    :param original_val:
    :param predicted_val:
    :param n:
    :param model_name:
    :param pred_type:
    :return:
    """
    # inverse transform and convert to millions
    original_val = to_millions(inverse_scaling(original_val))
    predicted_val = to_millions(inverse_scaling(predicted_val))
    original_val_list = []
    predicted_val_list = []
    for k in range(n):
        i = random.randint(0, len(predicted_val) - 1)
        original_val_list.append(original_val[i])
        predicted_val_list.append(predicted_val[i])

    original_val_df = pd.DataFrame(original_val_list)
    predicted_val_df = pd.DataFrame(predicted_val_list)

    actual_vs_predicted = pd.concat([original_val_df, predicted_val_df], axis=1)

    actual_vs_predicted.plot(kind="bar", fontsize=12, color=['g', 'b'], width=0.7)
    plt.title(
        "\nUsing Categorical and Numerical Features\n" + model_name + " : Actual " + pred_type + "VS Predicted " + pred_type + "(Random)")
    plt.ylabel('Gross (In Millions)', fontsize=14)
    plt.ylabel('Gross (In M', fontsize=14)
    plt.xticks([])
    plt.legend(["Actual ", "Predicted"], loc='center left', bbox_to_anchor=(1, 0.8))
    plt.grid()
    plt.show()


def preprocessing_catagory(data):
    """
    labeling category
    :param data:
    :return:
    """
    data_c = 0
    for i in range(len(catagory_features)):
        new_data = data[catagory_features[i]]
        new_data_c = preprocessing_categorical(new_data)
        if i == 0:
            data_c = new_data_c
        else:
            data_c = np.append(data_c, new_data_c, 1)
    return data_c


def preprocessing_numerical(data):
    """
    min-max normalization
    :param data:
    :return:
    """
    data_list_numerical = list(zip(data['director_facebook_likes'], data['actor_1_facebook_likes'],
                                   data['actor_2_facebook_likes'], data['actor_3_facebook_likes'],
                                   data['cast_total_facebook_likes'], data['budget']))

    data_numerical = np.array(data_list_numerical)
    data_numerical = preprocessing_numerical_minmax(data_numerical)
    return data_numerical


def preprocessed_agregated_data(database):
    """
    aggregate data
    :param database:
    :return:
    """
    numerical_data = preprocessing_numerical(database)
    categorical_data = preprocessing_catagory(database)
    all_data = np.append(numerical_data, categorical_data, 1)
    return all_data


def regression_without_cross_validation(model, train_data, train_target, test_data):
    """
    regression model training
    :param model:
    :param train_data:
    :param train_target:
    :param test_data:
    :return:
    """
    model.fit(train_data, train_target.ravel())
    prediction = model.predict(test_data)
    return prediction


def regr_without_cross_validation_train_test_perform_plot(model, data, target, model_name, pred_type):
    """
    regression without cross validation
    :param model:
    :param data:
    :param target:
    :param model_name:
    :param pred_type:
    :return:
    """
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.3, random_state=0)
    predicted_gross = regression_without_cross_validation(model, train_data, train_target, test_data)
    regression_scores(test_target, predicted_gross, model_name)
    prediction_performance_plot(test_target, predicted_gross, model_name, 200, 250, 0, "seq", pred_type)
    prediction_performance_plot(test_target, predicted_gross, model_name, 0, 0, 100, "random", pred_type)
    print_original_vs_predicted(test_target, predicted_gross, 0, 0, 10, "random", pred_type)
    bar_plot_original_vs_predicted_rand(test_target, predicted_gross, 20, model_name, pred_type)


path = "movie_metadata.csv"
data = data_clean(path)
target_gross = data['gross']
target_imdb_score = data['imdb_score']
database = data.drop('gross', 1)
database.info()
target_gross = preprocessing_numerical_minmax(target_gross.values.reshape(-1, 1))
preprocessed_data = preprocessed_agregated_data(database)
print("feature calculation complete\n")

regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)

regression_with_cross_validation(
    regr,
    preprocessed_data,
    target_gross,
    5,
    "Random Forest Regression",
    "(Movie Gross Prediction)")

regr_without_cross_validation_train_test_perform_plot(
    regr,
    preprocessed_data,
    target_gross,
    "Random Forest Regression",
    "(Movie Gross Prediction)")
