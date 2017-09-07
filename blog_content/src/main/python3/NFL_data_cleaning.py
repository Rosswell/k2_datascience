# general necessities
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# cleaning the data
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# linear classifier models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# nonlinear classification models
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

# ensemble classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# Grid Search for evaluating multiple parameters of the models while testing them
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# misc
from collections import OrderedDict

# web scraping
from bs4 import BeautifulSoup as BS
import requests
import csv
import sqlite3
import calendar


import pandas as pd
import sqlite3
import numpy as np
import math


class dataInitializer():
    def __init__(self):
        conn = sqlite3.connect('/Users/ross.blanchard/PycharmProjects/k2_datascience/EDA_Project/nfl_db.sqlite')
        self.cursor = conn.cursor()

    def close_conn(self):
        self.cursor.close()
        print("DB Connection closed.")

    def reopen_conn(self):
        conn = sqlite3.connect('nfl_db.sqlite')
        self.cursor = conn.cursor()
        print("DB Connection reopened.")

    def create_prediction_table(self, source_table, destination_table):
        """
        Creates a table if it doesn't already exist for training a model. not very dynamic right now
        :param source_table: source table name, string
        :param destination_table: prediction table name, string
        :return: void
        """
        source_query = "SELECT * FROM " + source_table
        self.cursor.execute(source_query)
        source_df = pd.DataFrame(self.cursor.fetchall())
        source_df.columns = [desc[0] for desc in self.cursor.description]

        # standard_col_names = ['id'] + ['team_id'] + ['opponent_id']
        # standard_data_types = [' integer'] +

        source_col_names = ['id'] + list(source_df.columns) + ['season']
        source_data_types = [' integer '] + [str(source_df[col].dtype).replace('object', ' text ').replace('int64', ' integer ').replace('float64', ' real ') for
         col in source_df.columns] + [' integer ']
        col_constraint = ['PRIMARY KEY AUTOINCREMENT NOT NULL, '] + (['NOT NULL, '] * len(source_col_names))
        column_string = ' ('
        for col_name, data_type, constraint in zip(source_col_names, source_data_types, col_constraint):
            column_string += col_name + data_type + constraint
        foreign_key = 'FOREIGN KEY(team_id) REFERENCES team_map(team_id), FOREIGN KEY(opponent_id) REFERENCES team_map(opponent_id)'
        destination_query = "CREATE TABLE IF NOT EXISTS " + destination_table + column_string
        self.cursor.execute(destination_query)
        print(destination_table + " table created successfully.")

    def fill_prediction_table(self, table):
        source_query = """select date, result, attempts, completions, percent, yards, yards_per_attempt, touchdowns, 
                          interceptions, longest_completion, sacks, qb_rating, player_name, team.team_id as team_id, 
                          opp.team_id 
                          as opp_id from """ + table + """ join team_map team on team.team=qbs.team
                          join team_map opp on opp.team=qbs.opponent"""
        self.cursor.execute(source_query)
        source_df = pd.DataFrame(self.cursor.fetchall())
        source_df.columns = [desc[0] for desc in self.cursor.description]
        source_df = self.convert_cols(source_df, table)

        col_names = list(source_df.columns)
        source_data_types = [str(source_df[col].dtype) for col in source_df.columns]
        for d_type in source_data_types:
            if d_type == 'object':
                source_df.drop(col_names[source_data_types.index(d_type)], 1, inplace=True)
        # split the data into data to aggregate and data to drop, add it to the predictions table, and use that for
        # models

    def fetchData(self, position):
        """
        Fetches a full table from the database, adjusts the columns into a ML model friendly format, and returns the df
        :param position: position to return the data from as a string
        :return: df of numeric data
        """
        query = "SELECT * FROM " + position
        self.cursor.execute(query)
        query_df = pd.DataFrame(self.cursor.fetchall())
        query_df.columns = [desc[0] for desc in self.cursor.description]
        # query_df = self.convertCols(query_df, position)
        return query_df

    def fetchTeams(self, df):
        """
        Takes a dataframe and maps it to the id, then returns the result for each team and opponent entry
        :param df: dataframe to get the teams from
        :return: a two column dataframe that contains the teams and opponents for each game
        """
        try:
            teamSeries = df['team_id']
            try:
                opponentSeries = df['opponent_id']
            except:
                return "DataFrame does not have an 'opponent_id' column to convert"
        except:
            return "DataFrame does not have a 'team_id' column to convert"

        teamMapQuery = "SELECT * FROM team_map"
        self.cursor.execute(teamMapQuery)
        teamMapDf = pd.DataFrame(self.cursor.fetchall())
        teamMapDf.columns = [desc[0] for desc in self.cursor.description]

        df = df.merge(teamMapDf, on='team_id', how='left')
        df = df.merge(teamMapDf, on='opponent_id', how='left')
        df['team'] = df['team_x']
        df['opponent'] = df['team_y']
        df.drop(['opponent_id', 'team_id', 'team_x', 'team_y'], 1, inplace=True)

        return df

    def fetchPlayers(self, df):
        """
        Takes a dataframe and maps it to the id, then returns the result for each team and opponent entry
        :param df: dataframe to get the teams from
        :return: a two column dataframe that contains the teams and opponents for each game
        """
        try:
            teamSeries = df['team_id']
            try:
                opponentSeries = df['opponent_id']
            except:
                return "DataFrame does not have an 'opponent_id' column to convert"
        except:
            return "DataFrame does not have a 'team_id' column to convert"

        playerMapQuery = "SELECT * FROM player_map"
        self.cursor.execute(playerMapQuery)
        playerMapDf = pd.DataFrame(self.cursor.fetchall())
        playerMapDf.columns = [desc[0] for desc in self.cursor.description]

        df = df.merge(playerMapDf, on='player_name', how='left')
        df.drop('player_id', 1, inplace=True)
        return df

    def cleanData(self, df, table_str):

        def dropCols(df):
            if 'sacks' in df.columns:
                df.drop('sacks', 1, inplace=True)
            elif 'id' in df.columns:
                df.drop('id', 1, inplace=True)
            df.drop(['team_x', 'team_y', 'result',
                     'opponent', 'team_id_y', 'team_id_x',
                     'date', 'player_name', 'team_id2_y', 'team_id2_x'], 1, inplace=True)
            return df

        def mapPlayersAndTeams(df):

            def capitalizeTeams(x):
                splitStr = x.split(' ')
                if len(splitStr) > 1:
                    splitStr[1] = splitStr[1].capitalize()
                else: return splitStr[0].capitalize()
                return ' '.join(splitStr)
            # team mapping
            teamMapQuery = "SELECT * FROM team_map"
            self.cursor.execute(teamMapQuery)
            teamMapDf = pd.DataFrame(self.cursor.fetchall())
            teamMapDf.columns = [desc[0] for desc in self.cursor.description]
            df['team'] = df['team'].apply(lambda x: capitalizeTeams(x))
            df['opponent'] = df['opponent'].apply(lambda x: capitalizeTeams(x))


            df = df.merge(teamMapDf, on='team', how='left')

            teamMapDf['opponent'] = teamMapDf['team']
            df = df.merge(teamMapDf, on='opponent', how='left')
            df['team_id'] = df['team_id_x']
            df['opponent_id'] = df['team_id_y']

            # player mapping
            playerMapQuery = "SELECT * FROM player_map"
            self.cursor.execute(playerMapQuery)
            playerMapDf = pd.DataFrame(self.cursor.fetchall())
            playerMapDf.columns = [desc[0] for desc in self.cursor.description]
            df = df.merge(playerMapDf, on='player_name', how='left')

            return df

        def longest_fix(x):
            if isinstance(x, str):
                return int(x[:-1])
            return int(x)

        def date_fix(series):
            laborDayDict = {
                2006: np.datetime64('2006-09-06'),
                2007: np.datetime64('2007-09-05'),
                2008: np.datetime64('2008-09-03'),
                2009: np.datetime64('2009-09-09'),
                2010: np.datetime64('2010-09-08'),
                2011: np.datetime64('2011-09-07'),
                2012: np.datetime64('2012-09-05'),
                2013: np.datetime64('2013-09-04'),
                2014: np.datetime64('2014-09-03'),
                2015: np.datetime64('2015-09-09'),
                2016: np.datetime64('2016-09-07'),
                2017: np.datetime64('2017-09-06')
            }
            weekList = []
            yearList = []
            for x in series.values:
                year = x.astype('datetime64[Y]').astype(int) + 1970
                month = x.astype('datetime64[M]').astype(int) % 12 + 1
                month -= 8
                if month < 0:
                    year -= 1
                    month += 12
                week = math.ceil((x - laborDayDict[year]) / np.timedelta64(1, 'W'))
                weekList.append(week)
                yearList.append(year)
            return pd.Series(weekList), pd.Series(yearList)

        def winloss_fix(x):
            return int(x.split(',')[0] == 'W')

        def score_fix(x):
            win_loss = x.split(',')[0]
            if win_loss == 'W':
                return int(x.split(',')[1].split('-')[0])
            return int(x.split(',')[1].split('-')[1])

        def sack_fix(series):
            sack_number_list = []
            sack_length_list = []
            for x in series.values:
                sack_number, sack_length = x.split('/')
                sack_number_list.append(int(sack_number))
                sack_length_list.append(int(sack_length))
            return pd.Series(sack_number_list), pd.Series(sack_length_list)

        def tar_yac_fix(x):
            if x == '--':
                return 0
            return int(x)

        df['date'] = df['date'].apply(lambda x: '20' + x[:2] + '/' + x[2:4] + '/' + x[4:])
        df['date'] = pd.to_datetime(df.date, format="%Y/%m/%d")
        df['score'] = df.result.apply(lambda x: score_fix(x))
        df['win_loss'] = df.result.apply(lambda x: winloss_fix(x))
        df['week'], df['season'] = date_fix(df['date'])

        if table_str == 'qbs':
            df['longest_completion'] = df.longest_completion.apply(lambda x: longest_fix(x))
            df['times_sacked'], df['sack_length'] = sack_fix(df['sacks'])
            df = df.drop('sacks', 1)

        if table_str == 'rbs':
            df['reception_long'] = df['reception_long'].apply(lambda x: longest_fix(x))
            df['rushing_long'] = df['rushing_long'].apply(lambda x: longest_fix(x))
            df['targets'] = df['targets'].apply(lambda x: tar_yac_fix(x))
            df['yards_after_catch'] = df['yards_after_catch'].apply(lambda x: tar_yac_fix(x))
            df = df.drop('id', 1)

        def dropPreseason(df):
            return df[df['week'] > 0]

        df = mapPlayersAndTeams(df)
        df = dropPreseason(df)
        df = dropCols(df)

        return df

    def aggregateData(self, df):

        name_list = list(df.player_id.unique())
        temp_df = df[:]
        for name in name_list:

            playerData = temp_df[temp_df['player_id'] == name][:].sort_values(['season', 'week'])
            nonAggCols = playerData[['week', 'season', 'team_id', 'opponent_id', 'player_id', 'win_loss']]
            aggCols = playerData.drop(['week', 'season', 'team_id', 'opponent_id', 'player_id'], 1)
            aggCols['agg_win_loss'] = aggCols['win_loss']
            aggCols.drop('win_loss', 1, inplace=True)
            aggDummyRow = pd.DataFrame({key: 0 for key in list(aggCols.columns)}, index=[0])
            aggColsWithDummyData = pd.concat([aggDummyRow, aggCols], axis=0, ignore_index=True)
            aggColsAggApplied = np.cumsum(aggColsWithDummyData, axis=0).reset_index(drop=True)
            aggPlayerData = pd.concat([aggColsAggApplied, nonAggCols.reset_index(drop=True)], axis=1,
                             join_axes=[aggColsAggApplied.index]).iloc[:-1]
            # creating dataframe for the firtst qb, otherwise concatting to existing data
            if name_list.index(name) != 0:
                trainingData = pd.concat([trainingData, aggPlayerData], axis=0, ignore_index=True)
            else:
                trainingData = aggPlayerData
            # trainingData.drop(['team_id2_y', 'team_id2_x'], 1, inplace=True)
        return trainingData

# ndc = dataInitializer()
# df = ndc.fetchData('qbs')
# cum_data = ndc.aggregateData(df)
# print(cum_data.head())

class mlOptimizer():
    def __init__(self, df, y_col, global_random_state=0):
        self.X = df.drop(y_col, 1)
        self.y = df[y_col]
        feature_number = len(df.columns) - 1
        self.global_random_state = global_random_state
        self.data = OrderedDict([('Model', []),
                                 ('Training Score', []),
                                 ('Test Score', []),
                                 ('Fit Time', []),
                                 ('Best Model Params', []),
                                 ('Best Model Instance', [])])
        self.classification_dict = dict(LogisticRegression={
            'name': 'Logistic Regression',
            'model': LogisticRegression(),
            'params': {
                'model__warm_start': [True, False],
                'model__C': [0.01, 0.1, 1, 10, 100]
            }
        }, LDA={
            'name': 'LDA',
            'model': LDA(),
            'params': {
                'model__solver': ['svd', 'lsqr', 'eigen']
            }
        }, GaussianNB={
            'name': 'Gaussian Naive Bayes',
            'model': GaussianNB(),
            'params': {}
        }, MultinomialNB={
            'name': 'Multinomial Naive Bayes',
            'model': MultinomialNB(),
            'params': {}
        }, QDA={
            'name': 'QDA',
            'model': QDA(),
            'params': {}
        }, DecisionTreeClassifier={
            'name': 'Decision Tree',
            'model': DecisionTreeClassifier(),
            'params': {
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [2, 5, 10]
            }
        }, SVC={
            'name': 'RBF SVC',
            'model': SVC(),
            'params': {
                'model__C': [0.1, 1, 10],
                'model__degree': [2, 3, 4, 5],
                'model__gamma': [0.1, 1, 10],
                'model__kernel': ['rbf']
            }
        }, KNeighborsClassifier={
            'name': 'K Neighbors Classifier',
            'model': KNeighborsClassifier(),
            'params': {
                'model__n_neighbors': list(range(1, feature_number + 1))
            }
        }, RandomForest={
            'name': 'Random Forest',
            'model': RandomForestClassifier(),
            'params': {
                "model__max_depth": [3, None],
                "model__max_features": list(range(1, feature_number, 2)),
                "model__min_samples_split": [5, 10, 15],
                "model__min_samples_leaf": list(range(1, feature_number, 2)),
                "model__bootstrap": [True, False],
                "model__criterion": ["gini", "entropy"]
            }
        }, GradientBoostingClassifier={
            'name': 'Gradient Boosting Classifier',
            'model': GradientBoostingClassifier(),
            'params': {
                'model__loss': ['deviance', 'exponential'],
                'model__n_estimators': [50, 100, 150],
                'model__min_samples_leaf': [5, 10, 15],
                'model__min_samples_split': [5, 10, 15],
                'model__max_features': [feature_number, feature_number - 1, feature_number - 2],
                'model__subsample': [0.4, 0.5, 0.6, 0.7, 0.8]
            }
        }, AdaBoostClassifier={
            'name': 'Ada Boost Classifier',
            'model': AdaBoostClassifier(),
            'params': {
                'model__n_estimators': list(range(100, 300, 50)),
                'model__learning_rate': list(np.arange(0.7, 1.1, 0.1))
            }
        })

    def create_data(self, df, y_col, test_percentage, bootstrap=False):
        """
        Splits a dataframe into x and y columns, and train and test sets
        :param bootstrap: whether to use bootstrap or not. default=False
        :param df: dataframe to split
        :param y_col: columns of Y values, string
        :param test_percentage: percentage of the data to use for testing
        :return: list of X_train, y_train, X_test, y_test
        """

        # maybe put in a check for consistent values in here in the future, but not right now, I'm tired
        x = df.drop(y_col, 1)
        y = df[y_col]

        return train_test_split(x, y, test_size=test_percentage,
                                random_state=self.global_random_state)
        # return data_suite

    def import_model(self, clf, clf_name):
        """
        Imports the model parameters to the classification dictionary
        :param clf: classification model
        :param clf_name: model name
        """

        if 'pca' in clf.best_estimator_.named_steps:
            clf_name = str(clf.best_estimator_.named_steps['pca'].n_components_) + " Component PCA + " + clf_name
        clf_df = pd.DataFrame(clf.cv_results_).sort_values('rank_test_score')
        test_score = clf_df['mean_test_score'].values[0]
        train_score = clf_df['mean_train_score'].values[0]
        fit_time = clf_df['mean_fit_time'].values[0]
        self.data['Model'].append(clf_name)
        self.data['Test Score'].append(test_score)
        self.data['Training Score'].append(train_score)
        self.data['Fit Time'].append(fit_time)
        self.data['Best Model Params'].append(clf.best_estimator_.named_steps['model'].get_params())
        self.data['Best Model Instance'].append(clf.best_estimator_)

    def view(self):
        return pd.DataFrame(self.data).sort_values('Test Score', ascending=False)

    def clear(self):
        self.data = OrderedDict([('Model', []),
                                 ('Training Score', []),
                                 ('Test Score', []),
                                 ('Fit Time', []),
                                 ('Best Model Params', [])])
        print("Ranking Cleared.")

    def plot(self):
        """
        Creates and displays a plot of the Classification Models' Test and Training Accuracies, as well as Fit Time.
        """

        index = [v for v in self.data['Model']]
        plotting_df = pd.DataFrame(self.data, index=index).sort_values('Test Score', ascending=False)
        #         mms = MinMaxScaler()
        #         plotting_df['Normalized Fit Time'] = mms.fit_transform(plotting_df['Fit Time'].values.reshape(-1, 1))
        #         plotting_df.drop(['Model', 'Fit Time'], 1, inplace=True)

        # plotting the data
        ax = plotting_df[['Test Score', 'Training Score']].plot(figsize=(15, 8),
                                                                fontsize=15, cmap='coolwarm', rot=90)
        # setting the text value parameters
        text_x = [-0.20, 0, 0.15]
        model_num = 0
        for y_values in plotting_df[['Test Score', 'Training Score']].values:
            for index, int_val in enumerate(y_values):
                # y_position = int_val - 0.02
                # if int_val < 0.1:
                y_position = int_val + 0.00
                ax.text(text_x[index] + model_num, y_position, str(round(int_val, 4)), color='black', fontweight='bold',
                        rotation=90)
            model_num += 1

        plt.title('Classification Model Comparison', size=20)
        plt.ylabel('Accuracy', size=15)
        plt.show()

    def optimizeModel(self, model, pca=False):
        """
        Runs all the linear or nonlinear models, with PCA preprocessing on or off, and writes them to the
        best classification model dictionary
        :param model: classification model to use as sklearn method, string
        :param pca: whether to use pca or not, boolean
        """

        model_name = self.classification_dict[model]['name']
        model_instance = self.classification_dict[model]['model']
        model_params = self.classification_dict[model]['params'].copy()
        pipeline_steps = [('model', model_instance)]

        if pca:
            pca_clf = PCA()
            pipeline_steps.insert(0, ('pca', pca_clf))
            model_params['pca__n_components'] = [3, 4, 5, 6]
        pipe = Pipeline(steps=pipeline_steps)
        grid = GridSearchCV(pipe, model_params) #, verbose=1)
        grid.fit(self.X, self.y)
        self.import_model(grid, model_name)

    def fetchGames(self, week, year):

        playerIdInt = 4000
        def getIds(key, team=True, away=False, playerIdInt=4000):
            conn = sqlite3.connect('/Users/ross.blanchard/PycharmProjects/k2_datascience/EDA_Project/nfl_db.sqlite')
            cursor = conn.cursor()
            if team:
                teamStr = key.capitalize()
                query = "SELECT team_id2 FROM team_map WHERE team='" + teamStr + "'"
                if away:
                    teamStr = '@ ' + teamStr
                    query = "SELECT team_id2 FROM team_map WHERE team='" + teamStr + "'"
                cursor.execute(query)
                team_id = cursor.fetchall()[0][0]
                conn.close()
                return team_id
            if isinstance(key, list):
                playerStr = [item.lower() for item in key]
                query = "SELECT player_id FROM player_map WHERE (player_name='" + playerStr[0] + "'" + \
                        ") OR (player_name='" + playerStr[1] + "')"
                cursor.execute(query)
                playerList = [item[0] for item in list(cursor.fetchall())]
                if len(playerList) < 2:
                    for nameStr in playerStr:
                        nameQuery = "SELECT player_id FROM player_map WHERE player_name='" + nameStr + "'"
                        cursor.execute(nameQuery)
                        results = list(cursor.fetchall())
                        if len(results) != 1:
                            createEntryQuery = "INSERT INTO player_map (player_name, player_id) values(\'{}\',\'{}\')".format(nameStr, playerIdInt)
                            playerIdInt += 1
                            cursor.execute(createEntryQuery)
                cursor.execute(query)
                playerList = [item[0] for item in list(cursor.fetchall())]
                conn.close()
                return playerList

        url = 'http://www.espn.com/nfl/schedule/_/' + 'year/' + year + '/_/' + 'week/' + week
        csvFilePath = '/Users/ross.blanchard/PycharmProjects/k2_datascience/blog_content/espnWeekData.csv'
        save = 'n'  # input('save? y/n\n')
        if save == 'y':
            page = requests.get(url)
            soup = BS(page.content, 'html.parser')
            htmlData = soup.find_all(id="sched-container")
            with open(csvFilePath, 'w') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(htmlData)
        else:
            with open(csvFilePath, 'r') as csvFile:
                reader = csv.reader(csvFile)
                htmlData = list(reader)[0][0]

        # getting game dates and home and away teams for the week
        awayTeams = []
        homeTeams = []
        dates = []
        startingQbs = []
        soup = BS(htmlData, 'lxml')
        dateList = soup.find_all(class_="table-caption")
        teamTableList = soup.find_all(class_='responsive-table-wrap')
        if len(teamTableList) > len(dateList):
            teamTableList[-2:-1] = teamTableList[-1:]
            del teamTableList[-1:]
        for dateIndex in range(len(dateList)):
            teamsPlayingOnDate = list(teamTableList[dateIndex].strings)[6:]
            totalTeamCount = int(len(teamsPlayingOnDate) / 12)
            for eachTeamIndex in range(totalTeamCount):
                awayTeamIndex = 2 + (eachTeamIndex * 12)
                homeTeamIndex = 5 + (eachTeamIndex * 12)
                awayTeams.append(teamsPlayingOnDate[awayTeamIndex])
                homeTeams.append(teamsPlayingOnDate[homeTeamIndex])
                dates.append(dateList[dateIndex].text)
        # getting starting QBs
        headerIndex = 0
        headerList = ['Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; Acoo Browser 1.98.744; .NET CLR 3.5.30729)',
                'Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; Acoo Browser 1.98.744; .NET CLR 3.5.30729)',
                'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; Acoo Browser; GTB5; Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1) ; InfoPath.1; .NET CLR 3.5.30729; .NET CLR 3.0.30618)',
                'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0; SV1; Acoo Browser; .NET CLR 2.0.50727; .NET CLR 3.0.4506.2152; .NET CLR 3.5.30729; Avant Browser)',
                'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)',
                'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
                'Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)',
                'Mozilla/5.0 (compatible; Yahoo! Slurp; http://help.yahoo.com/help/us/ysearch/slurp)',
                'Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246',
                'Mozilla/5.0 (Linux; Android 7.0; Pixel C Build/NRD90M; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/52.0.2743.98 Safari/537.36']
        for awayTeam, homeTeam in zip(awayTeams, homeTeams):

            for team in [awayTeam, homeTeam]:
                headers = {'user-agent': headerList[headerIndex]}
                headerIndex += 1
                if headerIndex > len(headers):
                    headerIndex = 0
                url = 'http://www.espn.com/nfl/team/depth/_/name/' + team
                depthChartHtml = requests.get(url, headers=headers)
                qbSoup = BS(depthChartHtml.content, 'lxml')
                for tag in qbSoup.find(class_='tablehead').contents:
                    if tag.td.text == 'QB':
                        startingQbs.append(tag.a.text)
        returnDict = {
            'team_id': [],
            'opponent_id': [],
            'player_id': [],
            'season': [],
            'week': []
        }
        for gameIndex in range(len(awayTeams)):
            team_id = getIds(awayTeams[gameIndex], True, False, playerIdInt)
            opponent_id = getIds(homeTeams[gameIndex], True, True, playerIdInt)
            player_id1, player_id2 = getIds([startingQbs[gameIndex * 2], startingQbs[gameIndex * 2 + 1]], False,
                                            False, playerIdInt)
            playerIdInt += 2
            returnDict['team_id'].extend([team_id, opponent_id])
            returnDict['opponent_id'].extend([opponent_id, team_id])
            returnDict['player_id'].extend([player_id1, player_id2])
            returnDict['season'].extend([year] * 2)
            returnDict['week'].extend([week] * 2)
        return pd.DataFrame(returnDict, index=list(range(len(returnDict['team_id']))))

    def predictWeek(self, knownPredictionValsDf, trainingDfX, trainingDfY):
        predictionData = pd.DataFrame(columns=list(trainingDfX.columns))
        for playerId in knownPredictionValsDf['player_id'].values:
            if int(playerId) in trainingDfX['player_id']:
                predictionData = predictionData.append(trainingDfX[trainingDfX['player_id'] == int(playerId)].sort_values(['season', 'week']).iloc[-1:])
            else:
                predictionData = predictionData.append(pd.DataFrame(np.zeros((1,len(trainingDfX.columns))), columns=list(trainingDfX.columns)))
        knownPredictionValsDf.index = predictionData.index
        predictionData.drop(list(knownPredictionValsDf.columns), 1, inplace=True)
        for col in list(knownPredictionValsDf.columns):
            predictionData[col] = knownPredictionValsDf[col]
        # predictionData.drop('win_loss', 1, inplace=True)

        models = self.view()['Best Model Instance'][:5]
        modelNames = self.view()['Model'][:5]
        conn = sqlite3.connect('/Users/ross.blanchard/PycharmProjects/k2_datascience/EDA_Project/nfl_db.sqlite')
        cursor = conn.cursor()
        teamPrintList = []
        for teamId in predictionData['team_id'].values:
            query = "select team from team_map where team_id='{}'".format(teamId)
            cursor.execute(query)
            team = cursor.fetchall()[0][0]
            if len(team.split(' ')) > 1:
                teamPrintList.append(team.split(' ')[1].upper())
            else:
                teamPrintList.append(team.upper())
        conn.close()
        predictionDisplayDf = pd.DataFrame(index = teamPrintList)
        for model, modelName in zip(models, modelNames):
            if isinstance(model, list):
                pipe = Pipeline(model[0], model[1])
                pipe.fit(trainingDfX, trainingDfY)
                prediction = pipe.predict(predictionData)
            else:
                clf = model
                clf.fit(trainingDfX, trainingDfY)
                prediction = clf.predict(predictionData)

            tempSeries = pd.Series(prediction, index=teamPrintList, name=modelName)
            predictionDisplayDf = pd.concat([predictionDisplayDf, tempSeries], axis=1)
        predictionDisplayDf['Aggregate Prediction'] = predictionDisplayDf.mean(axis=1)
        plottingDf = predictionDisplayDf.copy()
        for col in predictionDisplayDf.columns:
            predictionDisplayDf[col] = np.where(predictionDisplayDf[col] > 0.5, "Win", "Loss")
        return predictionDisplayDf, plottingDf

# test cases
ndc = dataInitializer()
df = ndc.fetchData('qbs')
df = ndc.cleanData(df, 'qbs')
cum_data = ndc.aggregateData(df)
new = mlOptimizer(cum_data, 'win_loss')
new.optimizeModel('LogisticRegression')
# new.optimize_model('RandomForestClassifier')
new.optimizeModel('LDA', True)
new.optimizeModel('LDA', False)
# new.optimize_model('AdaBoostClassifier', False)
newgames = new.fetchGames('1', '2017')
x_train = cum_data.drop('win_loss', 1)
y_train = cum_data['win_loss']
new.predictWeek(newgames, x_train, y_train)
# print(df.head())