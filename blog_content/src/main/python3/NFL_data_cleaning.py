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
import seaborn as sns

# web scraping
from bs4 import BeautifulSoup as BS
import requests
import sqlite3
import math


class dataInitializer:
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
                else:
                    return splitStr[0].capitalize()
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
            # creating dataframe for the first qb, otherwise concatting to existing data
            if name_list.index(name) != 0:
                trainingData = pd.concat([trainingData, aggPlayerData], axis=0, ignore_index=True)
            else:
                trainingData = aggPlayerData
        return trainingData


class mlOptimizer:
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

    def connectToDb(self):
        conn = sqlite3.connect('/Users/ross.blanchard/PycharmProjects/k2_datascience/EDA_Project/nfl_db.sqlite')
        return conn.cursor()

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

    def plotPredictions(self, weekGameData, aggData, weekTitles):
        """
        Creates and displays a plot of the Classification Models' Test and Training Accuracies, as well as Fit Time.
        """
        x_train = aggData.drop('win_loss', 1)
        y_train = aggData['win_loss']
        confidences, predictions = optimizer.predictWeek(weekGameData, x_train, y_train)
        plottingValues, plottingAnnots = self.combineGames(confidences, predictions)

        plt.figure(figsize=(15, 15))

        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 10), gridspec_kw={'height_ratios': [.3, 3, .3]})
        plotTitle = f.suptitle('NFL Week 2 Game Predictions', fontsize=14)
        rowStart = 0
        rowEndList = np.cumsum(weekTitles['iterations'].values)

        for title, rowEnd, axis in zip(weekTitles['date'].values, rowEndList, [ax1, ax2, ax3]):
            sns.heatmap(plottingValues.iloc[rowStart:rowEnd],
                        annot=plottingAnnots.iloc[rowStart:rowEnd],
                        linewidths=.5, cmap='RdYlGn', cbar=False, fmt="s",
                        ax=axis, vmin=0.0, vmax=1.0)
            axis.set_title(title)
            rowStart = rowEnd

        for ax in [ax1, ax2, ax3]:
            tl = ax.get_xticklabels()
            ax.set_xticklabels(tl, rotation=60, size=10)
            tly = ax.get_yticklabels()
            ax.set_yticklabels(tly, rotation=0)

        plt.tight_layout()
        plotTitle.set_y(0.95)
        plotTitle.set_x(0.55)
        f.subplots_adjust(top=0.90)
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
        grid = GridSearchCV(pipe, model_params, verbose=1)
        grid.fit(self.X, self.y)
        self.import_model(grid, model_name)

    def getIds(self, key, c, newPlayerId=None, team=True, away=False):

        ## NOTE: should just be able to pass all teams/players and return a list of ids instead of this malarky
        if team:
            teamStr = key.capitalize()
            query = "SELECT team_id2 FROM team_map WHERE team='" + teamStr + "'"
            if away:
                teamStr = '@ ' + teamStr
                query = "SELECT team_id2 FROM team_map WHERE team='" + teamStr + "'"
            c.execute(query)
            team_id = c.fetchall()[0][0]
            return team_id

        # returning a player_id from a player_name, type checking is probably unnecessary since the other case already returns
        if isinstance(key, list):
            playerStr = [item.lower() for item in key]
            query = "SELECT player_id FROM player_map WHERE (player_name='" + playerStr[0] + "'" + \
                    ") OR (player_name='" + playerStr[1] + "')"
            c.execute(query)
            playerList = [item[0] for item in list(c.fetchall())]
            if len(playerList) < 2:
                for nameStr in playerStr:
                    nameQuery = "SELECT player_id FROM player_map WHERE player_name='" + nameStr + "'"
                    c.execute(nameQuery)
                    results = list(c.fetchall())
                    if len(results) != 1:
                        createEntryQuery = "INSERT INTO player_map (player_name, player_id) values(\'{}\',\'{}\')".format(
                            nameStr, newPlayerId)
                        newPlayerId += 1
                        c.execute(createEntryQuery)
            c.execute(query)
            playerList = [item[0] for item in list(c.fetchall())]
            return playerList

    def parseDatesAndTeams(self, soup, results):

        awayTeams = []
        homeTeams = []
        dates = []
        startingQbs = []
        teamCountOnDate = []
        resultsMap = {}

        dateList = soup.find_all(class_="table-caption")
        teamTables = soup.find_all(class_='responsive-table-wrap')

        # attempting to anticipate bye weeks
        if len(teamTables) > len(dateList):
            teamTables[-2:-1] = teamTables[-1:]
            del teamTables[-1:]

        # getting home and away teams, sorted by order of appearance
        for dateIndex in range(len(dateList)):
            teamList = teamTables[dateIndex].find_all(class_="team-name")
            teamCountOnDate.append(int(len(teamList) / 2))
            dates.append(dateList[dateIndex].text)

            for table in teamList:
                if teamList.index(table) % 2 == 0:
                    homeTeams.append(table.abbr.string)
                else:
                    awayTeams.append(table.abbr.string)

        # parses and returns starting QBs if looking for future games, and results if looking for past games
        if results:
            for awayTeam, homeTeam in zip(awayTeams, homeTeams):
                for team in [awayTeam, homeTeam]:
                    url = 'http://www.espn.com/nfl/team/depth/_/name/' + team
                    depthChartHtml = requests.get(url)
                    qbSoup = BS(depthChartHtml.content, 'lxml')
                    for tag in qbSoup.find(class_='tablehead').contents:
                        if tag.td.text == 'QB':
                            startingQbs.append(tag.a.text)
            return [awayTeams, teamCountOnDate, homeTeams, dates, startingQbs]

        else:
            for table in teamTables:
                gameResults = table.find_all(class_='home')
                for game in gameResults:
                    gameResultStrings = game.nextSibling.string.replace(',', '').split(' ')
                    resultsMap[gameResultStrings[0]] = gameResults[1]
            return [awayTeams, teamCountOnDate, homeTeams, dates, resultsMap]

    def fetchFutureGames(self, week, year):

        # this dict is for the teams and the QBs playing
        returnDataDict = {
            'awayTeamId': [],
            'homeTeamId': [],
            'homePlayerId': [],
            'awayPlayerId': [],
            'season': [],
            'week': []
        }

        # this dict is for the date titles for future plot visualization
        returnTitleDict = {
            'date': [],
            'iterations': []
        }

        url = 'http://www.espn.com/nfl/schedule/_/' + 'year/' + year + '/week/' + week
        page = requests.get(url)
        soup = BS(page.content, 'html.parser')

        # getting game dates and home and away teams for the week
        awayTeams, teamCountOnDate, homeTeams, dates, startingQbs = self.parseDatesAndTeams(soup, True)

        c = self.connectToDb()
        c.execute("SELECT MAX(player_id) from player_map")
        newPlayerId = c.fetchall()[0][0]
        for gameIndex in range(len(awayTeams)):
            # getting the ids of players and teams
            awayTeamId = self.getIds(awayTeams[gameIndex], c, newPlayerId, team=True, away=False)
            homeTeamId = self.getIds(homeTeams[gameIndex], c, newPlayerId, team=True, away=True)
            startingQbList = [startingQbs[gameIndex * 2], startingQbs[gameIndex * 2 + 1]]
            awayPlayerId, homePlayerId = self.getIds(startingQbList, c, newPlayerId, team=False, away=False)

            # filling the map with game data
            returnDataDict['awayTeamId'].append(awayTeamId)
            returnDataDict['homeTeamId'].append(homeTeamId)
            returnDataDict['homePlayerId'].append(homePlayerId)
            returnDataDict['awayPlayerId'].append(awayPlayerId)
            returnDataDict['season'].append(year)
            returnDataDict['week'].append(week)
            newPlayerId += 1

        # filling the title map with dates and number of teams playing on the date
        returnTitleDict['date'] = dates
        returnTitleDict['iterations'] = teamCountOnDate

        return [pd.DataFrame(returnDataDict, index=list(range(len(returnDataDict['awayTeamId'])))),
                pd.DataFrame(returnTitleDict, index=list(range(len(returnTitleDict['date']))))]

    def fetchGameResults(self, week, year):

        # this dict is for the teams and the results
        returnDataDict = {
            'awayTeamId': [],
            'homeTeamId': [],
            'homeScore': [],
            'awayScore': [],
            'season': [],
            'week': []
        }

        # this dict is for the date titles for future plot visualization
        returnTitleDict = {
            'date': [],
            'iterations': []
        }

        url = 'http://www.espn.com/nfl/schedule/_/' + 'year/' + year + '/week/' + week
        page = requests.get(url)
        soup = BS(page.content, 'html.parser')

        awayTeams, teamCountOnDate, homeTeams, dates, resultsMap = self.parseDatesAndTeams(soup, True)

        c = self.connectToDb()
        for gameIndex in range(len(awayTeams)):
            # getting the ids of teams
            awayTeamId = self.getIds(awayTeams[gameIndex], c, team=False, away=False)
            homeTeamId = self.getIds(homeTeams[gameIndex], c, team=True, away=True)

            # filling the map with game data
            returnDataDict['homeScore'].extend(resultsMap[homeTeams[gameIndex]])
            returnDataDict['awayScore'].extend(resultsMap[awayTeams[gameIndex]])
            returnDataDict['awayTeamId'].append(awayTeamId)
            returnDataDict['homeTeamId'].append(homeTeamId)
            returnDataDict['season'].append(year)
            returnDataDict['week'].append(week)

        # filling the title map with dates and number of teams playing on the date
        returnTitleDict['date'] = dates
        returnTitleDict['iterations'] = teamCountOnDate

        return [pd.DataFrame(returnDataDict, index=list(range(len(returnDataDict['awayTeamId'])))),
                pd.DataFrame(returnTitleDict, index=list(range(len(returnTitleDict['date']))))]

    def predictWeek(self, fetchedGameData, trainingDfX, trainingDfY):
        """
        Uses the top 5 prediction models to create dataframes of win/loss predictions and confidence scores
        :param fetchedGameData:
        :param trainingDfX:
        :param trainingDfY:
        :return:
        """

        totalPlayerIds = list(np.append(fetchedGameData['homePlayerId'].values, fetchedGameData['awayPlayerId'].values))
        totalTeamIds = list(np.append(fetchedGameData['homeTeamId'].values, fetchedGameData['awayTeamId'].values))
        predictionData = pd.DataFrame(columns=list(trainingDfX.columns))

        # Aggregate data sorting for the QBs starting on the week
        for playerId in np.append(fetchedGameData['homePlayerId'].values, fetchedGameData['awayPlayerId'].values):
            if int(playerId) in trainingDfX['player_id']:
                predictionData = predictionData.append(trainingDfX[trainingDfX['player_id'] == int(playerId)].sort_values(['season', 'week']).iloc[-1:])
            else:
                predictionData = predictionData.append(pd.DataFrame(np.zeros((1, len(trainingDfX.columns))), columns=list(trainingDfX.columns)))

        # resetting the index and replacing the incorrect opponent ids with the right ones
        predictionData.index = list(range(0, 32))
        predictionData.drop('opponent_id', 1, inplace=True)
        opponentIdFiller = []
        for playerId in totalPlayerIds:
            if totalPlayerIds.index(playerId) < (len(totalPlayerIds) / 2):
                gameRow = fetchedGameData[fetchedGameData['homePlayerId'] == playerId]
                opponentIdFiller.append(gameRow['homeTeamId'].values[0])
            else:
                gameRow = fetchedGameData[fetchedGameData['awayPlayerId'] == playerId]
                opponentIdFiller.append(gameRow['awayTeamId'].values[0])
        predictionData['opponent_id'] = opponentIdFiller

        models = self.view()['Best Model Instance'][:5]
        modelNames = self.view()['Model'][:5]

        ## NOTE: should just have a map stored so we don't have to make all these queries all the time
        # gets the team abbreviations for the graph
        conn = sqlite3.connect('/Users/ross.blanchard/PycharmProjects/k2_datascience/EDA_Project/nfl_db.sqlite')
        cursor = conn.cursor()
        teamPrintList = []
        for teamId in totalTeamIds:
            query = "select team from team_map where team_id='{}'".format(teamId)
            cursor.execute(query)
            team = cursor.fetchall()[0][0]
            if len(team.split(' ')) > 1:
                teamPrintList.append(team.split(' ')[1].upper())
            else:
                teamPrintList.append(team.upper())
        conn.close()

        # predicting the games and their associated confidence scores
        confidencesDf = pd.DataFrame(index=teamPrintList)
        predictionsDf = pd.DataFrame(index=teamPrintList)
        for model, modelName in zip(models, modelNames):
            if isinstance(model, list):
                pipe = Pipeline(model[0], model[1])
                pipe.fit(trainingDfX, trainingDfY)
                predictions = pipe.predict(predictionData)
                confidences = list(pipe.predict_proba(predictionData))
            else:
                clf = model
                clf.fit(trainingDfX, trainingDfY)
                predictions = clf.predict(predictionData)
                confidences = list(clf.predict_proba(predictionData))

            confidenceOfPrediction = []
            for pred, conf in zip(predictions, confidences):
                if pred == 0:
                    confidenceOfPrediction.append(conf[0])
                else:
                    confidenceOfPrediction.append(conf[1])

            confidencesDf = pd.concat([confidencesDf, pd.Series(confidenceOfPrediction, index=teamPrintList, name=modelName).rename(modelName)],
                                      axis=1)
            predictionsDf = pd.concat([predictionsDf, pd.Series(predictions, index=teamPrintList, name=modelName)],
                                      axis=1)
        predictionsDf['Aggregate Prediction'] = predictionsDf.mean(axis=1)
        confidencesDf['Aggregate Prediction'] = confidencesDf.mean(axis=1)

        return confidencesDf, predictionsDf

    def combineGames(self, confidenceDf, predictionDf):

        # creating the combined maps
        confidenceMap = OrderedDict([(key, []) for key in confidenceDf.columns])
        plottingLabelsMap = OrderedDict([(key, []) for key in confidenceDf.columns])
        # creating the combined teams index
        index = list(confidenceDf.index)
        combinedTeamIndex = [homeTeam + ' vs. ' + awayTeam for homeTeam, awayTeam in zip(index[::2], index[1::2])]
        cols = list(confidenceDf.columns)

        for colIndex in range(len(confidenceDf.columns)):
            for gameIndexStart in range(0, len(confidenceDf), 2):
                homeRowIndex = gameIndexStart
                awayRowIndex = gameIndexStart + 1
                homeConfidence = confidenceDf.iloc[homeRowIndex: homeRowIndex+1, colIndex].values[0]
                awayConfidence = confidenceDf.iloc[awayRowIndex: awayRowIndex+1, colIndex].values[0]
                homePrediction = predictionDf.iloc[homeRowIndex: homeRowIndex+1, colIndex].values[0]
                awayPrediction = predictionDf.iloc[awayRowIndex: awayRowIndex+1, colIndex].values[0]
                maxConf = max(homeConfidence, awayConfidence)
                minConf = min(homeConfidence, awayConfidence)
                # getting the total Confidence in the game call by assessing whether the predictions agree or not
                if homePrediction == awayPrediction:
                    totalConfidence = maxConf * (1 - minConf)
                else:
                    totalConfidence = maxConf * minConf
                confidenceMap[cols[colIndex]].append(totalConfidence)
                # writes the home team to the label map if it has the higher confidence, otherwise write the away team
                if maxConf == homeConfidence:
                    plottingLabelsMap[cols[colIndex]].append(index[gameIndexStart] +
                                                             '\nConfidence: ' + str(round(totalConfidence, 2)))
                else:
                    plottingLabelsMap[cols[colIndex]].append(index[gameIndexStart + 1] +
                                                             '\nConfidence: ' + str(round(totalConfidence, 2)))

        return [pd.DataFrame(confidenceMap, index=combinedTeamIndex),
                pd.DataFrame(plottingLabelsMap, index=combinedTeamIndex)]


""" Tests """

dataConnection = dataInitializer()
qbDf = dataConnection.fetchData('qbs')
qbDf = dataConnection.cleanData(qbDf, 'qbs')
aggQbData = dataConnection.aggregateData(qbDf)
optimizer = mlOptimizer(aggQbData, 'win_loss')

optimizer.optimizeModel('LogisticRegression', pca=False)
optimizer.optimizeModel('LogisticRegression', pca=True)
optimizer.optimizeModel('LDA', pca=True)
optimizer.optimizeModel('LDA', pca=False)
optimizer.optimizeModel('QDA', pca=True)
optimizer.optimizeModel('QDA', pca=False)
optimizer.optimizeModel('GaussianNB', pca=False)
optimizer.optimizeModel('GaussianNB', pca=True)
# optimizer.optimizeModel('MultinomialNB', pca=False)
optimizer.optimizeModel('DecisionTreeClassifier', pca=False)
optimizer.optimizeModel('DecisionTreeClassifier', pca=True)
optimizer.optimizeModel('SVC', pca=False)
optimizer.optimizeModel('SVC', pca=True)
optimizer.optimizeModel('KNeighborsClassifier', pca=False)
optimizer.optimizeModel('KNeighborsClassifier', pca=True)
optimizer.optimizeModel('RandomForest', pca=False)
optimizer.optimizeModel('GradientBoostingClassifier', pca=False)
optimizer.optimizeModel('AdaBoostClassifier', pca=False)
weeklyGameData, weeklyGamePlotTitles = optimizer.fetchFutureGames('2', '2017')
# optimizer.fetchGameResults('1', '2017')
optimizer.plotPredictions(weeklyGameData, aggQbData, weeklyGamePlotTitles)
