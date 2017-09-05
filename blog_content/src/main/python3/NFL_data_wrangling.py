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
        query_df = self.convertCols(query_df, position)
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

    def convertCols(self, df, table_str):

        def dropCols(df):
            if 'sacks' in df.columns:
                df.drop('sacks', 1, inplace=True)
            elif 'id' in df.columns:
                df.drop('id', 1, inplace=True)
            df.drop(['team_x', 'team_y', 'result',
                     'opponent', 'team_id_y', 'team_id_x',
                     'date', 'player_name'], 1, inplace=True)
            return df

        def mapPlayersAndTeams(df):
            # team mapping
            teamMapQuery = "SELECT * FROM team_map"
            self.cursor.execute(teamMapQuery)
            teamMapDf = pd.DataFrame(self.cursor.fetchall())
            teamMapDf.columns = [desc[0] for desc in self.cursor.description]


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

        df['date'] = pd.to_datetime(df.date, format="%m/%d/%y")
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
        return trainingData

# ndc = dataInitializer()
# df = ndc.fetchData('qbs')
# cum_data = ndc.aggregateData(df)
# print(cum_data.head())