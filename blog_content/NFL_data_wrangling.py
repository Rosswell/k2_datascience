import pandas as pd
import sqlite3

class data_creation():
    def __init__(self):
        conn = sqlite3.connect('nfl_db.sqlite')
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



    def convert_cols(self, df, table_str):

        def longest_fix(x):
            if isinstance(x, str):
                return int(x[:-1])
            return int(x)

        def date_fix(series):
            for x in series.values:
                year = x.astype('datetime64[Y]').astype(int) + 1970
                month = x.astype('datetime64[M]').astype(int) % 12 + 1
                if month < 3:
                    year -= 1
                    month += 12
                month -= 7
            return month, year

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
        df['month'], df['season'] = date_fix(df['date'])

        if table_str == 'qbs':
            df['longest_completion'] = df.longest_completion.apply(lambda x: longest_fix(x))
            df['times_sacked'], df['sack_length'] = sack_fix(df['sacks'])
            df = df.drop(['sacks', 'result'], 1)

        if table_str == 'rbs':
            df['reception_long'] = df['reception_long'].apply(lambda x: longest_fix(x))
            df['rushing_long'] = df['rushing_long'].apply(lambda x: longest_fix(x))
            df['targets'] = df['targets'].apply(lambda x: tar_yac_fix(x))
            df['yards_after_catch'] = df['yards_after_catch'].apply(lambda x: tar_yac_fix(x))
            df = df.drop(['id', 'result'], 1)

        return df


db = data_creation()
# db.create_prediction_table('qbs', 'qb_predictions')
db.fill_prediction_table('qbs')
db.close_conn()