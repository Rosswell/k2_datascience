import pandas as pd
import sqlite3


def main(table_choice):
    table_choice = table_choice.lower()
    if 'qb' in table_choice:
        table_str = 'qbs'
    elif 'rb' in table_choice:
        table_str = 'rbs'

    query = 'SELECT * FROM ' + table_choice
    conn = sqlite3.connect('nfl_db.sqlite')
    c = conn.cursor()
    c.execute(query)
    df = pd.DataFrame(c.fetchall())
    df.columns = [desc[0] for desc in c.description]

    df['date'] = pd.to_datetime(df.date, format="%m/%d/%y")
    df['score'] = df.result.apply(lambda x: score_fix(x))
    df['win_loss'] = df.result.apply(lambda x: winloss_fix(x))
    df['month'] = df.date.apply(lambda x: month_fix(x.month))

    if table_str == 'qbs':
        df['longest_completion'] = df.longest_completion.apply(lambda x: longest_fix(x))
        df['times_sacked'], df['sack_length'] = sack_fix(df['sacks'])
        df = df.drop(['sacks', 'result'], 1)

    if table_str == 'rbs':
        df['rushing_long'] = df['rushing_long'].apply(lambda x: longest_fix(x))
        df['targets'] = df['targets'].apply(lambda x: tar_yac_fix(x))
        df['yards_after_catch'] = df['yards_after_catch'].apply(lambda x: tar_yac_fix(x))
        df = df.drop(['id', 'result'], 1)

    return df


def longest_fix(x):
    if isinstance(x, str):
        return int(x[:-1])
    return x


def month_fix(x):
    if x < 3:
        x = 12 + x
    return x - 7


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

if __name__ == '__main__':
    main('qbs')
