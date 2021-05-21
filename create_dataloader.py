import re
import json
from csv import reader
# import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def load_df(path, index_col=None):
    """
    path is the string path to the dataset
    returns pandas df
    """
    file_type = path[-3:]
    delim = '\t' if file_type == 'tsv' else ','
    df = pd.read_csv(path, header=0, delimiter=delim,
                     index_col=index_col, parse_dates=['date'])
    return df


def replace_html_apostrophes(s):
    new_s = s.lower()
    new_s = new_s.replace("&#039;", "'")
    contraction_dict = {"ain't": "is not", "isn't": "is not", "we're": "were",
                        "weren't": "were not", "aren't": "are not", "can't": "cannot",
                        "wasn't": "was not", "won't": "will not", "don't": "do not",
                        "shouldn't": "should not", "doesn't": "does not", "you're": "you are",
                        "'cause": "because", "could've": "could have", "i'm": "i am",
                        "i've": "i have", "would've": "would have", "haven't": "have not"}
    for k, v in contraction_dict.items():
        new_s = new_s.replace(k, v)
    return new_s


def clean_numbers(s):
    if bool(re.search(r'\d', s)):
        s = re.sub('[0-9]{5}', '#####', s)
        s = re.sub('[0-9]{4}', '####', s)
        s = re.sub('[0-9]{3}', '###', s)
        s = re.sub('[0-9]{2}', '##', s)
    return s


def clean_string(s):
    new_s = replace_html_apostrophes(s)
    new_s = clean_numbers(new_s)
    return new_s


def clean_data(df):
    """
    takes pandas df
    returns pandas df
    """

    # do string cleaning steps for review
    df['review'] = df.apply(lambda row: clean_string(row['review']), axis=1)

    # code NAs as a "Not Entered" category
    df.loc[df['condition'].isna(), 'condition'] = 'Not Entered'

    # creates ratings category by binning ratings
    df['rating_category'] = 'Postive'
    df.loc[df['rating'] < 7, 'rating_category'] = 'Neutral'
    df.loc[df['rating'] < 4, 'rating_category'] = 'Negative'

    # create daily useful count
    max_date = df['date'].max()
    df['useful_daily'] = df['usefulCount'] / \
        ((max_date - df['date']).dt.days + 1)

    return df


def make_cleanish_df(tsv_filepath, output_path):
    """
    paths include filenames
    """
    df = load_df(tsv_filepath, index_col=0)
    df = clean_data(df)
    df.to_csv(output_path, index=False)


class DrugReviewDataset(Dataset):
    def __init__(self, csv_file, x_colname, target_colname, tokenizer=None):
        """
        the following are assumed about csv_file:
            - headers are in first row
            - there is a column called 'date'
            - there is a column called 'review' which contains the text data
        """
        self.x = []
        self.target = []
        with open(csv_file, 'r') as f:
            data = list(reader(f))

        target_colnum = data[0].index(target_colname)
        x_colnum = data[0].index(x_colname)
        for row in data[1:]:
            self.target.append(row[target_colnum])

            if tokenizer:
                self.x.append(tokenizer(row[x_colnum]))
            else:
                self.x.append(row[x_colnum])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        """
        idx can be a list or tensor if integers
        """
        example = (self.target[idx], self.x[idx])

        return example


# import pandas as pd


def load_df(path, index_col=None):
    """
    path is the string path to the dataset
    returns pandas df
    """
    file_type = path[-3:]
    delim = '\t' if file_type == 'tsv' else ','
    df = pd.read_csv(path, header=0, delimiter=delim,
                     index_col=index_col, parse_dates=['date'])
    return df


def replace_html_apostrophes(s):
    new_s = s.lower()
    new_s = new_s.replace("&#039;", "'")
    contraction_dict = {"ain't": "is not", "isn't": "is not", "we're": "were",
                        "weren't": "were not", "aren't": "are not", "can't": "cannot",
                        "wasn't": "was not", "won't": "will not", "don't": "do not",
                        "shouldn't": "should not", "doesn't": "does not", "you're": "you are",
                        "'cause": "because", "could've": "could have", "i'm": "i am",
                        "i've": "i have", "would've": "would have", "haven't": "have not"}
    for k, v in contraction_dict.items():
        new_s = new_s.replace(k, v)
    return new_s


def clean_numbers(s):
    if bool(re.search(r'\d', s)):
        s = re.sub('[0-9]{5}', '#####', s)
        s = re.sub('[0-9]{4}', '####', s)
        s = re.sub('[0-9]{3}', '###', s)
        s = re.sub('[0-9]{2}', '##', s)
    return s


def clean_string(s):
    new_s = replace_html_apostrophes(s)
    new_s = clean_numbers(new_s)
    return new_s


def clean_data(df):
    """
    takes pandas df
    returns pandas df
    """

    # do string cleaning steps for review
    df['review'] = df.apply(lambda row: clean_string(row['review']), axis=1)

    # code NAs as a "Not Entered" category
    df.loc[df['condition'].isna(), 'condition'] = 'Not Entered'

    # creates ratings category by binning ratings
    df['rating_category'] = 'Postive'
    df.loc[df['rating'] < 7, 'rating_category'] = 'Neutral'
    df.loc[df['rating'] < 4, 'rating_category'] = 'Negative'

    # create daily useful count
    max_date = df['date'].max()
    df['useful_daily'] = df['usefulCount'] / \
        ((max_date - df['date']).dt.days + 1)

    return df


def make_cleanish_df(tsv_filepath, output_path):
    """
    paths include filenames
    """
    df = load_df(tsv_filepath, index_col=0)
    df = clean_data(df)
    df.to_csv(output_path, index=False)


class DrugReviewDataset(Dataset):
    def __init__(self, csv_file, x_colname, target_colname, tokenizer=None):
        """
        the following are assumed about csv_file:
            - headers are in first row
            - there is a column called 'date'
            - there is a column called 'review' which contains the text data
        """
        self.x = []
        self.target = []
        with open(csv_file, 'r') as f:
            data = list(reader(f))

        target_colnum = data[0].index(target_colname)
        x_colnum = data[0].index(x_colname)
        for row in data[1:]:
            self.target.append(row[target_colnum])

            if tokenizer:
                self.x.append(tokenizer(row[x_colnum]))
            else:
                self.x.append(row[x_colnum])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        """
        idx can be a list or tensor if integers
        """
        example = (self.target[idx], self.x[idx])

        return example


class DrugReviewDatasetPlus(Dataset):
    """
    tokens is assumed to be in the dataset as the main X column
    it's expected to be a stringified list of tokens
    """
    DEFAULT_OPTIONAL_COLS = (["pos_encoding", "dep_encoding",
                              "shape_encoding", "lemmas"])
    DEFAULT_ENCODING_COLS = ({'pos_encoding': 'pos_encoding_count',
                              'dep_encoding': 'dep_encoding_count',
                              'shape_encoding': 'shape_encoding_count'})

    def __init__(self, csv_file, target_colnum='rating_category',
                 optional_cols=DEFAULT_OPTIONAL_COLS,
                 encoding_cols=DEFAULT_ENCODING_COLS,
                 s3_bucket=None):
        """
        the following are assumed about csv_file:
            - headers are in first row
            - there is a column called 'review' which contains the text data
            - there are many optional columns as well
            - optional cols is a list of cols to also keep in X
        """
        self.target = []
        self.X = []
        self.encodings = {}
        self.feature_names = optional_cols + ['tokens']

        if s3_bucket:
            response = s3_client.get_object(Bucket=bucket, Key=csv_file)
            data = json.loads(response['Body'])

        with open(csv_file, 'r') as f:
            print(f"reading file {csv_file}")
            data = reader(f)
            headers = next(data)

            # required cols
            try:
                target_colnum = headers.index(target_colnum)
                review_colnum = headers.index('tokens')
            except ValueError:
                print("target_colnum and review must be in the first row of the csv")
                raise

            # additional possible cols
            colnums = {}
            for col in self.feature_names:
                try:
                    colnums[col] = headers.index(col)
                except ValueError:
                    print(f"{col} was not found in the first row of your csv," +
                          "make sure that each element in optional_cols and" +
                          "encoding_cols, including the defaults, can be found " +
                          "in the first row of your csv.")
                    raise

            # get encoding colnums
            encoding_colnums = {}
            for encoding_col, count_col in encoding_cols.items():
                try:
                    idx = headers.index(count_col)
                except ValueError:
                    print(f"{col} was not found in the first row of your csv," +
                          "make sure that each element in optional_cols and" +
                          "encoding_cols, including the defaults, can be found " +
                          "in the first row of your csv.")
                    raise

                # also get numbers from this loop since they're row invarient
                # self.encodings[encoding_col] = int(data[1][idx]) + 1
                self.encodings[encoding_col] = idx

            # get data rows
            for i, line in enumerate(data):
                if not i % 10000:
                    print(f"loading line {i}")
                # features
                try:
                    self.target.append(line[target_colnum])
                except IndexError:
                    print(
                        f"not enough columns in {i+1} row of csv. skipping...")
                    continue

                try:
                    row = {}
                    for col, idx in colnums.items():
                        # make one-hot encodings if necessary
                        if col in self.encodings:
                            if i == 0:
                                self.encodings[col] = int(line[idx]) + 1
                                # idx = self.encodings[col]
                            one_hots = []
                            for token in json.loads(line[idx]):
                                one_hot = [0] * self.encodings[col]
                                one_hot[token] = 1
                                one_hots.append(json.dumps(one_hot))

                            row[col] = one_hots

                        else:
                            row[col] = line[idx]
                except IndexError:
                    print(f"not enough columns in {i+1} row of csv")
                    raise

                self.X.append(row)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        idx can be a list or tensor if integers
        """
        example = (self.target[idx], self.X[idx])

        return example


def get_dataloader(data_file, batch_size, shuffle, collate=None,
                   optional_cols=[], encoding_cols={}):
    """
    datafile: path to input file (should be a csv)
    batch_size: (int) parameter for DataLoader class
    shuffle: (bool) parameter for DataLoader class
    collage: (fn) parameter for DataLoader class
    split: (bool) specifies if there is to be a train-validation split on data
    """
    print("get dataloader called")

    ds = DrugReviewDatasetPlus(
        data_file, optional_cols=optional_cols, encoding_cols=encoding_cols)
    dataloader = DataLoader(ds, batch_size=batch_size,
                            shuffle=shuffle, collate_fn=collate)

    return dataloader


def save_dataset(dataloader, filepath):
    data = {x: dataloader.dataset.x,
            target: dataloader.dataset.target}
    with open(filepath, 'w') as f:
        s = json.dumps(data)
        f.write(s)


# if __name__ == '__main__':

#     # ds = DrugReviewDataset('/home/nselman/ml/drugproject/tiny_train.csv', 'review', 'rating_category')
#     # print(ds[3])

#     dl = get_dataloader('/home/nselman/ml/drugproject/tiny_train.csv', 'rating_category', 5, True)
#     for i, (tar, rev) in enumerate(dl):
#         print(tar, rev)
#         if i == 3:
#             break


# class DrugReviewDatasetPlus(Dataset):
#     """
#     tokens is assumed to be in the dataset as the main X column
#     it's expected to be a stringified list of tokens
#     """
#     DEFAULT_OPTIONAL_COLS = (["pos_encoding", "dep_encoding",
#                               "shape_encoding", "lemmas"])
#     DEFAULT_ENCODING_COLS = ({'pos_encoding': 'pos_encoding_count',
#                               'dep_encoding': 'dep_encoding_count',
#                               'shape_encoding': 'shape_encoding_count'})

#     def __init__(self, csv_file, target_colnum='rating_category',
#                  optional_cols=DEFAULT_OPTIONAL_COLS,
#                  encoding_cols=DEFAULT_ENCODING_COLS,
#                  s3_bucket=None):
#         """
#         the following are assumed about csv_file:
#             - headers are in first row
#             - there is a column called 'review' which contains the text data
#             - there are many optional columns as well
#             - optional cols is a list of cols to also keep in X
#         """
#         self.target = []
#         self.X = []
#         self.encodings = {}
#         self.feature_names = optional_cols + ['tokens']

#         if s3_bucket:
#             response = s3_client.get_object(Bucket=bucket, Key=csv_file)
#             data = json.loads(response['Body'])

#         with open(csv_file, 'rb') as f:
#             print(f"reading file {csv_file}")
#             data = list(reader(f))

#             # required cols
#         try:
#             target_colnum = data[0].index(target_colnum)
#             review_colnum = data[0].index('tokens')
#         except ValueError:
#             print("target_colnum and review must be in the first row of the csv")
#             raise

#         # additional possible cols
#         colnums = {}
#         for col in self.feature_names:
#             try:
#                 colnums[col] = data[0].index(col)
#             except ValueError:
#                 print(f"{col} was not found in the first row of your csv," +
#                       "make sure that each element in optional_cols and" +
#                       "encoding_cols, including the defaults, can be found " +
#                       "in the first row of your csv.")
#                 raise

#         # get encoding colnums
#         encoding_colnums = {}
#         for encoding_col, count_col in encoding_cols.items():
#             try:
#                 idx = data[0].index(count_col)
#             except ValueError:
#                 print(f"{col} was not found in the first row of your csv," +
#                       "make sure that each element in optional_cols and" +
#                       "encoding_cols, including the defaults, can be found " +
#                       "in the first row of your csv.")
#                 raise

#             # also get numbers from this loop since they're row invarient
#             self.encodings[encoding_col] = int(data[1][idx]) + 1

#         # get data rows
#         for i, line in enumerate(data[1:]):
#             if not i % 100:
#                 print(f"loading line {i}")
#             # features
#             try:
#                 self.target.append(line[target_colnum])
#             except IndexError:
#                 print(f"not enough columns in {i+1} row of csv. skipping...")
#                 continue

#             try:
#                 row = {}
#                 for col, idx in colnums.items():
#                     # make one-hot encodings if necessary
#                     if col in self.encodings:
#                         one_hots = []
#                         for token in json.loads(line[idx]):
#                             one_hot = [0] * self.encodings[col]
#                             one_hot[token] = 1
#                             one_hots.append(json.dumps(one_hot))

#                         row[col] = one_hots

#                     else:
#                         row[col] = line[idx]
#             except IndexError:
#                 print(f"not enough columns in {i+1} row of csv")
#                 raise

#             self.X.append(row)

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         """
#         idx can be a list or tensor if integers
#         """
#         example = (self.target[idx], self.X[idx])

#         return example


# def get_dataloader(data_file, batch_size, shuffle, collate=None,
#                    optional_cols=[], encoding_cols={}):
#     """
#     datafile: path to input file (should be a csv)
#     batch_size: (int) parameter for DataLoader class
#     shuffle: (bool) parameter for DataLoader class
#     collage: (fn) parameter for DataLoader class
#     split: (bool) specifies if there is to be a train-validation split on data
#     """
#     print("get dataloader called")

#     ds = DrugReviewDatasetPlus(
#         data_file, optional_cols=optional_cols, encoding_cols=encoding_cols)
#     dataloader = DataLoader(ds, batch_size=batch_size,
#                             shuffle=shuffle, collate_fn=collate)

#     return dataloader


# def save_dataset(dataloader, filepath):
#     data = {x: dataloader.dataset.x,
#             target: dataloader.dataset.target}
#     with open(filepath, 'w') as f:
#         s = json.dumps(data)
#         f.write(s)


# if __name__ == '__main__':

#     # ds = DrugReviewDataset('/home/nselman/ml/drugproject/tiny_train.csv', 'review', 'rating_category')
#     # print(ds[3])

#     dl = get_dataloader('/home/nselman/ml/drugproject/tiny_train.csv', 'rating_category', 5, True)
#     for i, (tar, rev) in enumerate(dl):
#         print(tar, rev)
#         if i == 3:
#             break
