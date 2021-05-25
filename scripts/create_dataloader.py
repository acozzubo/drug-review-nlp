import re
import json
from csv import reader
# import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class DrugReviewDataset(Dataset):
    def __init__(self, csv_file, x_colname, target_colname, tokenizer=None):
        """[summary]

        Args:
            csv_file ([str]): csv file with x_colname and target_colname in first row
            x_colname ([str]): colname of review column
            target_colname ([str]): colname of target column
            tokenizer ([tokenizer object], optional): possibly returned by pytorch's get_tokenizer
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
    This is the dataset object that's actually used.
    It's built such that in can handle an arbitrary number of additional features
    The features are returned as a dictionary
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
        """[summary]

        Args:
            csv_file (str): target_col, optional cols, and encoding cols must all be in the first
                row of csv file
            target_colnum (str, optional): [description]. Defaults to 'rating_category'.
            optional_cols (list of strs, optional): lists optional columns
            encoding_cols (dict{str: str}, optional): for columns that will be turned into one-hot encodings
                previously these needed to map the column names to the columns containing the length of the
                encoding.  However, since that logic has been moved to the collate function this is no longer
                necessary
            s3_bucket (str, optional): artifact of when we naively setup everything to run on amazon
                little did we know it was a huge waste of time
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
                        row[col] = line[idx]
                        # one hot encodings used to be made here but it required too much memory so
                        # logic was moved to the collate function

                        # make one-hot encodings if necessary
                        # if col in self.encodings:
                        #     if i == 0:
                        #         # print("col encoding", col)
                        #         # print("idx", idx)
                        #         # print("line", line[self.encodings[col]])
                        #         self.encodings[col] = int(
                        #             line[self.encodings[col]]) + 1
                        #         # idx = self.encodings[col]
                        #     one_hots = []
                        #     for token in json.loads(line[idx]):
                        #         one_hot = [0] * self.encodings[col]
                        #         one_hot[token] = 1
                        #         one_hots.append(json.dumps(one_hot))

                        #     row[col] = one_hots

                        # else:
                        #     row[col] = line[idx]
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
    Builds dataloader object

    Args:
        data_file (str): path to input file (should be a csv created by pre-processing notebook)
        batch_size (int): [description]
        shuffle (bool): [description]
        collate (fn, optional): collate function from collate.py. if you're using any encoding columns
        use collate_rnn_plus.  otherwise collate_rnn should be fine.  Defaults to None.
        optional_cols (list of strs, optional): lists optional columns
        encoding_cols (dict{str: str}, optional): for columns that will be turned into one-hot encodings
            previously these needed to map the column names to the columns containing the length of the
            encoding.  However, since that logic has been moved to the collate function this is no longer
            necessary


    Returns:
        DataLoader: iternator that will return data in a dict from with keys (labels, reviews, etc)
    """

    print("get dataloader called")

    ds = DrugReviewDatasetPlus(
        data_file, optional_cols=optional_cols, encoding_cols=encoding_cols)
    dataloader = DataLoader(ds, batch_size=batch_size,
                            shuffle=shuffle, collate_fn=collate)

    return dataloader
