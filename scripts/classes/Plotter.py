import altair as alt
import pandas as pd
import csv
import os
import shutil


class Plotter():
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.plots_dir = f"{self.results_dir}/plots"

    def setup_dir(self):
        try:
            print("making plots dir...", self.plots_dir)
            os.mkdir(self.plots_dir)
        except FileExistsError as e:
            print(f"Directory {self.plots_dir} already exists...")

    def run_all(self):
        self.setup_dir()
        self.plot_accuracy_lines()
        self.make_confusion_matrix()
        self.plot_lines(['cohens_kappa', 'accuracy'])

    def plot_accuracy_lines(self):

        # get data

        preds_dir = f"{self.results_dir}/preds/"
        files = [f for f in os.listdir(preds_dir) if 'test' not in f]

        epoch_dict = {}
        max_epoch = -1
        for file in files:
            with open(f"{preds_dir}/{file}", 'r') as f:
                data = list(csv.reader(f))
            data = list(zip(*data))
            data = {c[0]: list(map(float, c[1:])) for c in data}

            correct = {0: 0, 1: 0, 2: 0}
            total = {0: 0, 1: 0, 2: 0}
            for p, l in zip(data['predictions'], data['labels']):
                total[l] += 1
                if p == l:
                    correct[l] += 1

            accs = {
                'negative': correct[0] / total[0],
                'neutral': correct[1] / total[1],
                'positive': correct[2] / total[2]
            }

            # extracts number from filename
            epoch_num = int(file.split('_')[-1][:-4])
            # print("epoch_number", epoch_num)
            if epoch_num > max_epoch:
                max_epoch = epoch_num
            epoch_dict[epoch_num] = accs

        data = {}
        # print("max epoch", max_epoch)
        for label in ('positive', 'negative', 'neutral'):
            col = [0] * (max_epoch + 1)
            for k, v in epoch_dict.items():
                col[k] = v[label]
            data[label] = col

        # produce chart
        chart = accuracy_lines(data)

        # save chart
        chart.save(f"{self.plots_dir}/accuracy_lines.html")

    def plot_lines(self, metrics):
        """
        metrics: list of metrics
        """
        # get data
        with open(f'{self.results_dir}/accuracies.csv', 'r') as f:
            data = csv.reader(f)
            header = next(data)
            split_col = header.index('split')
            valid_data = [[c for c in header if c != 'split']]
            train_data = [[c for c in header if c != 'split']]
            for row in data:
                if row[split_col] == 'train':
                    row = [float(c) for c in row if c != 'train']
                    train_data.append(row)
                elif row[split_col] == 'valid':
                    row = [float(c) for c in row if c != 'valid']
                    valid_data.append(row)

        # train
        train_data = list(zip(*train_data))
        train_cols = {c[0]: list(map(float, c[1:]))
                      for c in train_data}
        train_inputs = {k: v for k, v in train_cols.items()
                        if k in metrics}

        # valid
        valid_data = list(zip(*valid_data))
        valid_cols = {c[0]: list(map(float, c[1:]))
                      for c in valid_data}
        valid_inputs = {k: v for k, v in valid_cols.items()
                        if k in metrics}

        # test
        with open(f'{self.results_dir}/test_accuracies.csv', 'r') as f:
            data = list(csv.reader(f))

        test_inputs = {h: float(c) for h, c in zip(*data) if h in metrics}

        # produce charts
        chart = training_lines(train_accuracies=train_inputs, valid_accuracies=valid_inputs,
                               test_accuracies=test_inputs, title="Accuracies Across Epochs")

        # save charts
        chart.save(f"{self.plots_dir}/training_lines.html")

    def make_confusion_matrix(self):
        # get data
        with open(f'{self.results_dir}/preds/test_preds.csv') as f:
            data = list(csv.reader(f))

        data = list(zip(*data))
        data_cols = {c[0]: list(map(float, c[1:])) for c in data}
        preds = data_cols['predictions']
        labels = data_cols['labels']

        # produce chart
        chart = confusion_matrix(
            preds, labels, title="Accuracies Confusion Matrix")

        # save chart
        chart.save(f"{self.plots_dir}/confusion_matrix.html")


def training_lines(title="", train_accuracies={}, valid_accuracies={}, test_accuracies={}):
    '''
    Inputs:
    training_accuracies: list of length n of the training accuracies obtained at each epoch
    validation_accuries: list of length n of the trianing accuracies obtained at each epoch
    test_accuracy: integer of the highest test score attained
    '''

    cols = {"value": [], "epoch": [], "metric_name": []}

    # training
    for metric, values in train_accuracies.items():
        n = len(values)
        cols["value"].extend(values)
        cols["epoch"].extend(i+1 for i in range(n))
        cols["metric_name"].extend(["train_" + metric]*n)

    # validation
    for metric, values in valid_accuracies.items():
        n = len(values)
        cols["value"].extend(values)
        cols["epoch"].extend(i+1 for i in range(n))
        cols["metric_name"].extend(["valid_" + metric]*n)

    # print(cols["value"][0])
    # print(type(cols["value"][0]))
    df = pd.DataFrame(cols)

    # test
    test_cols = {"value": [], "epoch": [], "metric_name": []}
    for metric, value in test_accuracies.items():
        n = len(values)
        test_cols["value"].extend([value] * n)
        test_cols["epoch"].extend(i+1 for i in range(n))
        test_cols["metric_name"].extend(["test_" + metric]*n)

    test_df = pd.DataFrame(test_cols)

    min_y_axis = df['value'].min() - 0.1

    line_chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("epoch:Q", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("value", scale=alt.Scale(domain=[min_y_axis, 1])),
        color='metric_name')

    test_line = alt.Chart(test_df).mark_line(strokeDash=[7, 1]).encode(
        x=alt.X("epoch:Q", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("value", scale=alt.Scale(domain=[min_y_axis, 1])),
        color='metric_name')

    return (line_chart + test_line).properties(
        title={'text': title})


def confusion_matrix(predicted, actual, title=""):
    df = pd.DataFrame({"predicted": predicted, "actual": actual})
    df_cells = df.groupby(["predicted", "actual"]
                          ).size().reset_index(name='count')
    df_actual = df.groupby(["actual"]).size().reset_index(name='actual_count')
    df = df_cells.merge(df_actual, how='left', on='actual',
                        left_index=False, right_index=False)
    df['norm'] = df['count'] / df['actual_count']
    # matrix['share'] = matrix['count'] / len(df)
    # max_count = max(matrix['count'])
    base = alt.Chart(df)
    heatmap = base.mark_rect().encode(
        x=alt.X('predicted:O', title="Predicted Class"),
        y=alt.Y('actual:O', title="Actual Class"),
        color=alt.Color('norm:Q', scale=alt.Scale(
                        scheme='greens', domain=[0, 1]),
                        legend=alt.Legend(
                        title="Predicted / Actual")
                        )
    )
    text = base.mark_text(align='center', baseline='middle').encode(
        x=alt.X('predicted:O'),
        y=alt.Y('actual:O'),
        text='norm',
        color=alt.value('black')
    )
    return (heatmap + text).properties(
        width=500, height=500,
        title={'text': title})


def accuracy_lines(data, title=""):
    '''
    data is indexable with 3 columns: positive, neutral, negative
    the values are lists os numbers
    '''
    # print("data", data)
    cols = {"accuracy": [], "epoch": [], "label": []}

    # reshape data
    for label, values in data.items():
        n = len(values)
        cols["accuracy"].extend(values)
        cols["epoch"].extend(i+1 for i in range(n))
        cols["label"].extend([label]*n)
    df = pd.DataFrame(cols)

    min_y_axis = df['accuracy'].min() - 0.1

    line_chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("epoch:Q", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("accuracy", scale=alt.Scale(domain=[min_y_axis, 1])),
        color='label')

    return (line_chart).properties(
        title={'text': "Accuracy by Class over Epochs"})

# if __name__ == "__main__":

#     plotter = Plotter("./test_results/lstm_small_results")
#     plotter.plot_lines(['accuracy', 'cohens_kappa'])


#     train = {"cohens_kappa": [.2, .6, .1, .4, .5],
#              "accuracy": [.2, .3, .5, .6, .8],
#              "balanced_accuracy": [.1, .4, .2, .1, .8]}
#     valid = {"cohens_kappa": [.3, .5, .2, .4, .7],
#              "accuracy": [.2, .4, .6, .6, .7],
#              "balanced_accuracy": [.2, .4, .5, .6, .9]}
#     test = {"cohens_kappa": .5,
#             "accuracy": .6,
#             "balanced_accuracy": .7}

#     training_lines(train_accuracies=train,
#                    valid_accuracies=valid, test_accuracies=test)
