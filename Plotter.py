import altair as alt
import pandas as pd
import csv


class Plotter():
    def __init__(self, results_dir):
        self.results_dir = results_dir

    def plot_lines(self, metrics):
        """
        metrics: list of metrics
        """
        pass
        # get data
        with open(f'{self.results_dir}/accuracies.csv', 'r') as f:
            valid_data = list(csv.reader(f))

        valid_data = list(zip(*valid_data))
        print("valid_data", valid_data)
        valid_cols = {c[0]: list(map(float, c[1:])) for c in valid_data}
        # print("valid_cols", valid_cols)

        inputs = {k: v for k, v in valid_cols.items()
                  if k in metrics}
        print("inputs", inputs)

        # produce charts
        chart = training_lines(valid_accuracies=inputs,
                               title="Validation Accuracies")

        # save charts
        chart.save('test.html')


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

    print(cols["value"][0])
    print(type(cols["value"][0]))
    df = pd.DataFrame(cols)

    # test
    test_cols = {"value": [], "epoch": [], "metric_name": []}
    for metric, value in test_accuracies.items():
        n = len(values)
        test_cols["value"].extend([value] * n)
        test_cols["epoch"].extend(i+1 for i in range(n))
        test_cols["metric_name"].extend(["test_" + metric]*n)

    test_df = pd.DataFrame(test_cols)

    print(df['value'].dtype)

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
    print("about to return")
    return (heatmap + text).properties(
        width=500, height=500,
        title={'text': title})


if __name__ == "__main__":

    plotter = Plotter("./test_results/lstm_small_results")
    plotter.plot_lines(['accuracy', 'cohens_kappa'])


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
