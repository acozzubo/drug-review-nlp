import shutil
import csv
import os
import datetime
import torch
from sklearn.metrics import cohen_kappa_score, accuracy_score, balanced_accuracy_score, confusion_matrix
from torch.nn.functional import log_softmax, cross_entropy, hinge_embedding_loss
import boto3


class Evaluator():
    """
    Creates files relating to evaluation of models during and after training
    """

    def __init__(self, valid_dataloader, test_dataloader,
                 root_dir='./', run_name=None, unpack_batch_fn=None):
        self.root_dir = root_dir
        if run_name:
            self.run_name = run_name
        else:
            self.run_name = "run_" + str(datetime.today())
        self.metrics = ['epoch', 'time_taken', 'accuracy',
                        'balanced_accuracy', 'cohens_kappa', 'cross_entropy', 'hinge_loss']
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.unpack_batch_fn = unpack_batch_fn
        self.primary_metric = 'accuracy'
        self.best_metric_score = float('-inf')
        self.unpack_dataloaders()

    def unpack_dataloaders(self):
        valid_labels = []
        for (data) in self.valid_dataloader:
            valid_labels.extend(data['labels'].tolist())
        self.valid_labels = torch.tensor(valid_labels)

        test_labels = []
        for (data) in self.test_dataloader:
            test_labels.extend(data['labels'].tolist())
        self.test_labels = torch.tensor(test_labels)

    def setup_dirs(self):
        self.eval_dir = f"{self.root_dir}/{self.run_name}"

        try:
            os.mkdir(self.eval_dir)
        except FileExistsError as e:
            print(f"Directory {self.eval_dir} already exists... deleting...")
            shutil.rmtree(self.eval_dir)
            print(f"Creating {self.eval_dir}...")
            os.mkdir(self.eval_dir)

        # create accuracies file
        self.acc_file = f'{self.eval_dir}/accuracies.csv'
        with open(self.acc_file, 'w') as acc_file:
            writer = csv.writer(acc_file)
            writer.writerow(self.metrics)

        # create params_file
        self.params_file = f"{self.eval_dir}/{self.run_name}.pt"

        # create predictions dir
        self.preds_dir = f"{self.eval_dir}/preds"
        os.mkdir(self.preds_dir)

    def save_params(self, model):
        torch.save(model.state_dict(), self.params_file)

    def calc_cross_entropy_loss(self, labels, inputs):
        return cross_entropy(inputs, labels).item()

    def calc_hinge_loss(self, labels, inputs):
        hinge_labels = torch.ones(len(labels), max(labels)+1) * -1
        for l, h in zip(labels, hinge_labels):
            h[l] = 1
        return hinge_embedding_loss(inputs, hinge_labels).item()

    def calc_cohens_kappa(self, labels, predictions):
        return cohen_kappa_score(labels, predictions)

    def calc_accuracy(self, labels, predictions):
        return accuracy_score(labels, predictions)

    def calc_balanced_accuracy(self, labels, predictions):
        """
        average recall for each class
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score
        """
        return balanced_accuracy_score(labels, predictions)

    def compute_all_accuracies(self, labels, predictions, inputs):
        accs = {}
        accs['cohens_kappa'] = self.calc_cohens_kappa(labels, predictions)
        accs['accuracy'] = self.calc_accuracy(labels, predictions)
        accs['balanced_accuracy'] = self.calc_balanced_accuracy(
            labels, predictions)
        accs['cross_entropy'] = self.calc_cross_entropy_loss(labels, inputs)
        accs['hinge_loss'] = self.calc_hinge_loss(labels, inputs)
        return accs

    def write_epoch(self, epoch, labels, predictions, inputs, time_taken):
        row = self.compute_all_accuracies(labels, predictions, inputs)
        row['epoch'] = epoch
        row['time_taken'] = time_taken
        with open(self.acc_file, 'a') as acc_file:
            writer = csv.DictWriter(acc_file, fieldnames=self.metrics)
            writer.writerow(row)
        return row[self.primary_metric]

    def make_predictions(self, model, dataloader, unpack_batch_fn=None):
        """
        unpack data is a callable that takes output from dataloader and builds
        the inputs from the model
        inputs must be returned as a tuple
        currently doesn't support kwargs even though kwargs are the best
        """
        model.eval()
        with torch.no_grad():
            all_predictions = []
            all_log_probs = []
            all_labels = []
            for idx, data in enumerate(dataloader):
                all_labels.extend[dataloader['labels'].tolist()]
                if unpack_batch_fn:
                    inputs = unpack_batch_fn(data)
                    log_probs = model(*inputs)
                else:
                    # not sure when you wouldn't need to unpack data
                    log_probs = model(data)

                # take raw predictions
                all_log_probs.extend(log_probs.tolist())

                # get specific prections
                predictions = log_probs.argmax(1)
                all_predictions.extend(predictions.tolist())

        return torch.tensor(all_log_probs), torch.tensor(all_predictions), torch.tensor(all_labels)

    def save_predictions(self, epoch, predictions, labels):
        if torch.is_tensor(predictions):
            predictions = predictions.tolist()
        if torch.is_tensor(labels):
            labels = labels.tolist()

        with open(f"{self.preds_dir}/epoch_{epoch}.csv", 'w+') as pred_file:
            writer = csv.writer(pred_file)
            writer.writerow(['preditions', 'labels'])
            writer.writerows(map(list, zip(*[predictions, labels])))

    def after_epoch(self, epoch, model, time_taken):
        log_probs, predictions, labels = self.make_predictions(
            model, self.valid_dataloader,
            unpack_batch_fn=self.unpack_batch_fn)
        score = self.write_epoch(
            epoch, labels, predictions, log_probs, time_taken)
        # self.execute_predictions()
        self.save_predictions(epoch, predictions, labels)
        if score > self.best_metric_score:
            self.best_metric_score = score
            self.save_params(model)
        return score

    def test_data(self, model, test_dataloader):
        model.eval()
        model.load_state_dict(torch.load(self.params_file))
        log_probs, predictions, labels = self.make_predictions(model, test_dataloader,
                                                               unpack_batch_fn=self.unpack_batch_fn)
        accs = self.compute_all_accuracies(
            labels, predictions, log_probs)
        with open(f"{self.eval_dir}/test_predictions.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(accs)

    def after_all(self, model):
        self.test_data(model, self.test_dataloader)
        # any plotting functions will go here
#
    # def save_to_s3(self, bucket):

    #     s3_client = boto3.client('s3', region_name='us-east-1')
    #     for file in os.listdir(self.root_dir):
    #         if os.path.isfile(file):
    #             s3_client.upload_file(
    #                 file, bucket, f"{self.root_dir}/{file}")
    #         else:
    #             for f in os.listdir(file):
    #                 s3_client.upload_file(
    #                     file, bucket, f"{self.root_dir}/{file}/{f}")

    # def upload_files(self, directory, bucket):
    #     s3_client = boto3.client('s3', region_name='us-east-1')
    #     for item in os.listdir(directory):
    #         print(item)
    #         path = f'{directory}/{item}'
    #         if os.path.isfile(path):
    #             s3_client.upload_file(path, bucket, path)
    #         else:
    #             self.upload_files(path, bucket)
