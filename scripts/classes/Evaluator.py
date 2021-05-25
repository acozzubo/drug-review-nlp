import shutil
import csv
import os
import datetime
import torch
from sklearn.metrics import cohen_kappa_score, accuracy_score, balanced_accuracy_score, confusion_matrix
from torch.nn.functional import cross_entropy, hinge_embedding_loss
try:
    import boto3
except Exception as e:
    print(e)


class Evaluator():
    """
    Executes all evaluation methods
    Includes methods to record training and validation accuracy after epochs and
    test accuracy after all epochs are finished.
    """

    def __init__(self, *, root_dir='./', run_name=None, unpack_batch_fn=None):
        """ Constructor for Evaluator object

        Args:
            root_dir (str, optional): Defaults to './'.
            run_name (str, optional): Name that will be used to create directories for saving files
            unpack_batch_fn (fn, optional): function required to unpack an batch from the a dataloader
                and pass to the model  (this shouldn't be an attribute)
        """
        self.root_dir = root_dir
        if run_name:
            self.run_name = run_name
        else:
            self.run_name = "run_" + str(datetime.today())
        self.metrics = ['epoch', 'time_taken', 'split', 'accuracy',
                        'balanced_accuracy', 'cohens_kappa', 'cross_entropy', 'hinge_loss']
        self.unpack_batch_fn = unpack_batch_fn
        self.primary_metric = 'cohens_kappa'  # should be made into param
        self.best_metric_score = float('-inf')

    def setup_dirs(self):
        """
        Sets up directories to store evaluation results
        Based on self.root_dir and self.run_name
        """

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
        self.test_acc_file = f'{self.eval_dir}/test_accuracies.csv'
        with open(self.test_acc_file, 'w') as acc_file:
            writer = csv.writer(acc_file)
            writer.writerow(self.metrics)

        # create params_file
        self.params_file = f"{self.eval_dir}/{self.run_name}.pt"

        # create predictions dir
        self.preds_dir = f"{self.eval_dir}/preds"
        os.mkdir(self.preds_dir)

    def save_params(self, model):
        """saves parameters for model

        Args:
            model (nn.Module)
        """
        torch.save(model.state_dict(), self.params_file)

    def calc_cross_entropy_loss(self, labels, inputs):
        """
        Args:
            labels (tensor): tensor of one dimension with correct labels
            inputs (tensor): tensor that is output of model

        Returns:
            float: cross entropy loss
        """
        return cross_entropy(inputs, labels).item()

    def calc_hinge_loss(self, labels, inputs):
        """
        Args:
            labels (tensor): tensor of one dimension with correct labels
            inputs (tensor): tensor that is output of model

        Returns:
            float: hinge loss
        """
        hinge_labels = torch.ones(len(labels), max(labels)+1) * -1
        for l, h in zip(labels, hinge_labels):
            h[l] = 1
        return hinge_embedding_loss(inputs, hinge_labels).item()

    def calc_cohens_kappa(self, labels, predictions):
        """
        Args:
            labels (tensor): tensor of one dimension with correct labels
            predictions (tensor): tensor of predictions

        Returns:
            float: cohens kappa
        """
        return cohen_kappa_score(labels, predictions)

    def calc_accuracy(self, labels, predictions):
        """
        Args:
            labels (tensor): tensor of one dimension with correct labels
            predictions (tensor): tensor of predictions

        Returns:
            float: cohens kappa
        """
        return accuracy_score(labels, predictions)

    def calc_balanced_accuracy(self, labels, predictions):
        """
        average recall for each class
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score

        Args:
            labels (tensor): tensor of one dimension with correct labels
            predictions (tensor): tensor of predictions

        Returns:
            float: cohens kappa
        """
        return balanced_accuracy_score(labels, predictions)

    def compute_all_accuracies(self, labels, predictions, inputs):
        """
        computes all accuracy metrics

        Args:
            labels (tensor): tensor of one dimension with correct labels
            predictions (tensor): tensor of predictions
            inputs (tensor): tensor that is output of model

        Returns:
            dict: maps metric name to corresponding score
        """
        accs = {}
        accs['cohens_kappa'] = self.calc_cohens_kappa(labels, predictions)
        accs['accuracy'] = self.calc_accuracy(labels, predictions)
        accs['balanced_accuracy'] = self.calc_balanced_accuracy(
            labels, predictions)
        accs['cross_entropy'] = self.calc_cross_entropy_loss(labels, inputs)
        accs['hinge_loss'] = self.calc_hinge_loss(labels, inputs)
        return accs

    def write_epoch(self, *, accuracy_file, epoch, labels, predictions, inputs, time_taken, split):
        """[summary]

        Args:
            accuracy_file (str): file where accuracies are stored
            epoch (int): number of epoch
            labels (list or tensor): correct labels
            predictions (list or tensor): predicted labels
            inputs (tensor): output of model forward
            time_taken (time.time()): time that the epoch took
            split (str): train or valid

        Returns:
            float: score of primary metric (id cohens kappa or accuracy)
        """

        row = self.compute_all_accuracies(labels, predictions, inputs)
        row['epoch'] = epoch
        row['time_taken'] = time_taken
        row['split'] = split
        with open(accuracy_file, 'a') as acc_file:
            writer = csv.DictWriter(acc_file, fieldnames=self.metrics)
            writer.writerow(row)
        return row[self.primary_metric]

    def make_predictions(self, model, dataloader, unpack_batch_fn=None):
        """
        Args:
            model (nn.Module)
            dataloader (DataLoader): iterator that spits out batches
            unpack_batch_fn (fn, optional): unpack data is a callable that takes output from dataloader and builds

        Returns:
            tuple of tensors: log_probs (which is a misnomer), predictions, labels
        """

        model.eval()
        with torch.no_grad():
            all_predictions = []
            all_log_probs = []
            all_labels = []
            for idx, data in enumerate(dataloader):
                all_labels.extend(data['labels'].tolist())
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

    def save_predictions(self, save_path, epoch, predictions, labels):
        """
        saves all predictions for a particular epoch

        Args:
            save_path (str): directory to save predictions
            epoch (int): epoch number
            predictions (list or tensor): predicted labels
            labels (list or tensor): actual labels
        """
        if torch.is_tensor(predictions):
            predictions = predictions.tolist()
        if torch.is_tensor(labels):
            labels = labels.tolist()

        fn = f"epoch_{epoch}.csv" if epoch != "test" else "test_preds.csv"

        with open(f"{save_path}/{fn}", 'w+') as pred_file:
            writer = csv.writer(pred_file)
            writer.writerow(['predictions', 'labels'])
            writer.writerows(map(list, zip(*[predictions, labels])))

    def after_epoch(self, *, epoch, model, time_taken, valid_dataloader, train_dataloader):
        """
        everything that should be run after an epoch

        Args:
            epoch (int): epoch number
            model (nn.Module)
            time_taken (time.time()): time taken during epoch
            valid_dataloader (DataLoader)
            train_dataloader (DataLoader)

        Returns:
            float: value of best_metric
        """
        # train results
        log_probs, predictions, labels = self.make_predictions(
            model, train_dataloader,
            unpack_batch_fn=self.unpack_batch_fn)
        score = self.write_epoch(accuracy_file=self.acc_file, epoch=epoch, labels=labels,
                                 predictions=predictions, inputs=log_probs, time_taken=time_taken, split='train')
        # validation results
        log_probs, predictions, labels = self.make_predictions(
            model, valid_dataloader,
            unpack_batch_fn=self.unpack_batch_fn)  # self.unpack_batch_fn should be a parameter
        score = self.write_epoch(accuracy_file=self.acc_file, epoch=epoch, labels=labels,
                                 predictions=predictions, inputs=log_probs, time_taken=time_taken, split='valid')
        # self.execute_predictions()
        self.save_predictions(self.preds_dir, epoch, predictions, labels)
        if score > self.best_metric_score:
            self.best_metric_score = score
            self.save_params(model)
        return score

    def test_data(self, *, model, test_dataloader):
        """

        Args:
            model (nn.Module)
            test_dataloader (DataLoader)
        """
        model.eval()
        model.load_state_dict(torch.load(self.params_file))
        log_probs, predictions, labels = self.make_predictions(model, test_dataloader,
                                                               unpack_batch_fn=self.unpack_batch_fn)  # self.unpack_batch_fn should be a parameter
        self.save_predictions(self.preds_dir, "test", predictions, labels)
        accs = self.compute_all_accuracies(
            labels, predictions, log_probs)
        with open(self.test_acc_file, 'w') as f:
            dictwriter = csv.DictWriter(f, fieldnames=list(accs.keys()))
            dictwriter.writeheader()
            dictwriter.writerow(accs)

    def after_all(self, *, model, time_taken, test_dataloader):
        """[summary]

        Args:
            model (nn.Module)
            time_taken not actually used! MUST REMOVE FROM CALLER
            test_dataloader (DataLoader)
        """
        self.test_data(
            model=model, test_dataloader=test_dataloader)
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
