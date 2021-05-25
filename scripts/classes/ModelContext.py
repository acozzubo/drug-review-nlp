try:
    import boto3
except Exception as e:
    print(e)
import torch
import os


class ModelContext():
    """
    Object that holds everything required to train and evaluate model
    """

    def __init__(self, *, model, trainer, evaluator, plotter,
                 train_dataloader, valid_dataloader, test_dataloader):
        """
        Constructor for mile object

        Args:
            model (nn.Module)
            trainer (Trainer)
            evaluator (Evaluator)
            plotter (Plotter)
            train_dataloader (DataLoader)
            valid_dataloader (DataLoader)
            test_dataloader (DataLoader)
        """
        self.trainer = trainer
        self.evaluator = evaluator
        self.plotter = plotter
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

    def run(self, num_epochs, log_interval):
        """
        runs all training, evaluating and plotting

        Args:
            num_epochs ([type]): [description]
            log_interval ([type]): [description]
        """

        # setup
        self.evaluator.setup_dirs()

        # train
        time_taken = self.trainer.train(model=self.model, evaluator=self.evaluator,
                                        num_epochs=num_epochs, log_interval=log_interval,
                                        train_dataloader=self.train_dataloader,
                                        valid_dataloader=self.valid_dataloader)

        # load best model
        self.model.load_state_dict(torch.load(self.evaluator.params_file))

        # eval
        self.evaluator.after_all(
            model=self.model, time_taken=time_taken, test_dataloader=self.test_dataloader)

        # plots
        self.plotter.run_all()

    def make_plots(self, directory):
        """
        builds plots in directory

        Args:
            directory (str): directory in which to build plots
        """

        self.plotter.plot(directory)

    def upload_files(self, directory, bucket):
        """
        NOT ACTUALLY USED SINCE WE DITCHED AWS
        """
        s3_client = boto3.client('s3', region_name='us-east-1')
        for item in os.listdir(directory):
            print(item)
            path = f'{directory}/{item}'
            if os.path.isfile(path):
                s3_client.upload_file(path, bucket, path)
            else:
                self.upload_files(path, bucket)

    def load_params(self, file):
        """

        Args:
            file (str): file should hold a state dict
        """
        self.model.load_state_dict(torch.load(file))
