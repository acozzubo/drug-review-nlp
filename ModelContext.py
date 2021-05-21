try:
    import boto3
except Exception as e:
    print(e)
import torch
import os


class ModelContext():
    def __init__(self, model, trainer, evaluator, plotter):
        self.trainer = trainer
        self.evaluator = evaluator
        self.plotter = plotter
        self.model = model

    def run(self, num_epochs, log_interval):
        # setup
        self.evaluator.setup_dirs()

        # train
        time_taken = self.trainer.train(self.model, self.evaluator,
                                        num_epochs, log_interval)

        # load best model
        self.model.load_state_dict(torch.load(self.evaluator.params_file))

        # eval
        self.evaluator.after_all(self.model, time_taken)

        # plots
        self.plotter.run_all()

    def make_plots(self, directory):
        # TODO
        self.plotter.plot(directory)
        pass

    def upload_files(self, directory, bucket):
        s3_client = boto3.client('s3', region_name='us-east-1')
        for item in os.listdir(directory):
            print(item)
            path = f'{directory}/{item}'
            if os.path.isfile(path):
                s3_client.upload_file(path, bucket, path)
            else:
                self.upload_files(path, bucket)

    def load_params(self, file):
        self.model.load_state_dict(torch.load(file))
