import wandb


class LogWriter:
    def __init__(self, suffix):
        self.suffix = suffix
        self.training_step = -1
        self.logs = {}

    def add_scalar(self, name: str, value, step: int):
        if step > self.training_step:
            self.training_step = step
            if self.logs:
                wandb.log(self.logs)
            self.logs = {}
        self.logs.update({name: value})

