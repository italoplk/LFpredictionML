
from Params import get_args
from DataSet import DataSet
from Trainer import Trainer
import wandb

def main():
    params = get_args()

    config_name = f"{params.model}_{params.batch_size}_{params.lr}"


    wandb.init(
        # set the wandb project where this run will be logged
        project="predictorUnet",
        # track hyperparameters and run metadata
        name=config_name,
        config={
            "learning_rate": params.lr,
            "architecture": f"{params.model}",
            "dataset": params.dataset_name,
            "epochs": params.epochs,
            "name": f"{config_name}"
        }
    )


    dataset = DataSet(params)
    dataset.split()
    # for lf in dataset.list_train.inner_storage:
    #     print(lf.name)

    # for lf in dataset.list_test.inner_storage:
    #     print(lf.name)


    Trainer(dataset, config_name, params)

    wandb.finish()








if __name__ == '__main__':
    main()



