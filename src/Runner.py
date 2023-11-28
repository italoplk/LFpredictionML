
from Params import get_args
from DataSet import DataSet


def main():
    params = get_args()

    dataset = DataSet(params)
    dataset.split()
    for lf in dataset.list_train:
        print(lf.name)

    for lf in dataset.list_test:
        print(lf)

    config_name = f"{params.model}_{params.batch_size}_{params.lr}"









if __name__ == '__main__':
    main()



