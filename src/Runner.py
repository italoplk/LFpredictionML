
from Params import get_args
from DataSet import DataSet


def main():
    params = get_args()

    dataset = DataSet(params)
    dataset.split()
    for lf in dataset.list_train:
        print(lf.lf_name)

    for lf in dataset.list_test:
        print(lf)

    print(len(dataset.list_train))








if __name__ == '__main__':
    main()



