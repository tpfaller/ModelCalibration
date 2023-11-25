import train
import mlflow

import numpy as np
from matplotlib import pyplot as plt

def grid_search():
    lr_array = np.array([1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6])
    # lr_array = np.linspace(lr_min, lr_max, num=lr_steps)
    return lr_array


def main():
    args = train.get_args_parser().parse_args()

    params = vars(args)
    grid = grid_search()
    # print(grid)
    # plt.plot(grid, "x")
    # plt.show()
    # plt.close()

    with mlflow.start_run():
        for lr in grid:
            params["lr"] = lr
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                # train.main(args=args)
                metrics = dict()
                metrics["best_acc"] = .5
                mlflow.log_metrics(metrics)

if __name__ == '__main__':
    main()
