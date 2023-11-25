import train
import mlflow

def grid_search():
    pass


def main():
    args = train.get_args_parser().parse_args()

    params = vars(args)

    with mlflow.start_run():

        mlflow.log_params(params)

        train.main(args=args)

if __name__ == '__main__':
    main()
