import os


def make_experiment(output_path, experiment_name):
    experiment_path = os.path.join(output_path, experiment_name)
    weight_path = os.path.join(experiment_path, 'weights')
    output_path = os.path.join(experiment_path, 'output')
    print(experiment_name)
    if os.path.exists(experiment_path):
        print('[INFO] experiment path exists')
    else:
        os.mkdir(experiment_path)

    if os.path.exists(weight_path):
        print('[INFO] weight path exists')
    else:
        os.mkdir(weight_path)

    if os.path.exists(output_path):
        print('[INFO] the output path exists')
    else:
        os.mkdir(output_path)

    print('[INFO] {} directory has been created'.format(experiment_path))

    return experiment_path, weight_path, output_path
