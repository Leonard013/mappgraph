# generating_train_test.py

import os
import json
import random
import warnings
warnings.filterwarnings('ignore')


def GeneratingTrainTest(dataset):
  '''
  Define set of hyper-parameters
  List of tuples (duration, overlap)
  '''
  params = [(5, 3), (4, 2), (3, 1), (2, 0), (1, 0)]

  root_path = dataset

  train_size = 0.8
  ## Split data into training and testing and save into a json file
  """
  input: list of filenames (samples) and the train size
  output: list of filenames for training, list of filenames of testing
  """

  def get_train_test(filenames, train_size):
    train_idx = random.sample(range(len(filenames)), int(len(filenames)*train_size))
    train_filenames = [filenames[i] for i in train_idx]

    test_filenames = list(set(filenames) - set(train_filenames))

    return (train_filenames, test_filenames)
  # loop over set of hyper-parameters
  for duration, overlap in params:
    param_folder = os.path.join(root_path, '%d_%d'%(duration, overlap))
    samples_folder = os.path.join(param_folder, 'samples')

    # initial a dictionary containing training and testing information
    train_test_info = dict()
    for app in os.listdir(samples_folder):
      if app.startswith('.') or app == 'desktop.ini':
        continue
      app_folder = os.path.join(samples_folder, app)
      filenames = os.listdir(app_folder)

      train_test_info[app] = get_train_test(filenames, train_size)

    # save train_test_info as json file
    saved_path = os.path.join(param_folder, 'train_test_info.json')
    with open(saved_path, 'w') as f:
      json.dump(train_test_info, f)



if __name__ == '__main__':
   GeneratingTrainTest(dataset = 'path/to/dataset' ) # path to the dataset foldere