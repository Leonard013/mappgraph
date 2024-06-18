import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# this path depends on your setup (need to contain sources folder)
root_path = '/Volumes/T9/map'
## Config setting
'''
Define set of hyper-parameters
List of tuples (duration, overlap)
'''
params = [(5, 3), (4, 2), (3, 1), (2, 0), (1, 0)]
## Create the mobile traffic chunks with the same length
# folder that contains source data
sources_folder = os.path.join(root_path, 'sources')
# loop over set of hyper-parameters
for duration, overlap in params:
  # folder that contain samples of one set of parameters
  param_folder = os.path.join(root_path, '%d_%d'%(duration, overlap))
  
  # check whether the data is already generated or not
  if not os.path.exists(param_folder):
    os.mkdir(param_folder)

    # create folder to contain samples
    samples_folder = os.path.join(param_folder, 'samples')
    os.mkdir(samples_folder)

    # loop over each app to generate samples
    for app in os.listdir(sources_folder):
      if app.startswith('.') or app == 'desktop.ini':
        continue
      print('App: ', app)
      app_sources_folder = os.path.join(sources_folder, app)

      # create folder contain samples for each app
      app_samples_folder = os.path.join(samples_folder, app)
      if not os.path.exists(app_samples_folder):
        os.mkdir(app_samples_folder)

      for filename in os.listdir(app_sources_folder):
        if filename.startswith('.') or filename == 'desktop.ini':
          continue
        print('Processing %s ...' % filename)
        index = 1

        file_path = os.path.join(app_sources_folder, filename)
        df = pd.read_csv(file_path, index_col=0)
        base = df['time'].iloc[0]
        end = df['time'].iloc[-1]
      
        while ((index - 1)*(duration - overlap) + duration)*60 + base < end:
          start_time = base + (index-1)*(duration - overlap)*60
          end_time = start_time + duration*60
          df_ = df[(df['time'] >= start_time) & (df['time'] <= end_time)].reset_index(drop=True)

          # save a sample as csv file
          sample_filename = "_".join(filename.split('_')[:-2]) + '_' + filename.split('_')[-2] + '_' + str(index) + '.csv'
          sample_path = os.path.join(app_samples_folder, sample_filename)
          df_.to_csv(sample_path, index=True)

          index += 1
    
      print('...................................................')