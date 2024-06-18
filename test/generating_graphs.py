import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import os
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')
## Graph generator functions
# Basic reprocessing
def basic_reprocessing(df, N):

  # remove dns protocol
  df = df[(df['source_port'] != 53) & (df['destination_port'] != 53) & 
        (df['source_port'] != 5353) & (df['destination_port'] != 5353) &
        (df['source_port'] != 137) & (df['destination_port'] != 137) &
        (df['source_port'] != 67) & (df['destination_port'] != 67) &
        (df['source_port'] != 68) & (df['destination_port'] != 68) &
        (df['source_port'] != 5355) & (df['destination_port'] != 5355)]
  
  # get IP address and port number of the service
  df['des_greater_src'] = df['destination_port'] - df['source_port']
  df1 = df[df['des_greater_src'] > 0]
  df2 = df[df['des_greater_src'] < 0]
  df1['destination'] = df1['source_address']
  df1['port'] = df1['source_port']
  df1['outgoing'] = 0
  df2['destination'] = df2['destination_address']
  df2['port'] = df2['destination_port']
  df2['outgoing'] = 1
  df = pd.concat([df1, df2], ignore_index=True).sort_values(by='time').reset_index(drop=True)

  # merge IP address into port (same tuple (IP, port) - same network destination)
  df['IP_port'] = list(zip(df['destination'], df['port']))

  df = df.drop(['source_address', 'destination_address', 'certificate', 'des_greater_src', 'source_port', 'destination_port', 'destination', 'port'], axis=1)

  # get N network destinations that have the most packets
  df_ = df.groupby(['IP_port'], as_index = False).agg({'length':['count']}).sort_values(by=[('length', 'count')], ascending=False)
  destinations = df_[:N]['IP_port']

  return df[df['IP_port'].isin(destinations)].reset_index(drop=True)

# Packet-based features
def pkt_reprocessing(df):
  df = df.drop(['time', 'stream_id', 'protocol'], axis=1).reset_index(drop=True)
  df = df.sort_values(by=['IP_port']).reset_index(drop=True)

  # return 3 series of packet: outgoing, incoming, both
  out_df = df[df['outgoing'] == 1].drop(['outgoing'], axis=1).reset_index(drop=True)
  in_df = df[df['outgoing'] == 0].drop(['outgoing'], axis=1).reset_index(drop=True)
  full_df = df.drop(['outgoing'], axis=1)

  return out_df, in_df, full_df

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def extract_pkt_features(df, type="complete"):
  features_df = df.groupby(['IP_port'], as_index = False).\
    agg({'length':['max', 'min', 'mean', 'mad', 'std', 'var', 'skew', pd.DataFrame.kurt, 'count', 
                   percentile(10), percentile(20), percentile(30), percentile(40), percentile(50), 
                   percentile(60), percentile(70), percentile(80), percentile(90)],
     })
    
  # rename columns
  feature_names = ['max', 'min', 'mean', 'mad', 'std', 'var', 'skew', 'kurt', 'pkt_num', '10per', '20per', '30per', '40per', '50per', '60per', '70per', '80per', '90per']
  features_df.columns = ['IP_port'] + [type + "_" + x for x in feature_names]

    
  return features_df
# Flow-based features
def flow_reprocessing(df):

  df['protocol'] = df['protocol'] == 'tcp'
  df['protocol'] = df['protocol'].astype('int')

  # sort by stream_id, protocol, time
  df = df.sort_values(by=['stream_id', 'protocol', 'time']).reset_index(drop=True)

  # merge packets into flows
  df =  df.groupby(['stream_id', 'protocol', 'IP_port'], as_index = False).\
              agg({'time':['min', 'max'],
                    'length':['sum', 'count']})
  
  df = df.drop(['stream_id'], axis=1)

  df.columns = ['protocol', 'IP_port', 'start', 'end', 'flow_length', 'pkt_num']
  
  # create duration of each flow
  df['duration'] = df['end'] - df['start']
  df = df.drop(['end', 'start'], axis=1)

  return df

def extract_flow_features(df):
  features_df = df.groupby(['IP_port'], as_index = False).\
    agg({'protocol':['mean', 'count'],
         'flow_length': ['mean'],
          'pkt_num': ['mean'],
         'duration': ['mean']
     })
    
  # rename columns
  features_df.columns = ['IP_port', 'protocol', 'flows_num', 'flow_length_mean', 'flow_pkt_num_mean', 'flow_duration_mean']

  return features_df
# Weights
def weights_reprocessing(df):
  df = df.drop(['stream_id', 'protocol', 'length', 'outgoing'], axis=1).reset_index(drop=True)
  return df
def weight(window_indx1, window_indx2):
  intersection = window_indx1.intersection(window_indx2)
  union = window_indx1.union(window_indx2)
  return len(intersection)
# Merge packet-based and flow-based features
def merge_features(df1, df2):
  features_df = pd.merge(df1, df2, on="IP_port")

  # sort by complete pkt number
  features_df = features_df.sort_values(by="complete_pkt_num", ascending=False).reset_index(drop=True)

  return features_df
# Main function to generate a graph from a traffic chunk
'''
input: mobile traffic chunck as a dataframe, N (the maximum nodes kept to build one graph), window (number of seconds used to build weight between two nodes)
output: two dataframe. One contains features of all nodes in a graph generated. The other contains weights between the nodes.
'''
def generate_features_weights(df, N, window):
  df = basic_reprocessing(df, N)

  #------------------------------ Generate features ------------------------------------
  # generate packet-based features
  out_df, in_df, complete_df = pkt_reprocessing(df)
  complete_df = extract_pkt_features(complete_df)
  out_df = extract_pkt_features(out_df, "out")
  in_df = extract_pkt_features(in_df, "in")
  pkt_features_df = pd.merge(pd.merge(complete_df, out_df, on="IP_port"), in_df, on="IP_port")
  # replace NaN by 0
  pkt_features_df = pkt_features_df.fillna(0)
  
  # generate flow-based features
  flow_df = flow_reprocessing(df)
  flow_features_df = extract_flow_features(flow_df)

  # merge packet-based and flow-based features df into a single features df
  features_df = merge_features(pkt_features_df, flow_features_df)

  #------------------------------ Generate weights ------------------------------------
  w_df = weights_reprocessing(df)
  w_df['time'] = (w_df['time']//window).astype('int')
  w_df = w_df.groupby('IP_port')['time'].agg(active= lambda x: set(x)).reset_index(drop=False)
  
  # create a dataframe of weights
  destination1_list = []
  destination2_list = []
  weight_list = []
  destinations = list(features_df['IP_port'])
  active_destinations = set()

  for i in range(len(destinations)):
    for j in range(i+1, len(destinations)):
      des1 = destinations[i]
      des2 = destinations[j]
      destination1_list.append(des1)
      destination2_list.append(des2)
      w = weight(w_df[w_df['IP_port'] == des1]['active'].values[0], w_df[w_df['IP_port'] == des2]['active'].values[0])
      weight_list.append(w)
      if w > 0:
        active_destinations = active_destinations.union({des1, des2})
  
  # get inactive destinations to remove
  inactive_destinations = list(set(destinations) - active_destinations)
  
  # create dataframe of edge weights
  weights_df = pd.DataFrame(
  {
  "source": destination1_list,
  "target": destination2_list,
  "weight": weight_list,
  }
  )

  weights_df = weights_df.sort_values(by="weight", ascending=False, ignore_index=True)

  # remove destinations that do not connect to any other destinations from features df
  features_df = features_df[~features_df['IP_port'].isin(inactive_destinations)]
  # add ip features
  features_df['ip1'] = features_df['IP_port'].apply(lambda x: int(x[0].split('.')[0]))
  features_df['ip2'] = features_df['IP_port'].apply(lambda x: int(x[0].split('.')[1]))
  features_df['ip3'] = features_df['IP_port'].apply(lambda x: int(x[0].split('.')[2]))
  features_df['ip4'] = features_df['IP_port'].apply(lambda x: int(x[0].split('.')[3]))

  # remove destinations that do not connect to any other destinations from weights df
  weights_df = weights_df[~weights_df['source'].isin(inactive_destinations) & ~weights_df['target'].isin(inactive_destinations)].reset_index(drop=True)

  # min-max normalize weights
  weights_df['weight'] = (weights_df['weight'] - weights_df['weight'].min())/(weights_df['weight'].max() - weights_df['weight'].min())

  return features_df, weights_df

'''
Input:
- app_src: folder that contain all traffic chunks (samples) of the app that we want to generate graphs
- filenames: list of filenames in app_src (training or testing)
Output: Generate graphs and save the graphs for all set of parameters (N and window) for just one app.
'''
def generate_graphs_one_app(app_src, filenames):

  feature_columns = ['IP_port'] + features + ['graph_id']
  weight_columns = ['source', 'target', 'weight', 'graph_id']
  
  features_df = pd.DataFrame([], columns=feature_columns)
  weights_df = pd.DataFrame([], columns=weight_columns)
  graph_id = 0
  # ----------------------------------------------------------------------------

  # loop over all traffic chunks of one app
  for filename in filenames:
    path = os.path.join(app_src, filename)
    df = pd.read_csv(path, index_col=0)
    df = df.sort_values(by='time')
      
    if df.empty:
      print('EMPTY')
      continue
        
    df['time'] = df['time'] - df['time'].iloc[0] # get base time

    #------------- generate one graph -----------------
    try:
      node_data, weights = generate_features_weights(df, N, window)
    except:
      print('WRONG')
      continue
      
    if weights.shape[0] > 1:
      graph_id = graph_id + 1 
      node_data['graph_id'] = graph_id
      weights['graph_id'] = graph_id

      #------------- add one graph into graphs of the app -----------------
      features_df = pd.concat([features_df, node_data], ignore_index=True)
      weights_df = pd.concat([weights_df, weights], ignore_index=True)
      #--------------------------------------------------------------------

  return [features_df, weights_df]
      
  print("================================================================END ONE APP================================================================")
'''
Input:
- A set of parameter: Duration and overlap
- Index: 0 if we want to generate graphs for training samples, 1 for testing samples 
'''
def generate_graphs(duration, overlap, index=0):

  # get train_test information
  path = os.path.join(root_path, '%d_%d'%(duration, overlap), 'train_test_info.json')
  with open(path, 'r') as f:
    train_test_info = json.load(f)
      
  samples_folder = os.path.join(root_path, '%d_%d'%(duration, overlap), 'samples')

  # initial a dictionary containing features and weights of graphs for all apps (app -> (features_df, weights_df))
  graphs = dict()

  idx = 0
  for app in os.listdir(samples_folder):
    if app.startswith('.') or app == 'desktop.ini':
      continue
    idx += 1
    print('Loading {} ... {}/{}'.format(app, idx, 101))
    
    app_src = os.path.join(samples_folder, app)
    filenames = train_test_info[app][index]

    graphs[app] = generate_graphs_one_app(app_src, filenames)
  
  return graphs

'''
Input: A dataframe containing features of nodes in a graph, dictionary contain mean-std of all features
Output: A dataframe of features after standardization
'''
def standardize_features(df, mean_std_dic):
  # standardize the features in dataframe
  for feature in mean_std_dic.keys():
    m, std = mean_std_dic[feature][0], mean_std_dic[feature][1]
    df[feature] = (df[feature] - m)/std
  
    # normalize ip feature
    df['ip1'] = df['ip1']/255
    df['ip2'] = df['ip2']/255
    df['ip3'] = df['ip3']/255
    df['ip4'] = df['ip4']/255
  
  return df


#Save graphs
def save_graphs(graphs, dataset='train_graphs'):
  
  #----------------------- create folder to save graphs ---------------------
  saved_graph_folder = os.path.join(root_path, '%d_%d'%(duration, overlap), dataset)
  if not os.path.exists(saved_graph_folder):
    os.mkdir(saved_graph_folder)

  N_folder = os.path.join(saved_graph_folder, 'N%d'%N)
  if not os.path.exists(N_folder):
    os.mkdir(N_folder)

  window_folder = os.path.join(N_folder, 't%d'%window)
  if not os.path.exists(window_folder):
    os.mkdir(window_folder)
      
  for app in graphs.keys():
    graph_app_folder = os.path.join(window_folder, app)      
    if not os.path.exists(graph_app_folder):
      os.mkdir(graph_app_folder)
    
    '''
    Save graphs for the app as two csv files (features.csv and weights.csv)
    '''
    features_path = os.path.join(graph_app_folder, 'features.csv')
    features_df = graphs[app][0]
    weights_path = os.path.join(graph_app_folder, 'weights.csv')
    weights_df = graphs[app][1]

    features_df.to_csv(features_path)
    weights_df.to_csv(weights_path)




## Generate and save graphs from the traffic chunks
features = ['complete_max', 'complete_min', 'complete_mean', 'complete_mad', 'complete_std', 'complete_var', 'complete_skew',
       'complete_kurt', 'complete_pkt_num', 'complete_10per', 'complete_20per', 'complete_30per', 'complete_40per', 'complete_50per', 
        'complete_60per', 'complete_70per', 'complete_80per', 'complete_90per', 'out_max', 'out_min', 'out_mean', 'out_mad', 'out_std',
        'out_var', 'out_skew', 'out_kurt', 'out_pkt_num', 'out_10per', 'out_20per', 'out_30per', 'out_40per', 'out_50per', 'out_60per',
        'out_70per', 'out_80per', 'out_90per', 'in_max', 'in_min', 'in_mean', 'in_mad', 'in_std', 'in_var', 'in_skew', 'in_kurt', 
        'in_pkt_num', 'in_10per', 'in_20per', 'in_30per', 'in_40per', 'in_50per', 'in_60per', 'in_70per', 'in_80per', 'in_90per', 
        'protocol', 'flows_num', 'flow_length_mean', 'flow_pkt_num_mean', 'flow_duration_mean', 'ip1', 'ip2', 'ip3', 'ip4'
       ]




## Config setting
root_path = '/Volumes/T9/map'

configurations = {
    'params': [(5, 3), (4, 2), (3, 1), (2, 0), (1, 0)], # (duration, Overlap)
    5: [(7,10),(10,10),(20,1),(20,5),(20,10),(30,10),(10000,10)], # (N, window)
    4: [(20,10)],
    3: [(20,10)],
    2: [(20,10)],
    1: [(20,10)],
    'k': [10, 20] # window
}


duration = 5
overlap = 3
N = 20
window = 10











#Generate graphs
# generate graphs for training dataset
training_graphs = generate_graphs(duration, overlap, index=0)
# generate graphs for testing dataset
testing_graphs = generate_graphs(duration, overlap, index=1)
#Standardize features of each node
# Get mean-std of each features in the training dataset

# define the initial empty dataframe
cols = ['IP_port'] + features + ['graph_id']
df = pd.DataFrame([], columns=cols)

# loop over train graphs
for app in training_graphs.keys():
  df_ = training_graphs[app][0]
  df = pd.concat([df, df_], axis=0)

# save mean and std of all featurs as dictionary
mean_std_dic = dict()
for feature in df.columns:
  if feature not in ['IP_port', 'ip1', 'ip2', 'ip3', 'ip4', 'protocol', 'graph_id']:
    mean_std_dic[feature] = (df[feature].mean(), df[feature].std())
# Standardization
for app in training_graphs.keys():
  training_graphs[app][0] = standardize_features(training_graphs[app][0], mean_std_dic)
  testing_graphs[app][0] = standardize_features(testing_graphs[app][0], mean_std_dic)






save_graphs(training_graphs, dataset='train_graphs')
save_graphs(testing_graphs, dataset='test_graphs')