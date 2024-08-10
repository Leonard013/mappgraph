
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from stellargraph import StellarGraph
import pandas as pd
import numpy as np
import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from stellargraph import StellarGraph
from stellargraph import datasets
from sklearn import model_selection
from IPython.display import display, HTML
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten,  BatchNormalization
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
## Loading graphs for training and testing
'''
Load graphs for one app
Input: app and folder that contains graphs of the app
Output: List of graphs (StellarGraph objects) and List of labels
'''
def graphs_one_app(app, graphs_folder):
  graphs = []

  app_graph_folder = os.path.join(graphs_folder, app)
  features_path = os.path.join(app_graph_folder, 'features.csv')
  weights_path = os.path.join(app_graph_folder, 'weights.csv')



  features_df = pd.read_csv(features_path, index_col=0)
  weights_df = pd.read_csv(weights_path, index_col=0)

  if features_df.empty:
        print(f"No data for {app} in features.csv")
        return [], []

  graph_num = features_df['graph_id'].iloc[-1]
  # loop over all graphs of the app
  for i in range(1, graph_num+1):
    feature_df = features_df[features_df['graph_id'] == i]
    feature_df = feature_df[['IP_port'] + features + ['graph_id']]
    feature_df = feature_df.set_index('IP_port')
    
    weight_df = weights_df[weights_df['graph_id'] == i].reset_index(drop=True)

    # drop graph_id column
    feature_df = feature_df.drop(['graph_id'], axis=1)
    weight_df = weight_df.drop(['graph_id'], axis=1)

    if weight_df.shape[0] > 0:
      graph = StellarGraph(feature_df, weight_df)
      graphs.append(graph)

  labels = [app]*graph_num
    
  return graphs, labels
'''
Load all graphs
Input: folder that contains graphs
Output: List of graphs (StellarGraph objects), List of graph_labels (dummy values) and List of labels (names of app)
'''
def generate_graphs(graphs_folder):
  # build a list of graphs and labels: note that only apply for more than 2 classes
  li = []
  labels = []
  idx = 0

  for app in apps:
    
    idx += 1
    print('Loading {} ... {}/{}'.format(app, idx, len(apps)))
    
    one_app_graphs, one_app_labels = graphs_one_app(app, graphs_folder)
    li.extend(one_app_graphs)
    labels.extend(one_app_labels)
    

  graph_labels = pd.get_dummies(labels)
  graphs = li

  print('...............................................................')

  return graphs, graph_labels, labels

## Training
class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 10 == 0 & epoch > 0:  # or save after some epoch, each k-th epoch etc.
            self.model.save(os.path.join(models_folder, "model_{}.hd5".format(epoch)))


features = ['complete_max', 'complete_min', 'complete_mean', 'complete_mad', 'complete_std', 'complete_var', 'complete_skew',
       'complete_kurt', 'complete_pkt_num', 'complete_10per', 'complete_20per', 'complete_30per', 'complete_40per', 'complete_50per', 
        'complete_60per', 'complete_70per', 'complete_80per', 'complete_90per', 'out_max', 'out_min', 'out_mean', 'out_mad', 'out_std',
        'out_var', 'out_skew', 'out_kurt', 'out_pkt_num', 'out_10per', 'out_20per', 'out_30per', 'out_40per', 'out_50per', 'out_60per',
        'out_70per', 'out_80per', 'out_90per', 'in_max', 'in_min', 'in_mean', 'in_mad', 'in_std', 'in_var', 'in_skew', 'in_kurt', 
        'in_pkt_num', 'in_10per', 'in_20per', 'in_30per', 'in_40per', 'in_50per', 'in_60per', 'in_70per', 'in_80per', 'in_90per', 
        'protocol', 'flows_num', 'flow_length_mean', 'flow_pkt_num_mean', 'flow_duration_mean', 'ip1', 'ip2', 'ip3', 'ip4'
       ]


## Config setting
N = 20
t = 10
k = 10  # the number of rows for the output tensor (k = 10, 20)
T = 5
overlap = 3 # note: overlap depends on T

root_path = 'path/to/dataset' # path to the dataset folder
apps = os.listdir(os.path.join(root_path, '%d_%d/samples'%(T, overlap)))


def main():

  train_graphs_folder = os.path.join(root_path, '%d_%d/train_graphs/N%d/t%d'%(T, overlap, N, t))
  test_graphs_folder = os.path.join(root_path, '%d_%d/test_graphs/N%d/t%d'%(T, overlap, N, t))


  train_graphs, train_graph_labels, _ = generate_graphs(train_graphs_folder)
  test_graphs, test_graph_labels, _ = generate_graphs(test_graphs_folder)
  train_size = len(train_graphs)

  graphs = train_graphs + test_graphs
  graph_labels = train_graph_labels.append(test_graph_labels, ignore_index=True)

  test_graph_labels = graph_labels[train_size:]
  generator = PaddedGraphGenerator(graphs=graphs)
  ## Build GNN
  layer_sizes = [1024, 1024, 1024, 512]

  dgcnn_model = DeepGraphCNN(
      layer_sizes=layer_sizes,
      activations=["tanh", "tanh", "tanh", "tanh"],
      k=k,
      bias=False,
      generator=generator,
  )
  x_inp, x_out = dgcnn_model.in_out_tensors()

  #------------------------------------------------------------------------------
  x_out = Conv1D(filters=256, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
  x_out = MaxPool1D(pool_size=2)(x_out)

  x_out = Conv1D(filters=512, kernel_size=5, strides=1)(x_out)

  x_out = Flatten()(x_out)

  x_out = Dense(units=1024, activation="relu")(x_out)
  x_out = Dropout(rate=0.25)(x_out)

  predictions = Dense(units=len(apps), activation="softmax")(x_out)

  #------------------------------------------------------------------------------

  model = Model(inputs=x_inp, outputs=predictions)

  # using exponentialDecay to decrease the learning rate after 10 epochs
  # lr =  initial_lr * decay_rate ^ (step / decay_steps)
  batch_size = 256
  decay_epoch = 20

  lr_schedule = keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=0.0001,
      decay_steps=(train_size//batch_size)*decay_epoch,
      decay_rate=0.9)
  optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

  model.compile(
      optimizer=optimizer, loss=categorical_crossentropy, metrics=["acc"],
  )
  model.summary()
  gen = PaddedGraphGenerator(graphs=graphs)

  train_gen = gen.flow(
      list(train_graph_labels.index - 1),
      targets=train_graph_labels.values,
      batch_size=256,
      symmetric_normalization=False,
  )

  test_gen = gen.flow(
      list(test_graph_labels.index - 1),
      targets=test_graph_labels.values,
      batch_size=256,
      symmetric_normalization=False,
  )

  # create and use callback:
  saver = CustomSaver()

  epochs = 150
  history = model.fit(
      train_gen, callbacks=[saver], epochs=epochs, verbose=1, validation_data=test_gen, shuffle=True,
  )
  ## Testing
  # use the model to predict on testing data and get predicted labels
  pred_prob = model.predict(test_gen)
  pred_labels = np.argmax(pred_prob, axis=-1) 
  pred_labels = [str(x+1) for x in list(pred_labels)]

  # get the true labels of the testing data
  test_labels = np.argmax(test_graph_labels.values, axis=-1)
  test_labels = [str(x+1) for x in list(test_labels)]

  # show the result as classification report
  print(classification_report(test_labels, pred_labels, target_names=apps, digits=4))
  sg.utils.plot_history(history,
                        return_figure=False)
  plt.savefig('/home/leonardo/GitHub/mappgraph/results/train_val_plot_%d_%d_%d_%d_%d.png'%(N, t, T, overlap, k))


if __name__ == "__main__":
    main()

