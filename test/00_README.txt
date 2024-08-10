README - Testing the Model

Step 1: Set Up the Directory Structure
--------------------------------------
1. Create a main directory that will serve as the root folder for your project.
2. Inside this root folder, create a subfolder named `raw_data`.
3. Within the `raw_data` folder, create additional subfolders, each named after a specific application.
4. Place the pcap files containing the application traffic inside their corresponding subfolders.

Your folder structure should look like this:

root_folder/
│
├── raw_data/
│   ├── App1/
│   │   ├── capture1.pcap
│   │   ├── capture2.pcap
│   │   └── ...
│   ├── App2/
│   │   ├── capture1.pcap
│   │   ├── capture2.pcap
│   │   └── ...
│   └── ...

Step 2: Run the Preprocessing Function
--------------------------------------
- Replace `root_path` in the script with the path to your root folder.
- Run the preprocessing function. This will convert all the pcap files in the `raw_data` folder to CSV format and place them into a sources folder.

Step 3: Execute the Python Scripts
----------------------------------
- After preprocessing, run the following Python scripts in order, making sure to replace paths with your root folder:

  1. `01_generating_sample.py`
  2. `02_generating_train_test.py`
  3. `03_generating_graphs.py`
  4. `04_train_GNN.py`

These scripts will generate samples, create training and testing datasets, generate graphs, and finally train the GNN (Graph Neural Network) model.
