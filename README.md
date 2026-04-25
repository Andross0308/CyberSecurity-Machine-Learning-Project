# CyberSecurity Machine Learning Project

### What does it do:
This project implements a Network Intrusion Detection System (NIDS) using Machine Learning. Receives the information of two CSVs that contains the information of network connections from the NSL-KDD DataSet, including if they are normal or anomalies, the purpose of the program is to analyze the information and determine if the file is normal or anomaly using 4 different models. 

### Used Technologies:
- Scikit-learn 1.8.0
- Xgboost 3.2.0
- Pandas 3.0.1
- Seaborn 0.13.2
- Matplotlib 3.10.8
- Numpy 2.4.3

### Project Structure:

├── data/ <br>
│   ├── KDDTest.arff <br>
│   └── KDDTrain.arff <br>

├── src/ <br>
│   ├── main.py <br>
│   ├── config.py <br>
│   ├── data_loader.py <br>
│   ├── PreProcessor.py <br>
│   ├── ModelManager.py <br>
│   └── Visualizer.py <br>
├── Results/ <br>
│   ├── Graph.png <br>
│   └── XBBMatrix.png <br>
└── README.md <br>
