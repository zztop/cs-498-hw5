import os

import numpy as np
import pandas as pd
from scipy.cluster import vq
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    all_data_set = None
    all_data_cluster = None
    class_dataset = {}
    class_file_data = {}
    all_labels = []
    split = 20
    k = 300
    for path, subdirs, files in os.walk('./HMP_Dataset'):
        if path == './HMP_Dataset' or 'MODEL' in path:
            continue
        # if 'soup' in path or 'Brush' in path or 'Walk' in path:
        #     pass
        # else:
        #     continue

        label = os.path.split(path)[1]
        all_labels.append(label)
        for file in files:
            # if 'Accelerometer-2012-06-06-09-04-41-walk-m5.txt' in file or \
            #         'Accelerometer-2011-05-30-20-57-19-walk-f1.txt' in file:
            #     continue
            print(os.path.join(path, file))
            file_data = pd.read_csv(os.path.join(path, file),
                                    names=None,
                                    na_values=['?'], delimiter=r"\s+").astype(float)

            no_of_split = int(file_data.iloc[:, 0].shape[0] / split)
            remainder = file_data.iloc[:, 0].shape[0] % split
            clean_split_data = file_data.iloc[:file_data.shape[0] - remainder]
            split_all_data = np.vsplit(clean_split_data, no_of_split)
            # split_all_data.append(file_data.iloc[-32:])
            class_data = None
            for i in range(0, len(split_all_data)):
                data = pd.concat(
                    [split_all_data[i].iloc[:, 0].T, split_all_data[i].iloc[:, 1].T, split_all_data[i].iloc[:, 2].T],
                    axis=0)
                if class_data is None:
                    class_data = data.to_frame().T
                else:
                    class_data.loc[i] = data.values
            class_file_data[(label, file)] = class_data

            # kmeans=vq.kmeans(class_data.values,40,2)
            # kmeans = KMeans(n_clusters=40)
            if label not in class_dataset:
                class_dataset[label] = class_data
            else:
                class_dataset[label] = class_dataset[label].append(class_data)
        new_frame = pd.DataFrame(class_dataset[label])
        new_frame["label"] = [label for x in range(0, new_frame.shape[0])]
        # new_frame['label'] = new_frame.apply(lambda row: label)
        if all_data_set is None:
            all_data_set = new_frame
        else:
            all_data_set = pd.concat([all_data_set, new_frame])

    all_data_cluster, distortion = vq.kmeans(all_data_set.drop(all_data_set.columns[[split * 3]], axis=1), k)

    all_hist = None

    for data_file in class_file_data:

        code_book = vq.vq(class_file_data[data_file], all_data_cluster)[0]
        result = pd.DataFrame(code_book).groupby(0).sum()
        code_list = {}
        for i in range(0, k):
            code_list[i] = 0
        for i in range(0, code_book.shape[0]):
            idx = code_book[i]
            code_list[idx] += 1
        code_list_frame = pd.DataFrame(code_list.values()).T
        code_list_frame["label"] = all_labels.index(data_file[0])
        if all_hist is None:
            all_hist = code_list_frame
        else:
            all_hist = all_hist.append(code_list_frame)
        # all_hist["label"] = [data_file[0] for x in range(0, len(code_list))]

    train_features, test_features, train_labels, test_labels = train_test_split(
        all_hist.drop(all_hist.columns[[k]], axis=1),
        all_hist.iloc[:, -1], test_size=0.20,
        random_state=42)

    rf = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf.fit(train_features, train_labels)
    predictions = rf.predict(test_features)

    yay = 0
    for idx, lbl in enumerate(test_labels):
        if lbl == predictions[idx]:
            yay += 1

    output = pd.DataFrame(confusion_matrix(test_labels, predictions)).to_html(
        'confusion_matrix_k' + str(k) + '_split_' + str(split) + '.html')

    print('Accuracy :{}'.format(100 * yay / len(test_labels)))

    print('done')
