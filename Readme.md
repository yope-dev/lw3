# Artificial Intelligence of Control Systems in Aircraft and Moving Vehicles

## **Andrii Popovych** СУ-402Ба Laboratory work №3

## About Laboratory Work

text

## Installing and execute

To install all necessary libraries, execute the following commands:

```bash
cd current/folder
pip3 install -r requirements.txt
python3 main.py
```

## Results

Shape of the dataset: `(150, 5)`

### Dataset Figure

![Dataset Figure](./images/dataset_figure.png "Dataset Figure")

### Histagram Dataset

![Dataset Figure](./images/histagram_dataset.png "Dataset Figure")

### Scatter Matrix

![Dataset Figure](./images/scatter_matrix_dataset.png "Dataset Figure")

| Name                          | Aaccuracy, % | Array of scores of the estimator for each run of the cross validation. |
 :---            |    ----: |   ---: |
| Logistic Regression           | 94.16        | 0.06508541396588878                                                    |
| Linear Discriminant Analysis  | 97.5         | 0.03818813079129868                                                    |
| KNeighbors Classifier         | 95.83        | 0.04166666666666669                                                    |
| Decision Tree Classifier      | 95.0         | 0.055277079839256664                                                   |
| Gaussian Naive Bayes          | 95.0         | 0.05527707983925667                                                    |
| Support Vector Classification | 98.33        | 0.03333333333333335                                                    |

Accuracy score: `96.66666666666667%`

Confusion Matrix:

```text
[[11  0  0]
 [ 0 12  1]
 [ 0  0  6]]
```

classification_report:

|                 | precision | recall | f1-score | support |
| :---            |    :----: |   ---: |      ---:|   ---:  |
| Iris-setosa     | 1.00      | 1.00   | 1.00     | 11      |
| Iris-versicolor | 1.00      | 0.92   | 0.96     | 13      |
| Iris-virginica  | 0.86      | 1.00   | 0.92     | 6       |
| accuracy        |           |        | 0.97     | 30      |
| macro avg       | 0.95      | 0.97   | 0.96     | 30      |
| weighted avg    | 0.97      | 0.97   | 0.97     | 30      |

## Project structure

| File Name        | Description                           |
|------------------|---------------------------------------|
| main.py          | The main script of the project        |
| requirements.txt | Required libraries and their versions |
| iris.csv         | Dataset                               |
| images/          | Folder for all images                 |
| Readme.md        | Project description file              |
