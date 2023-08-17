<div style="page-break-after: always; visibility: hidden">\pagebreak</div>

<p class="bigger-heading">Introduction</p>

In this project, we will be reviewing various classification techniques as well as their variants as to find out which is the best for our use case

<br>

**Objective of analysis:** Develop a robust and accurate predictive model that can determine whether a patient's breast cancer is benign or malignant. This predictive capability will aid in early and precise diagnosis, contributing to improved patient care and outcomes

**Objective of the machine learning models:** Predict the malignancy or benign nature of breast cancer in patients by analysing various tumor attributes such as cell size, cell shape, clump thickness, and other pertinent factors

<br>

Code as well as its output is given at the end. To view the raw files, visit [this](https://github.com/msr8/classification-project) github repository

<!-- To predict wether in a patient, the scale of the breast cancer is benign or malignant, given various attributes about the tumour (size and shape of cell, clump thickness, etc) -->




<div style="page-break-after: always; visibility: hidden">\pagebreak</div>





<p class="bigger-heading">About the data</p>

**Title:** Wisconsin Breast Cancer Database (January 8, 1991)
**Number of Instances:** 699
**Number of Attributes:** 10 plus the class attribute
**Missing attribute values:** 16 (denoted by "?")
**Class distribution:** benign (458 ie 65.5%) and malignant (241 ie 34.5%)

<br>

## Attribute Information

| Attribute                   | Role    | Domain             | Missing values                            |
| --------------------------- | ------- | ------------------ | ----------------------------------------- |
| Sample code number          | ID      | id number          | <span style="color: #ff0000">false</span> |
| Clump Thickness             | Feature | 1 - 10             | <span style="color: #ff0000">false</span> |
| Uniformity of Cell Size     | Feature | 1 - 10             | <span style="color: #ff0000">false</span> |
| Uniformity of Cell Shape    | Feature | 1 - 10             | <span style="color: #ff0000">false</span> |
| Marginal Adhesion           | Feature | 1 - 10             | <span style="color: #ff0000">false</span> |
| Single Epithelial Cell Size | Feature | 1 - 10             | <span style="color: #00ff00">true </span> |
| Bare Nuclei                 | Feature | 1 - 10             | <span style="color: #ff0000">false</span> |
| Bland Chromatin             | Feature | 1 - 10             | <span style="color: #ff0000">false</span> |
| Normal Nucleoli             | Feature | 1 - 10             | <span style="color: #ff0000">false</span> |
| Mitoses                     | Feature | 1 - 10             | <span style="color: #ff0000">false</span> |
| Class                       | Target  | benign, malignant  | <span style="color: #ff0000">false</span> |

<br>

## Sources

Dr. William H. Wolberg (physician)
University of Wisconsin Hospitals
Madison, Wisconsin
USA

Donor: Olvi Mangasarian (mangasarian@cs.wisc.edu)

Received by David W. Aha (aha@cs.jhu.edu)

<br>

## Citations

O. L. Mangasarian and W. H. Wolberg: "Cancer diagnosis via linear programming", SIAM News, Volume 23, Number 5, September 1990, pp 1 & 18

William H. Wolberg and O.L. Mangasarian: "Multisurface method of pattern separation for medical diagnosis applied to breast cytology", Proceedings of the National Academy of Sciences, U.S.A., Volume 87, December 1990, pp 9193-9196

O. L. Mangasarian, R. Setiono, and W.H. Wolberg: "Pattern recognition via linear programming: Theory and application to medical diagnosis", in: "Large-scale numerical optimization", Thomas F. Coleman and YuyingLi, editors, SIAM Publications, Philadelphia 1990, pp 22-30

K. P. Bennett & O. L. Mangasarian: "Robust linear programming discrimination of two linearly inseparable sets", Optimization Methods and Software 1, 1992, 23-34 (Gordon & Breach Science Publishers)





<div style="page-break-after: always; visibility: hidden">\pagebreak</div>





<p class="bigger-heading">Code Summary</p>

The full code as well as its output is given at the end. View it as a jupyter notebook [here](https://github.com/msr8/classification-project/blob/main/main.ipynb)

<br>

## Preprocessing

- Since there are only 16 records in the dataset containing missing values, **they are dropped**
- In the class column, **binary encoding** is performed by replacing "benign" with 0 and "malignant" with 1

<br>

## Splitting data

- Data is **split into training and testing sets** (80% and 20% respectively) using the `sklearn.model_selection.train_test_split` function

<br>

## K Nearest Neighbours (KNN)
- **A KNN classifier object is created** of the `sklearn.neighbours.KNeighborsClassifier` class
- The model **is trained** on the training data using the `.fit` method
- **Predictions are made** from the features of the training data using the `.predict` method
- **Accuracy is calculated** using the `sklearn.metrics.accuracy_score` function
- All the above steps are **repeated** for values of K from 1-100
- The accuracies of the models with varying Ks is are **plotted** onto a line graph using the `matplotlib.pyplot.plot` function

<br>

## Support Vector Machines (SVM)
- **An SVM classifier object is created** of the `sklearn.svm.SVC` class
- The model **is trained** on the training data using the `.fit` method
- **Predictions are made** from the features of the training data using the `.predict` method
- **Accuracy is calculated** using the `sklearn.metrics.accuracy_score` function
- All the above steps are **repeated** for different kernels (linear, polynomial, radial basis function, and sigmoid)
- The accuracies of the models with varying kernels is are **plotted** onto a bar graph using the `matplotlib.pyplot.plot` function

<br>

## Decision Trees
- **A DTree classifier object is created** of the `sklearn.tree.DecisionTreeClassifier` class
- The model **is trained** on the training data using the `.fit` method
- **Predictions are made** from the features of the training data using the `.predict` method
- **Accuracy is calculated** using the `sklearn.metrics.accuracy_score` function
- A **visualisation** of the decision tree is plotted using the `sklearn.tree.plot_tree` function
- All the above steps are **repeated** for different criterions (linear, polynomial, radial basis function, and sigmoid)
- The accuracies of the models with varying criterions is are **plotted** onto a bar graph using the `matplotlib.pyplot.plot` function





<div style="page-break-after: always; visibility: hidden">\pagebreak</div>





<p class="bigger-heading">Findings & Analysis</p>

## K Nearest Neighbour (KNN)
![KNN](https://raw.githubusercontent.com/msr8/classification-project/main/assets/knn.png)

The accuracy remains above 0.9, and doesn't fluctuate much in higher values of `k`

During the testing, in k ranging from 1 to 100, the lowest accuracy was ~91.2% and the highest accuracy was ~95.7%

<div style="page-break-after: always; visibility: hidden">\pagebreak</div>

## Support Vector Machine (SVM)
![SVM](https://raw.githubusercontent.com/msr8/classification-project/main/assets/svm.png)

The accuracy is very high using linear, polynomial, or rbf (radial basis function) kernels, but worse than random when using the sigmoid kernel

During the testing, the highest accuracy was obtained by the polynomial and rbf kernels (~94.9%), and the worst accuracy was obtained by the sigmoid kernel (~39.4%)

<div style="page-break-after: always; visibility: hidden">\pagebreak</div>

## Decision Trees

<!-- <div style="display:flex; justify-content: center;">
    <img width=70% src="https://raw.githubusercontent.com/msr8/classification-project/main/assets/dtree.png" alt="DTree">
</div> -->

![DTree](https://raw.githubusercontent.com/msr8/classification-project/main/assets/dtree.png)

The accuracy was very high regardless of which criterion was picked

During the testing, the highest accuracy was obtained by the entropy and log_loss criterion (~95.6%), while the worst accuracy was obtained by the gini criterion (~93.4%)

The visualisation of the models created by the different criterions is given on the next page (zoom in to read the text)

![DTree](https://raw.githubusercontent.com/msr8/classification-project/main/assets/dtree-representations.png)
<!-- ![DTree](/Users/mark/Documents/github/classification-project-github/assets/dtree-representations%20copy.png) -->





<div style="page-break-after: always; visibility: hidden">\pagebreak</div>





<p class="bigger-heading">Conclusion</p>

In conclusion, although choosing any model (except SVM with the sigmoid kernel) would be a good choice, SVM with a linear kernel is the best choice for our data set since it achieved the highest accuracy (ie ~96.4%)

<br>

## Possible flaws

- **Binary classes:** Since there were only two classes in the dataset, even randomly selecting data could lead to an accuracy of 50%
- **Lack of outliers:** There are not many outliers present in the data, which can cause the models to classify an outlier incorrectly
- **Low number of malignant tumours:** The ratio of the records of benign tumours to malignant tumours is almost 2:1, which can cause the models to classify an malignant tumour incorrectly

<br>

## Suggestions for next steps

- Redo our analysis with multiple random states
- Use different training and testing splits