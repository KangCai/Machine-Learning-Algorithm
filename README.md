# Machine-Learning-Algorithm

**注意: 每个文件只有开始的 class 是模型本身，其它代码都是用来测试的，每个模型的实现都在 100 行以内**

**Note: Only the class at the beginning of each file is the model itself, the rest of the code is for testing, and the implementation of each model is within 100 lines**

---

### 1. Logistic Regression

**File** - [logistic_regression.py](https://github.com/KangCai/Machine-Learning-Algorithm/blob/master/logistic_regression.py)

**Cost Function** -
 
 <img src="https://latex.codecogs.com/gif.latex?H(p,q)=-\sum_{i}^{&space;}[y^{(i)}\&space;log\&space;h_\theta(x^{(i)})&plus;(1-y^{(i)})\&space;log(1-h_\theta(x^{(i)})))]&space;" />

**Optimization Algorithm** - Gradient descent method

<img src="https://user-images.githubusercontent.com/11633302/67949923-a1cafb00-fc23-11e9-9249-363cfee52f87.png" width=80%/>

---

### 2. Support Vector Machine

**File** - [support_vector_machine.py](https://github.com/KangCai/Machine-Learning-Algorithm/blob/master/support_vector_machine.py)

**Example** -

<img src="https://user-images.githubusercontent.com/11633302/67949933-a394be80-fc23-11e9-8b7e-bbeaa6670c3e.png" width=60%/>

**Cost Function** -

<img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;max\&space;W(\alpha)&=\sum_{i=1}^{n}\alpha-\frac{1}{2}\sum_{i,j=1}^{n}y_iy_ja_ia_j(K(x_i,x_j))\\&space;s.t.\sum_{i=1}^{n}y_ia_i&=0&space;\\&space;0&space;\leq&space;a_i&space;\leq&space;C&(i=1,2...n)&space;\end{aligned}" />

**Optimization Algorithm** - Sequential minimal optimization (SMO)

---

### 3. Perception

**File** - [perception.py](https://github.com/KangCai/Machine-Learning-Algorithm/blob/master/perception.py)

**Example** -

<img src="https://user-images.githubusercontent.com/11633302/67949931-a2fc2800-fc23-11e9-8825-d5a61778148f.png" width=50%/>

---

### 4. Naive Bayes

**File** - [naive_bayes.py](https://github.com/KangCai/Machine-Learning-Algorithm/blob/master/naive_bayes.py)

**Example** -

<img src="https://user-images.githubusercontent.com/11633302/67949927-a2fc2800-fc23-11e9-8386-ead2ef4322bb.png" width=80%/>

---

### 5. K-Nearest Neighbor

**File** - [k_nearest_neighbor.py](https://github.com/KangCai/Machine-Learning-Algorithm/blob/master/k_nearest_neighbor.py) | [util_kd_tree.py](https://github.com/KangCai/Machine-Learning-Algorithm/blob/master/util_kd_tree.py)

**Example** -

<img src="https://user-images.githubusercontent.com/11633302/67949920-a099ce00-fc23-11e9-90f2-3720c5ce7830.png" width=50%/>

---

### 6. Decision Tree

**File** - [decision_tree.py](https://github.com/KangCai/Machine-Learning-Algorithm/blob/master/decision_tree.py)

**Optimization Algorithm** - Generalized Iterative Scaling (GIS)

**Example** -

<img src="https://user-images.githubusercontent.com/11633302/67949914-a0013780-fc23-11e9-9b99-8633380d3164.png" width=45%/>

---

### 7. Random Forest

**File** - [random_forest.py](https://github.com/KangCai/Machine-Learning-Algorithm/blob/master/random_forest.py) | | [decision_tree.py](https://github.com/KangCai/Machine-Learning-Algorithm/blob/master/decision_tree.py)

**Example** -

<img src="https://user-images.githubusercontent.com/11633302/67949932-a394be80-fc23-11e9-9b6d-e04c5bf85849.png" width=70%/>

---

### 8. Gradient Boosting Decision Tree

**File** - [gradient_boosting_decision_tree.py](https://github.com/KangCai/Machine-Learning-Algorithm/blob/master/gradient_boosting_decision_tree.py) | [decision_tree.py](https://github.com/KangCai/Machine-Learning-Algorithm/blob/master/decision_tree.py)

<img src="https://user-images.githubusercontent.com/11633302/67949915-a0013780-fc23-11e9-81a5-36bd7b6c9e9c.png" width=25%/>

---

### 9. Linear Discriminant Analysis

**File** - [linear_discriminant_analysis.py](https://github.com/KangCai/Machine-Learning-Algorithm/blob/master/linear_discriminant_analysis.py)

---

### 10. Maximum Entropy

**File** - [maximum_entropy.py](https://github.com/KangCai/Machine-Learning-Algorithm/blob/master/maximum_entropy.py)

**Example** -

<img src="https://user-images.githubusercontent.com/11633302/67949926-a2639180-fc23-11e9-8830-4b7da5dc5e1c.png" width=80%/>

---

### 11. Gaussian Discriminant Analysis

**File** - [gaussian_discriminant_analysis.py](https://github.com/KangCai/Machine-Learning-Algorithm/blob/master/gaussian_discriminant_analysis.py)

<img src="https://user-images.githubusercontent.com/11633302/67949918-a099ce00-fc23-11e9-8550-953cc6f344f1.png" width=50%/>

---

### 12. Principal Component Analysis

**File** - [principal_component_analysis.py](https://github.com/KangCai/Machine-Learning-Algorithm/blob/master/principal_component_analysis.py)

**Example** -

<img src="https://user-images.githubusercontent.com/11633302/67949929-a2fc2800-fc23-11e9-9c9d-1baf77b219be.png" width=50%/>

---

### 13. K-means

**File** - [kmeans.py](https://github.com/KangCai/Machine-Learning-Algorithm/blob/master/kmeans.py) | [util_kd_tree.py](https://github.com/KangCai/Machine-Learning-Algorithm/blob/master/util_kd_tree.py)
