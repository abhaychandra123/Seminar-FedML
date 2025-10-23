# **Sampling in Federated Learning (FL) p**

## **1\. Problem Definition**

Incorporation of new clients with potentially diverse data distributions and computational capabilities, poses a significant challenge to the stability and efficiency of these distributed learning networks.

## **2\. Paper's contrib**

This paper outlines strategies for effective client selection strategies and solutions for ensuring system scalability and stability. Using the example of images from optical quality inspection, it offers insights into practical approaches.

## **3\. Intro**

FL allows to collaboratively train a shared model while retaining all training data locally on the devices.

### **Introduction of New Devices**

* **How to handle this?**  
* **Why is it good?**  
  * to enhance the diversity of the data pool, thereby improving the robustness and performance of the collective learning process. Each new client contributes unique insights and perspectives, enriching the network and making the model more comprehensive and better suited for real-world applications.

### **How to Tackle Sparsity?**

1. either generate synthetic  
2. breaking down existing data silos can increase both the volume and diversity of available data

**HERE FL HELPS**

### **Example: Manufacturing Data**

* Data from sensors, processes, quality control.  
* BUT sensitive  
* *(then gives summary of FedAvg)*

## **4\. Literature Review: Client Selection Strategies**

### **A. Client Selection Strategies**

#### **Initial Approach: Random Sampling**

* STARTED with random sampling  
* “The authors observed that incorporating additional clients beyond a certain threshold yields diminishing returns as the additional communication overhead exceeds the performance gains, but did not investigate further into client selection strategies.”  
* This fails to account for data heterogeneity.  
* **THIS LEADS TO:** incomplete communication rounds, or the risk of over-sampling frequently provided data.

#### **Resource-Aware Selection**

* on basis of available computation power, the network connection as well as a possible battery operation

#### **Performance-Aware Selection**

* consider accuracy or training loss in addition to resource constraints  
* One paper \[26\] show clients with HIGHER LOCAL LOSS can IMPROVE CONVERGENCE, as they have bigger potential for imp

#### **Contribution-Based Selection**

* on basis of client for selection \[27\]  
* \[28\] use SHAPLEY value ??  
* \[29\] looks at gradient space and data space for each client  
* \[30\] converge even if rounds are interrupted by C leav or join  
* \[31\] weight selection strategies in federated learning, OEWS

### **Clustering Approaches**

#### **Why use clustering?**

1. form clusters for further training as separate groups, thereby improving local accuracy.  
2. Others use clustering to select a diverse set of clients for the next training round, with the goal of developing a more robust global model.

#### **Ref \[32\]**

* choose cluster based on  
  1. resource: cpu or ram  
  2. data : quality  
* DBSCAN-based selection ??  
  * of clients performs better than randomly selecting the same number of clients.  
* **DRAWBACK?** rely on correctness of info, only tested on MNIST (HEREEEEE CONTRIII)

#### **Ref \[33\]**

* cluster based on  
  * label of local faults of computer room, use Hamming distance to deter simi  
* **ONLY useful if:** all possible error cases are predefined and displayed in a binary representation.  
* **NOT SUITABLE:** for recognizing similarities from only the underlying data or weights.

#### **Ref \[35\] \- CLUSTERED FL (CFA)**

* clients are iter bipartitioned based on COSINE of grad changes  
* **ADV:**  
  * no need to know num of clust in adv  
  * scalable interpretability of results  
* **DOWNSIDE:**  
  1. recursive bipartitioning clustering demands significant computational and communication resources  
  2. clients are randomly selected each round, the algorithm can be computationally inefficient and may even completely fail, (\[37\] tell why)

#### **Ref \[38\]**

* similar to CFA  
* but performs clustering after predetermined no of comm rounds  
* init global \> fine-tune of data of all C \> create cluster with this info and assign clients

#### **Ref \[39\] (by GHOSH sir only)**

* improves efficiency of CFA  
* **ITERATIVE FCA**  
* rand gen CLUSTER CENTROID (meaning??) \-\> assign clients to minimize defined loss func  
* ACC increase BUT  
* **DOWNSIDE:**  
  1. comm cost unchanged  
  2. comp effort is more even as all update must be transmitted and cluster reformed in each round  
  3. success depended lot on init of cluster centroid

#### **Ref \[40\]**

* Cosine similarity followed by affinity propagation clustering has been shown to effectively cluster clients in image classification settings,

#### **Limitations of Academic Solutions**

* there are some more academic soln but fail on real data (eg: clients may join or leave the system at any time, or introduce entirely new data, adding layers of complexity that are not fully accounted for in controlled, static environments.)

## **5\. This Paper's Contribution**

### **Assumptions**

1. disregard the hardware limitations of the clients, as it is assumed that companies have sufficient computing resources and stable network connections. model perf \>\> duration of comm round  
2. assume a cross-silo problem definition, characterized by a relatively small number of clients

### **What will they do?**

* examining various client selection techniques to lay the groundwork for improving the integration of new clients in future work.  
* **IMP CAUSE:** careless integration of new clients into a stable system can lead to unwanted performance drops until the system stabilizes again with new clients.

### **Methodology**

* use on real world data par and NNs  
* simple dataset first then IMAGENET  
* analyzing only the final layer’s activations is often sufficient for effective clustering

### **Dataset**

* cabins of mini trucks: with diff parts , material ,  
* testing purposes, we split the data set so that each client has only one color and one type of windshield, in addition to the class without windshield for every client.  
* ALSO simulate newcomers

### **Specifications**

* A pretrained EfficientNet B4 \[44\] was used as the frozen backbone, with one fully connected layer replaced to match the number of classes. The training was done with SGD as the optimizer with a learning rate of 0.001, 200 communication rounds and 5 local epochs.

## **6\. Results in Paper**

### **3 Cluster Select Methods:**

1. Agglomerative clustering  
2. HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) \[45\]  
3. BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) \[46\]

### **Observations**

* RESULTS in ppr  
* fluctuations in BIRCH maybe due to sensitivity to param, high dimensionality \+ small sample size

### **Why Agglo GOAT (among these 3\)**

* Agglomerative clustering performs best in our scenario due to its ability to capture complex hierarchical structures in relatively small, but high-dimensional dataset. Its flexibility in distance metrics, robustness to small sample sizes, and stability in high-dimensional spaces make it well-suited to effectively clustering the final layer activations of your neural network.

### **Scenarios & Evaluation**

* Saved Weights from 2 scenarios:  
  1. one in which all clients participated in every communication round, and another where  
  2. the experiment began with 8 clients (representing all classes but limited to red and blue colors), gradually adding new clients after an initial training phase.  
* **OBS:**  
* **EVAL OF CLIENT SELECTION STRAT:**  
  * In comparison to random selection, we evaluate metric-based methods (specifically, highest training loss) and cluster-based methods (focussing on agglomerative clustering).’

## **7\. Conclusion**

As we move forward, it will be crucial to focus on developing more nuanced client selection and clustering techniques that can further enhance the performance and fairness of FL systems and extend it to other architectures.  
