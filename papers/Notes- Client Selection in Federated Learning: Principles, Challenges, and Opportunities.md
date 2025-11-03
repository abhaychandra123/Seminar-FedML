# Client Selection in Federated Learning

Random sampling may not fully exploit the local updates from heterogeneous clients, resulting in lower model accuracy, slower convergence rate, degraded fairness, etc.

started off with normal intro to Fl

FL categories : cross-silo FL and cross-device (??)
FL. Cross-silo FL targets collaborative learning among
several organizations, while cross-device FL targets ML across
large populations, e.g., mobile devices

better client selection leads to :
1. Improved model accuracy
2. Faster convergence rates
3. Enhanced fairness among clients
4. Fairness
5. Robustness (??)
6. Reduced training overheads

Research questions :
1) How does FL clients behave that affects the client
selection performance? (Section III).
2) What is the principle behind existing FL client selection
algorithms to prioritize FL clients? (Section IV).
3) What is the current practice to implement an FL client
selection algorithm? (Section V).
4) What are the challenges to realize an effective FL client
selection algorithm? (Section VI).
5) What are the research opportunities to boost
performance of FL client selection? (Section VII).


## Literature Review
basically focusing of 'utility of client'  and based on that selecting 

## heterogenity
FL clients exhibit system and statistical heterogeneity. 
### System: based on hardware
1. system capability: time to run the AI model differs
2. comm capability: diff transmission (LTE vs wifi, location)
3. other: 

### Statistical: unbalanced data, non-IID, varying data quality
1. massively distributed: large number of clients with small local data, like gboard
2. unbalanced data: different clients have different amount of data
3. non iid data: data distribution varies across clients (iid meaning: Each client’s data does not represent the overall distribution)
   
## Selection Strategies
find ulities based on statistical and system heterogeneity and mulitiply to get overall utility score
### statistical utility
based on data sample based and model based
1. data sample based: client’s local data to quantify
   1. NUM of data samples per client, work when IID *app-one* (see (2) in paper)
   2. high score to clients that are divergent far from the model. ie , whose L2 norm of the L2-norm of the data sample k’sgradient in bin Bi of FL client i.  is HIGH  BAD: compute more *app-two*
   3. data sample wigth large loss -> large gradient -> more imporant , better coz loss is available so compute overhead ruduce *app-three*
   4. just sum the loss , no need to normalize *app-four*
2. model based: compare MODEL weights/gradients
   1.  normalized model divergence : avg diff bt (model w of client i) and global model, model div less then update insignificant  *app-five*
   2.  % of same sign w bt global and local model : mlower percentage of same- sign weights results in better communication efficiency upon converges.(??) counter intuitive , as this shows *app-six*
   3.  Clients whose weight updates differ greatly from the global model are considered more important for selection because they bring new, diverse information that helps the model learn better and avoid stagnation.
   --  difference in the direction and magnitude of client updates from the global model *app-seven*
   4. compare change of local model before and after local training : large change more imporant *app-eight*
   5.  L2-norm of the model’s gradients *app-nine*
   6.  dot product of the local and global model gradients : small dot product means more imporant , those whose updates are more different from the global model, as they provide new information and help avoid stagnation *app-ten*
   
### system utility
based on client hardware configuration
1. DEADLINE BASED: set to avoid slow clients
    1.  HARD deadlien :clients with a time t longer than T are removed from the  FL aggregation, where t is total round time of client i that includes the local training, transmission, compression, etc. *app-eleven*
    2.  SOFT deadline :  equals 1 (i.e., no punishment) for nonstragglers and increases exponentially for stragglers. *app-twelve*

T can represent other metrics , like if target the computation speed, in FLOPs; target transmission bandwidth, then in Mb/s

(??) any technique based on RAM ,CPU, battery , etc?

### how to schedule
see not practical to calc utility after each training round, nly be deter- mined after it has participated in a training round. Therefore, a mainstream approach is to forecast a client’s utility along the training stage and update/rectify its utility measure once it is selected to join the training round (??)

**system util X stat util X fairness util X robustness util**  like this u can add more factors too (??) INSTEAD OF MULTIPLYING , CAN WE DO OTHER OPERATIONS, like WEIGHTED SUM, etc? (??)
exists! eiffel[24] UTIL= adding the loss value of its local model, the local data size, the computation power, the resource demand, and the age of update, with adjustable weights

## Implementation 
### Simulation
1.  synthetic data (MNIST, CIFAR10, FEMNIST, Shakespeare) and partitions to diff clients
2.   real world data (LEAF, Google speech commands, Google keyboard data,) Reddit, OpenImage,


### FL Frameworks
1. FL from scratch: Pytorch, TensorFlow
2. TensorFlow Federated (TFF)  https://www.tensorflow.org/federated --> THIS TOO
3. FedScale https://arxiv.org/pdf/2105.11367 -->BEGIN
4. LEAF https://arxiv.org/pdf/1812.01097 https://leaf.cmu.edu
5. FedML https://fedml.ai https://arxiv.org/abs/2007.13518

| Framework | Best For                           | Ease of Setup (2-Day) | Custom Client Selection              | ML Framework                 |
| --------- | ---------------------------------- | --------------------- | ------------------------------------ | ---------------------------- |
| ✅ Flower  | Prototyping, Research, Flexibility | Very Easy             | Very Easy (Override 1 method)        | Agnostic (PyTorch, TF, etc.) |
| FedML     | All-in-one MLOps Platform          | Medium                | Medium (Have to find the right API)  | Agnostic                     |
| TFF       | Deep research on FL fundamentals   | Very Hard             | Hard (Requires learning TFF's model) | TensorFlow only              |
| FedScale  | Large-scale deployment simulation  | Hard                  | Medium-Hard                          | Agnostic                     |
| LEAF      | Benchmarking Datasets              | N/A                   | N/A (It's not a framework)           | N/A                          |

## Challenges
most assume that client always available, but in real world clients may drop out due to various reasons, dataset for device scarcity no avilable (??) (AREA OF RESEARCH)
google obv less acc during day, 

FAIRNESS prob: Always selecting the prioritized clients tends to result in suboptimal performance as underrepresented clients may never have the chance to be selected (??) HOW TO SOLVE THIS? (AREA OF RESEARCH) MABs? exloration vs exploitation here ?

Worse, heterogeneity is different across different regions and application scenarios. Toh global algo difficult, like somewhere net speed fast , but else place slow (??) (AREA OF RESEARCH)

## Research Opportunities
1. optimal no of selected clients:
currently empirical, no theoretical analysis
gboard usses 100 , many show more is better  with diminishing returns,
but more clients = more comm cost
2. theo perf guarantee: current studies mostly on experiments , bias . some work demonstrates that clients with more
divergent models from the global are preferred (e.g., [16]
and [23]), while some work shows the opposite (e.g., [20]
and [59]).  OPPOSITE FINDINGS

3. Benchmark and Eval Metrics : no standard benchmark datasets and eval metrics for client selection strategies (??) (AREA OF RESEARCH)
   NOT COMPARABLE
4. pplication scenarios
often require a large number of clients, envision large-scale,
open testbeds for FL research,

## Conclusion
more selection paradigms(without utility func) : fuzzy logic-based client selection [84] and reinforcement learning-based client selection 