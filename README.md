# Motivation
This project was created as part of the lecture 'Advanced Deep Learning' at the Seoul National University. In this lecture, we talked about state-of-the-art approaches in AI from different topics. We also wrote in total 20 paper reviews - 2 for each topic. Within our project, we decided to focus on self-supervised learning and unsupervised domain adaptation. In detail, we focused on the task of improving violence detection in CCTV footage. Please find the final report and final presentation in the [report directory](report/).

# Abstract
Closed-circuit television (CCTV) has demonstrated its effectiveness in crime prevention, especially when it is actively monitored. However, the deployment of automatic violence detection for real-life CCTV footage encounters challenges due to the lack of camera surveillance datasets containing crimes and the domain shift caused by the installation environment and camera quality. To enhance violence detection capabilities, we explore an unsupervised domain adaptation approach to address domain discrepancy and employ a self-supervised learning approach to refine feature representation. In addition, we propose a novel framework to combine the two approaches with the same optimization objective. The proposed framework achieves an impressive improvement of 11% in the UCF-fighting dataset.

# Experiments Collection

## General Settings

- Model: MoviNetA1

  - Causal = False
  - pretrained = True

- Frame Rates

  - Hockey: None
  - UCF: 5
  - Num_frames = 16

  

## Experiment 1 - Finetuning



## Experiment 2 - Self Supervised Learning

**SSL**
| Experiment   | Accuracy | Confusion Matrix                                             |
| ------------ | -------- | ------------------------------------------------------------ |
| Source Val   | 0.97     |  [[0.9584615384615385, 0.04153846153846154], [0.01, 0.99]]    |
| Source Test  | 0.97     | [[0.9657794676806084, 0.034220532319391636], [0.016166281755196306, 0.9838337182448037]]    |
| Target Train | 0.66     | [[0.6337209302325582, 0.36627906976744184], [0.31402439024390244, 0.6859756097560976]] |
| Target Val   | 0.76     | [[0.6649672250546249, 0.3350327749453751], [0.14547304170905392, 0.854526958290946]] |
| Target Test  | 0.78    | [[0.6595918367346939, 0.3404081632653061], [0.10458360232408005, 0.89541639767592]] |



## Experiment 3 - Domain Alignment

**Baseline Performance**

| Experiment   | Accuracy | Confusion Matrix                                             |
| ------------ | -------- | ------------------------------------------------------------ |
| Source Val   | 0.99     | [[1.0, 0.0], [0.011538461538461539, 0.9884615384615385]]     |
| Source Test  | 0.98     | [[1.0, 0.0], [0.020015396458814474, 0.9799846035411856]]     |
| Target Train | 0.64     | [[0.4205128205128205, 0.5794871794871795], [0.13114754098360656,0.8688524590163934]] |
| Target Val   | 0.57     | [[0.22402597402597402, 0.775974025974026], [0.06516464471403813,0.9348353552859618]] |
| Target Test  | 0.58     | [[0.36530398322851154,0.6346960167714885],[0.18856059093631464,0.8114394090636854]] |



### **Results Optuna Search for Optimal Hyperparameters**

- Batch Size: 10 (due to memory limitations)
- lr = 0.0006949557944820544
- mmd_weighting_factor=0.567610878889856
- domain_alignment_loss = Wasserstein
- Mean_pooling = True
- Augmentations = False

| Experiment   | Accuracy | Confusion Matrix                                             |
| ------------ | -------- | ------------------------------------------------------------ |
| Source Val   | 0.99     | [[0.9938461538461538,0.006153846153846154], [0.013846153846153847,0.9861538461538462]] |
| Source Test  | 0.98     | [[0.9954372623574145,0.0045627376425855515], [0.01924557351809084, 0.9807544264819091]] |
| Target Val   | 0.83     | [[0.6961334120425029, 0.30386658795749705], [0.040901213171577126, 0.9590987868284229]] |
| Target Test  | 0.68     | [[0.8270440251572327, 0.17295597484276728], [0.46526252745058894, 0.5347374725494111]] |
| Target Train | 0.72     | [[0.6594594594594595, 0.34054054054054056], [0.23809523809523808, 0.7619047619047619]] |

**Result MMD**

Source Val:

0.96, Confusion Matrix: [[0.9953846153846154, 0.004615384615384616], [0.07538461538461538, 0.9246153846153846]]

Source Test:

0.981139337952271, Confusion Matrix: [[1.0, 0.0], [0.037721324095458045, 0.962278675904542]]

Target Train:

Accuracy: 0.6653796653796654, Confusion Matrix: [[0.7783783783783784, 0.22162162162162163], [0.44761904761904764, 0.5523809523809524]]

Target Val

0.7114491405081448, Confusion Matrix: [[0.7185655253837072, 0.2814344746162928], [0.2956672443674177, 0.7043327556325824]]

Target Test

0.5874716390999346, Confusion Matrix: [[0.8317610062893082, 0.16823899371069181], [0.6568177280894391, 0.343182271910561]]

## Experiment 4 - Domain Alignment + Self Supervised Learning

**Baseline Performance**

| Experiment   | Accuracy | Confusion Matrix                                             |
| ------------ | -------- | ------------------------------------------------------------ |
| Source Val   | 0.99     | [[1.0, 0.0], [0.022307692307692306, 0.9776923076923076]]    |
| Source Test  | 0.99     | [[0.9977186311787072, 0.0022813688212927757], [0.02155504234026174, 0.9784449576597383]]    |
| Target Train | 0.59     | [[0.33121019108280253, 0.6687898089171974], [0.14285714285714285, 0.8571428571428571]] |
| Target Val   | 0.57     | [[0.22402597402597402, 0.775974025974026], [0.06516464471403813,0.9348353552859618]] |
| Target Test  | 0.70     | [[0.43248979591836734, 0.5675102040816327], [0.02356358941252421, 0.9764364105874758]] |

**UDA**

| Experiment   | Accuracy | Confusion Matrix                                             |
| ------------ | -------- | ------------------------------------------------------------ |
| Source Val   | 0.97     | [[1.0, 0.0], [0.055384615384615386, 0.9446153846153846]]  |
| Source Test  | 0.98     | [[1.0, 0.0], [0.03002309468822171, 0.9699769053117783]]  |
| Target Train | 0.70     | [[0.7467532467532467, 0.2532467532467532], [0.34104046242774566, 0.6589595375722543]] |
| Target Val   | 0.72     | [[0.5647001699441612, 0.4352998300558388], [0.12156663275686673, 0.8784333672431333]] |
| Target Test  | 0.77     | [[0.6953469387755102, 0.3046530612244898], [0.16026468689477083, 0.8397353131052292]] |

**SSL** (experiment ssl 88 or 89)
| Experiment   | Accuracy | Confusion Matrix                                             |
| ------------ | -------- | ------------------------------------------------------------ |
| Source Val   | 0.97     |  [[0.9584615384615385, 0.04153846153846154], [0.01, 0.99]]    |
| Source Test  | 0.97     | [[0.9657794676806084, 0.034220532319391636], [0.016166281755196306, 0.9838337182448037]]    |
| Target Train | 0.66     | [[0.6337209302325582, 0.36627906976744184], [0.31402439024390244, 0.6859756097560976]] |
| Target Val   | 0.76     | [[0.6649672250546249, 0.3350327749453751], [0.14547304170905392, 0.854526958290946]] |
| Target Test  | 0.78    | [[0.6595918367346939, 0.3404081632653061], [0.10458360232408005, 0.89541639767592]] |

### Results
**UDA + SSL** (experiment ssl 88 or 89)
| Experiment   | Accuracy | Confusion Matrix                                             |
| ------------ | -------- | ------------------------------------------------------------ |
| Source Val   | 0.93     |  [[0.99, 0.01], [0.13307692307692306, 0.8669230769230769]]   |
| Source Test  | 0.95     |  [[0.9536121673003802, 0.04638783269961977], [0.06312548113933796, 0.9368745188606621]]   |
| Target Train | 0.70     | [[0.7844311377245509, 0.2155688622754491], [0.3933933933933934, 0.6066066066066066]] |
| Target Val   | 0.79     | [[0.7768875940762321, 0.2231124059237679], [0.193794506612411, 0.8062054933875891]] |
| Target Test  | 0.81    | [[0.8081632653061225, 0.19183673469387755], [0.19625564880568108, 0.8037443511943189]] |
