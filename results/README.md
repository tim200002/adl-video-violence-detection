<!-- results -->
## Results
### Fine-tuning on Kinetics 600
#### Training Blocks: Conv1 -> Blocks -> Conv7 -> MLP -> FC
| Conv1 | Blocks | Conv7 | MLP | FC | Epoch 1 ACC | Epoch 2 ACC |
|:-----:|:------:|:-----:|:---:|:--:|:---:|:---:|
| O | O | X | X | O | **96.33** | 95.69 |
| X | O | X | X | O | 95.76 | **96.13** |
| X | O | O | X | O | 96.10 | 95.76 |
| X | O | O | O | O | 95.62 | 95.69 |
| O | O | O | O | O | 94.97 | 94.53 |
| X | O | X | O | O | 94.43 | 94.74 |
| X | X | O | O | O | 94.70 | 94.67 |
| X | X | X | O | O | 94.43 | 93.65 |
| O | X | X | X | O | 92.53 | 93.28 |
| X | X | X | X | O | 88.62 | 92.22 |


### Fine-tuning on Hockey Dataset
#### Training Blocks: Conv1 -> Blocks -> Conv7 -> MLP -> FC
| Conv1 | Blocks | Conv7 | MLP | FC | Epoch 1 ACC | Epoch 2 ACC | Epoch 3 ACC |
|:-----:|:------:|:-----:|:---:|:--:|:---:|:---:|:---:|
| O | O | X | X | O | 99.35 | **99.85** | 99.62 |
| X | O | X | X | O | **99.50** | 99.73 | |
| O | O | O | O | O | 99.15 | 98.88 | |
| X | X | X | X | O | 93.23 | 93.88 | 94.15 |

### Fine tune on Hockey -> Test on Kinetics 600
99.85% -> 76.37% (-23.48%)