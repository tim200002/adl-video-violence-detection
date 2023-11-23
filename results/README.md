<!-- results -->
## Results
### Fine-tuning on Kinetics 600
#### Training Blocks: Conv1 -> Blocks -> Conv7 -> MLP -> FC
| Conv1 | Blocks | Conv7 | MLP | FC | Epoch 1 ACC | Epoch 2 ACC |
|:-----:|:------:|:-----:|:---:|:--:|:---:|:---:|
| O | O | X | X | O | 96.33 | 95.69 |
| X | O | X | X | O | 95.76 | 96.13 |
| X | O | O | X | O | 96.10 | 95.76 |
| X | O | O | O | O | 95.62 | 95.69 |
| O | O | O | O | O | 94.97 | 94.53 |
| X | O | X | O | O | 94.43 | 94.74 |
| X | X | O | O | O | 94.70 | 94.67 |
| X | X | X | O | O | 94.43 | 93.65 |
| O | X | X | X | O | 92.53 | 93.28 |
| X | X | X | X | O | 88.62 | 92.22 |
