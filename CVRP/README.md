
# ReLD CVRP

This code is implemented based on [ELG](https://github.com/gaocrr/ELG) and [POMO](https://github.com/yd-kwon/POMO).

## How to Run 
### Training
To train from scratch, set the `load_checkpoint` parameter in `config.yml` to `null` and run:
```shell
python train.py
```
 
### Evaluating on Synthetic Data
Run the following script to reproduce Table 3:
```shell
python test.py
```
### Evaluating on VRPLIB
Modify the `vrplib_set` parameter in `config.yml` to specify the desired dataset:​

* set-X for evaluations corresponding to Table 4.

* set-XXL for evaluations corresponding to Table 8.

Run the following script:
```
python test_vrplib.py
```
## Acknowledgments
* https://github.com/gaocrr/ELG
* https://github.com/yd-kwon/POMO
* https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD
