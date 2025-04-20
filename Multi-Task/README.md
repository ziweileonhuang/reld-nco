
# ReLD Multi-Task

This code is implemented based on [MVMoE](https://github.com/RoyalSkye/Routing-MVMoE).

## How to Run

### Training
To convert a model from the original MVMoE repository into a ReLD variant:

- Remove normalization layers in the encoder by setting `--norm=none`.
- Add identity connections with feed-forward networks in the decoder by including the `--ffidt` flag.

```shell
# Default: --problem_size=100 --pomo_size=100 --gpu_id=0
# ReLD-MTL 
python train.py --problem=Train_ALL --model_type=MTL --norm=none --ffidt

# ReLD-MoEL 
python train.py --problem=Train_ALL --model_type=MOE --num_experts=4 --routing_level=node --routing_method=input_choice --norm=none --ffidt
```

### Evaluation

```shell
# ReLD-MTL 
python test.py --problem=ALL --model_type=MTL --norm=none --ffidt --checkpoint={MODEL_PATH}

# ReLD-MoEL 
python test.py --problem=ALL --model_type=MOE_LIGHT --num_experts=4 --routing_level=node --routing_method=input_choice --norm=none --ffidt --checkpoint={MODEL_PATH}
```

## Acknowledgments
* https://github.com/RoyalSkye/Routing-MVMoE
