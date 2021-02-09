# Bridging Adversarial Robustness and Optimization Bias

The evaluation code for maximally robust classifiers using optimization bias
and the Fourier-Linf attack from  the paper
**[Bridging the Gap Between Adversarial Robustness and Optimization Bias]()**
, F. Faghri, D. J. Fleet, C. Vasconcelos, F. Pedregosa, N. Le Roux, 2021.

## Dependencies
see `requirements.txt`

## Run

To train and evaluate a linear model using gradient descent and evaluate 
against Linf attack run:

```
python -m train --config.dim 100 --config.num_train 10\
  --config.num_test 400 --config.optim.lr 0.1 --config.temperature 0.0001\
  --config.adv.lr 0.1 --config.adv.norm_type linf --config.adv.eps_tot 0.5\
  --config.adv.eps_iter 0.5 --config.adv.niters 100 --config.log_keys\
  '("risk/train/loss","risk/train/zero_one","risk/test/zero_one","risk/train/adv/linf","weight/norm/l1")'\
  --config.optim.name gd --config.optim.niters 1000 --config.model.arch linear\
  --config.model.regularizer none --config.log_interval 1
```

To train a linear convolutional model and evaluate against Fourier-Linf attack 
run:
```
python -m train --config './config.py' --config.seed 2 --config.temperature\
  0.0001 --config.num_test 1 --config.dim 100 --config.num_train 3\
  --config.adv.norm_type dftinf --config.adv.lr 0.1 --config.adv.niters 1\
  --config.adv.pre_normalize --config.adv.post_normalize\
  --config.adv.eps_from_cvxpy --config.adv.step_dir dftinf_sd\
  --config.log_interval 1000 --config.log_keys\
  '("risk/train/zero_one","risk/train/adv/dftinf","weight/linear/norm/dft1","margin/dft1")'\
  --config.enable_cvxpy --config.model.arch conv_linear_exp\
  --config.model.nlayers 2 --config.model.regularizer none --config.optim.name gd\
  --config.optim.lr 0.1 --config.optim.niters 10000\
  --config.log_dir "runs/X"
```

To run the Fourier-Linf attack against CIFAR-10 models, clone the 
[AutoAttack](https://github.com/fra31/auto-attack) repository and replace
`autoattack/autoattack.py` and `autoattack/autopgd_pt.py` with copies in the 
`autoattack` directory in this repository and copy the new file 
`autoattack/norm.py` to `autoattack/`. Modifications are based on the 
AutoAttack commit 
[9b264b5](https://github.com/fra31/auto-attack/commit/9b264b52bb65c373373727f532865a5551ab9c02).

Then to evaluate the Fourier-Linf robustness of a model (e.g.  robust Linf 
model of `Wu2020Adversarial_extra`)  run:
```
mkdir -p runs/X && python eval_cifar10.py\
  --model_name Wu2020Adversarial_extra  --model_norm Linf  --eps 8\
  --path runs/X
```

## Hyperparameters

The grid file `grid/linear.py` specifies hyperparameter configurations used in 
experiments.

## Reference

If you found this code useful, please cite the following paper:

## License

This code is based on the 
[robust_optim](https://github.com/google-research/google-research/tree/master/robust_optim)
project released under the
[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).
