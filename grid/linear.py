"""Grids for linear experiments."""
from collections import OrderedDict
import numpy as np


def nameit(name, args):
  assert isinstance(args, list)
  new_args = []
  for a in args:
    assert isinstance(a, tuple)
    new_args += [tuple(['%s.%s' % (name, a[0])] + list(a[1:]))]
  return new_args


def get_shared_params(adv_norm_type, dual_norm_type, attack_step_dir):
  d_over_n = [1, 2, 4, 8, 16, 32]  # separable >= 1
  dim = 100
  num_train = [int(dim / p) for p in d_over_n]

  # Config params
  shared_params = []
  shared_params += [
    ('config', './config.py'),
    ('seed', list(range(3))),
  ]

  # Data hyper-parameters
  shared_params += [
    ('temperature', 0.0001),
    ('num_test', 500),
    ('dim', dim),
    ('num_train', num_train),
  ]

  # Adversarial configuration: test
  shared_params += nameit('adv', [
    ('norm_type', adv_norm_type),
    ('lr', 0.1),
    ('niters', 10),
    # ('eps_iter', attack_eps),  # Overwritten by cvxpy
    # ('eps_tot', attack_eps),  # Overwritten by cvxpy
    ('pre_normalize', True),  # multi attacks
    ('post_normalize', True),
    ('eps_from_cvxpy', True),
    ('step_dir', attack_step_dir),
  ])

  # Logging to standard output
  shared_params += [
    ('log_interval', 10000),  # 1000),
    ('log_keys', '\'("%s")\'' % ('","'.join([
      'risk/train/zero_one',
      'risk/train/adv/%s' % adv_norm_type,
      'weight/linear/norm/%s' % dual_norm_type,
      'margin/%s' % dual_norm_type,
    ]))),
    # Compare with cvxpy
    ('enable_cvxpy', True),
  ]
  return shared_params



def experiment_linear_lp(adv_norm_type, dual_norm_type, baseline_norm_types,
                         attack_step_dir):
  """Configuration for deep linear models with regularizer and adversarial training."""
  module_name = 'train'
  # log_dir = 'runs_linear_%s' % adv_norm_type
  # log_dir = 'runs_linear_postnorm_%s' % adv_norm_type
  log_dir = 'runs_linear_postnorm_randinit_%s' % adv_norm_type
  exclude = '*'

  shared_params = get_shared_params(adv_norm_type, dual_norm_type,
                                    attack_step_dir)

  # No 0 regularization coefficient
  reg_coeff = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
  # reg_coeff = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
  # Between 1e-3 and 1e-1 for d/n=10 the adv robustness drops
  # reg_coeff += [3e-3, 5e-3, 3e-2, 5e-2, 3e-1, 5e-1]

  # Model hyper-parameters
  linear_noreg_model_params = nameit('model', [
    ('arch', 'linear'),
    ('regularizer', 'none'),
  ])
  linear_reg_model_params = nameit('model', [
    ('arch', 'linear'),
    ('regularizer', ['w_%s' % b for b in baseline_norm_types] +
     ['w_%s' % dual_norm_type]),
    ('reg_coeff', reg_coeff),
  ])

  params = []

  # cvxpy solution
  # njobs=3*6*1=18
  cvxpy_params = nameit('optim', [
    ('name', 'cvxpy'),
    ('norm', dual_norm_type),
    ('niters', 10000),
    ('lr', 0),  # keep cvxpy sol fixed
  ])
  params += [OrderedDict(shared_params+linear_noreg_model_params+cvxpy_params)]

  # njobs=3*6*2=36
  # CD with line search doesn't work right, so not including it
  gd_ls = nameit('optim', [
    ('name', 'gd_ls'),  # ['gd_ls', 'cd_ls']),
    ('niters', 10000),
    ('bound_step', True),
  ])
  params += [OrderedDict(shared_params+linear_noreg_model_params+gd_ls)]

  # Implicit bias with fixed lr
  # GD with fixed lr performs similar to line search, so we don't include them
  # njobs=3*6*11=198
  # gd_fixed_lr = nameit('optim', [
  #   ('name', 'gd'),
  #   ('niters', 10000),
  #   ('lr', [
  #     1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1,
  #     3e-1, 1
  #   ]),
  # ])
  # params += [OrderedDict(shared_params+linear_noreg_model_params+gd_fixed_lr)]

  # njobs=3*6*19=342
  cd_fixed_lr = nameit('optim', [
    ('name', ['cd', 'signgd']),
    ('niters', 10000),
    ('lr', [
      1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1,
      3e-1, 1, 2, 3, 6, 9, 10, 20, 30, 50
    ]),
  ])
  params += [OrderedDict(shared_params+linear_noreg_model_params+cd_fixed_lr)]

  # Explicit regularization with line search
  # njobs=3*6*20*4*2=2880
  explicit_reg = nameit('optim', [
    ('name', 'fista'),
    ('niters', 10000),
    ('bound_step', True),
    ('step_size', [1, 10, 100, 1000]),
  ])
  params += [OrderedDict(shared_params+linear_reg_model_params+explicit_reg)]

  # Adversarial training with line search
  # njobs=3*6*5=90
  adv_train_params = nameit('optim', [
    ('name', 'gd_ls'),
    ('niters', 10000),
    ('bound_step', True),
  ])
  adv_train_params += nameit('optim', nameit('adv_train', [
    ('enable', True),
    ('norm_type', adv_norm_type),
    ('lr', 0.1),
    ('niters', 10),  # niters, 1000
    ('pre_normalize', True),
    ('post_normalize', True),
    ('step_dir', attack_step_dir),
    ('eps_from_cvxpy', True),
  ]))
  adv_train_params = OrderedDict(
    shared_params+linear_noreg_model_params+adv_train_params)

  params += [adv_train_params]

  return params, log_dir, module_name, exclude


def experiment_linear_linf(_):
  """Run linear experiments against L1, L2 and Linf attacks."""
  # Attack epsilon is manually set according to the norm of the min-norm
  # solution found using cvxpy for d/n=10. That is max-margin=1/min-norm.
  # Min linf-norm solution found (norm=0.0422)
  # Min l2-norm solution found (norm=0.3411)
  # Min l1-norm solution found (norm=1.8497)
  # Min l4-norm solution found (norm=0.0002)
  # Min l1.5-norm solution found (norm=0.5274)
  return experiment_linear_lp(
    adv_norm_type='linf',
    dual_norm_type='l1',
    baseline_norm_types=['l2'],
    attack_step_dir='sign_grad')


def experiment_linear_l2(_):
  # Attack epsilon is manually set according to the norm of the min-norm
  # solution found using cvxpy for d/n=10. That is max-margin=1/min-norm.
  # Min linf-norm solution found (norm=0.0422)
  # Min l2-norm solution found (norm=0.3411)
  # Min l1-norm solution found (norm=1.8497)
  # Min l4-norm solution found (norm=0.0002)
  # Min l1.5-norm solution found (norm=0.5274)
  return experiment_linear_lp(
    adv_norm_type='l2',
    dual_norm_type='l2',
    baseline_norm_types=['l1'],
    attack_step_dir='grad')


def experiment_linear_l1(_):
  """Run linear experiments against L1, L2 and Linf attacks."""
  # Attack epsilon is manually set according to the norm of the min-norm
  # solution found using cvxpy for d/n=10. That is max-margin=1/min-norm.
  # Min linf-norm solution found (norm=0.0422)
  # Min l2-norm solution found (norm=0.3411)
  # Min l1-norm solution found (norm=1.8497)
  # Min l4-norm solution found (norm=0.0002)
  # Min l1.5-norm solution found (norm=0.5274)
  return experiment_linear_lp(
    adv_norm_type='l1',
    dual_norm_type='linf',
    baseline_norm_types=['l2'],
    attack_step_dir='grad_max')


def experiment_linear_tradeoff_linf(_):
  """Configuration for deep linear models with regularizer and adversarial training."""
  adv_norm_type = 'linf'
  dual_norm_type = 'l1'
  # Min l1-norm solution found (norm=0.6876)
  attack_eps = 1/0.6876
  attack_step_dir = 'sign_grad'
  module_name = 'train'
  log_dir = 'runs_linear_tradeoff_%s' % adv_norm_type
  exclude = '*'

  d_over_n = [32]
  dim = 100
  num_train = [int(dim / p) for p in d_over_n]

  # Config params
  shared_params = []
  shared_params += [
    ('config', './config.py'),
    ('seed', list(range(3))),
  ]

  # Data hyper-parameters
  shared_params += [
    ('temperature', 0.0001),
    ('num_test', 500),
    ('dim', dim),
    ('num_train', num_train),
  ]

  # Adversarial configuration: test
  shared_params += nameit('adv', [
    ('norm_type', adv_norm_type),
    ('lr', 0.1),
    ('niters', 10),
    # ('eps_iter', attack_eps),  # Overwritten by cvxpy
    # ('eps_tot', attack_eps),  # Overwritten by cvxpy
    ('pre_normalize', True),  # multi attacks
    ('post_normalize', True),
    ('eps_from_cvxpy', True),
    ('step_dir', attack_step_dir),
  ])

  # Logging to standard output
  shared_params += [
    ('log_interval', 10000),  # 1000),
    ('log_keys', '\'("%s")\'' % ('","'.join([
      'risk/train/zero_one',
      'risk/train/adv/%s' % adv_norm_type,
      'weight/linear/norm/%s' % dual_norm_type,
      'margin/%s' % dual_norm_type,
    ]))),
    # Compare with cvxpy
    ('enable_cvxpy', True),
  ]
  params = []

  # reg_coeff = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
  reg_coeff = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
  # Between 1e-3 and 1e-1 for d/n=10 the adv robustness drops
  reg_coeff += [3e-3, 5e-3, 3e-2, 5e-2, 3e-1, 5e-1]

  # Model hyper-parameters
  linear_noreg_model_params = nameit('model', [
    ('arch', 'linear'),
    ('regularizer', 'none'),
  ])
  linear_reg_model_params = nameit('model', [
    ('arch', 'linear'),
    ('regularizer', ['w_%s' % dual_norm_type]),
    ('reg_coeff', reg_coeff),
  ])

  # Explicit regularization with line search
  # njobs=3*6*20*4*2=2880
  explicit_reg = nameit('optim', [
    ('name', 'fista'),
    ('niters', 10000),
    ('bound_step', True),
    ('step_size', [1, 10, 100, 1000]),
  ])
  params += [OrderedDict(shared_params+linear_reg_model_params+explicit_reg)]

  # Adversarial training with line search
  for i in [1] + list(np.arange(0.1, 2, 0.2)):  # [0.1, 0.3, 0.5, 0.7, 1, 1.3]:
    adv_train_params = nameit('optim', [
      ('name', 'gd_ls'),
      ('niters', 10000),
      ('bound_step', True),
    ])
    adv_train_params += nameit('optim', nameit('adv_train', [
      ('enable', True),
      ('norm_type', adv_norm_type),
      ('lr', 0.1),
      ('niters', 10),  # niters, 1000
      ('pre_normalize', True),
      ('post_normalize', True),
      ('step_dir', attack_step_dir),
      ('eps_iter', float(attack_eps) * i),
      ('eps_tot', float(attack_eps) * i),
    ]))
    params += [OrderedDict(
      shared_params+linear_noreg_model_params+adv_train_params)]

  return params, log_dir, module_name, exclude


def experiment_linear_conv(_):
  """Configuration for deep convolutional linear models against DFT_inf attack."""
  # Min dft1-norm solution found (norm=1.9895)
  adv_norm_type = 'dftinf'
  dual_norm_type = 'dft1'
  baseline_norm_types = ['l1', 'linf']
  attack_step_dir = 'dftinf_sd'  # 'dftinf'

  module_name = 'train'
  log_dir = 'runs_linear_conv_%s' % adv_norm_type
  exclude = '*'

  d_over_n = [1, 2, 4, 8, 16, 32]  # separable >= 1
  dim = 100
  num_train = [int(dim / p) for p in d_over_n]

  # Config params
  shared_params = []
  shared_params += [
    ('config', './config.py'),
    ('seed', list(range(3))),
  ]

  # Data hyper-parameters
  shared_params += [
    ('temperature', 0.0001),
    ('num_test', 1),  # 500
    ('dim', dim),
    ('num_train', num_train),
  ]

  # Adversarial configuration: test
  shared_params += nameit('adv', [
    ('norm_type', adv_norm_type),
    # ('lr', 0.1),
    ('niters', 1),  # 10
    # ('eps_iter', attack_eps),  # Overwritten by cvxpy
    # ('eps_tot', attack_eps),  # Overwritten by cvxpy
    ('pre_normalize', True),  # multi attacks
    ('post_normalize', True),
    ('eps_from_cvxpy', True),
    ('step_dir', attack_step_dir),
  ])

  # Logging to standard output
  shared_params += [
    ('log_interval', 10000),  # 1000),
    ('log_keys', '\'("%s")\'' % ('","'.join([
      'risk/train/zero_one',
      'risk/train/adv/%s' % adv_norm_type,
      'weight/linear/norm/%s' % dual_norm_type,
      'margin/%s' % dual_norm_type,
    ]))),
    # Compare with cvxpy
    ('enable_cvxpy', True),
  ]

  # No 0 regularization coefficient
  reg_coeff = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

  # Model hyper-parameters
  linear_noreg_model_params = nameit('model', [
    ('arch', 'linear'),
    ('regularizer', 'none'),
  ])
  linear_reg_model_params = nameit('model', [
    ('arch', 'linear'),
    ('regularizer', ['w_%s' % b for b in baseline_norm_types] +
     ['w_%s' % dual_norm_type]),
    ('reg_coeff', reg_coeff),
  ])
  deep_linear_params = nameit('model', [
    ('arch', 'deep_linear'),
    ('nlayers', 2),
    ('regularizer', 'none'),
  ])
  conv_linear_params = nameit('model', [
    ('arch', 'conv_linear'),
    ('nlayers', 2),
    ('regularizer', 'none'),
  ])

  params = []

  # cvxpy solution
  cvxpy_params = nameit('optim', [
    ('name', 'cvxpy'),
    ('norm', dual_norm_type),
    ('niters', 10000),
    ('lr', 0),  # keep cvxpy sol fixed
  ])
  params += [OrderedDict(shared_params+linear_noreg_model_params+cvxpy_params)]

  # GD line search implicit bias
  gd_ls = nameit('optim', [
    ('name', 'gd_ls'),
    ('niters', 10000),
    ('bound_step', True),
  ])
  params += [OrderedDict(shared_params+deep_linear_params+gd_ls)]
  params += [OrderedDict(shared_params+conv_linear_params+gd_ls)]

  # CD, SignGD implicit bias
  cd_fixed_lr = nameit('optim', [
    ('name', ['cd', 'signgd']),
    ('niters', 10000),
    ('lr', [
      1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1,
      3e-1, 1, 2, 3, 6, 9, 10, 20, 30, 50
    ]),
  ])
  params += [OrderedDict(shared_params+linear_noreg_model_params+cd_fixed_lr)]

  # Explicit regularization with line search
  explicit_reg = nameit('optim', [
    ('name', 'fista'),
    ('niters', 10000),
    ('bound_step', True),
    ('step_size', [1, 10, 100, 1000]),
  ])
  params += [OrderedDict(shared_params+linear_reg_model_params+explicit_reg)]

  # Adversarial training with line search
  adv_train_params = nameit('optim', [
    ('name', 'gd_ls'),
    ('niters', 10000),
    ('bound_step', True),
  ])
  adv_train_params += nameit('optim', nameit('adv_train', [
    ('enable', True),
    ('norm_type', adv_norm_type),
    # ('lr', 0.1),
    ('niters', 1),  # niters, 1000
    ('pre_normalize', True),
    ('post_normalize', True),
    ('step_dir', attack_step_dir),
    ('eps_from_cvxpy', True),
  ]))
  params += [OrderedDict(
    shared_params+linear_noreg_model_params+adv_train_params)]

  return params, log_dir, module_name, exclude


def experiment_linear_conv_constant_lr(_):
  """Configuration for deep convolutional linear models against DFT_inf attack."""
  # Min dft1-norm solution found (norm=1.9895)
  adv_norm_type = 'dftinf'
  dual_norm_type = 'dft1'
  attack_step_dir = 'dftinf_sd'  # 'dftinf'

  module_name = 'train'
  # log_dir = 'runs_linear_conv_constant_lr_%s' % adv_norm_type
  log_dir = 'runs_linear_conv_constant_lr_normfix_%s' % adv_norm_type
  exclude = '*'

  # d_over_n = [1, 2, 4, 8, 16, 32]  # separable >= 1
  d_over_n = [16, 32]  # separable >= 1
  dim = 100
  num_train = [int(dim / p) for p in d_over_n]

  # Config params
  shared_params = []
  shared_params += [
    ('config', './config.py'),
    ('seed', list(range(3))),
  ]

  # Data hyper-parameters
  shared_params += [
    ('temperature', 0.0001),
    ('num_test', 1),  # 500
    ('dim', dim),
    ('num_train', num_train),
  ]

  # Adversarial configuration: test
  shared_params += nameit('adv', [
    ('norm_type', adv_norm_type),
    # ('lr', 0.1),
    ('niters', 1),  # 10
    # ('eps_iter', attack_eps),  # Overwritten by cvxpy
    # ('eps_tot', attack_eps),  # Overwritten by cvxpy
    ('pre_normalize', True),  # multi attacks
    ('post_normalize', True),
    ('eps_from_cvxpy', True),
    ('step_dir', attack_step_dir),
  ])

  # Logging to standard output
  shared_params += [
    ('log_interval', 10000),  # 1000),
    ('log_keys', '\'("%s")\'' % ('","'.join([
      'risk/train/zero_one',
      'risk/train/adv/%s' % adv_norm_type,
      'weight/linear/norm/%s' % dual_norm_type,
      'margin/%s' % dual_norm_type,
    ]))),
    # Compare with cvxpy
    ('enable_cvxpy', True),
  ]

  # Model hyper-parameters
  conv_linear_params = nameit('model', [
    ('arch', 'conv_linear'),
    ('nlayers', 2),
    ('regularizer', 'none'),
  ])

  params = []

  # Conv linear constant lr
  cd_fixed_lr = nameit('optim', [
    ('name', ['gd']),
    ('niters', 100000),
    ('lr', [
      1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1,
      3e-1, 1, 2, 3, 6, 9, 10, 20, 30, 50
    ]),
  ])
  params += [OrderedDict(shared_params+conv_linear_params+cd_fixed_lr)]

  return params, log_dir, module_name, exclude


def experiment_linear_conv_ls(_):
  """Configuration for deep convolutional linear models against DFT_inf attack."""
  # Min dft1-norm solution found (norm=1.9895)
  adv_norm_type = 'dftinf'
  dual_norm_type = 'dft1'
  attack_step_dir = 'dftinf_sd'  # 'dftinf'

  module_name = 'train'
  # log_dir = 'runs_linear_conv_ls_%s' % adv_norm_type
  log_dir = 'runs_linear_conv_ls_normfix_%s' % adv_norm_type
  exclude = '*'

  d_over_n = [1, 2, 4, 8, 16, 32]  # separable >= 1
  dim = 100
  num_train = [int(dim / p) for p in d_over_n]

  # Config params
  shared_params = []
  shared_params += [
    ('config', './config.py'),
    ('seed', list(range(3))),
  ]

  # Data hyper-parameters
  shared_params += [
    ('temperature', 0.0001),
    ('num_test', 1),  # 500
    ('dim', dim),
    ('num_train', num_train),
  ]

  # Adversarial configuration: test
  shared_params += nameit('adv', [
    ('norm_type', adv_norm_type),
    # ('lr', 0.1),
    ('niters', 1),  # 10
    # ('eps_iter', attack_eps),  # Overwritten by cvxpy
    # ('eps_tot', attack_eps),  # Overwritten by cvxpy
    ('pre_normalize', True),  # multi attacks
    ('post_normalize', True),
    ('eps_from_cvxpy', True),
    ('step_dir', attack_step_dir),
  ])

  # Logging to standard output
  shared_params += [
    ('log_interval', 10000),  # 1000),
    ('log_keys', '\'("%s")\'' % ('","'.join([
      'risk/train/zero_one',
      'risk/train/adv/%s' % adv_norm_type,
      'weight/linear/norm/%s' % dual_norm_type,
      'margin/%s' % dual_norm_type,
    ]))),
    # Compare with cvxpy
    ('enable_cvxpy', True),
  ]

  # Model hyper-parameters
  conv_linear_params = nameit('model', [
    ('arch', 'conv_linear'),
    ('nlayers', 2),
    ('regularizer', 'none'),
  ])

  params = []

  # GD line search implicit bias
  gd_ls = nameit('optim', [
    ('name', 'gd_ls'),
    ('niters', 100000),
    ('bound_step', True),
  ])
  params += [OrderedDict(shared_params+conv_linear_params+gd_ls)]

  return params, log_dir, module_name, exclude
