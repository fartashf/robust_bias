"""Evaluate multiple models against adversarial attacks.

Example:
python eval_cifar10.py --model_name grid --path runs/run_autoattack
"""
import logging
import argparse
import os
import sys
sys.path.append('../auto-attack/')
sys.path.append('../robustbench/')
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import autoattack
from autoattack import AutoAttack
from robustbench.data import load_cifar10
from robustbench.utils import load_model
import numpy as np
import joblib
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def eval_batch(model, x_test, y_test, eps):
  n_correct = {}
  num_test = x_test.shape[0]
  eps = eps/255

  adversary = AutoAttack(model, norm='dftinf', eps=eps, version='custom',
                         attacks_to_run=['apgd-ce', 'apgd-dlr'])
  adversary.apgd.n_restarts = 1
  # adversary.apgd.n_iter = 1
  x_adv = adversary.run_standard_evaluation(x_test, y_test)
  n_correct['dftinf'] = adversary.clean_accuracy(x_adv, y_test)*num_test

  n_correct['clean'] = adversary.clean_accuracy(x_test, y_test)*num_test

  return n_correct


def eval_dataset(model, eps, batch_size):
  transform_chain = transforms.Compose([transforms.ToTensor()])
  item = datasets.CIFAR10(
    root='./data', train=False, transform=transform_chain, download=True)
  test_loader = data.DataLoader(
    item, batch_size=batch_size, shuffle=False, num_workers=0)
  n_correct = {}
  num_samples = 0
  for i, (x_test, y_test) in enumerate(test_loader):
    logging.info('>>>> %d/%d', i, len(test_loader))
    x_test = x_test.to('cuda:0')
    y_test = y_test.to('cuda:0')
    n_correct_cur = eval_batch(model, x_test, y_test, eps)
    for k, v in n_correct_cur.items():
      n_correct[k] = n_correct.get(k, 0)+v
    num_samples += x_test.shape[0]
  accuracy = {k:v*100/num_samples for k,v in n_correct.items()}
  logging.info(str(accuracy))
  return accuracy


def make_grid(args):
  models = [
    ('Wu2020Adversarial_extra', 'Linf'),
    ('Carmon2019Unlabeled', 'Linf'),
    ('Wu2020Adversarial', 'L2'),
    ('Augustin2020Adversarial', 'L2'),
    ('Standard', 'Linf'),
  ]

  run_id = 0
  for model_name, norm in models:
    # for eps in np.arange(0, args.eps_max, 0.5):
    eps = args.eps
    cmd = 'mkdir -p {}/{} && \\\n'.format(args.path, run_id)
    cmd += 'python eval_cifar10.py\\\n'
    cmd += '  --model_name {}\\\n'.format(model_name)
    cmd += '  --model_norm {}\\\n'.format(norm)
    cmd += '  --eps {}\\\n'.format(eps)
    cmd += '  --path {}/{}\\\n'.format(args.path, run_id)
    cmd += '  > {}/{}/log 2>&1 &'.format(args.path, run_id)
    cmd += 'wait'
    with open('jobs/{}.sh'.format(run_id), 'w') as f:
      print(cmd, file=f)
    run_id += 1


def run_all(args):
  models = [
    ('Wu2020Adversarial_extra', 'Linf'),
    ('Carmon2019Unlabeled', 'Linf'),
    ('Wu2020Adversarial', 'L2'),
    ('Augustin2020Adversarial', 'L2'),
    ('Standard', 'Linf'),
  ]

  accuracy = {}
  for model_name, model_norm in models:
    model = load_model(model_name=model_name, norm=model_norm)
    model = model.to('cuda:0')
    accuracy[(model_name, model_norm)] = eval_dataset(
      model, args.eps, args.batch_size)
    logging.info('%s %s', model_name, model_norm)
    logging.info(str(accuracy))
  logging.info(str(accuracy))


def main():
  parser = argparse.ArgumentParser(description='Eval CIFAR10')
  parser.add_argument('--path', type=str, default='runs/X')
  parser.add_argument('--model_name', type=str, default='all')
  parser.add_argument('--model_norm', type=str, default='')
  parser.add_argument('--eps', type=float, default=8)
  parser.add_argument('--batch_size', type=float, default=250)
  parser.add_argument('--eps_max', type=float, default=10)

  args = parser.parse_args()

  if args.model_name == 'all':
    run_all(args)
  elif args.model_name == 'grid':
    make_grid(args)
  else:
    model = load_model(model_name=args.model_name, norm=args.model_norm)
    model = model.to('cuda:0')
    accuracy = eval_dataset(model, args.eps, args.batch_size)
    with open(os.path.join(args.path, 'log.jb'), 'wb') as f:
      joblib.dump({'args': dict(vars(args).items()), 'accuracy': accuracy}, f)


if __name__ == '__main__':
  main()
