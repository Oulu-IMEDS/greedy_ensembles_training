# Greedy Bayesian Posterior Approximation with Deep Ensembles

This repository is the official implementation of Greedy Bayesian Posterior Approximation with Deep Ensembles by A.
Tiulpin and M. B. Blaschko. (2021)

<center>
<img src="https://github.com/mipt-oulu/oaprogression/blob/master/assets/main_figure.png" width="900"/> 
</center>


## Installation

In the root of the codebase:

```
conda env create -f env.yaml
conda activate grde
pip install -e .
```

## Results
We conducted our main evaluations on 3 architectures: PreResNet164, VGG16BN, and WideResNet28x10. 
LSUN and SVHN datasets were used as out-of-distribution. The following table illustrates the main results of the paper on PreResNet164:
</br>
</br>
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-fymr{border-color:inherit;font-weight:bold;text-align:left;vertical-align:top}
.tg .tg-7btt{border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-fymr" rowspan="2">Dataset</th>
    <th class="tg-fymr" rowspan="2">Method</th>
    <th class="tg-7btt" colspan="2">SVHN</th>
    <th class="tg-7btt" colspan="2">LSUN</th>
  </tr>
  <tr>
    <td class="tg-fymr">AUC</td>
    <td class="tg-fymr">AP</td>
    <td class="tg-fymr">AUC</td>
    <td class="tg-fymr">AP</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky" rowspan="2">CIFAR10</td>
    <td class="tg-0pky">Deep Ensembles</td>
    <td class="tg-0pky">0.94</td>
    <td class="tg-0pky">0.96</td>
    <td class="tg-0pky">0.93</td>
    <td class="tg-0pky">0.89</td>
  </tr>
  <tr>
    <td class="tg-0pky">Ours</td>
    <td class="tg-fymr">0.95</td>
    <td class="tg-fymr">0.97</td>
    <td class="tg-fymr">0.95</td>
    <td class="tg-fymr">0.94</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="2">CIFAR100</td>
    <td class="tg-0pky">Deep Ensembles</td>
    <td class="tg-0pky">0.79</td>
    <td class="tg-0pky">0.88</td>
    <td class="tg-0pky">0.86</td>
    <td class="tg-0pky">0.81</td>
  </tr>
  <tr>
    <td class="tg-0pky">Ours</td>
    <td class="tg-fymr">0.82</td>
    <td class="tg-fymr">0.90</td>
    <td class="tg-fymr">0.87</td>
    <td class="tg-fymr">0.85</td>
  </tr>
</tbody>
</table>
</br>
</br>

## Reproducing the results: training

### CIFAR
We ran our main experiments for ensembles of size 11 on 400 with Nvidia V100 GPUs 
(thanks to [Aalto Triton](https://scicomp.aalto.fi/triton/) and [CSC Puhti](https://docs.csc.fi/computing/overview/) clusters). We launched 1 experiment (i.e.
ensemble) per GPU. One can try to re-run our codes on a single-gpu machine using the script located in the `experiments/replicate.sh`. 
It is possible to check the performance for some individual setting with a single seed as follows (must be run from `experiments/`):

* PreResNet164 on CIFAR10:
```
python -m gde.train \
        experiment=cifar_resnet \
        model.name=PreResNet164 \
        data.num_classes=10 \
        ensemble.greedy=true \
        ensemble.ens_size=11 \
        ensemble.diversity_lambda=3 
```

* VGG16BN on CIFAR10
```
python -m gde.train \
        experiment=cifar_vgg \
        model.name=VGG16BN 
        data.num_classes=10 \
        ensemble.greedy=true \
        ensemble.ens_size=11 \
        ensemble.diversity_lambda=5 
```

* WideResNet28x10 on CIFAR10
```
python -m gde.train \
       experiment=cifar_wide_resnet \
       model.name=WideResNet28x10 \
       data.num_classes=10 \
       ensemble.greedy=true \
       ensemble.ens_size=11 \
       ensemble.diversity_lambda=1 
```

To train models on CIFAR100, simply replace `data.num_classes=10` to `data.num_classes=100`,
and `ensemble.diversity_lambda` the values from the paper.

### MNIST and two moons
* MNIST:
```
python -m gde.train experiment=mnist
```
* Two moons
```
python -m gde.train experiment=two_moons_fc_net
```

## Reproducing the results: testing
For convenience, we have provided the script for running standardized evaluation for CIFAR10/100 and MNIST.
To evaluate the results. Assume `experiments/workdir/` contains the snapshots structured into subfolders, then
the following code will run the OOD evaluation, and will create the results stored as pandas dataframes in `experiments`:

```
python -u -m gde.eval_results \
          --arch PreResNet164 \
          --dataset cifar10 \
          --ens_size 11 \
          --seed 5 \
          --workdir workdir/
```

One can loop over seeds to get the results over multiple runs.


