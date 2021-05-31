#!/bin/bash -l

declare -a seeds=(5 10 21 42 84)
declare -a classes=(10 100)
declare -a diversity_lambdas=(0.001 0.005 0.01 0.05 0.1 0.5 1 3 5 7 10)


EXPERIMENT=cifar_resnet
MODEL=PreResNet164
ENS_SIZE=11

# This is important to initialize conda
# We assume that it is installed in /home/$USER
. "/home/$USER/anaconda3/etc/profile.d/conda.sh"
conda activate grde


i=1
for seed in "${seeds[@]}"
do
    for num_classes in "${classes[@]}"
    do
        # Greedy first
        for diversity_lambda in "${diversity_lambdas[@]}"
        do
            python -m gde.train model.name=${MODEL} \
                    seed=${seed} \
                    ensemble.ens_size=${ENS_SIZE} \
                    data.num_classes=${num_classes} \
                    ensemble.diversity_lambda=${diversity_lambda} \
                    ensemble.greedy=true \
                    experiment=${EXPERIMENT} \
                    experiment_cat=${i}_cifar${num_classes}_${ENS_SIZE}_${MODEL}_greedy
            i=$((i+1))
        done

        # Now just plain deep ensembles
        python -m gde.train model.name=${MODEL} \
                seed=${seed} \
                ensemble.ens_size=${ENS_SIZE} \
                data.num_classes=${num_classes} \
                ensemble.diversity_lambda=0\
                ensemble.greedy=false \
                experiment=${EXPERIMENT} \
                experiment_cat=${i}_cifar${num_classes}_${ENS_SIZE}_${MODEL}_greedy
        i=$((i+1))
    done
done