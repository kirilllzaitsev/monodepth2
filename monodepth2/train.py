# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from monodepth2.options import MonodepthOptions
from monodepth2.trainer import Trainer

options = MonodepthOptions()
opts = options.parse()
# import yaml

# with open("latest_config.yaml", "w") as f:
#     yaml.dump(vars(opts), f, default_flow_style=False)
# exit(0)

if __name__ == "__main__":
    trainer = Trainer(opts, disable_exp=opts.tune_params)
    if opts.tune_params:
        import optuna
        from optuna.trial import TrialState
        study = optuna.create_study(direction="minimize")
        study.optimize(trainer.tune_params, n_trials=7)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        best_params = study.best_params
        best_loss = study.best_value
        print("Best Hyperparameters:", best_params)
        print("Best Loss:", best_loss)
    else:
        trainer.train()
