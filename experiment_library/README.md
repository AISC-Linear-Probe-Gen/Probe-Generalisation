# The Impact of Off-Policy Training Data on Probe Performance
See anotebooks/DataPipeline.ipynb to get the datasets of inputs, outputs, labels and activations (but this is old so below script is better).
- See scripts/generate_datasets.py for automating this, reading in a dataset config file from the configs folder.
- For running the above do ```uv run scripts/generate_datasets.py --config bias_config``` for running configs/bias_config.yaml
- Resulting datasets of labels and activations are stored at: https://huggingface.co/lasrprobegen

See notebooks/TrainProbes.ipynb to train and evaluate probes on the datasets.
- Experiment results are written to: https://wandb.ai/LasrProbeGen, these are read when generating result plots.
- See scripts/get_probe_hyperparams_and_results.py for automating this, with the experiments config being specified in the file.
- For running the above do ```nohup uv run scripts/get_probe_hyperparams_and_results.py > output.log 2>&1 &``` and check the output with ```tail -f output.log```


# Citation
This repo is based on a fork of github.com/SamDower/LASR-probe-gen
```
@misc{kirch2025impactoffpolicytrainingdata,
      title={The Impact of Off-Policy Training Data on Probe Generalisation}, 
      author={Nathalie Kirch and Samuel Dower and Adrians Skapars and Ekdeep Singh Lubana and Dmitrii Krasheninnikov},
      year={2025},
      eprint={2511.17408},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2511.17408}, 
}
```
