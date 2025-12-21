# Inference and Evaluation

## Batch Inference

To evaluate the model on a dataset, use the `batch_inference.py` script. It is significantly more efficient in large-scale evaluation compared to `demo.py`, supporting batched inference, multi-GPU inference.

An example of running this script with four GPUs is as follows:

```bash
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=4  batch_inference.py duration_s=8 dataset=vggsound model=small_16k num_workers=8
```

You may need to update the data paths in `config/eval_data/base.yaml`. 
More configuration options can be found in `config/base_config.yaml` and `config/infer_config.yaml`.


## Obtaining Quantitative Metrics

For evaluation, install requirements by following instructions in: [av-benchmark](https://github.com/hkchengrex/av-benchmark), and [kadtk](https://github.com/YoonjinXD/kadtk). <br/>
After running <i>av-benchmark</i>, update the path of `gt_cache` in  `config/eval_data/base.yaml`.
