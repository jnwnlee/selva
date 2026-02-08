# Inference and Evaluation

## Batch Inference

To evaluate the model on a dataset, use the `batch_inference.py` script. It supports batched inference, multi-GPU inference.

An example of running this script with four GPUs is as follows:

```bash
torchrun batch_inference.py duration_s=8 dataset=vggsound model=small_16k num_workers=8
```
Use appropirate options for torchrun, e.g., `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=10 torchrun --standalone --nproc_per_node=1 batch_inference.py`.

You may need to update the data paths in `config/eval_data/base.yaml`. 
More configuration options can be found in `config/base_config.yaml` and `config/infer_config.yaml`.


## Inference on Custom Videos

Run an inference with your own video: for preparation,
1. Put the video files at `inference/video` and text prompts at `inference/json`. The filename of corresponding video and json should be the same. Follow the format of example videos that we've provided.
2. Change the dataset name in inference config file `config/infer_config.yaml`: `dataset: infer_video`.
Note that we do not officially support quantitative evaluation for custom videos.


## Quantitative Evaluation on Benchmarks

For evaluation, install requirements by following instructions in: [av-benchmark](https://github.com/hkchengrex/av-benchmark), and [kadtk](https://github.com/YoonjinXD/kadtk). <br/>
After running <i>av-benchmark</i>, update the path of `gt_cache` in  `config/eval_data/base.yaml`.


1. Download the demanded dataset (for VGG-MonoAudio, [huggingface link](https://huggingface.co/datasets/jnwnlee/vgg-monoaudio)).
2. Clone av-benchmark `git clone https://github.com/hkchengrex/av-benchmark`, and extract latents:
    ```bash
    python training/preproc_eval_cache.py --av_benchmark_dir path/to/av-benchmark
    ```
    Modify other arguments (`batch_size`, `num_worker`) and `metadata` inside the .py file.
3. Update config files: `config/eval_data/base.yaml` and `config/infer_config.yaml`.
4. `batch_inference.py` will handle the metric calculation after inference. Check the `hydra.run.dir` folder of `infer_config.yaml`.
