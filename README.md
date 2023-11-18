
## BLIP4CIR with bi-directional training

[![arXiv](https://img.shields.io/badge/paper-wacv2024-cyan)](#) 
[![arXiv](https://img.shields.io/badge/arXiv-2303.16604-red)](https://arxiv.org/abs/2303.16604)

The official implementation for **Bi-directional Training for Composed Image Retrieval via Text Prompt Learning**.

> The link and citation for the WACV proceeding version will be updated after its release.

## Citation
If you find this code useful for your research, please consider citing our work.
```bibtex
@article{liu2023_bi,
  title={Bi-directional Training for Composed Image Retrieval via Text Prompt Learning},
  author={Liu, Zheyuan and Sun, Weixuan and Hong, Yicong and Teney, Damien and Gould, Stephen},
  journal={arXiv preprint arXiv:2303.16604},
  year={2023}
}
```

## News and Upcoming Updates

* **Nov-2023** We have released our code and pre-trained checkpoints.

## About

<p align="center">
  <img src="demo_imgs/model_blip_finetune.png" height="150">
</p>

<p align="center">
  <img src="demo_imgs/model_combiner.png" height="150">
</p>

## Setting up

TODO

A brief introduction on each scripts in `/src` is in [CLIP4Cir - Usage](https://github.com/ABaldrati/CLIP4Cir/tree/master#usage).

## Training

Our methods is built on top of CLIP4Cir with a two-stage training pipeline, with stage-I being the BLIP text encoder finetuning, and the subsequent stage-II being the combiner training. Please check our paper for details.

### for Fashion-IQ

#### BLIP text encoder finetuning

```bash
# Optional: comet experiment logging --api-key and --workspace
python src/clip_fine_tune.py --dataset FashionIQ \
                             --api-key <your comet api> --workspace <your comet workspace> \
                             --num-epochs 20 --batch-size 128 \
                             --blip-max-epoch 10 --blip-min-lr 0 \
                             --blip-learning-rate 5e-5 \
                             --transform targetpad --target-ratio 1.25 \
                             --save-training --save-best --validation-frequency 1 \
                             --experiment-name BLIP_cos10_loss_r.40_5e-5
```

#### Combiner training

```bash
# Optional: comet experiment logging --api-key and --workspace
# Load the blip text encoder weights finetuned in the previous step in --blip-model-path
python src/combiner_train.py --dataset FashionIQ \
                             --api-key <your comet api> --workspace <your comet workspace> \
                             --num-epochs 300 --batch-size 512 --blip-bs 32 \
                             --projection-dim 2560 --hidden-dim 5120  --combiner-lr 2e-5 \
                             --transform targetpad --target-ratio 1.25 \
                             --save-training --save-best --validation-frequency 1 \
                             --blip-model-path <BLIP text encoder finetuned weights path>/saved_models/tuned_blip_best.pt \
                             --experiment-name Combiner_loss_r.50_2e-5__BLIP_cos10_loss_r_.40_5e-5
```

### for CIRR

#### BLIP text encoder finetuning

```bash
# Optional: comet experiment logging --api-key and --workspace
python src/clip_fine_tune.py --dataset CIRR \
                             --api-key <your comet api> --workspace <your comet workspace> \
                             --num-epochs 20 --batch-size 128 \
                             --blip-max-epoch 10 --blip-min-lr 0 \
                             --blip-learning-rate 5e-5 \
                             --transform targetpad --target-ratio 1.25 \
                             --save-training --save-best --validation-frequency 1 \
                             --experiment-name BLIP_5e-5_cos10_loss_r.1
```
#### Combiner training

```bash
# Optional: comet experiment logging --api-key and --workspace
# Load the blip text encoder weights finetuned in the previous step in --blip-model-path
python src/combiner_train.py --dataset CIRR \
                             --api-key <your comet api> --workspace <your comet workspace> \
                             --num-epochs 300 --batch-size 512 --blip-bs 32 \
                             --projection-dim 2560 --hidden-dim 5120 --combiner-lr 2e-5 \
                             --transform targetpad --target-ratio 1.25 \
                             --save-training --save-best --validation-frequency 1 \
                             --blip-model-path <BLIP text encoder finetuned weights path>/saved_models/tuned_blip_mean.pt \
                             --experiment-name Combiner_loss_r.10__BLIP_5e-5_cos10_loss_r.1
```

## Validating and Testing

### Checkpoints

The following weights shall reproduce our results reported in Tables 1 and 2 (hosted on OneDrive, check the SHA1 hash against the listed value):

| checkpoints | Combiner (for `--combiner-path`) | BLIP text encoder (for `--blip-model-path`) |
|------------|----------|-------------|
| Fashion-IQ <br />`SHA1` | [combiner.pt](https://1drv.ms/u/s!AgLqyV5O53gxt8onMBM4dP_yezpcNQ?e=jTu9Gu) <br />`4a1ba45bf52033c245c420b30873f68bc8e60732`  | [tuned_blip_best.pt](https://1drv.ms/u/s!AgLqyV5O53gxt8oo0AF4kSHKWmJgtg?e=c793Sg) <br />`80f0db536f588253fca416af83cb50fab709edda`   |
| CIRR <br />`SHA1`      | [combiner_mean.pt](https://1drv.ms/u/s!AgLqyV5O53gxt8oqSfGkANa0U-pW-A?e=ohCgln) <br />`327703361117400de83936674d5c3032af37bd7a` | [tuned_blip_mean.pt](https://1drv.ms/u/s!AgLqyV5O53gxt8orEuV7r87WkyU5Jg?e=Up9aGw) <br />`67dca8a1905802cfd4cd02f640abb0579f1f88fd`   |

### Reproducing results

To validate on checkpoints, please see below:

#### Fashion-IQ

```bash
python src/validate.py --dataset fashionIQ \
                       --combining-function combiner \
                       --combiner-path <combiner trained weights path>/combiner.pt \
                       --blip-model-path <BLIP text encoder finetuned weights path>/tuned_blip_best.pt
```
#### CIRR

For validation split:

```bash
python src/validate.py --dataset CIRR \
                       --combining-function combiner \
                       --combiner-path <combiner trained weights path>/combiner_mean.pt \
                       --blip-model-path <BLIP text encoder finetuned weights path>/tuned_blip_mean.pt
```

For test split, the following command will generate `recall_submission_combiner-bi.json` and `recall_subset_submission_combiner-bi.json` at `/submission/CIRR/` for submission:

```bash
python src/cirr_test_submission.py --submission-name combiner-bi \
                                   --combining-function combiner \
                                   --combiner-path <combiner trained weights path>/combiner_mean.pt \
                                   --blip-model-path <BLIP text encoder finetuned weights path>/tuned_blip_mean.pt
```

Our generated `.json` files are also available [here](/submission/CIRR/). To try submitting and receiving the test split results, please refer to [CIRR test split server](https://cirr.cecs.anu.edu.au/).

## Further development

### Hyperparameters

 * reversed loss scale
 * cosine learning rate schedule


## License
MIT License applied. Please also check the licenses from [CLIP4Cir](https://github.com/ABaldrati/CLIP4Cir/blob/master/LICENSE) and [BLIP](https://github.com/salesforce/BLIP/blob/main/LICENSE.txt) as our code is based on theirs.

## Contact

 * Raise a new [GitHub issue](https://github.com/Cuberick-Orion/Bi-Blip4CIR/issues/new)
 * [Contact us](mailto:zheyuan.liu@anu.edu.au?subject=Regarding_Bi-BLIP4Cir)