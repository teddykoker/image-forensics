# ImageForensics - *Unofficial* PyTorch Implementation

**Learning Manipulation-Invariant Image Similarity for Detecting Re-Use of Images
in Scientific Publications**

Paper: https://arxiv.org/abs/1802.06515<br>
Website: https://hms-idac.github.io/ImageForensics<br>

## Example Detections

![correct](figures/correct.png)

## Requirements

 * Tested on Ubuntu 18.04.3 LTS, Python 3.7.5, PyTorch 1.2.0


## Download Data

```shell
make download
```

## Train

```shell
python train.py
```

## Test

View test examples

```shell
python test.py
```

## Example Incorrect Predictions

### Should Detect Same

![wrong same](figures/same.png)

### Should Detect Different

![wrong diff](figures/diff.png)


