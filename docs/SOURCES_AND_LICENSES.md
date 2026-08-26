# Sources and licenses

This document records the source, version and license status of data, model
weights and software used by the released experiment. It is documentation, not
legal advice. Users remain responsible for reviewing the linked license terms
for their intended use.

## Data

| Resource | Use | Source | License/status |
|---|---|---|---|
| MOVi-A 128x128 validation | 50-video synthetic benchmark, RGB, masks, depth, calibration and simulator diagnostics | https://github.com/google-research/kubric/tree/main/challenges/movi and `gs://kubric-public/tfds/movi_a/128x128/1.0.0/` | Kubric repository and MOVi generation materials are Apache License 2.0. The public MOVi documentation does not state a separate license in the dataset page; users should retain Kubric attribution and confirm any institutional redistribution requirements. This repository downloads but does not redistribute TFRecords. |
| MOVi-D 128x128 validation | Fixed-camera control for the camera-pose extension; RGB, masks, depth, intrinsics, camera pose and simulator metadata | https://github.com/google-research/kubric/tree/main/challenges/movi and `gs://kubric-public/tfds/movi_d/128x128/1.0.0/` | Generated and published with Kubric under the upstream Apache-2.0 project terms; no separate dataset-page license was identified. TFRecords are downloaded but are not part of the source-code release. |
| MOVi-E 128x128 validation | Moving-camera confirmatory data for the camera-pose extension with the same oracle geometry channels | https://github.com/google-research/kubric/tree/main/challenges/movi and `gs://kubric-public/tfds/movi_e/128x128/1.0.0/` | Generated and published with Kubric under the upstream Apache-2.0 project terms; no separate dataset-page license was identified. TFRecords are downloaded but are not part of the source-code release. |
| CLEVR-style MOVi-A assets | Shapes, colors, materials and fixed-camera synthetic scenes | https://github.com/google-research/kubric/blob/main/challenges/movi/README.md | Documented as part of MOVi-A/Kubric; subject to the upstream terms above. |

Recommended citation: Klaus Greff et al., “Kubric: A Scalable Dataset
Generator,” CVPR 2022, https://arxiv.org/abs/2203.03570.

## Model and pretrained weights

| Resource | Version/identifier | Source | License/status |
|---|---|---|---|
| ResNet-18 architecture and torchvision implementation | torchvision 0.17.2 | https://github.com/pytorch/vision and https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18 | BSD-style torchvision license; bundled notice in `third_party_licenses/torchvision/`. |
| Frozen pretrained checkpoint | `ResNet18_Weights.IMAGENET1K_V1`; SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` | https://download.pytorch.org/models/resnet18-f37072fd.pth | Distributed by PyTorch/torchvision. The weights were trained on ImageNet-1K. PyTorch notes that pretrained weights can carry their own dataset-derived terms; users should separately assess ImageNet terms for their use. The checkpoint is downloaded and not redistributed here. |
| ImageNet-1K | Upstream training data for the frozen checkpoint only; not downloaded by this experiment | https://www.image-net.org/ | ImageNet access/use terms are controlled by ImageNet. No ImageNet images are included. |

## Direct Python dependencies

| Package | Pinned version | Role | License | Official source |
|---|---:|---|---|---|
| numpy | 1.26.4 | Arrays, geometry, feature storage | BSD-3-Clause plus bundled notices | https://numpy.org/ |
| Pillow | 12.3.0 | PNG crop/mask I/O and gallery thumbnails | MIT-CMU/HPND-style | https://python-pillow.org/ |
| tfrecord | 1.14.6 | TensorFlow-free TFRecord reader | MIT | https://github.com/vahidk/tfrecord |
| torch | 2.2.2 | Frozen neural encoder runtime | BSD-3-Clause plus notices | https://pytorch.org/ |
| torchvision | 0.17.2 | ResNet-18 definition, weights and transforms | BSD-style | https://github.com/pytorch/vision |
| scikit-learn | 1.7.1 | Logistic regression and metrics | BSD-3-Clause | https://scikit-learn.org/ |
| joblib | 1.5.1 | Compact model serialization | BSD-3-Clause | https://joblib.readthedocs.io/ |

## Locked transitive dependencies

| Package | Version | License | Official source |
|---|---:|---|---|
| crc32c | 2.8 | LGPL-2.1-or-later; bundled third-party notices | https://github.com/ICRAR/crc32c |
| filelock | 3.32.2 | MIT | https://github.com/tox-dev/py-filelock |
| fsspec | 2026.7.0 | BSD-3-Clause | https://github.com/fsspec/filesystem_spec |
| Jinja2 | 3.1.6 | BSD-3-Clause | https://github.com/pallets/jinja |
| MarkupSafe | 3.0.3 | BSD-3-Clause | https://github.com/pallets/markupsafe |
| mpmath | 1.3.0 | BSD-3-Clause | https://mpmath.org/ |
| networkx | 3.6.1 | BSD-3-Clause | https://networkx.org/ |
| protobuf | 7.35.1 | BSD-3-Clause | https://github.com/protocolbuffers/protobuf |
| scipy | 1.17.1 | BSD-3-Clause plus bundled notices | https://scipy.org/ |
| sympy | 1.14.0 | BSD-3-Clause | https://www.sympy.org/ |
| threadpoolctl | 3.6.0 | BSD-3-Clause | https://github.com/joblib/threadpoolctl |
| typing_extensions | 4.16.0 | PSF-2.0 | https://github.com/python/typing_extensions |

Exact installation pins are in `requirements-lock.txt`. License texts captured
from the tested wheel metadata are retained under `third_party_licenses/`.

## Experiment code and generated outputs

The original experiment code and generated tables/gallery have no open-source
license assigned by this release. See `LICENSE.md`. Generated records derived
from MOVi-A, MOVi-D, or MOVi-E should retain upstream attribution and remain subject to applicable
upstream terms. No trademark rights are granted.
