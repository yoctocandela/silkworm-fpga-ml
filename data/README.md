# Dataset

This project uses the **Silkworm Diseases v1** dataset published by
*Silkworm Annotation* on Roboflow Universe.

The dataset is **NOT redistributed in this repository**.

## How to get it

Download the YOLOv8-format export from:

> <https://universe.roboflow.com/silkworm-annotation-y6ztu/silkworm-diseases-y0bro/dataset/1>

After extracting, the folder layout used by the scripts in this repo is:

```
data/
└── silkworm_dataset_yolov8/
    ├── data.yaml
    ├── README.dataset.txt
    ├── README.roboflow.txt
    ├── train/
    │   ├── images/   (*.jpg)
    │   └── labels/   (*.txt — YOLO format)
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

If you keep the dataset in the repository root (next to `src/`,
`reports/`, etc.), the existing scripts will find it via the relative path
`silkworm_dataset_yolov8/`. If you keep it inside `data/`, edit the
`DATASET_DIR` constant in `src/fpga_classical_best.py` and the
`dataset_path` argument in `src/preprocess.py` accordingly.

## License and attribution

The dataset is released under **CC BY 4.0** by *Silkworm Annotation*.
When using it (for any purpose, including this project), please credit
the dataset owner. A suitable attribution line is:

> Silkworm Annotation, *Silkworm Diseases — v1*. Roboflow Universe, 2024.
> Licensed under CC BY 4.0.
> https://universe.roboflow.com/silkworm-annotation-y6ztu/silkworm-diseases-y0bro/dataset/1

## Dataset statistics used in this work

After YOLO bounding-box cropping, grayscale conversion and 64×64 resize
(for v1) or RGB resize (for v2), the working pools used in the experiments
are:

| Split | Grasserie (class 0) | Healthy (class 1) | Total |
| ----- | -------------------: | -----------------: | -----: |
| Train |                4 000 |              4 174 |  8 174 |
| Valid |                  264 |                249 |    513 |
| Test  |                  249 |                271 |    520 |

The 89 / 6 / 6 train/valid/test split is the one provided in the original
Roboflow export and is preserved verbatim — no shuffle is performed across
split boundaries, eliminating leakage.

## Class-label note (important)

The Roboflow `data.yaml` lists the classes in the following order:

```yaml
names: ['Grasserie', 'Healthy']
```

so **class `0` = Grasserie (diseased)** and **class `1` = Healthy**. The
YOLO label files agree with this convention (Grasserie images have label
id `0`, Healthy images have label id `1`).

An earlier version of `src/strict_validation.py` had these directions
transposed in its metric labels; the bug was caught during the validation
audit and corrected in `src/strict_validation_fixed.py`. All numbers
reported in the project deliverables come from the corrected pipeline.
