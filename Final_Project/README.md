# Self Driving Vehicle Navigating A Dynamic Environment
- Team Members: Aswath Muthuselvam(118286204), Shailesh Pranav Rajendran(11828)
- Date: 04/29/2022
- Course: ENPM690 - Robot Learning

# Setup code
## Setup dataset
1. Download dataset from https://level-5.global/data/prediction/
2. Organize your `<datatset-folder>` folder in this structure:
```bash
$ tree -L 2
.
├── aerial_map
│   ├── aerial_map.png
│   ├── feedback.txt
│   ├── LICENSE
│   └── nearmap_images
├── meta.json
├── scenes
│   ├── sample_chopped_100
│   ├── sample.zarr
│   ├── train
│   └── train.zarr
└── semantic_map
    ├── feedback.txt
    ├── LICENSE
    └── semantic_map.pb
```


## Setup environment
```bash
# ~/.bashrc
export L5KIT_DATA_FOLDER = <datatset-folder>
```


# Run the code
- Supervised Learning:
```bash
#Training:
jupyter nbconvert --to python code/supervised_learning/train.ipynb

#Inference:
run code/supervised_learning/open_loop_test.ipynb

```

- Reinforcement Learning:
```bash
#Training:
run code/reinforcement_learning/gym_environment_play_and_learn.ipynb
```

# Outputs
Youtube video of simualtion output is available in this [link]().

