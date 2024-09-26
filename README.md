# Joint Multi-Unimodal Attention using Fusion for images.
Joint multiple features in Unimodal attention model with fusion.

the dataset is structured as follow:

```
temp_lfw
|
|__ CK
|    |__ 10 REP
|    |   |__ A
|    |   |   |__ images_0.npy
|    |   |   |__ images_1.npy
|    |   |       |__ ...
|    |   |__ L
|    |   |__ LA
|    |__ 20 REP
|    |__ 30 REP
|__JAFFE
```
there are 3 main files:
 - main.py: Script to train, validate and test with LOSO protocol and PCA
 - main_noloso.py: Script to train, validate and test without LOSO protocol and PCA
 - main_nopca.py: Script to train, validate and test without LOSO protocol and without PCA (using a FC layer instead).

for each main file there are env variables to be defined at the begining.

```
pca_size = int(os.getenv('PCA_SIZE', 150))
epochs = int(os.getenv('EPOCHS', 50))
nro_rep = int(os.getenv('NRO_REP', 10))
kind_rep = os.getenv('KIND_REP', 'L')
dataset_target = os.getenv('DATASET_TARGET', 'JAFFE')
folder_path_rep = os.getenv('FOLDER_PATH_REP', 'temp2')
```


## Setup

Build an image using `Dockerfile_pytorch`, create a container with that image and run the script `python main.py`.