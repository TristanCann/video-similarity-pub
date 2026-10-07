# Codebase for Contrastive learning for video reuse detection

This codebase supports the paper Contrastive learning for video reuse detection, which uses a constrastive learning approach to identify similar visuals between two sets of videos.

The method begins by filtering each video down to a series of representative frames, i.e. those that capture all unique visual signatures in a given video. Then, the sets of representative frames are compared by the Siamese network to identify cases of repeated visuals.

We demonstrated the utility of this method using [Imagenet100](https://huggingface.co/datasets/clane9/imagenet-100) for training and short video clips collected from [Pixabay](https://pixabay.com/) for testing.

## File descriptions and running order
The following scripts will reproduce all models and results from the paper. Each script contains a description of input parameters and an illustrative example of running the script.

- `train.py`: This script trains the contrastive model over one of three backbones (densenet 121, resnet50 or clip-vit-base-patch16) on a dataset of images sampled from Imagenet100.

- `make_composite_videos.py`: This script processes the short video clips from Pixabay into a series of edited videos composed of six 10-second segments. This assumes a directory of mp4 files to sample from. Other file types in the directory will be ignored.

- `test.py`: This script applies the trained model on a set of videos and records relevant information for the reporting of final model performance.

- `Evaluate_results.py`: This script applies the evaluation process to the labels for the generated synthetic data. It calculates the TP,FP,FN,TN and saves them to a csv.