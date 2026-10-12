# Codebase for Contrastive learning for video reuse detection

This codebase supports the paper "Contrastive learning for video reuse detection" by Arjun Biswas and Tristan J.B. Cann. It applies a constrastive learning approach to identify similar visuals between two sets of videos.

The method begins by filtering each video down to a series of representative frames, i.e. those that capture all unique visual signatures in a given video. Then, the sets of representative frames are compared by the Siamese network to identify cases of repeated visuals.

We demonstrated the utility of this method using [Imagenet100](https://huggingface.co/datasets/clane9/imagenet-100) for training and short video clips collected from [Pixabay](https://pixabay.com/) for testing.

## File descriptions and running order
The following scripts will reproduce all models and results from the paper. Each script contains a description of input parameters and an illustrative example of running the script.

- `composite_videos`: This folder would contain the composite videos used for testing, but they are omitted from this repository for copyright and file size purposes.
	- `Syntethic-Video-Mapping-Groundtruth.csv`: A file that links the composite video names to the underlying ground truth clips that they are comprised of. 

- `data`: This folder collects all relevant data used as inputs for the various scripts, and processed outputs required to reproduce each of the figures.
	- `example_frame.jpg`: An example image to make the augmentation examples. This image was taken by the authors.

- `figs`: This folder contains output figures as produced by each of the included scripts.
	- `blur.jpg`, `brightness.jpg`, `contrast.jpg`, `rotated.jpg`, `saturation.jpg`, `text_overlay.jpg`: Augmented versions of `example_frame.jpg`, included in Fig. 3.

- `Results`: This folder contains similarity results from the comparisons between the composite videos and clips.
	- `Downsampled_Representative_Frames.csv`: Statistics for the number of frames for each video before and after identifying representative frames.
	- `Results_of_Comparison_metrics-Pixabay-{BACKBONE}-{THRESHOLD}-{THRESHOLD}-{THRESHOLD}.csv`: A set of files for each backbone and threshold to assess the performance at recovering the clips that make up each composite video.

- `generate_augementation_examples.py`: This script produces the example images to show each of the augmentations applied during the trainig process, as included in Fig. 3.

- `train.py`: This script trains the Siamese network over one of three backbones (densenet 121, resnet50 or clip-vit-base-patch16) on a dataset of images sampled from Imagenet100.

- `make_composite_videos.py`: This script processes the short video clips from Pixabay into a series of edited videos composed of six 10-second segments. This assumes a directory of mp4 files to sample from. Other file types in the directory will be ignored.

- `test.py`: This script applies the trained model on a set of composite videos and clips, and records relevant information for the reporting of final model performance.

- `Evaluate_results.py`: This script applies the evaluation process to the labels for the generated synthetic data. It calculates the TP,FP,FN,TN and saves them to a csv.

- `plot_figures.ipynb`: A jupyter notebook to compile the remaining figures from the results. Most figures produce `.eps`, although those with transparent elements are exported as `.pdf`.