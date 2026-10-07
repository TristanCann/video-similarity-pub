import numpy as np
import pandas as pd
import os
import torch
import warnings
import cv2
from train import * # Here we import the Dataset and Siamesenetwork classes.
import tqdm
import gc

warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TestModel():
    def __init__(self,backbone,video_dir,representative_frame_threshold,comparison_threshold,output_dir,calculate_representative=False,batch_size=32):

        '''This function will evaluate the model to compare videos find similar clips between two sets of videos. It takes in the following parameters:
        Inputs :  
            1. backbone : str = The name of the backbone used to train the contrastive learning model. This determines the filename of model weights to load.
            2. video_dir : list[str] = The directory where the videos are stored [clip directory, composite directory]. 
                                The videos should be stored in the following format: video_dir/video_name.mp4. 
                                This parameter should be provided even in calculate_representative=False to determine the locations for the previous representative frames.
            3. representative_frame_threshold : float = The distance threshold to determine if a frame is representative of a video.
            4. comparison_threshold : float = The distance threshold to determine if two frames are similar.
            5. output_dir: str = The folder name for all outputs. This will automatically be appended with the backbone name and similarity thresholds.
            6. calculate_representative : bool = Whether to calculate the representative frames for the videos. Default is False, i.e. use the previously identified representative frames.
            7. batch_size: int = The number of frames to be processed in a single batch. The default is 32, but can be increased where more VRAM is available.'''
        
        self.backbone = backbone
        self.model = Siamesenetwork(self.backbone).to(device)
        self.model_name = f"{self.backbone}_twinnetwork"
        self.model.load_state_dict(torch.load(self.model_name))
        self.calculate_representative = calculate_representative
        self.video_dir = video_dir
        self.representative_frame_threshold = representative_frame_threshold
        self.comparison_threshold = comparison_threshold
        self.output_dir = f"{output_dir}_{backbone}_{representative_frame_threshold}_{comparison_threshold}"
        self.batch_size = batch_size


    def embed_images_iter(self,fname,images): 
        '''This function runs a set of images through the contrastive learning model and returns the final embedded representation.
        Inputs:
            1. fname : str = filename of the video, used when saving model outputs.
            2. images : np.array = a Nx224x224x3 image array.
        Outputs:
            1. output: np.array = a batch size x Y output array from the contrastive learning model. Y depends on the model backbone.
            2. images: np.array = the images from this batch.'''

        # There can be memory errors for longer videos on smaller gpus, we embed the frames in batches for efficiency.

        self.model.eval()
        num_outputs = 0
        for start in range(0, images.shape[0], self.batch_size):
            batch = images[start:start + self.batch_size]

            # Format the images correctly - make sure the colour channels are correct and stored as a torch object.
            batch = torch.from_numpy(batch)
            batch = batch.to(device).float()

            with torch.no_grad():
                if self.backbone == "clip-vit-base-patch16":
                    batch_proc = self.model.processor(images=batch, return_tensors="pt", padding=True, truncation=True).to(device)
                    output_conv = self.model.forward_once(batch_proc)
                else:
                    batch = batch/255.0
                    batch = batch.permute(0,3,1,2)
                    output_conv = self.model.forward_once(batch)
                output = self.model.fc(output_conv).cpu().numpy()

            del batch,output_conv
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            num_outputs += 1

            yield output, images[start:start+self.batch_size]
    
    
    def compare_embeddings(self,reference_embedding,candidate_embeddings):
        '''This function compares the embedding of a single image against a set of embeddings from all other images.
        Inputs:
            1. reference_embedding : np.array = an embedded vector from the contrastive learning model.
            2. candidate_embeddings : np.array = a compiled array of all candidate embeddings to compare the reference image against.
        Outputs:
            1. matches : None or np.array = All frames within the required similarity threshold. None is returned if no frames are matched.
            2. sims : None or np.array = All similarities within the required similarity threshold. None is retured if no frames are matched.'''

        # Calculate the similarity between a single reference embedding and a set of candidate embeddings.
        normalized_1 = torch.nn.functional.normalize(torch.from_numpy(reference_embedding),dim=0)
        normalized_2 = torch.nn.functional.normalize(torch.from_numpy(candidate_embeddings),dim=-1)
        sims = torch.nn.functional.pairwise_distance(normalized_1,normalized_2,p=2).numpy()

        # Identify any similarities that are less than the threshold.
        mask = sims <= self.comparison_threshold 
        matches, sims_np = None, None
        if mask.any():
            matches = np.where(mask != 0)[0]
            sims_np = sims[mask]
        
        del reference_embedding, candidate_embeddings, sims, mask
        torch.cuda.empty_cache()

        # Determine what to return: matches and similarities if some have been found, otherwise None.
        if type(matches) != type(None):
            return matches, sims_np
        else:
            return None, None
        

    def get_representative_frames_iter(self,video_name,all_images):
        """Extract a subsample of the provided video frames that gives a coverage of similarity. For each image, search the similar images and then removes them 
        from the pool of images to search. This reduces computation and more quickly generates representative frames.
        Inputs:
            video_name : str = The filename for the video, used when saving to file.
            all_images : np.array = The video frames to be compared. The dimensions are N x 224 x 224 x 3.
        Outputs:
            None : This method does not return any outputs. The representative frames and similarities are written to file. """

        representative_frames = []
        representative_embeddings = []

        # Produce and run through the iterator of image embeddings and the sets of associated images.
        for embeddings, images in self.embed_images_iter(video_name,all_images):
            cand_frame = embeddings[0]  # This is only to avoid a referencing error.

            ## Remove any frames that match the previous representative frames.
            for rep in representative_embeddings:
                indices,sims = self.compare_embeddings(rep,embeddings)

                # Remove the frames that match the existing representative frame.
                if type(indices) != type(None):
                    mask = np.ones(images.shape[0],dtype=bool)
                    mask[indices] = False
                    images = images[mask]  
                    embeddings = embeddings[mask,:]

            ## Find any new representative frames in this batch.
            while embeddings.shape[0] > 1: # Stop at 1 so there is no error in compare_embeddings when embeddings is empty.
                # Save the next frame and remove it from the set to compare.
                representative_frames.append(images[0])
                representative_embeddings.append(embeddings[0])
                cand_frame = embeddings[0]
                images = images[1:]
                embeddings = embeddings[1:]

                # Compare the new representative frame against all remaining frames.
                indices,sims = self.compare_embeddings(cand_frame,embeddings)

                # Remove the frames that match the new representative frame.
                if type(indices) != type(None):
                    mask = np.ones(images.shape[0],dtype=bool)
                    mask[indices] = False
                    images = images[mask] 
                    embeddings = embeddings[mask,:]

            # Save the last element as a representative frame in this case.
            if embeddings.shape[0] == 1:
                representative_frames.append(images[0])
                representative_embeddings.append(embeddings[0])

            del embeddings, cand_frame

        # Save the embeddings to file.
        with open(f'{self.output_dir}/{video_name.split(".")[0]}_{self.backbone}_rep_embeddings.np','wb') as f:
            np.save(f,np.vstack(representative_embeddings))

        # Save the representative frames to file.
        OUTPUT_DIR = os.path.join(self.output_dir,f"Representative_{self.backbone}",video_name.split(".")[0])

    
    def get_directory_representative_frames(self,directory):
        """Extract the representative frames for all video files in a given directory. These frames are saved, along with their model embeddings.
        Inputs:
            directory : str = Target directory containing the video files.
        Outputs:
            None : This method does not return any outputs."""
        
        # Create the output directories if they don't already exist.
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            os.makedirs(f'{self.output_dir}/batches')

        for video in tqdm.tqdm(os.listdir(directory)):
            if video.endswith('.mp4'):

                # Load all frames
                frames = []
                cap = cv2.VideoCapture(os.path.join(directory,video))
                while True:
                    ret,frame = cap.read()
                    if ret == False:  # We can break when there is no frame to be found, i.e. the end of the video.
                        break
                    else:
                        frames.append(cv2.resize(frame,(224,224)))
                cap.release()

                # Calculate the representative frames.
                self.get_representative_frames_iter(video,np.stack(frames))

                del frames
                torch.cuda.empty_cache()
                gc.collect()


    def find_common_frames(self,video1,video2):
        """Find and return all common frames between video1 and video2.
        Inputs:
            video1 : str = Filename for the query video, i.e. the one matches are sought for.
            video2 : str = Filename for the reference video, i.e. the one matches are sought against.
        Outputs:
            all_matches : list of tuples = List of all common frames, numbered by video2, along with their similarity scores."""
        
        # Load in the frame embeddings.
        video1_embeddings = np.load(f"{self.output_dir}/{video1.split('.')[0]}_{self.backbone}_rep_embeddings.np")
        video2_embeddings = np.load(f"{self.output_dir}/{video2.split('.')[0]}_{self.backbone}_rep_embeddings.np")

        # Compare the embeddings.
        all_matches = []
        for id,frame in enumerate(video1_embeddings):
            matches,sims = self.compare_embeddings(frame,video2_embeddings) # These are the frame indices that match under the current thresholds.
            
            if type(matches) != type(None):
                all_matches += [(id,m,s) for m,s in zip(matches,sims)]

        # Return the list of all matched frames.
        return all_matches

    
    def find_common_frames_directory(self,query_video,reference_directory):
        """Find and return all matched frames between a query video and all reference videos in a given directory.
        Inputs:
            query_video : str = File/directory name for the reference video for comparison.
            reference_directory : str = Directory path containing all videos to check against.
        Outputs:
            matches : list of tuples = List of matching frames for each reference video in the directory. Each tuple contains (frame number,matched video name,matched frame,similarity)."""

        # Get the set of reference video names.
        reference_videos = [f for f in os.listdir(reference_directory) if f.endswith('.mp4')]

        # Get the matches for each reference video.
        matches = []
        for vid in reference_videos:
            matches += [(frame,vid,m,s) for frame,m,s in self.find_common_frames(query_video,vid)]

        # Return the matches.
        return matches
    
    
    def format_model_results(self,matches):
        """Given a set of matched frames, sorted in order of similarity, prepare the match data to a dataframe for output.
        Inputs:
            matches : list of tuples = List of matching frames for each reference video in the directory. Each tuple contains (frame,vid,matched frame,sim).
        Outputs:
            results : pd.DataFrame = A dataframe containing a structured version of all match data."""

        # Matches contains an unsorted list of tuples (query frame, reference video, reference frame, similarity).
        return pd.DataFrame({'frame_id':[m[0] for m in matches],'mapped_video_name':[m[1] for m in matches],'mapped_video_frame':[m[2] for m in matches],'similarity':[m[3] for m in matches]})
    

    def run(self):
        """Run the full testing pipeline."""

        # Check whether the representative frames need to be calculated (based on model parameter), if they do then calculate them.
        if self.calculate_representative:
            print(f'Calculating representative frames for {self.video_dir[0]}')
            self.get_directory_representative_frames(self.video_dir[0])
            print(f'Calculating representative frames for {self.video_dir[1]}')
            self.get_directory_representative_frames(self.video_dir[1])

        # Get the set of query videos to iterate over.
        query_vids = [f for f in os.listdir(self.video_dir[1]) if f.endswith('.mp4')]

        # For each query video.
        print('Identifying matched frames.')
        for query in tqdm.tqdm(query_vids):

            # Find the representative frame matches.
            matches = self.find_common_frames_directory(query,self.video_dir[0])

            # Format the results and save them to file.
            results = self.format_model_results(matches)

            if not os.path.exists(f'{self.output_dir}/Results'):
                os.makedirs(f'{self.output_dir}/Results')

            results.to_csv(f"{self.output_dir}/Results/matches_{self.backbone}_{query.split('.')[0]}.csv")
        
        del self.model
        if device == 'cuda':
            torch.cuda.empty_cache()



if __name__ == "__main__":
    # Use this to control the parameters you wish to test. Model options are densenet121, resnet50, clip-vit-base-patch16.
    clip_dir = "pixabay-vids"
    comp_dir = "composite_videos"
    for backbone in ["resnet50"]:
        for thresh in [0.1]:
            frame_thresh = thresh
            comp_thresh = thresh
            output_dir = "evaluation"
            calc_rep = True
            batch_size = 32
            print("Model backbone",backbone)
            TestModel(backbone=backbone,video_dir = [clip_dir,comp_dir],representative_frame_threshold = frame_thresh,comparison_threshold = comp_thresh,output_dir = output_dir,calculate_representative = calc_rep,batch_size=batch_size).run()
