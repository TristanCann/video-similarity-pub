import os
import numpy as np
import pandas as pd
import cv2
import tqdm
import traceback
import code

class MakeVideos():

	def __init__(self,ouput_dir,clip_dir,sample_length,target_length,num_videos,width,height,fps):
		'''This function initializes the class MakeVideos. It takes in the following parameters:
		Inputs :
			ouput_dir : str = The directory where the final videos will be saved.
			clip_dir : str = The directory where the clips are stored.
			sample_length: int = The length of each sampled clip extract in seconds.
			target_length : int = The length of the final video in seconds.
			num_videos : int = The number of videos to be generated.
			width : int = The width of the final video.
			height : int = The height of the final video.
			fps : int = The frames per second of the final video.'''
		
		self.OUT_DIR = ouput_dir
		self.CLIP_DIR = clip_dir
		self.SAMPLE_LENGTH = sample_length
		self.TARGET_LENGTH = target_length
		self.NUM_VIDEOS = num_videos
		self.WIDTH = width
		self.HEIGHT = height
		self.FPS = fps
	

	def choose_frames_and_construct_videos(self):
		'''
		This function chooses random clips from the clips directory and constructs videos from them. It also saves the mapping of the clips used in each video to a csv file.
		'''

		os.makedirs(self.OUT_DIR,exist_ok=True)

		patch = cv2.imread("Patch.PNG") # Read the filler frame once only.

		## Get the list of all files in the clip directory, we will sample from here.
		clip_files = [i for i in os.listdir(self.CLIP_DIR) if i.endswith('.mp4')]

		## Set up storage for which clips are included in a given sampled video. The mapping is a dictionary with the video name as the key and the list of clips as the value.
		Videos_clips_mapping = {}

		# Check for which videos have already been made. This allows for resuming sampling if the process in interrupted.
		completed = [int(f.split('_')[2].split('.')[0]) for f in os.listdir(self.OUT_DIR) if f.endswith('.mp4')]
		for vid_no in tqdm.tqdm(range(self.NUM_VIDEOS)):
				try:
					if vid_no+1 not in completed:

						# Load the mapping data already completed.
						try:
							df = pd.read_csv(f"{self.OUT_DIR}/Synthetic-Video-Mapping-Groundtruth.csv")
							Videos_clips_mapping = {k:v for k,v in zip(df["Synthetic_Video"],df["Original_Clips"])}

						except FileNotFoundError:
							pass  # We don't need to do anything in this case, there are no previous videos to load the data for.
						
						# We fix the random seed as the video number to ensure reproducibility after interruptions.
						np.random.seed(vid_no)
						
						## Sample clips for the video, roughly one per 10 seconds.
						sampled_clips = list(map(str,np.random.choice(clip_files,self.TARGET_LENGTH//10,replace=False)))  ## Note we do not allow replacement in the selection process.
						Videos_clips_mapping[f"Test_video_{vid_no+1}"] = sampled_clips
						
						## Make the clip order and add in some filler space. Add clips+1 so it could be in between all OP clips.
						video_order = ['filler']*(len(sampled_clips)+1) + sampled_clips
						np.random.shuffle(video_order)

						final_frames = []
						frame_limits = []
						for vid in video_order:
							frame_arrays = []

							if vid == 'filler':
								final_frames.append('filler')
								frame_limits.append('filler')
							else:

								## Convert the video to a set of frames as images. These are PIL objects.
								container = cv2.VideoCapture(f'{self.CLIP_DIR}/{vid}')
								while True:
									ret,frame = container.read()
									if not ret:  # ret determines if the frame read was successful.
										break
									else:
										frame_arrays.append(cv2.resize(np.array(frame),(self.WIDTH,self.HEIGHT)))  # We mostly downsample here for downstream efficiency.
								container.release()
								
								## Pick a random start point for the clip slice. We want to sample exactly 10 seconds.
								start = np.random.randint(0,len(frame_arrays)-(10*self.FPS))
								duration = 10*self.FPS

								if duration <= 0:
									sampled_clips.remove(vid)
									Videos_clips_mapping[f"Test_video_{vid_no+1}"] = ','.join(sampled_clips)

								if start+duration <= len(frame_arrays):
									end = start+duration
								else:
									end = len(frame_arrays)

								frame_limits.append((start,end))

					## Now we replace the 'filler' frames to fill out the video duration.
					part_sizes = [1]*sum([f == 'filler' for f in final_frames])
					missing_frames = 60*self.FPS - sum([f[1]-f[0] for f in frame_limits if f != 'filler']) # This works for the new, more efficient version using only frame boundaries.
					
					while sum(part_sizes) < missing_frames:
						part_sizes[np.random.randint(len(part_sizes))] += 1

					# Now edit frame_limits to record the part sizes in one place.
					out_limits = []
					k = 0
					for f in frame_limits:
						if f == 'filler':
							out_limits.append(('filler',part_sizes[k]))
							k += 1
						else:
							out_limits.append(f)

					fourcc = cv2.VideoWriter_fourcc(*'mp4v')
					video = cv2.VideoWriter(f"{self.OUT_DIR}/Test_video_{vid_no+1}.mp4",fourcc,self.FPS,(self.WIDTH,self.HEIGHT))

					# Use the previously determined frame ranges to extract short sections of each sampled clip.
					for vid,(start,stop) in zip(video_order,out_limits):
						frame_arrays = []
						if start == 'filler':  # We are adding filler frames.
							for l in range(stop):
								video.write(patch)
						else:  # We load and slice the video again.
							container = cv2.VideoCapture(f'{self.CLIP_DIR}/{vid}')
							while True:
								ret,frame = container.read()
								if not ret:  # ret determines if the frame read was successful.
									break
								else:
									frame_arrays.append(cv2.resize(np.array(frame),(self.WIDTH,self.HEIGHT)))  # We mostly downsample here for downstream efficiency.
							container.release()

							frame_arrays = frame_arrays[start:stop]

							for f in frame_arrays:
								video.write(f)
							
					#cv2.destroyAllWindows()
					video.release()
				
					## Save the mapping to a csv file.
					keys = list(Videos_clips_mapping.keys())
					values = list(Videos_clips_mapping.values())

					data = [[keys[i],values[i]] for i in range(len(keys))]
					df = pd.DataFrame(data=data,columns = ["Synthetic_Video","Original_Clips"])
					df.to_csv(f"{self.OUT_DIR}/Synthetic-Video-Mapping-Groundtruth.csv")
				except Exception as e:
                                        print(f"Exception in vid {vid_no} and exception",e)
                                        traceback.format_exc()
                                        continue


if __name__ == "__main__":
	MakeVideos(ouput_dir="composite_videos",clip_dir="pixabay-vids",target_length=60,num_videos=125,width=1920,height=1080,fps=24).choose_frames_and_construct_videos()
