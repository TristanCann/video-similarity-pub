import pandas as pd
import numpy as np
import os
import code
import ast
# Groundtruth dataset the contains the mapping between the Pixabay clips used and the video

dataset_groundtruth = pd.read_csv("composite_videos/Synthetic-Video-Mapping-Groundtruth.csv")
model = 'resnet50' #'clip-vit-base-patch16' #'densenet121'
#rep_frame_threshold = 0.6
#match_frame_threshold = 0.6

filter_to_threshold = [
	0.1,
	0.2,
	0.3,
	0.4,
	0.5,
	0.6000000000000001,
	0.7000000000000002,
	0.8000000000000003,
	0.9000000000000001
]

# Aim is to go through each evaluation file and check for common clips between the groundtruth and the matched clips
# Results can be in the form of 2 metrics 
#    * True Positives : Common Files between groundtruth and the Evaluation
#    * False Positives : File selected by the algorithm to be in the clip when it is not in the groundtruth
#    * False Negatives : File not Selected by the algorithm when it is present in the groundtruth
#    * True Negatvies : File not selected by both the algorithm and the groundtruth

def create_results_file(dataset_groundtruth, path, output_path, filter_to_threshold=0.6):

    filenames = [i for i in os.listdir(path)]

    os.makedirs(output_path, exist_ok=True)

    dataset_comparison_results = pd.DataFrame()

    for j in filenames:

        dataset_evaluation = pd.read_csv(f"{path}/{j}")

        # Filter the matches to a smaller threshold (mimics choosing a smaller match_frame_threshold).
        dataset_evaluation = dataset_evaluation[dataset_evaluation['similarity']<=filter_to_threshold]
        
        video_names_found_evaluation = [f.split('.')[0] for f in list(set(dataset_evaluation['mapped_video_name']))]

        # Clip Names Extracted now we can find common filenames

        Evaluation_row = "Test_video_" + j.split("_")[-1].split(".")[0]
        extracted_row = dataset_groundtruth[dataset_groundtruth["Synthetic_Video"]==Evaluation_row].iloc[0]["Original_Clips"]
        groundtruth_clips = [i.split(".mp4")[0].strip() for i in ast.literal_eval(extracted_row.replace("'",'"'))]  # Conversion here is required to ensure the 

        # We are calculating the metrics and for errors we are also storing the filenames
        total_groundtruth = len(set(groundtruth_clips))
        true_positives_len = len(set(groundtruth_clips).intersection(set(video_names_found_evaluation)))
        false_positives = ",".join(set(video_names_found_evaluation) - set(groundtruth_clips))
        false_positives_len = len(set(video_names_found_evaluation) - set(groundtruth_clips))
        false_negatives = ",".join(set(groundtruth_clips) - set(video_names_found_evaluation))
        false_negatives_len = len(set(groundtruth_clips) - set(video_names_found_evaluation))

        #code.interact(local=locals())    

        # Storing things in a dataframe
        df_temp = pd.DataFrame(data= [[j,total_groundtruth,true_positives_len,false_positives_len,false_positives,false_negatives_len,false_negatives]],columns=["Filename","True Clips","True Positives","False Positives Num","False Positives","False Negatives Num","False Negatives"])

        dataset_comparison_results = pd.concat([dataset_comparison_results,df_temp])

    dataset_comparison_results.to_csv(f"{output_path}/Results_of_Comparison_metrics-Pixabay-{model}-{filter_to_threshold}-{filter_to_threshold}-{filter_to_threshold}.csv")

'''def create_top_k(dataset_groundtruth, path, output_path):

    filenames = [i for i in os.listdir(path) if (not (i.startswith("Synthetic"))) and i.split("_")[1]=="all"]
    #print(filenames)

    dataset_comparison_results = pd.DataFrame()
    r10 = []
    r5 = []
    r1 = []

    r10_1 = []
    r5_1 = []
    r1_1 = []
    os.makedirs(output_path, exist_ok=True)

    for filename in filenames:

        dataset_evaluation = pd.read_csv(f"{path}/{filename}")
        video_names_found_evaluation = []
        similarity_found_evaluation = []
        video_score_mapping = {}

        
        # For each frame there may be multiple matches comma separated

        for i,j in dataset_evaluation.iterrows():
            video_names = j["mapped_video_name"].split(",")
            video_names = [k.strip() for k in video_names]
            video_names_found_evaluation.extend(video_names)
            similarity = j["similarity"].split(",")
            similarity = [k.strip() for k in similarity]
            similarity = [float(k) for k in similarity]
            similarity_found_evaluation.extend(similarity)
        
        for i,j  in zip(video_names_found_evaluation, similarity_found_evaluation):
            if i not in video_score_mapping:
                video_score_mapping[i] = [j]
            else:  
                video_score_mapping[i].append(j)
        
        video_score_mapping = {key:np.min(value) for key, value in video_score_mapping.items()}
        video_score_mapping = sorted(video_score_mapping.items(), key=lambda x: x[1])
        video_score_mapping = dict(video_score_mapping)
        Evaluation_row = "Test_video_" + filename.split("_")[4].split(".")[0]
        extracted_row = dataset_groundtruth[dataset_groundtruth["Synthetic_Video"]==Evaluation_row].iloc[0]["Original_Clips"]
        groundtruth_clips = [i.split(".mp4")[0].strip() for i in extracted_row.split(",")]

        top_all = list(video_score_mapping.keys())[:12]
        present10 = [i for i in groundtruth_clips if i in top_all]
        present5 = [i for i in groundtruth_clips if i in top_all[:6]]
        present1 = [i for i in groundtruth_clips if i in top_all[:1]]

        if len(present10)>=1:
            r_single_10 = (len(present10)/len(groundtruth_clips))
            r_single_10_1 = 1
        else:
            r_single_10 = 0
            r_single_10_1 = 0
        
        if len(present5)>=1:
            r_single_5 = len(present5)/len(groundtruth_clips)
            r_single_5_1 = 1
        else:
            r_single_5 = 0
            r_single_5_1 = 0  
        
        if len(present1)>=1:
            r_single_1 = len(present1)/len(groundtruth_clips)
            r_single_1_1 = 1
        else:
            r_single_1 = 0
            r_single_1_1 = 0

        r10.append(r_single_10)
        r5.append(r_single_5)
        r1.append(r_single_1)

        r10_1.append(r_single_10_1)
        r5_1.append(r_single_5_1)
        r1_1.append(r_single_1_1)
    
    r10 = np.array(r10)
    r5 = np.array(r5)
    r1 = np.array(r1)

    r10_1 = np.array(r10_1)
    r5_1 = np.array(r5_1)
    r1_1 = np.array(r1_1)
    
    print(r10.shape,r5.shape,r1.shape)
    print("Recall 12 Proportion: ",np.round(r10,2))
    print("Recall 6 proportion: ",np.round(r5,2)) 
    print("Recall 1 proportion: ",np.round(r1,2))

    print("Recall 12 : ",np.mean(r10_1))
    print("Recall 6 : ",np.mean(r5_1)) 
    print("Recall 1 : ",np.mean(r1_1))
    #dataset_comparison_results = pd.DataFrame(data= [[filenames[i],r10[i],r5[i],r1[i]] for i in range(len(filenames))],columns=["Filename","R10","R5","R1"])
    #dataset_comparison_results.to_csv(f"{output_path}/Results_of_Comparison_metrics-Pixabay-new-topk.csv")'''
        

if __name__ == "__main__":

    out_path = 'Results'

    for thresh in filter_to_threshold:
        result_path = f'Results/evaluation_{model}_{thresh}_{thresh}/Results'
        create_results_file(dataset_groundtruth, result_path, out_path, thresh)
    #create_top_k(dataset_groundtruth, result_path, out_path)