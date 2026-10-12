import numpy as np
import os
import torch
from torchvision import models
from torch.utils.data import Dataset as DS
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import time
import warnings
import cv2
import json
from sklearn.metrics import confusion_matrix
import torchvision.transforms.functional as TF
import torch.utils.checkpoint as cp
from transformers import CLIPProcessor,CLIPModel
warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from PIL import Image

print(device)

class Dataset(DS):

    '''This is the dataset class for the training and testing of the model. It is the format used by pytorch dataloader to load the images and labels.'''
    
    def __init__(self,X,y):
        self.data = (X,y)
    
    def __len__(self):
        return self.data[0].shape[0]
    
    def __getitem__(self,ind):
        x1 = self.data[0][ind,0]
        x2 = self.data[0][ind,1]
        y = self.data[1][ind]
        return x1,x2,y



class Siamesenetwork(torch.nn.Module):
    
    '''This is the model proposed. It is a siamese network with a variable backbone. The model is trained on the Imagenet 100 dataset. 
    The convnet and the fc layers are sharing weights.
    Inputs : A pair of images of shape 3 x 224 x 224.'''

    def __init__(self,backbone):
        super().__init__()

        self.sigmoid = torch.nn.Sigmoid()
        self.backbone = backbone
        if self.backbone == 'clip-vit-base-patch16':
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        if self.backbone == 'densenet121':
            # We are using the densenet121 model as the backbone and removing the last layer
            self.densenet = models.densenet121(weights='DEFAULT')
            self.convnet = torch.nn.Sequential(
                *list(self.densenet.children())[:-1]
            )

            self.fc = torch.nn.Sequential(
                torch.nn.Linear(1024,1024),
                torch.nn.ReLU(inplace=True)
            )
            
            self.convnet.apply(self.init_weights)
        
        elif self.backbone == 'resnet50':
            self.densenet = models.resnet50(weights='DEFAULT')
            self.convnet = torch.nn.Sequential(
                *list(self.densenet.children())[:-1]
            )

            self.fc = torch.nn.Sequential(
                torch.nn.Linear(2048,1024),
                torch.nn.ReLU(inplace=True)
            )
            
            self.convnet.apply(self.init_weights)
        
        elif self.backbone == 'clip-vit-base-patch16':
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

            self.fc = torch.nn.Sequential(
                torch.nn.Linear(512,512),
                torch.nn.ReLU(inplace=True)
            )
            
        self.fc.apply(self.init_weights)
    

    def init_weights(self, m):
        """Initialize weights for Linear layers using Xavier uniform initialization and set biases to 0.01."""

        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


    def forward_once(self,x):
        """Run a forward pass of the convolutional network for a single input, applying global average pooling to produce a feature representation."""

        if self.backbone == 'clip-vit-base-patch16':
           # x_proc = self.processor(images=x, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
               output1 = self.clip.get_image_features(**x).pooler_output
        else:
            output1 = self.convnet(x)
            output1 = output1.mean(dim = (-2,-1))
        return output1
    

    def forward(self,input1,input2):
        """Run a forward pass for a pair of inputs. Each input is processed through the convolutional backbone and a fully connected layer. Returns both the final outputs and the intermediate convolutional features for each input."""

        output_conv_1 = self.forward_once(input1)
        output_conv_2 = self.forward_once(input2)
        output1 = self.fc(output_conv_1)
        output2 = self.fc(output_conv_2)
        return output1,output2,output_conv_1,output_conv_2



class TrainModel():

    def __init__(self,path,backbone,batch_size,epochs,lr,distances,threshold=0.6,augment=False,augmentation_details={},seed = 42):

        ''' This is the wrapper class for training the contrastive learning model.
        Inputs : 
            1. path :  str = Directory name of the imagenet 100 data.  
            2. backbone: str = Name of the model backbone to train on. Options are densenet121, resnet50, clip-vit-base-patch16.
            3. batch_size : int = Number of images per training batch.
            4. epochs : int = Number of epochs to train the model.
            5. lr : float = Learning rate of the model.
            6. distances : list = List of sampling distances for choosing the labels from the dataset. For each value n, take every nth class.
            7. threshold : float = Threshold for the distance between the embeddings for predicting similar images.
            8. augment : bool = Whether to augment the images or not.
            9. augmentation_details : dict = Dictionary containing the details of the augmentations.
            10. seed: int = Seed to fix numpy.random state.
            
            augmentations_details (example) = {
                "augmentations":[<list of augmentations between 1-6. See below for the list of augmentations>],
                "augmentation_percentage":<float between 0-1. Indicates how much of the data to augment>,
                "augmentation_parameters":<list of parameters for the augmentations in order of level,angle,text>
                }
                
            List of augmentations :
            1. Brightness : Adjustment level sampled from a Normal(1,0.5) distribution.
            2. Saturation : Adjustment level sampled from a Normal(1,0.5) distribution.
            3. Sharpness : Adjustment level sampled from a Normal(1,0.5) distribution.
            4. Gaussian Blur : Random choice of kernel size from {5,7,9}.
            5. Rotation : Angle sampled uniformly from [-30,30].
            6. Text Overlay : Requires text to be overlayed on the image as an input.'''

        self.path = path
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.distances = distances
        self.backbone = backbone
        self.model = Siamesenetwork(self.backbone).to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=self.lr)
        self.threshold = threshold
        self.augment = augment
        self.augmentation_details = augmentation_details
        self.seed = seed
        if self.backbone == 'clip-vit-base-patch16':
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")


    def prepare_dataset(self):

        ''' This function return a pytorch dataloader which has access to all the images pairs and the labels.
        Inputs : 
        1. Path : str = Path where the images are and labels are loaded. It assumes a json mapping of the image names to the labels.
        2. batch_size : int = Number of images per batch.
        3. distances : list = List of sampling distances for choosing the labels from the dataset. For each value n, take every nth class.

        Outputs : 
        1. train_loader : pytorch dataloader object = Object that contains batches of image pairs and labels. Labels are 1 for different and 0 for same.
        2. test_loader : pytorch dataloader object = Object that contains batches of image pairs and labels. Labels are 1 for different and 0 for same.
        '''

        classes = [i for i in os.listdir(f"{self.path}/train") if not i.endswith(".json")]
        total_images = [len(list(os.listdir(f"{self.path}/train/{i}"))) for i in classes]

        print(f"Total number of images {sum(total_images)}")

        # load the json file which contains all the labels. The structure is every directory in the main image directory represents a label and has corresponding images of those labels.
        with open(f"{self.path}/Labels.json",'r') as file:
            labels = json.load(file)
        
        #  We are choosing labels from the dataset at distance d where d belongs to the input parameter distances.
        labels_per_distance = []
        
        # Here for every d in distances we are selecting the classes that will be chosen from the Imagenet 100 directory.
        for i in self.distances:
            for j in range(0,len(labels.keys()),i):
                labels_per_distance.append(list(labels.keys())[j])
        
        # Pairing of labels into similar and disimilar pairs. Similar pairs are images with themself. Dissimilar pairs are chosen randomly from another class.
        similar_pairs = [[i,i] for i in labels_per_distance]
        dissimilar_pairs = [[i,np.random.choice([x for x in labels_per_distance if x!=i])] for i in labels_per_distance]
        
        # Now we will sample the 400 images from the directories and create the pairs.
        indices_similar = np.random.choice(range(1300),400,replace=False)
        indices_different = np.random.choice(range(1300),400,replace=False)
        similar_pairs_images = np.array([list(zip(np.array(os.listdir(f"{self.path}/train/{i[0]}"))[indices_similar] , np.array(os.listdir(f"{self.path}/train/{i[1]}"))[indices_similar])) for i in similar_pairs])
        dissimilar_pairs_images = np.array([list(zip(np.array(os.listdir(f"{self.path}/train/{i[0]}"))[indices_different] , np.array(os.listdir(f"{self.path}/train/{i[1]}"))[indices_different])) for i in dissimilar_pairs])

        # Save a the sampled image filenames for future reference.
        with open(f'similar_class_images_{self.backbone}_{self.seed}.txt', 'w') as f:
            for pair in similar_pairs_images:
                f.write(f'{pair[0]}\n')  # The pair is an image with itself.
        
        with open(f'dissimilar_class_images_{self.backbone}_{self.seed}.txt', 'w') as f:
            for pair in dissimilar_pairs_images:
                f.write(f'{pair[0],pair[1]}\n')  ## This needs to record both images.

        # Save the images sampled to file for future comparisons.
        similar_pairs = []

        print(f"Image shapes are Similar pairs : {similar_pairs_images.shape}. Dissimilar pairs are : {dissimilar_pairs_images.shape}.")

        # Finally flattening the image pairs into num_class x num_images x 2.
        similar_pairs_images = similar_pairs_images.reshape(-1,2)
        dissimilar_pairs_images = dissimilar_pairs_images.reshape(-1,2)

        print(f"Image shapes after flattening are Similar pairs : {similar_pairs_images.shape}. Dissimilar pairs are : {dissimilar_pairs_images.shape}.")

        # Creating labels for the pairs.
        similar_labels = np.ones((similar_pairs_images.shape[0]))
        dissimilar_labels = np.zeros((dissimilar_pairs_images.shape[0]))

        # Finally creating the dataset.
        dataset = np.vstack([similar_pairs_images,dissimilar_pairs_images])
        labels = np.hstack([similar_labels,dissimilar_labels])

        print(f"Final Dataset Shape : {dataset.shape}. Labels Shape : {labels.shape}.")

        # Spliting the dataset into train and test split and creating the dataloaders.
        X_train, X_test, y_train, y_test = train_test_split(dataset,labels,test_size=0.3,shuffle=True,random_state=43)
        train_data = Dataset(X_train,y_train)
        test_data = Dataset(X_test,y_test)
        train_loader = DataLoader(train_data,batch_size=64,shuffle=True,num_workers=4)
        test_loader = DataLoader(test_data,batch_size=64,shuffle=True,num_workers=4)

        return train_loader,test_loader
    

    def contrastive_loss(self,y, d, m=1):

        ''''Distance-based constrastive loss proposed by Hadsell et al., 2006.
        Inputs : 
        1. y : tensor = Pairwise distances between the embeddings.
        2. d : tensor = Binary labels for the pairs.
        3. m : float = Margin for the loss.
        
        ouptut loss : tensor  = Contrastive loss'''

        loss_pos = torch.mean(y**2)
        delta = m - y
        delta = torch.clamp(delta, min=0, max=None)
        loss_neg = torch.mean(delta**2)
        
        # Combine losses based on labels.
        loss = torch.mean((1-d) * loss_pos + (d) * loss_neg)
        loss.requires_grad = True
        
        return loss
    

    def transform_images(self, images):
        '''
        Function to transform images with varying augmentation levels per image.
        
        Inputs:
            1. images : tensor = Tensor of shape (N, 3, 224, 224).
        Outputs:
            1. images : tensor = Tensor of shape (N, 3, 224, 224).
        '''

        operations = self.augmentation_details.get("augmentations", None)
        batch_size = images.size(0)
        transformed_images = []

        # Convert to NumPy for CV2 text operation.
        image_numpy = images.permute(0, 2, 3, 1).cpu().numpy()

        for i in range(batch_size):
            img = image_numpy[i]
            img_tensor = images[i]
            

            if operations:
                if 6 in operations:
                    text = self.augmentation_details['augmentation_parameters']['text']
                    img_cv = img.copy()
                    img_cv = cv2.putText(
                        img=img_cv,
                        text=text,
                        org=(50, 50),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(255, 255, 255),
                        thickness=3,
                        lineType=cv2.LINE_AA
                    )
                    img_tensor = torch.from_numpy(img_cv).permute(2,0,1)

                if 1 in operations:
                    level = float(np.clip(np.random.normal(1.5, 0.5),0.2,3))
                    
                    img = TF.adjust_brightness(img_tensor, level)
                if 2 in operations:
                    level = float(np.clip(np.random.normal(1.5, 0.5),0.2,3.0))
                    
                    img = TF.adjust_saturation(img_tensor, level)
                if 3 in operations:
                    level = float(np.clip(np.random.normal(1.5, 0.5),0.2,3.0))
                    
                    img = TF.adjust_sharpness(img_tensor, level)
                if 4 in operations:
                    kernel_size = np.random.choice([5,7,9])  # or sample radius.
                    
                    img = TF.gaussian_blur(img_tensor, int(kernel_size))
                if 5 in operations:
                    angle = np.random.uniform(-30, 30)
                    img = TF.rotate(img_tensor, angle)

            transformed_images.append(img_tensor)

        # Stack into a single tensor of shape (N, 224, 224,3).
        return torch.stack(transformed_images)


    def process_images_for_transformation(self,images):

        '''Function to transform images based on the augmentation details. We take the first n% of the images after shuffling and augment them.
        Inputs :
            1. images : numpy ndarray = Array of images of shape N x 224 x 224 x 3.
        Outputs :
            1. images_aug : numpy ndarray = Array of images of shape N x 224 x 224 x 3.'''
        
        # Randomly shuffle the images so the augmentation can be applied to only the first percentage.
        #np.random.shuffle(images)

        # Determine which images should be augmented.
        images_aug = None
        images = images.copy()
        
        if self.augmentation_details.get("augmentation_perc",None) is not None:
            images_aug = images[:int(self.augmentation_details["augmentation_perc"]*images.shape[0])]
        images_aug = images_aug[:,:,:]
        images_aug = torch.from_numpy(images_aug).permute(0,3,1,2)
        
        # Apply the transformations and convert back to the proper format.

        img_trans = self.transform_images(images_aug)
        img_trans_numpy = np.array(img_trans.permute(0,2,3,1))
        
        images[:int(self.augmentation_details["augmentation_perc"]*images.shape[0]),:,:] = img_trans_numpy
        return images
    
    

    def load_images_from_path(self,imagepaths):

        '''Utility function to load images from the path and return the images as tensors.
        Inputs :
            1. imagepaths : list = List of image paths.
        Outputs :
            1. images : tensor = Tensor of images of shape N x 224 x 224 x 3.
        '''
        if self.backbone == "clip-vit-base-patch16":
          images = np.array([cv2.cvtColor(cv2.resize(cv2.imread(f"{self.path}/train/{i.split('_')[0]}/{i}"),(224,224),interpolation=cv2.INTER_CUBIC),cv2.COLOR_BGR2RGB) for i in imagepaths]) 
        else:
          images = np.array([cv2.cvtColor(cv2.resize(cv2.imread(f"{self.path}/train/{i.split('_')[0]}/{i}"),(224,224),interpolation=cv2.INTER_CUBIC),cv2.COLOR_BGR2RGB)/255.0 for i in imagepaths]) 
 
        return images
    
    def predict_with_test_loader(self,test_loader):

        '''This function predicts the embeddings for the test set and returns the embeddings and the ground truth labels.
        Inputs : 
            1. test_loader : pytorch dataloader object = Dataloader object for the test set.
        Outputs :
            1. predictions1 : list = List of embeddings for the first image in the pair.
            2. predictions2 : list = List of embeddings for the second image in the pair.
            3. truth : list = List of ground truth labels.'''

        predictions1 = []
        predictions2 = []
        truth = []
        self.model.eval()
        with torch.no_grad():
            for x1,x2,y in test_loader:
                x1 = self.load_images_from_path(x1)
                x2 = self.load_images_from_path(x2)

                if self.backbone == 'clip-vit-base-patch16':
                    x1 = self.processor(images=x1, return_tensors="pt", padding=True, truncation=True).to(device)
                    x2 = self.processor(images=x2, return_tensors="pt", padding=True, truncation=True).to(device)

                else:
                    x1 = torch.from_numpy(x1)
                    x2 = torch.from_numpy(x2)
                    x1 = x1.permute(0,3,1,2)
                    x2 = x2.permute(0,3,1,2)
                    x1 = x1.to(device).float()
                    x2 = x2.to(device).float()
                
                truth.append(y)
                prediction1,prediction2,_,_ = self.model(x1,x2)
                prediction1 = torch.nn.functional.normalize(prediction1,dim=1)
                prediction2 = torch.nn.functional.normalize(prediction2,dim=1)
                for i in range(len(prediction1)):
                    predictions1.append(prediction1[i])
                    predictions2.append(prediction2[i])
        return predictions1,predictions2,truth


    def train(self):

        '''This function trains the model and returns the model object and stores the weights in the directory. It also return results on the test set.
        Image paths are only stored in the dataloaders. Train function loads the images per batch and trains the model.'''
        
        # Seed the numpy RNG.
        np.random.seed(self.seed)

        train_loader,test_loader = self.prepare_dataset()
        self.model.train()
        losses = []
        start  = time.time()
        # Training the model.
        for epoch in range(self.epochs):
            for batch_num,inputs in enumerate(train_loader):
                self.optimizer.zero_grad()
                x1,x2,y = inputs

                # Convert paths to tensors.
                if self.backbone == 'clip-vit-base-patch16':
                    x1 = self.load_images_from_path(x1)
                    x2 = self.load_images_from_path(x2)
                else:
                    x1 = self.load_images_from_path(x1)
                    x2 = self.load_images_from_path(x2)
                if self.augment:
                    x1 = self.process_images_for_transformation(x1)

                if self.backbone == 'clip-vit-base-patch16':
                    x1 = [Image.fromarray(img.astype(np.uint8)) for img in x1]
                    x2 = [Image.fromarray(img.astype(np.uint8)) for img in x2]
                    x1 = self.processor(images=x1, return_tensors="pt", padding=True, truncation=True).to(device)
                    x2 = self.processor(images=x2, return_tensors="pt", padding=True, truncation=True).to(device)

                else:
                    x1 = torch.from_numpy(x1)
                    x2 = torch.from_numpy(x2)

                    # Load the tensors to the GPU or device. Finally convert them to N x 3 x 224 x 224 format as required by the model.
                    x1 = x1.permute(0,3,1,2)
                    x2 = x2.permute(0,3,1,2)
                    x1 = x1.float().to(device)
                    x2 = x2.float().to(device)

                y = y.to(device)

                # Model outputs are normalized and then the pairwise distance is calculated.
                output1,output2,_,_ = cp.checkpoint(self.model,x1,x2)
                normalized_1 = torch.nn.functional.normalize(output1,dim=-1)
                normalized_2 = torch.nn.functional.normalize(output2,dim=-1)
                output = torch.nn.functional.pairwise_distance(normalized_1,normalized_2,p=2)
                #if batch_num%25 == 24:
                    #print("output distances normalized",output)
                    #print("output distances unormalized",torch.nn.functional.pairwise_distance(output1,output2,p=2))
                    #print("labels",y)

                # Finally loss is calculated and backpropagated.
                loss = self.contrastive_loss(output.float(),y,m=2)
                loss.backward()
                losses.append(loss.item())
                self.optimizer.step()
                print("Batch Number : ",batch_num)
                if batch_num%25 == 24:
                    print('Epoch %d | Loss %6.2f ' % (epoch, sum(losses)/len(losses)))
                    losses = []
        
        torch.save(self.model.state_dict(), f"{self.backbone}_{self.seed}_twinnetwork")
        end = time.time()
        print("Training Complete and model saved. Time taken : ",end-start)

        print("Evaluating the model on the test set")

        # Get the predictions on the test set.
        predictions1,predictions2,truth = self.predict_with_test_loader(test_loader)

        predictions_1 = torch.stack(predictions1)
        predictions_2 = torch.stack(predictions2)
        truth = torch.cat(truth)
        truth = truth.cpu().numpy()

        sim = torch.norm(predictions_1 - predictions_2, dim=1).cpu()
        print("Distances in test",sim)
        print("Actual labels",truth)
        for threshold in np.arange(0.1,0.9,0.1):
            predicted_labels = (sim <= threshold).long()
            predicted_labels = predicted_labels.cpu().numpy()
            # Calculate and save the confusion matrix.
            cm = confusion_matrix(truth,predicted_labels)
            print(f"Confusion Matrix : {cm}")
        np.save(f"{backbone}_confusion_matrix.npy",np.array(cm))



if __name__ == "__main__":
    # Choose the type of backbone. Options are densenet121, resnet50, clip-vit-base-patch16
    backbone = 'resnet50'

    # Example of training the model - arameter descriptions below
    ''' 
        1. path :  str = Directory name of the imagenet 100 data.  
        2. backbone: str = Name of the model backbone to train on. Options are densenet121, resnet50, clip-vit-base-patch16.
        3. batch_size : int = Number of images per training batch.
        4. epochs : int = Number of epochs to train the model.
        5. lr : float = Learning rate of the model.
        6. distances : list = List of sampling distances for choosing the labels from the dataset. For each value n, take every nth class.
        7. threshold : float = (Optional, default=0.6) Threshold for the distance between the embeddings for predicting similar images.
        8. augment : bool = (Optional, default=False) Whether to augment the images or not.
        9. augmentation_details : dict = (Optional, default={}) Dictionary containing the details of the augmentations.
        10. seed: int = (Optional, default=42) Seed to fix numpy.random state.
    '''

    TrainModel("Imagenet-100",backbone,64,5,1e-3,[5],augment=True,augmentation_details={"augmentations":[1,2,3,4,5,6],"augmentation_perc":0.2,"augmentation_parameters":{"text":"Hello world"}},seed=42).train()
