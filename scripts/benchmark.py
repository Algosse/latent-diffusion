import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision.transforms import transforms

from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

ROOT_PATH = "/home/alban/ImSeqCond"

dataset = 'SIAR'

class CustomMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean((x - y) ** 2, dim=(1, 2, 3))
    
# CLIP metric
import clip
import torch.nn as nn
from torchvision.transforms import Normalize, Resize, CenterCrop, ToTensor

class CustomCLIP(nn.Module):
    """ Use to compute CLIP score between two images 
    
        Images shoud be PIL images in the range [0, 255] with the shape W x H x 3
    """
    
    def __init__(self, model="ViT-B/32"):
        super().__init__()
        self.model, _ = clip.load(model, device=device)
        self.preprocess = transforms.Compose([
            Resize(self.model.visual.input_resolution, interpolation=BICUBIC),
            CenterCrop(self.model.visual.input_resolution),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        
    def encode(self, image):
        image = self.preprocess(image).to(device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        return image_features
    
    def score(self, image1, image2):
        E1 = self.encode(image1)
        E2 = self.encode(image2)
        score = torch.cosine_similarity(E1, E2, dim=-1)
        return score
    
    def forward(self, image1, image2):
        """ 
            Args:
                image1 (torch.Tensor): Image 1, shape (B, C, H, W)
                image2 (torch.Tensor): Image 2, shape (B, C, H, W)
        """
        return self.score(image1, image2)

import lpips

class CustomLPIPS(nn.Module):
    
    def __init__(self, model='vgg'):
        super().__init__()
        self.model = lpips.LPIPS(net=model).to(device)
        
    def forward(self, image1, image2):
        """ 
            Args:
                image1 (torch.Tensor): Image 1, shape (B, C, H, W). Should be in the range [-1, 1]
                image2 (torch.Tensor): Image 2, shape (B, C, H, W). Should be in the range [-1, 1]
        """
        
        d = self.model(image1, image2)
        
        return d.squeeze()

class Benchmark():
    
    def __init__(self, model, dataloader, mse=True, clip=False, lpips=False, cond_key='label'):
        self.model = model
        self.dataloader = dataloader
        
        self.metrics = {
            'mse': CustomMSE() if mse else None,
            'clip': CustomCLIP() if clip else None,
            'lpips': CustomLPIPS() if lpips else None,
        }
        
        self.results = {key: None for key in self.metrics.keys()}
        
        self.cond_key = cond_key

    def evaluate(self):

        metrics_used = [key for key, metric in self.metrics.items() if metric is not None]
        print(f"Evaluating model on metrics {', '.join(metrics_used)}")
        
        self.model.eval()
        self.model.to(device)
        
        scores = {key: [] for key in self.metrics.keys()}
        
        min_scores = {key: float('inf') for key in self.metrics.keys()}
        min_scores_im = {key: None for key in self.metrics.keys()}
        max_scores = {key: float('-inf') for key in self.metrics.keys()}
        max_scores_im = {key: None for key in self.metrics.keys()}
        
        for data in tqdm(self.dataloader):
                
                im = torch.tensor(data['data']).permute(0, 3, 1, 2).to(device)
                label = data[self.cond_key]
                
                with torch.no_grad():
                
                    prediction = self.sample(label)
        
                    for i in range(prediction.shape[0]):
                        for name, metric in self.metrics.items():
                            if metric is not None:
                                im_2, pred = (im[i].unsqueeze(0) + 1) / 2, (prediction[i].unsqueeze(0) + 1) / 2
                                score = metric(pred, im_2)
                                
                                if name == 'lpips':
                                    scores[name].append(score.unsqueeze(0))
                                else:
                                    scores[name].append(score)
                                
                                if score.min() < min_scores[name]:
                                    min_scores[name] = score.min()
                                    min_scores_im[name] = data['name'][score.argmin().item()]
                                
                                if score.max() > max_scores[name]:
                                    max_scores[name] = score.max()
                                    max_scores_im[name] = data['name'][score.argmax().item()]

        for name, metric in self.metrics.items():
            if metric is not None:
                metric_scores = torch.cat(scores[name])
                print(f"Metric {name} score: {metric_scores.mean().item()}, lowest score: {metric_scores.min().item()} for image {min_scores_im[name]}, highest score: {metric_scores.max().item()} for image {max_scores_im[name]}")
                self.results[name] = {
                    'mean': metric_scores.mean().item(),
                    'min': metric_scores.min().item(),
                    'max': metric_scores.max().item(),
                    'min_im': min_scores_im[name],
                    'max_im': max_scores_im[name],
                }
        
        print("Find results in 'results' attribute")
        
        return self.results
    
    def sample(self, data):
        """ Overwrite this method to sample from the model with data as conditioning """
        pass
    
    def evaluate_metric(self, name):
        
        print(f"Evaluating model on metric {name}")
        
        self.model.eval()
        self.model.to(device)
        
        metric = self.metrics[name]
        scores = {key: [] for key in self.metrics.keys()}
        
        min_score = float('inf')
        max_score = float('-inf')
        
        for data in tqdm(self.dataloader):
            
            im = torch.tensor(data['data']).permute(0, 3, 1, 2).to(device)
            label = data[self.cond_key]
            
            with torch.no_grad():
            
                prediction = self.sample(label) # (B, C, H, W)
                
                batch_scores = []
                for i in range(prediction.shape[0]):
                    batch_scores.append(metric(prediction[i].unsqueeze(0), im[i].unsqueeze(0)))
            
                score = torch.stack(batch_scores)
            
            if score.min() < min_score:
                min_score = score.min()
                min_score_im = data['name'][score.argmin().item()]
            
            if score.max() > max_score:
                max_score = score.max()
                max_score_im = data['name'][score.argmax().item()]
            
            scores.append(score)

        scores = torch.cat(scores)
        print(f"Mean score: {scores.mean().item()}, lowest score: {scores.min().item()} for image {min_score_im}, highest score: {scores.max().item()} for image {max_score_im}")
        
        return {
            'mean': scores.mean().item(),
            'min': scores.min().item(),
            'max': scores.max().item(),
            'min_im': min_score_im,
            'max_im': max_score_im,
        }

if __name__ == '__main__':
        

    class Baseline(nn.Module):
        """ 
            Naive approach to solve the restoration problem
        """
        
        def __init__(self):
            super(Baseline, self).__init__()
        
        def forward(self, x):
            """ 
                Forward pass
                Args:
                    x (torch.tensor): input image sequence. size: (batch_size, 10, 3, 256, 256)
                Returns:
                    torch.tensor: restored image
            """
            out = torch.mean(x, dim=1)
            
            return out.permute(0, 3, 1, 2)

    class BaselineBenchmark(Benchmark):
            
            def __init__(self, model, dataloader, mse=True, clip=False, lpips=False, cond_key='label'):
                super().__init__(model, dataloader, mse, clip, lpips, cond_key)
                
            def sample(self, data):
                """ Overwrite this method to sample from the model with data as conditioning """
                
                data = data.to(device)
                
                return self.model(data)

    from ldm.data.siar import SIAR

    dataset = SIAR(root=os.path.join(ROOT_PATH, 'data', dataset), set_type='val', resolution=256)
    print(f"Dataset size: {len(dataset)}")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=12)

    benchmark = BaselineBenchmark(Baseline(), dataloader, mse=True, clip=True, lpips=True, cond_key='label')
        
    benchmark.evaluate()