#!/usr/bin/env python
# coding: utf-8

# # Sequence restoration with Latent Diffusion Models

# In[36]:


import matplotlib.pyplot as plt
import matplotlib
#%matplotlib inline

import os

import numpy as np
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms.functional import resize

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Let's also check what type of GPU we've got.

# In[37]:


import os
os.chdir("/home/alban/ImSeqCond/latent-diffusion/")


# Load it.

# In[38]:


#@title loading utils
import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config


def load_model_from_config(config, ckpt=None):
    model = instantiate_from_config(config.model)
    
    if ckpt is not None:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt)#, map_location="cpu")
        sd = pl_sd["state_dict"]
        m, u = model.load_state_dict(sd, strict=False)
    else:
        print("Instantiated model from config")
        
    model.cuda()
    model.eval()
    return model

#cond_key = 'label'
cond_key = 'LR_image'

model_folder = "/home/alban/ImSeqCond/latent-diffusion/logs_saved/2023-12-21T00-11-09_config_siar_sr"
checkpoint = "epoch=000036.ckpt"

files = os.listdir(os.path.join(model_folder, "configs"))
config_file = ""
for file in files:
    if file.endswith("project.yaml"):
        config_file = file
        break

if config_file == "":
    raise ValueError("No config file found")

def get_model():
    config = OmegaConf.load(os.path.join(model_folder, 'configs', config_file))
    model = load_model_from_config(config, os.path.join(model_folder, "checkpoints", checkpoint))
    return model


# In[39]:


from ldm.models.diffusion.ddim import DDIMSampler

model = get_model()
sampler = DDIMSampler(model)

# count model parameters
params = sum([p.numel() for p in model.parameters() if p.requires_grad])
print(f"Model has {params/1e6:.2f}M parameters")


# In[40]:


# Load some custom data
from ldm.data.siar import SIAR

dataset = SIAR("/home/alban/ImSeqCond/data/SIAR", set_type='val', resolution=256, max_sequence_size=10, downscale_f=4)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)


# And go. Quality, sampling speed and diversity are best controlled via the `scale`, `ddim_steps` and `ddim_eta` variables. As a rule of thumb, higher values of `scale` produce better samples at the cost of a reduced output diversity. Furthermore, increasing `ddim_steps` generally also gives higher quality samples, but returns are diminishing for values > 250. Fast sampling (i e. low values of `ddim_steps`) while retaining good quality can be achieved by using `ddim_eta = 0.0`.

# In[41]:


i = 5

images_indexes = [i]
n_samples_per_image = 6

ddim_steps = 20
ddim_eta = 0.0
scale = 3  # for unconditional guidance


all_samples = list()

with torch.no_grad():
    with model.ema_scope():
        
        """ uc = model.get_learned_conditioning(
            {model.cond_stage_key: torch.zeros(n_samples_per_image, 3, 64, 64).cuda().to(model.device)}
            ) """
        
        """ uc = model.get_learned_conditioning(
            torch.zeros(n_samples_per_image, 3, 4, 64, 64).cuda().to(model.device)
            ) """

        for image_index in images_indexes:
            print(f"rendering {n_samples_per_image} examples of images '{image_index}' in {ddim_steps} steps and using s={scale:.2f}.")
            
            if cond_key == 'LR_image':
                xc = rearrange(torch.tensor(dataset[image_index]['LR_image']), 'h w c -> c h w').unsqueeze(0).repeat(n_samples_per_image, 1, 1, 1)
            elif cond_key == 'label':
                xc = rearrange(torch.tensor(dataset[image_index][cond_key]), 's h w c -> c s h w').unsqueeze(0).repeat(n_samples_per_image, 1, 1, 1, 1)
            else:
                raise ValueError(f"Unknown cond_key '{cond_key}'")

            c = model.get_learned_conditioning(xc.to(model.device))

            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=c,
                                             batch_size=n_samples_per_image,
                                             shape=[3, 64, 64],
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             #unconditional_conditioning=uc,
                                             eta=ddim_eta)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0,
                                         min=0.0, max=1.0)
            all_samples.append(x_samples_ddim)


# display as grid
grid = torch.stack(all_samples, 0)
grid = rearrange(grid, 'n b c h w -> (n b) c h w')
grid = make_grid(grid, nrow=n_samples_per_image)

# to image
grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
Image.fromarray(grid.astype(np.uint8))


# In[42]:


def plot_image(data, predict=None):
    """ For a single data point, plot the ground truth image, the input images and the predicted image
    Args:
        gt (torch.tensor): ground truth image
        input (torch.tensor): input images
        predict (torch.tensor): predicted image
    """
    gt, input = data['data'], data['label']
    
    label_images = data['label'].shape[0]
    
    fig, axes = plt.subplots(2, 6, figsize=(20, 10))

    axes[0, 0].imshow(gt)
    axes[0, 0].set_title("Ground truth")
    
    for i in range(label_images):
        axes[i//5, i%5 + 1].imshow(input[i])
        axes[i//5, i%5 + 1].set_title("Input " + str(i+1))
        
    if predict is not None:
        axes[1, 0].imshow(predict)
        axes[1, 0].set_title("Predicted")
        
    plt.show()


# In[43]:


def prepare_for_plot(data, all_samples=None):
    
    data_prepared = dict()
    for key, value in data.items():
        if key in ['data', 'label']:
            data_prepared[key] = (value + 1) / 2
    
    predict_prepared = rearrange(all_samples[0][0], 'c h w -> h w c')
    #predict_prepared = (predict_prepared + 1) / 2
    predict_prepared = predict_prepared.cpu().detach().numpy()
    
    return data_prepared, predict_prepared


# In[44]:


plot_image(*prepare_for_plot(dataset[i], all_samples))


# In[45]:


# STUDY OF THE LATENT SPACE

""" cond = c[0]

# convert cond in 0, 1
cond = (cond - cond.min()) / (cond.max() - cond.min())
cond = rearrange(cond, 'c h w -> h w c')
cond = cond.detach().cpu().numpy()

cond_decode = model.decode_first_stage(c[0].unsqueeze(0))

cond_decode = torch.clamp((cond_decode+1.0)/2.0,
                                        min=0.0, max=1.0)
cond_decode = rearrange(cond_decode.squeeze(), 'c h w -> h w c')
cond_decode = cond_decode.detach().cpu().numpy()


fig, axes = plt.subplots(1, 2, figsize=(20, 10))

axes[0].imshow(cond)
axes[0].set_title("Cond in latent space")

axes[1].imshow(cond_decode)
axes[1].set_title("Cond in pixel space") """


# In[46]:


from benchmark import Benchmark

class BenchmarkLDM(Benchmark):
    
    def __init__(self, model, dataloader, mse=True, clip=False, lpips=False, cond_key='label'):
        super().__init__(model, dataloader, mse, clip, lpips, cond_key)
    
    def sample(self, data, ddim_steps=20, ddim_eta=0.0, scale=1):
        """ Method used to sample from the model with the data as conditionning 
            Args:
                data (torch.tensor): conditionning data. size: (batch_size, 3, W, H) or (batch_size, 10, 3, W, H)
            Output:
                torch.tensor: restored image. size: (batch_size, 3, W, H)
        """
        
        if self.cond_key == 'LR_image':
            xc = rearrange(torch.tensor(data), 'b h w c -> b c h w')
        elif self.cond_key == 'label':
            xc = rearrange(torch.tensor(data), 'b s h w c -> b c s h w')
        else:
            raise ValueError(f"Unknown cond_key '{cond_key}'")

        c = model.get_learned_conditioning(xc.to(model.device))

        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                            conditioning=c,
                                            batch_size=data.shape[0],
                                            shape=[3, 64, 64],
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            eta=ddim_eta)

        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0,
                                        min=0.0, max=1.0)
    
        return x_samples_ddim


# In[47]:


benchmark = BenchmarkLDM(model, dataloader, mse=True, clip=True, lpips=True, cond_key=cond_key)


# In[48]:


results = benchmark.evaluate()


# In[49]:


def rescale(data):
    """ Rescale data between 0 and 1 from -1 and 1
        Args:
            data (torch.tensor): data to rescale
        Output:
            torch.tensor: rescaled data
    """
    return {
        'data': (data['data'] + 1) / 2,
        'label': (data['label'] + 1) / 2,
        'name': data['name'],
    }


# In[ ]:


# GENERATE SAMPLES

from PIL import Image
import numpy as np

output_folder = os.path.join(model_folder, 'test_predictions')

print(output_folder)

for i in range(min(1, len(dataset))):
    
    j = np.random.randint(len(dataset))
    data = dataset[i]
    
    y = data[cond_key]

    predict = benchmark.sample(y[None,...],20)
    #predict = model.predict(y.unsqueeze(0).to(device))
    out = predict.detach().cpu()
    
    out = out[0].transpose(0,1).transpose(1,2)
    
    if os.path.exists(output_folder) == False:
        os.makedirs(output_folder)
        
    # rescale data
    data = rescale(data) # scale between 0 and 1
    
    plot_image(data, out)
    
    out_im = (out.numpy()* 255).astype(np.uint8) # rescale to 0-255
    
    im_pil = Image.fromarray(out_im)
    im_pil.save(os.path.join(output_folder, f'{data["name"]}.png'))

