# Stable-Diffusion-FineTuning
# Text to Image Generation with Amazon SageMaker

[![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/vivek7208/Stable-Diffusion-FineTuning/blob/master/stable-diffusion-v2.ipynb)
[![Preview In nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/vivek7208/Stable-Diffusion-FineTuning/blob/master/stable-diffusion-v2.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vivek7208/Stable-Diffusion-FineTuning/blob/master/stable-diffusion-v2.ipynb)


This repository contains a Jupyter notebook that demonstrates how to use Amazon SageMaker and the Stable Diffusion model to generate images from text prompts.

## Notebook Overview

The notebook is divided into two main sections:

### 1. Generating Images from Text

This section shows how to use the JumpStart API to generate highly realistic and artistic images from any text prompt. This could be used for a variety of applications, such as product design, ecommerce, realistic art generation, and more.

The notebook uses a Stable Diffusion model for this task. Stable Diffusion is a text-to-image model that creates photorealistic images from a text prompt. It works by training to remove noise that was added to a real image. This de-noising process generates a realistic image. These models can also generate images from text alone by conditioning the generation process on the text.

Here are examples of images produced for the following prompts:

**Prompt: "Country estate with an impressionist touch"**

![Country estate with an impressionist touch](https://github.com/vivek7208/Stable-Diffusion-FineTuning/assets/65945306/d63fd56c-880b-43bb-9783-1a316213db61)


**Prompt: "Astronaut on a horse"**

![Astronaut on a horse](https://github.com/vivek7208/Stable-Diffusion-FineTuning/assets/65945306/ac9fec59-3e26-4b21-b49f-aabd98bf85d3)


**Prompt: "Chase scene on the streets of Los Santos, sports car weaving through traffic, police in pursuit, neon lights, dynamic action, rain-slicked streets reflecting city lights, GTA V theme, digital painting, concept art, trending on DeviantArt, high resolution, art by WLOP, Maciej Kuciara"**

![Chase scene on the streets of Los Santos](https://github.com/vivek7208/Stable-Diffusion-FineTuning/assets/65945306/91c3ca13-4d00-46af-85d0-e39d61032ccd)


**Prompt: "A photo of a Doppler dog with a hat"**

![A photo of a Doppler dog with a hat](https://github.com/vivek7208/Stable-Diffusion-FineTuning/assets/65945306/22c6228a-72b4-4bcd-9a91-cdd128e6b964)


### 2. Fine-tuning the Model

The second part of the notebook demonstrates how to fine-tune the Stable Diffusion model on a custom dataset. This could be useful for generating custom art, logos, NFTs, and other personalized images.

#### Fine-tune the Model on a New Dataset

The model can be fine-tuned to any dataset of images. It works very well even with as little as five training images.

The fine-tuning script is built on the script from [dreambooth](https://dreambooth.github.io/). The model returned by fine-tuning can be further deployed for inference. Below are the instructions for how the training data should be formatted.

##### Input: 
A directory containing the instance images, dataset_info.json and (optional) directory class_data_dir. Images may be of .png or .jpg or .jpeg format. dataset_info.json file must be of the format {'instance_prompt':<<instance_prompt>>,'class_prompt':<<class_prompt>>}. If with_prior_preservation = False, you may choose to ignore 'class_prompt'. class_data_dir directory must have class images. If with_prior_preservation = True and class_data_dir is not present or there are not enough images already present in class_data_dir, additional images will be sampled with class_prompt.

##### Output: 
A trained model that can be deployed for inference. The s3 path should look like s3://bucket_name/input_directory/. Note the trailing / is required.

Here is an example format of the training data.

```
input_directory
    |---instance_image_1.png
    |---instance_image_2.png
    |---instance_image_3.png
    |---instance_image_4.png
    |---instance_image_5.png
    |---dataset_info.json
    |---class_data_dir
        |---class_image_1.png
        |---class_image_2.png
        |---class_image_3.png
        |---class_image_4.png
```

#### Prior preservation, instance prompt and class prompt: 
Prior preservation is a technique that uses additional images of the same class that we are trying to train on. For instance, if the training data consists of images of a particular dog, with prior preservation, we incorporate class images of generic dogs. It tries to avoid overfitting by showing images of different dogs while training for a particular dog. Tag indicating the specific dog present in instance prompt is missing in the class prompt. For instance, instance prompt may be "a photo of a riobugger cat" and class prompt may be "a photo of a cat". You can enable prior preservation by setting the hyper-parameter with_prior_preservation = True.

We provide a default dataset of cat images. It consists of eight images (instance images corresponding to instance prompt) of a single cat with no class images. It can be downloaded from here. If using the default dataset, try the prompt "a photo of a riobugger cat" while doing inference in the demo notebook.

Fine-tuning is a process where a pre-trained model is further trained on a new dataset. The aim is to adapt the pre-existing model (which has already learned useful features from a larger dataset) to new data. In this case, we fine-tune the Stable Diffusion model on a custom dataset. The notebook guides you through this process step-by-step, from retrieving the training artifacts, setting the training parameters, to starting the training process.

## Getting Started

### Requirements

- An AWS account with appropriate permissions to create and manage Amazon SageMaker resources.
- An `ml.t3.medium` instance for running the notebook.
- An `ml.p3.2xlarge` or `ml.g4dn.2xlarge` instance for deploying the model. If `ml.g5.2xlarge` is available in your region, we recommend using it as it has more GPU memory and supports generating larger, better quality images.

### Setup

1. Clone this repository to your local machine.
2. Open the notebook in Amazon SageMaker Studio or Notebook instance.
3. Run the cells in order to execute the notebook.

## Detailed Process

### Running the Notebook

1. Install necessary libraries: The notebook begins by installing necessary libraries, including `ipywidgets` and `sagemaker`.

2. Import libraries and set up the SageMaker client: The notebook then imports necessary libraries and sets up the SageMaker client to communicate with the SageMaker service.

3. Generate Images from Text: This section demonstrates how to generate images from any text prompt using the Stable Diffusion model.

4. Fine-tuning the Model: This section demonstrates how to fine-tune the model on a custom dataset. 

## License

This project is licensed under the MIT License.
