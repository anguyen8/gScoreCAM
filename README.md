# gScoreCAM: What is CLIP looking at?

_**tldr:** Based on the observations that [CLIP ResNet-50](https://github.com/openai/CLIP) channels are very noisy compared to typical ImageNet-trained ResNet-50, and most saliency methods obtain pretty low object localization scores with CLIP. By visualizing the top 10% most sensitive (highest-gradient) channels, our gScoreCAM obtains the state of the art weakly supervised localization results using CLIP (in both ResNet and ViT versions)._


**Official Implementation** for the paper [gScoreCAM: What is CLIP looking at?]() (2022) by Peijie Chen, Qi Li, Saad Biaz, Trung Bui, and Anh Nguyen.
If you use this software, please consider citing:

    @article{chen2022gScoreCAM,
      title={gScoreCAM: What is CLIP looking at?},
      author={Peijie Chen, Qi Li, Saad Biaz, Trung Bui, and Anh Nguyen},
      publisher={arXiv},
      year={2022}
    }

:star2: Interactive [Colab demo](https://colab.research.google.com/drive/13BRR5eiOE0zIrdc9Fy6uFciJ6l13PVg8?usp=sharing) :star2:


### Prerequisite
Install annconda following the [anaconda installation documentation](https://docs.anaconda.com/anaconda/install/).
Create an enviroment with all required packages with the following command :
```bashscript
conda env create -f gscorecam_env.yml
```

### Interative CLI
Other than the Colab demo above, we provide a interative command line tool for testing different visualization methods.
You may use it with:
```bashscript
python visualize_cam.py --cam-version [CAM version] --image-folder [path to testing images] --image-src [name of the datset]
```
For example, test gScoreCAM on COCO images.
```
python visualize_cam.py --cam-version gscorecam --image-folder path_to_coco --image-src coco
```
After the program will prompt you with a question asking if you would like to go for specific class or random class, you could simply tpye the class name or press enter for random.

### Evaluation code
In order to use the evaluation code, you will need to download the meta data from [Google Drive](https://drive.google.com/file/d/1S6JPTDNJnlr3it2ox3i8gAR3Bv9VuWOk/view?usp=sharing). We extract the metat data of IamgeNetv2, COCO, and PartImageNet into `.hdf5` format for convenience. 
#### COCO evalutaion
You may run the evalution code with the following command:
```
python evaluate_cam.py info-ground-eval --model-name RN50x16 --cam-version gscorecam --image-src coco --image-folder path_to_image --meta-file meta_data/coco_val_instances_stats.hdf5
```
You may need to change the path accordingly.
#### PartImageNet evaluation
Similar to COCO evaluation, simply run:
```
python eval_partsImageNet.py info-ground-eval --model-name RN50x16 --cam-version gscorecam --image-src coco --image-folder path_to_image --meta-file meta_data/partsImageNet_parts_test.hdf5
```

#### ImageNetv2 evaluation
To evaluate ImageNetv2, we use Junsuk choe's evaluation script directly. Please first follow [this repo](https://github.com/clovaai/wsolevaluation) to download the data and evaluation script.

You may get the heatmap with the following command:
 ```
 python wsol_compute_heatmap.py main --model RN50x16 --method gscorecam --dataset imagenet --is-clip
 ```
Then you can evaluate on these heatmaps.


