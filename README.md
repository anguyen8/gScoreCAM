# gScoreCAM: What is CLIP looking at?


### Interactive demo ([Colab](https://colab.research.google.com/drive/13BRR5eiOE0zIrdc9Fy6uFciJ6l13PVg8?usp=sharing))


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


