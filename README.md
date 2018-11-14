# separate-cnn-for-brain-tumor-segmentation

This repro first transforms [U-Net brain tumor project](https://github.com/zsdonghao/u-net-brain-tumor) from tensorlayer into tensorflow, and test it on tensorflow1.10 and [BRATS2017](https://www.med.upenn.edu/sbia/brats2017/data.html).

The location of dataset and files shall be:

```bash
data
 --MICCAI_BraTS17_Data_Training
   --HGG
   --LGG
load_data.py
train.py
model_separate.py
...
```
Where the model_separate.py stores the model U-net designs and my future work. Because I am studying [Xception](http://cn.arxiv.org/abs/1610.02357), so I named this file model_separate.py.

In training part, I shuffle the data loaded and set the batch size 10, so the input to model is \[10, 240, 240, 4\]. I don't think study in the whole MRI image is wise, which may lead to unbalance in the feature learning explained in this [paper](http://cn.arxiv.org/abs/1706.01805). I will consider crop it before sending to the model, like cencter crop.

The load_data.py produce two arrays \[?, 240, 240, 4] and \[?, 240, 240, 1], the \[flair, T1, T1c, T2] and \[groud truth] separately.

### The example data in temp folder
I put seceral cases in the temp folder for testing usage. Pay attention to the path in load_data.py and run
```bash
python train.py
```
If the 1-dice loss ranges around 0.2, I think the model is studying.

### Future work
- Changes the way of pre-processing images
- Design separate CNNs for brain tumor segmentation
- Learn the GAN method

