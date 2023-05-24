# Usage

Download the FSC-147 dataset. Then rename the image folder to `image` and the json file of annotation to `annotation.json`. If you want to use the `randomselect`, please make diractory named `encoder`. If you use the `main.sh`, you just needs to download the FSC-147 and rename relevant folders and files. The `encoder` diractory will be made automatically. And after the train, there will be a model file called `vit-sam-changed-l1loss.pth` generated.

The `train.txt` an the  `valid.txt` is randomly generated. However, if we generate them but not remove, the code will not regenerate them again to keep the test in the same environment.

Since we do not need density maps, it is not necessary to unzip this folder from the FSC-147 dataset.