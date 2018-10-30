# Feasibility Analysis Framework for Machine Learning

This project serves as a framework to quickly test a range of machine learning models on given data set with segmentation of test data into sets of increasing amount of data partitioned from the original dataset.

The framework works based on the retraining algorithm implementation available in tensorflow

## Executing tests

1. Make a modules file and write down all the models you want to use for training (ex. modules.csv)
2. Load all your files in a test folder (the framework generates a .temp folder in that folder for storing segments/partitions of data, intermediate and final results)
3. Execute the tests as

   ```bash
   python analyse.py --image_dir=D:/Temp/flower_photos --modules=D:/Temp/modules.csv --trainingSteps=100 --segmentSize=50
   ```

4. The results can then be obtained from "D:\Temp\flower_photos\\.temp" folder. The results folder will contain the final models for different segment size and algorithms, and the .logs folder will contain the logs of the training of all the models.