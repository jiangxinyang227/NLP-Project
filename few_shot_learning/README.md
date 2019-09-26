### few-shot learning
##### Each folder contains an implementation of each model.
* data_helper. data processing
* model. model construction
* trainer. train model
* metrics. performance metrics
* config.json  Configuration files for model parameters and training parameters

#### induction_network
* paper: Few-Shot Text Classification with Induction Network

#### relation_network
* paper: Learning to Compare: Relation Network for Few-Shot Learning

#### prototypical_network
* paper: Prototypical Networks for Few-shot Learning

#### siamese_network
* paper: Siamese Neural Networks for One-shot Image Recognition


#### ARSC data set
* the data from Amazon Review Data Set, arranged by Alibaba Group 
* citation: ***Image-based recommendations on styles and substitutes J. McAuley, C. Targett, J. Shi, A. van den Hengel SIGIR, 2015***
* citation: ***Mo Yu, Xiaoxiao Guo, Jinfeng Yi, Shiyu Chang, Saloni Potdar, Yu Cheng, Gerald Tesauro, Haoyu Wang, and Bowen Zhou. 2018. Diverse few-shot text classification with multiple metrics***

#### note
 
* You can only use 2-way, and if you need to use other way, you can modify the data_helper.py file.
* Shot should not be more than 10, because there are few comments under some categories. 
