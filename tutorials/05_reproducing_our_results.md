# Reproducing our results
To reproduce our results there are various steps that need to be taken. The first step is making sure that the user
has downloaded the respective download folders on our main Github folder. Here we politely refer the user to our tutorial on 
[how to get started](https://github.com/informagi/REL/tree/master/tutorials/01_How_to_get_started.md). Now that we have obtained
the required folder structure, which includes the necessary training, validation and test files, we need [to train our Entity Disambiguation model](https://github.com/informagi/REL/tree/master/tutorials/deploy_REL_new_Wiki/04_04_training_your_own_ED_model.md). 

To train a ED model for Wikipedia 2014, no extra changes are required. To obtain results for Wikipedia 2019, we note that the parameter `"dev_f1_change_lr"` needs to 
be changed to `0.88`. After following these steps, the user now has access to a Entity Disambiguation model. In our paper we divide our results into three tables, which
can be reproduced by following the [linked tutorial](https://github.com/informagi/REL/tree/master/tutorials/03_Evaluate_Gerbil.md) and by making sure to use the
corresponding settings:
 1. Configure the GERBIL platform to perform either the Entity Linking or Disambiguation step.
 2. Configure the server to use the correct model path for Wiki 2014 or 2019.