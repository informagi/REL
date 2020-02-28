# Reproducing our results
To reproduce our results there are various steps that need to be taken. The first step is making sure that the user
has downloaded the respective download folders on our main Github folder. Here we politely refer the user to our tutorial on 
[how to get started](https://github.com/informagi/REL/tree/master/tutorials/01_How_to_get_started.md). Now that we have obtained
the required folder structure, which includes the necessary training, validation and test files, we need [to train our Entity Disambiguation model](https://github.com/informagi/REL/tree/master/tutorials/deploy_REL_new_Wiki/04_04_training_your_own_ED_model.md). Here we note
that we changed the parameter `"dev_f1_change_lr"` to `0.88` in the configuration dictionary for the Wikipedia 2019 corpus. For the Wikipedia 2014 corpus, the default parameters were used, meaning 
no change has to be made by the user. The final step consists of [evaluating on GERBIL](https://github.com/informagi/REL/tree/master/tutorials/03_Evaluate_Gerbil.md).