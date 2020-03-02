# Reproducing our results
To reproduce our results there are various steps that need to be taken. The first step is making sure that the user
has downloaded the respective download folders on our main Github folder. Here we politely refer the user to our tutorial on 
[how to get started](https://github.com/informagi/REL/tree/master/tutorials/01_How_to_get_started.md). Now that we have obtained
the required folder structure, which includes the necessary training, validation and test files, we need [to train our Entity Disambiguation model](https://github.com/informagi/REL/tree/master/tutorials/deploy_REL_new_Wiki/04_04_training_your_own_ED_model.md). 

Now that we have obtained our ED model, we may start evaluation. In our paper we divide our results into three tables. To obtain Entity Disambiguation results on GERBIL (Table 2), the user needs to follow the [linked tutorial](https://github.com/informagi/REL/tree/master/tutorials/03_Evaluate_Gerbil.md). and
set the respective `mode` to 'ED'. To obtain Entity Linking strong matching results on GERBIL (Table 1), we refer to the same tutorial, but now the respective variable `mode` should be
set to 'EL'. To obtain ED local results (Table 3), the user needs to follow the same tutorial as for [training the Entity Disambigation]](https://github.com/informagi/REL/tree/master/tutorials/deploy_REL_new_Wiki/04_04_training_your_own_ED_model.md), but
instead of setting the `mode` in the dictionary `config` to 'train', it should be set to 'eval' and the `model_path` should refer to the respective model.
 
Finally, for each table we divide our results into REL 2014 and 2019. First of, to obtain results
for Wikipedia 2014, no extra changes need to be made. For Wikipedia 2019, however, we note that the parameter `"dev_f1_change_lr"` needs to
 be changed to `0.88`.