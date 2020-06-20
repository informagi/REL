from REL.training_datasets import TrainingEvaluationDatasets
from REL.entity_disambiguation import EntityDisambiguation

base_url = "/users/vanhulsm/Desktop/projects/data/"
wiki_version = "wiki_2014"

# 1. Load datasets # '/mnt/c/Users/mickv/Google Drive/projects/entity_tagging/deep-ed/data/wiki_2019/'
datasets = TrainingEvaluationDatasets(base_url, wiki_version).load()

# 2. Init model, where user can set his/her own config that will overwrite the default config.
config = {
    "mode": "eval",
    "model_path": "{}/{}/generated/model".format(
        base_url, wiki_version
    ),
}
model = EntityDisambiguation(base_url, wiki_version, config)

# 3. Evaluate pre-finetuning.
model.evaluate({k: v for k, v in datasets.items()})

#TODO: Go through paper (see chat Faegheh) for preprocessing stuff.

#TODO: If train and not found (during data generation), remove data point. How is it done for AIDA? Or are they ignored
# during training? look it up.

#TODO: Are there any data points that should be ignored? What if no entity available.

# 4. Fine-tune ED model.
#TODO We fine-tune our model, but first we evaluate it as is.
#TODO: Store as NEW model.

