from REL.training_datasets import TrainingEvaluationDatasets
from REL.entity_disambiguation import EntityDisambiguation

base_url = "C:/Users/mickv/Google Drive/projects/entity_tagging/deep-ed/data/"
wiki_version = "wiki_2019"

# 1. Load datasets # '/mnt/c/Users/mickv/Google Drive/projects/entity_tagging/deep-ed/data/wiki_2019/'
datasets = TrainingEvaluationDatasets(base_url, wiki_version).load()

# 2. Init model, where user can set his/her own config that will overwrite the default config.
config = {
    "mode": "eval",
    # "model_path": "{}/generated/model_12_01_2020".format(base_url),
    # "model_path": "{}/generated/model_27_01_2020".format(base_url),
    # "model_path": "{}/generated/model_w2v_08_02_2019_1".format(base_url),
    "model_path": "{}/{}/generated/model_colab_28_02_2019".format(
        base_url, wiki_version
    ),
    # "model_path": "{}/generated/model_GPU_28_02_2019".format(base_url),
    "dev_f1_change_lr": 0.915,  # 0.88,#0.915,
}
model = EntityDisambiguation(base_url, wiki_version, config)

# 3. Train or evaluate model
if config["mode"] == "train":
    model.train(
        datasets["aida_train"], {k: v for k, v in datasets.items() if k != "aida_train"}
    )
else:
    model.evaluate({k: v for k, v in datasets.items() if "train" not in k})
