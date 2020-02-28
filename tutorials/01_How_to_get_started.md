# How to get started
Throughout the project we will constantly use a variable, which we called `base_url`. This variable refers to your data
folder and will be used to store the folders that need to be downloaded from our main Github page, but will also be used to 
store e.g. generated training and test files.

```python
base_url = "/home/user/project_folder"
```

The files that are provided on our main Github page can be copied and downloaded into this folder, thus obtaining
the folder structure below. Throughout this set of tutorials the `generic` folder will always be required, but the user
may choose to work with either the 2014, 2019 or their own Wikipedia corpus.

```
.
├── generic
└─── wiki_2014
|   ├── basic_data
|      └── anchor_files
|   └── generated
└─── wiki_2019
|   ├── basic_data
|      └── anchor_files
|   └── generated
```