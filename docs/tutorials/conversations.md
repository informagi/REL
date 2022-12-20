# Conversational entity linking

The `crel` submodule the conversational entity linking tool trained on the [ConEL-2 dataset](https://github.com/informagi/conversational-entity-linking-2022#conel-2-conversational-entity-linking-dataset).

Unlike existing EL methods, `crel` is developed to identify both named entities and concepts.
It also uses coreference resolution techniques to identify personal entities and references to the explicit entity mentions in the conversations.

This tutorial describes how to start with conversational entity linking on a local machine.

For more information, see the original [repository on conversational entity linking](https://github.com/informagi/conversational-entity-linking-2022).

## Start with your local environment

### Step 1: Download models

First, download the models below:

- **MD for concepts and NEs**: 
	+ [Click here to download models](https://drive.google.com/file/d/1OoC2XZp4uBy0eB_EIuIhEHdcLEry2LtU/view?usp=sharing)
	+ Extract `bert_conv-td` to your `base_url`
- **Personal Entity Linking**:
	+ [Click here to download models](https://drive.google.com/file/d/1-jW8xkxh5GV-OuUBfMeT2Tk7tEzvH181/view?usp=sharing)
	+ Extract `s2e_ast_onto` to your `base_url`

Additionally, conversational entity linking uses the wiki 2019 dataset. For more information on where to place the data and the `base_url`, check out [this page](../how_to_get_started). If setup correctly, your `base_url` should contain these directories:


```bash
.
└── bert_conv-td
└── s2e_ast_onto
└── wiki_2019
```


### Step 2: Example code

This example shows how to link a short conversation. Note that the speakers must be named "USER" and "SPEAKER".


```python
>>> from REL.crel.conv_el import ConvEL
>>> 
>>> cel = ConvEL(base_url="C:/path/to/base_url/")
>>> 
>>> conversation = [
>>>     {"speaker": "USER", 
>>>     "utterance": "I am allergic to tomatoes but we have a lot of famous Italian restaurants here in London.",}, 
>>> 
>>>     {"speaker": "SYSTEM", 
>>>     "utterance": "Some people are allergic to histamine in tomatoes.",},
>>> 
>>>     {"speaker": "USER", 
>>>     "utterance": "Talking of food, can you recommend me a restaurant in my city for our anniversary?",},
>>> ]
>>> 
>>> annotated = cel.annotate(conversation)
>>> [item for item in annotated if item['speaker'] == 'USER']
[{'speaker': 'USER',
  'utterance': 'I am allergic to tomatoes but we have a lot of famous Italian restaurants here in London.',
  'annotations': [[17, 8, 'tomatoes', 'Tomato'],
   [54, 19, 'Italian restaurants', 'Italian_cuisine'],
   [82, 6, 'London', 'London']]},
 {'speaker': 'USER',
  'utterance': 'Talking of food, can you recommend me a restaurant in my city for our anniversary?',
  'annotations': [[11, 4, 'food', 'Food'],
   [40, 10, 'restaurant', 'Restaurant'],
   [54, 7, 'my city', 'London']]}]

```

