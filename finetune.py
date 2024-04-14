import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification
import pandas as pd
from datasets import Dataset,load_metric
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoModelForMaskedLM,
    AutoModel,
    LlamaTokenizer,
    LlamaForSequenceClassification,
    pipeline,
    BartForSequenceClassification,
    BartTokenizerFast,
)
import evaluate
from trl import SFTTrainer
from sklearn.utils.class_weight import compute_class_weight
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
import tqdm as tqdm

from torch.utils.data import DataLoader
from transformers.trainer_utils import IntervalStrategy
from sklearn.utils import resample



# #config 

max_seq_length = 2048 
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
device_map = {"": 0}

tune_task = "fine-tune" # "few-shot" or "zero-shot"
training_data = "lima" # "full" , "lima"
task = "argument-classification" # "stance-classification" # argument-classification
add_data = False
add_gpt_pro = False
add_lima_data = False
add_formal_gpt_data = False
add_test_lima_data = False
add_more_ukp_args = False
# add_more_ukp_args_val = True80.
# add_ibm_argqual_data = True
add_argqual_ibm_data = True ## THIS IS GOOD FOR ARG!!!
llm = "mistral" # "yi" , # mistral, bart # llama
add_system_message = True
add_topic = False
DO_TRAIN= False
DO_PRED= True
DO_EVAL= True 
                                
"""
Stance Datasets:
ukp_human       gpt_pro_all_stance          argqual_stance_human
Arg Datasets:
ukp_human       gpt_pro_all          arg_spoken_human

"""

test_data  = "ukp_human"  #arg_spoken_human,ukp_human,gpt_pro_all  argqual_stance_human   gpt_pro_all_stance   # "ibm" # "gpt" # "ukp" # "speech" "gpt_pro", "gpt_pro_all" "ibm_spoken" "ibm_coling"
num_epochs = 2
if task == "argument-classification": 
    num_labels = 2
else:
    num_labels = 3

model_path_name = llm+ "_arg_v7"
# arg_stance_v21 is best for stance

# for llama arg "arg_fixed_v90" is the best



# df_test = pd.read_csv('/home/arman/finetunedata/asonam_paper_data/ukp/ukp_test.csv',names=["text","class"])
# df_test["text"] = df_test["text"].str.split("Text: ").str[1]     
# df_test = df_test.dropna()
# df_test = df_test.rename(columns={"text":"sentence","class":"target"})

# # get value counts for each target
# print(df_test['target'].value_counts())
# print(stop)
#parser = argparse.ArgumentParser()
#parser.add_argument("--max_seq_length", type=int, default=2048)
#parser.add_argument("--dtype", type=str, default=None)
#parser.add_argument("--load_in_4bit", type=bool, default=True)
#parser.add_argument("--device_map", type=dict, default={"": 0})
#parser.add_argument("--tune_task", type=str, default="fine-tune")
#parser.add_argument("--training_data", type=str, default="lima")
#parser.add_argument("--task", type=str, default="argument-classification")
#parser.add_argument("--add_data", type=bool, default=False)
#args = parser.parse_args()

#max_seq_length = args.max_seq_length
#dtype = args.dtype
#load_in_4bit = args.load_in_4bit
#device_map = args.device_map
#tune_task = args.tune_task
#training_data = args.training_data
#task = args.task
#add_data = args.add_data


system_message= """
Premise: A statement that provides evidence, reasons, or support.
Conclusion: A statement that is being argued for or claimed based on the premises.

Argument/NoArgument Transition Network:
Start State --Token matches Premise Definition--> Premise State Augmentation (Premise sub-network) --Token matches Conclusion definition--> Conclusion State Augmentation (Conclusion sub-network) ----> Argument State ----> End State
Start State --Token matches Conclusion definition--> Conclusion State Augmentation (Conclusion sub-network) ----> Premise State Augmentation (Premise sub-network) ----> Argument State ----> End State
Start State --Token matches Premise Definition--> Premise State Augmentation (Premise sub-network) --Token does not match Conclusion Definition--> NoArgument State -> End State
Start State --Token matches Conclusion definition--> Conclusion State Augmentation (Conclusion sub-network) --Token does not match Premise Definition--> NoArgument State ----> End State
Start State ----> NoArgument State ----> End State
Start State --Token does not match Premise Definition--> NoArgument State ----> End State
Start State --Token does not match Conclusion Definition--> NoArgument State ----> End State

Premise State Augmentation (Premise sub-network) ----> Premise Content State ----> Premise Conjunction State ----> Premise State ----> Premise End State
Conclusion State Augmentation (Premise sub-network) ----> Conclusion Content State ----> Conclusion Conjunction State ----> Conclusion State ----> Conclusion End State

Argument State ----> Action: Classify as Argument ----> Argument State
NoArgument State ----> Action: Classify as NoArgument ----> NoArgument State

Follow this chain of thought reasoning and apply the transition network rules and systematically determine whether a given sentence is an argument or not, based on the presence or absence of premises and claims.
If the sentence is an argument, output only 'Argument' and your task is finished.
If the sentence is not an argument, output only 'NoArgument' and your task is finished."""

id2labels = {
    "NoArgument":0,
    "Argument_for":1,
    "Argument_against":2
}



def find_all_linear_names(model):
    cls = torch.nn.Linear #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            if 'lm_head' in lora_module_names: # needed for 16-bit
                lora_module_names.remove('lm_head')
    return list(lora_module_names)





def get_additional_data():
    ibm_argqual_claim_good_train = pd.read_csv("/home/arman/claim_topic_data/ibm_argqual_claim_good_train.csv")
    ibm_argqual_claim_bad_train = pd.read_csv("/home/arman/claim_topic_data/argqual_lowquality_topic.csv")
    ukp_claim_no_topic = pd.read_csv("/home/arman/claim_topic_data/ukp_claim_no_topic.csv")
    # argqual_lowquality_notopic = pd.read_csv("/home/arman/claim_topic_data/argqual_lowquality_notopic.csv")

    ibm_argqual_claim_bad_train = ibm_argqual_claim_bad_train[ibm_argqual_claim_bad_train["set"] == "train"]
    ukp_claim_no_topic = ukp_claim_no_topic[ukp_claim_no_topic["set"] == "train"]
    # argqual_lowquality_notopic = argqual_lowquality_notopic[argqual_lowquality_notopic["set"] == "train"]

    ibm_argqual_claim_good_train = ibm_argqual_claim_good_train[["argument","topic","label"]]
    ibm_argqual_claim_bad_train = ibm_argqual_claim_bad_train[["argument","topic","label"]]
    ukp_claim_no_topic = ukp_claim_no_topic[["sentence","topic","annotation"]]
    # argqual_lowquality_notopic = argqual_lowquality_notopic[["argument","target"]]

    ibm_argqual_claim_good_train = ibm_argqual_claim_good_train.rename(columns={"argument":"sentence","label":"class"})
    ibm_argqual_claim_bad_train = ibm_argqual_claim_bad_train.rename(columns={"argument":"sentence","label":"class"})
    ukp_claim_no_topic = ukp_claim_no_topic.rename(columns={"sentence":"sentence","annotation":"class"})
    # argqual_lowquality_notopic = argqual_lowquality_notopic.rename(columns={"argument":"sentence","target":"target"})


    df_train = pd.concat([ibm_argqual_claim_good_train,ibm_argqual_claim_bad_train,ukp_claim_no_topic])
    # df_train["text"] = df_train["topic"] + "Text: " + df_train["sentence"]
    df_train["text"] = df_train["sentence"]
    df_train = df_train[["text","class"]]
    df_train = df_train.dropna()
    # if class != NoArgument than 1 else 0
    df_train["class"] = df_train["class"].apply(lambda x: 1 if x != "NoArgument" else 0)

    return df_train

def get_additional_data_val():
    ibm_argqual_claim_good_val = pd.read_csv("/home/arman/claim_topic_data/ibm_argqual_claim_good_val.csv")
    ibm_argqual_claim_bad_val = pd.read_csv("/home/arman/claim_topic_data/argqual_lowquality_topic.csv")
    ukp_claim_no_topic = pd.read_csv("/home/arman/claim_topic_data/ukp_claim_no_topic.csv")

    ibm_argqual_claim_bad_val = ibm_argqual_claim_bad_val[ibm_argqual_claim_bad_val["set"] == "dev"]
    ukp_claim_no_topic = ukp_claim_no_topic[ukp_claim_no_topic["set"] == "val"]
    
    ibm_argqual_claim_good_val = ibm_argqual_claim_good_val[["argument","topic","label"]]
    ibm_argqual_claim_bad_val = ibm_argqual_claim_bad_val[["argument","topic","label"]]
    ukp_claim_no_topic = ukp_claim_no_topic[["sentence","target","annotation"]]

    ibm_argqual_claim_good_val = ibm_argqual_claim_good_val.rename(columns={"argument":"sentence","label":"class"})
    ibm_argqual_claim_bad_val = ibm_argqual_claim_bad_val.rename(columns={"argument":"sentence","label":"class"})
    ukp_claim_no_topic = ukp_claim_no_topic.rename(columns={"sentence":"sentence","annotation":"class"})

    df_val = pd.concat([ibm_argqual_claim_good_val,ibm_argqual_claim_bad_val,ukp_claim_no_topic])
    # df_val["text"] = df_val["topic"] + "Text: " + df_val["sentence"]
    df_val["text"] = df_val["sentence"]
    df_val = df_val[["text","class"]]
    # if class != NoArgument than 1 else 0
    df_val["class"] = df_val["class"].apply(lambda x: 1 if x != "NoArgument" else 0)


    # drop nan rows
    df_val = df_val.dropna()
    return df_val

if DO_TRAIN:
    if llm =="llama":
        tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        model = LlamaForSequenceClassification.from_pretrained('meta-llama/Llama-2-7b-hf', num_labels=num_labels)


    if llm == "yi":
        tokenizer = LlamaTokenizer.from_pretrained('01-ai/Yi-6B')
        model = LlamaForSequenceClassification.from_pretrained('01-ai/Yi-6B', num_labels=num_labels)

    if llm == "mistral":
        tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
        model = AutoModelForSequenceClassification.from_pretrained('mistralai/Mistral-7B-v0.1', num_labels=num_labels)

    if llm == "bart":
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        model = BartForSequenceClassification.from_pretrained("facebook/bart-large",num_labels=num_labels)

    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.unk_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))



    def create_input_sequence(sample):
       return tokenizer(sample["text"], truncation=True,max_length=2048,return_token_type_ids=False)
      

    ### LIMA TRAIN ### 
    ### Using the Less is More approach ### 
    if training_data == "lima":
        data = pd.read_csv('/home/arman/lima_class_data/ukp_lima_train_v3.csv')
        data = data[['topic','sentence','annotation_real']]
        if task == "stance-classification":
            data = data.rename(columns={"annotation_real": "class","sentence":"text"})
            #map labels
            data["class"] = data["class"].map(id2labels)
            data = data[["text","class","topic"]]

        
        else:
            data['text'] = data['sentence']
            data = data.rename(columns={"annotation_real": "class"})
            data["text"] = data["text"].astype(str)
            if add_topic:
                data = data[['text','class','topic']]
            else:
                data = data[["text","class"]]

            data["class"] = data["class"].apply(lambda x: 1 if x != "NoArgument" else 0) 
            print(data['class'].value_counts())

    

        if add_more_ukp_args:
            ukp_df = pd.read_csv('/home/arman/finetunedata/asonam_paper_data/ukp/ukp_train.csv')
            ukp_df.columns = ['text', 'class']
            ukp_df["text"] = ukp_df["text"].astype(str)
            ukp_df["text"] = ukp_df["text"].str.split("Text: ").str[1]
            ukp_df = ukp_df[["text","class"]]

            ukp_df = ukp_df[~ukp_df["text"].isin(data["text"])]

            ukp_df_noarg = ukp_df[ukp_df['class'] == "NoArgument"].sample(n=250)
            ukp_df_noarg["class"] = ukp_df_noarg["class"].apply(lambda x: 1 if x != "NoArgument" else 0)


            data = pd.concat([data,ukp_df_noarg],ignore_index=True)

    


        if add_lima_data:
            data_add = pd.read_csv('/home/arman/finetunedata/asonam_paper_data/ukp/ukp_train.csv')
            data_add.columns = ['text', 'class']
            data_add["text"] = data_add["text"].astype(str)
            data_add['topic'] = data_add['text'].str.split("Target: ").str[1].str.split(" Text: ").str[0]
            data_add["text"] = data_add["text"].str.split("Text: ").str[1]
            print(data_add["text"].head() )
            
            # # randomly select 337 no_arg rows from data_add
            data_add = data_add[~data_add["text"].isin(data["text"])]
            data_add_none = data_add[data_add['class']=="NoArgument"].sample(n=250) #337

            data_add_none = data_add_none[["text","class","topic"]]
            # make sure the data is not in the original data
            data_add_none = data_add_none[~data_add_none["text"].isin(data["text"])]
            # convert noargument to 0
            if task == "stance-classification":
                data_add_none["class"] = data_add_none["class"].apply(lambda x: id2labels[x])
            else:
                data_add_none["class"] = data_add_none["class"].apply(lambda x: 1 if x != "NoArgument" else 0)
            #concat
            data = pd.concat([data,data_add_none],ignore_index=True)
    

    if training_data == "full":
        data = pd.read_csv('/home/arman/finetunedata/asonam_paper_data/ukp/ukp_train.csv')
        data.columns = ['text', 'class']
        data['topic'] = data['text'].str.split("Target: ").str[1].str.split(" Text: ").str[0]
        data["text"] = data["text"].str.split("Text: ").str[1]
        data["text"] = data["text"].astype(str)



    if training_data == "ibm":
        data = pd.read_csv('/home/arman/finetunedata/asonam_paper_data/ibm/ibm_argqual_train.csv')
        data.columns = ['text', 'class']
        data["text"] = data["text"].astype(str)
    
    if training_data == "speech":
        data = pd.read_csv('/home/arman/speech_train_v1.csv')
        data.columns = ['text', 'class']
        data["text"] = data["text"].astype(str)

    if add_data:
        append_data = get_additional_data()
        if data is None:
            data = pd.read_csv('/home/arman/lima_class_data/ukp_lima_train.csv')
            data = data[['topic','sentence','annotation']]
            data['text'] = data['sentence']
            data = data.rename(columns={"annotation": "class"})
            data["text"] = data["text"].astype(str)
            data = data[["text","class"]]

        data = pd.concat([data,append_data],ignore_index=True)
        
    train_dataset = Dataset.from_pandas(data)

    if add_system_message:
        if task == "stance-classification":
            train_dataset = train_dataset.map(lambda examples: {"text": [f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n" + "Topic: '" + topic +"' Text: '" +sentence  + "' [/INST] " for topic,sentence in zip(examples['topic'],examples['text'])]}, batched=True)
        if task == "argument-classification" and add_topic ==False:
            train_dataset = train_dataset.map(lambda examples: {"text": [f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n" + "Text: '" + prompt +"' [/INST] " for prompt in examples['text']]}, batched=True)
        if task == "argument-classification" and add_topic == True:
            train_dataset = train_dataset.map(lambda examples: {"text": [f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n" + "Topic: '" + topic +"' Text: '" +sentence  + "' [/INST] " for topic,sentence in zip(examples['topic'],examples['text'])]}, batched=True)
    else:
        if task == "stance-classification":
            train_dataset = train_dataset.map(lambda examples: {"text": [f"Target: '" + topic +" Text:'" + sentence+"'" for topic,sentence in zip(examples['topic'],examples['text'])]}, batched=True)
        else:
            train_dataset = train_dataset.map(lambda examples: {"text": [f"Text:'" + sentence+"'" for sentence in examples['text']]}, batched=True)   
    
    train_dataset = train_dataset.map(create_input_sequence, batched=True)#,remove_columns=["text"])
    train_dataset = train_dataset.rename_column("class", "labels")
    train_dataset.set_format('torch')
    train_dataset_tokenized = train_dataset.shuffle(seed=1234)  # Shuffle dataset here

    if task == "stance-classification":
        if training_data == "full":
            val = pd.read_csv('/home/arman/finetunedata/asonam_paper_data/ukp/ukp_val.csv',names=["text","class"])
            val["text"] = val["text"].astype(str)
            val['topic'] = val['text'].str.split("Target: ").str[1].str.split(" Text: ").str[0]
            val["text"] = val["text"].str.split("Text: ").str[1]
            val["text"] = val["text"].astype(str)
            val = val[["text","class","topic"]]
            #map
            val["class"] = val["class"].map(id2labels)

        if training_data == "lima":
            val = pd.read_csv('/home/arman/lima_class_data/ukp_lima_val_stance.csv')
            val = val.dropna(subset=["text"])
            val = val[['text','class','topic']]
            val["class"] = val["class"].map(id2labels)
            print(val['class'].value_counts())
        
        
        if add_argqual_ibm_data:
            arg_mapping = {
                "con":2,
                "pro":1,
                "none":0

            }
             
            df_add = pd.read_csv('/home/arman/lima_class_data/argqual_ibm_2.csv')
            df_add = df_add[df_add['test']==True]
            
            # argument to text, stance to label
            df_add = df_add.rename(columns={"argument":"text","stance":"class"})
            df_add = df_add[df_add['class'] == 'none']
            df_add["class"] = df_add["class"].map(arg_mapping)
            df_add = df_add[["text","class","topic"]]
            val = pd.concat([val,df_add],ignore_index=True)

        if add_test_lima_data:
                
                val_ukp = pd.read_csv('/home/arman/finetunedata/asonam_paper_data/ukp/ukp_val.csv',names=["text","class"])
                val_ukp["text"] = val_ukp["text"].astype(str)
                print(val_ukp.head())
                # print any nan values in text
                print(val_ukp["text"].isna().sum())
                val_ukp['topic'] = val_ukp['text'].str.split("Target: ").str[1].str.split(" Text: ").str[0]
                val_ukp["text"] = val_ukp["text"].str.split("Text: ").str[1]
                val_ukp = val_ukp[["text","topic","class"]]

                val_ukp = val_ukp[~val_ukp["text"].isin(val["text"])]
                val_none = val_ukp[val_ukp['class']=="NoArgument"].sample(n=100) #80 @260 was good
                # map id2labels
                val_none["class"] = val_none["class"].map(id2labels)
                val = pd.concat([val,val_none],ignore_index=True)
                print(val["class"].value_counts())

    if task == "argument-classification":
        ### Val Lima ###
        if training_data == "lima":
            val = pd.read_csv('/home/arman/lima_class_data/ukp_lima_val_stance.csv')
            val = val.dropna(subset=["text"])
            # map
            val["class"] = val["class"].apply(lambda x: 1 if x != "NoArgument" else 0)
            print(val['class'].value_counts())
        
        
        if add_argqual_ibm_data:
            arg_mapping = {
                "con":1,
                "pro":1,
                "none":0

            }
             
            df_add = pd.read_csv('/home/arman/lima_class_data/argqual_ibm_2.csv')
            df_add = df_add[df_add['test']==True]
            
            # argument to text, stance to label
            df_add = df_add.rename(columns={"argument":"text","stance":"class"})
            df_add = df_add[df_add['class'] == 'none']

            df_add["class"] = df_add["class"].map(arg_mapping)
            df_add = df_add[["text","class",'topic']]

            val = pd.concat([val,df_add],ignore_index=True)
            

        if training_data == "lima1":
            val = pd.read_csv('/home/arman/lima_class_data/ukp_lima_val.csv')
            val = val[['topic','sentence','annotation']]
            # val['text'] = val['topic'] + "Text: " + val['sentence']
            val['text'] = val['sentence']
            val = val.rename(columns={"annotation": "class"})
            val["text"] = val["text"].astype(str)
            val = val[["text","class"]]
            print(val["class"].value_counts())

            test_ukp = pd.read_csv('/home/arman/finetunedata/asonam_paper_data/ukp/ukp_val.csv')
            test_ukp.columns = ['text', 'class']
            test_ukp["text"] = test_ukp["text"].astype(str)
            test_ukp["text"] = test_ukp["text"].str.split("Text: ").str[1]
            test_ukp = test_ukp[["text","class"]]
            # how many of val are in test ukp
            print("this how many overlap")
            print(len(val[val["text"].isin(test_ukp["text"])]))

        
        if training_data == "full":
            val = pd.read_csv('/home/arman/finetunedata/asonam_paper_data/ukp/ukp_val.csv',names=["text","class"])
            val["text"] = val["text"].astype(str)
            val['topic'] = val['text'].str.split("Target: ").str[1].str.split(" Text: ").str[0]
            val["text"] = val["text"].str.split("Text: ").str[1]
            val["text"] = val["text"].astype(str)
            val = val[["text","class","topic"]]
            #map
            val["class"] = val["class"].apply(lambda x: 1 if x != "NoArgument" else 0)


    validation_dataset = Dataset.from_pandas(val)    
    if add_system_message:
        if task == "stance-classification":
            validation_dataset = validation_dataset.map(lambda examples: {"text": [f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n" + "Target: '" + topic +"' Text: '" +sentence  + "' [/INST] " for topic,sentence in zip(examples['topic'],examples['text'])]}, batched=True)
        if task == "argument-classification" and add_topic ==False:
            validation_dataset = validation_dataset.map(lambda examples: {"text": [f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n" + "Text: '" + prompt +"' [/INST] " for prompt in examples['text']]}, batched=True)
        if task == "argument-classification" and add_topic == True:
            validation_dataset = validation_dataset.map(lambda examples: {"text": [f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n" + "Target: '" + topic +"' Text: '" +sentence  + "' [/INST] " for topic,sentence in zip(examples['topic'],examples['text'])]}, batched=True)
    else:
        if task == "stance-classification":
            validation_dataset = validation_dataset.map(lambda examples: {"text": [f"Target: '" + topic +" Text:'" + sentence+"'" for topic,sentence in zip(examples['topic'],examples['text'])]}, batched=True)
        else:
            validation_dataset = validation_dataset.map(lambda examples: {"text": [f"Text:'" + sentence+"'" for sentence in examples['text']]}, batched=True) 
    validation_dataset = validation_dataset.map(create_input_sequence)#, batched=True,remove_columns=["text"])
    validation_dataset = validation_dataset.rename_column("class", "labels")
    validation_dataset.set_format('torch')
    #shuffle
    validation_dataset = validation_dataset.shuffle(seed=1234)  # Shuffle dataset here

    print(validation_dataset['labels'])
    print(train_dataset_tokenized['labels'])    
    llm_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    modules = find_all_linear_names(model)


    lora_config = LoraConfig(
        r=8,   #8 or 32 or 64                    
        lora_alpha= 32, # 32 or 64 or 16 
        target_modules=modules,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS"
                    )

    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

    def compute_metrics(eval_pred):
        # All metrics are already predefined in the HF `evaluate` package
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        f1_metric= evaluate.load("f1")
        accuracy_metric = evaluate.load("accuracy")

        logits, labels = eval_pred # eval_pred is the tuple of predictions and labels returned by the model
        if llm == "bart":
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)
        
        print(f"predictions: {predictions}; labels: {labels}")
        
        precision = precision_metric.compute(predictions=predictions, references=labels, average='weighted')["precision"]
        recall = recall_metric.compute(predictions=predictions, references=labels, average='weighted')["recall"]
        f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')["f1"]
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        
        # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores. 
        return {"precision": precision, "recall": recall, "f1-score": f1, 'accuracy': accuracy}
        
    model = model.cuda()

    lr = 1e-4
    batch_size = 4

    training_args = TrainingArguments(
        output_dir="/home/arman/results/mistral-lora-token-classification", 
        num_train_epochs=num_epochs,  # demo
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        lr_scheduler_type="constant", #cosine
        # lr_scheduler_kwargs=lr_scheduler_config,
        learning_rate=3e-05, #0.00001,#3e-05,
        warmup_steps=500,
        # warmup_ratio = 0.03,
        max_grad_norm= 0.3,
        weight_decay=0.1,
        evaluation_strategy=IntervalStrategy.STEPS,
        label_smoothing_factor=0.1,
        optim="paged_adamw_32bit", #"adafactor", #paged_adamw_32bit
        # gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        fp16=True,
        logging_dir="/home/arman/logs",
        logging_steps=100,
        # metric_for_best_model="eval_f1-score",
        # greater_is_better=True,
        save_total_limit=50,
        eval_steps=100,
        load_best_model_at_end=True,
        neftune_noise_alpha=5#0.1
    )

    llm_trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tokenized,
        eval_dataset=validation_dataset,
        data_collator=llm_data_collator,
        compute_metrics=compute_metrics,
    )

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    llm_trainer.train()
    if llm == "llama":
        llm_trainer.model.save_pretrained("/home/arman/models/llama2/"+model_path_name+"/")
        tokenizer.save_pretrained("/home/arman/models/llama2/"+model_path_name+"/")
    if llm == "yi":
        llm_trainer.model.save_pretrained("/home/arman/models/yi/"+model_path_name+"/")
        tokenizer.save_pretrained("/home/arman/models/yi/"+model_path_name+"/")
    
    if llm == "mistral":
        llm_trainer.model.save_pretrained("/home/arman/models/mistral/"+model_path_name+"/")
        tokenizer.save_pretrained("/home/arman/models/mistral/"+model_path_name+"/")

    if llm == "bart":
        llm_trainer.model.save_pretrained("/home/arman/models/bart/"+model_path_name+"/")
        tokenizer.save_pretrained("/home/arman/models/bart/"+model_path_name+"/")


if DO_PRED:
    if llm == "llama_base":
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        model = LlamaForSequenceClassification.from_pretrained('meta-llama/Llama-2-7b-hf', num_labels=num_labels,torch_dtype=torch.float16, device_map=device_map,)
    if llm == "llama":
        tokenizer = AutoTokenizer.from_pretrained("/home/arman/models/llama2/"+model_path_name+"/")
        model = LlamaForSequenceClassification.from_pretrained("/home/arman/models/llama2/"+model_path_name+"/", num_labels=num_labels,torch_dtype=torch.float16, device_map=device_map,)

    if llm=="yi":
        tokenizer = AutoTokenizer.from_pretrained('/home/arman/models/yi/'+model_path_name+'/')
        model = LlamaForSequenceClassification.from_pretrained('/home/arman/models/yi/'+model_path_name+'/', num_labels=num_labels,torch_dtype=torch.float16, device_map=device_map,)
    if llm=="mistral":
        tokenizer = AutoTokenizer.from_pretrained('/home/arman/models/mistral/'+model_path_name+'/')
        model = AutoModelForSequenceClassification.from_pretrained('/home/arman/models/mistral/'+model_path_name+'/', num_labels=num_labels,torch_dtype=torch.float16, device_map=device_map,)
    
    if llm=="bart":
        tokenizer = AutoTokenizer.from_pretrained('/home/arman/models/bart/'+model_path_name+'/')
        model = BartForSequenceClassification.from_pretrained('/home/arman/models/bart/'+model_path_name+'/', num_labels=num_labels,torch_dtype=torch.float16, device_map=device_map,)

    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.unk_token
    model.config.pad_token_id = tokenizer.pad_token_id
    device = torch.device("cuda")

    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    id2labels = {
            "NoArgument":0,
            "Argument_for":1,
            "Argument_against":2
    }
    def create_input_sequence(sample):        
       return tokenizer(sample["text"], padding=True,truncation=True,max_length=2048,return_token_type_ids=False,return_tensors="pt").to(device)

    predictions = []
    confidences = []
    labels = []
    

    if test_data == "ukp":
        if task == "stance-classification":
            df_test = pd.read_csv('/home/arman/finetunedata/asonam_paper_data/ukp/ukp_test.csv',names=["text","class"])
            df_test["topic"] = df_test["text"].str.split("Target: ").str[1].str.split(" Text: ").str[0]

        #Now strip the text column to only include the text
            df_test["text"] = df_test["text"].str.split("Text: ").str[1]
            df_test = df_test.dropna()
            df_test = df_test.rename(columns={"class":"target"})
            print(df_test.head())
        else:
            df_test = pd.read_csv('/home/arman/finetunedata/asonam_paper_data/ukp/ukp_test.csv',names=["text","class"])
            df_test["text"] = df_test["text"].str.split("Text: ").str[1]     
            df_test = df_test.dropna()
            df_test = df_test.rename(columns={"class":"target"})
            print(df_test.head())

    if test_data == "ukp_human":
        df_test = pd.read_csv('/home/arman/finetunedata/asonam_paper_data/ukp/ukp_test_untouched_human.csv')

        if task == "stance-classification":
            # if human_eval is 1 than leave target as is else, set to noargument
            df_test["target"] = df_test.apply(lambda x: "NoArgument" if x['human_eval'] == 0 else x['target'],axis=1)
            # target map to labels
            print(df_test)
            df_test["target"] = df_test["target"].map(id2labels)
            
            df_test = df_test[['text','target','topic']]

        else:
            df_test = df_test[['text','human_eval','topic']]
        df_test = df_test.rename(columns={"human_eval":"target"})


    if test_data == "ibm":
        df_test = pd.read_csv('/home/arman/finetunedata/asonam_paper_data/ibm/ibm_argqual_test.csv')
        df_test = df_test.rename(columns={"argument":"text","label":"target"})
        df_test = df_test[['text','target']]
        # get everything after "Text: "
        df_test["text"] = df_test["text"].str.split("Text: ").str[1]

    if test_data == "ibm_train":
        df_test = pd.read_csv('/home/arman/ibm_train.csv')
        df_test = df_test[['candidate','label']]
        df_test = df_test.rename(columns={"candidate":"text","label":"target"})

    if test_data =="ibm_spoken":
        df_test = pd.read_csv('/home/arman/finetunedata/asonam_paper_data/ibm/arg_spoken.csv')
        df_test = df_test[['sentence','label','test']]
        df_test = df_test[df_test['test'] == True]
        df_test = df_test.rename(columns={"sentence":"text","label":"target"})

    if test_data == "ibm_coling":
        df_test = pd.read_csv('/home/arman/finetunedata/asonam_paper_data/ibm/ibm_coling.csv',names=["id","context","topic","sentence","q","confidence","label","source"])
        df_test = df_test[['sentence','label']]
        df_test = df_test.rename(columns={"sentence":"text","label":"target"})

    if test_data == "ibm_argqual":
        df_test1 = pd.read_csv('/home/arman/finetunedata/asonam_paper_data/ibm/ibm_arg_qual_all.csv')

        df_test1 = df_test1[['argument','rank','context','stance','topic', 'Manual']]
        df_test1 = df_test1.rename(columns={"argument":"text","stance":"target"})
        # df_test = df_test[df_test['Manual'] == 1]
        # replace "-" in context with " "
        df_test1["context"] = df_test1["context"].str.replace("-"," ")
        # rename context to text
        # if target == "con" change to 2 else 1
        df_test1["target"] = df_test1["target"].apply(lambda x: 2 if x == "con" else 1)

        # randomly select 500 examples less than 0.5 rank and 500 above
        df_test_low = df_test1[df_test1['rank'] < 0.5].sample(n=500)
        df_test = pd.concat([df_test_low,df_test1[df_test1['rank'] > 0.5].sample(n=500)])

    if test_data == "arg_spoken_human":
        df_test = pd.read_csv('/home/arman/finetunedata/asonam_paper_data/ibm/arg_spoken_human.csv')
        df_test = df_test[['sentence','human_eval','topic']]
        df_test = df_test.rename(columns={"sentence":"text","human_eval":"target"})

    if test_data == "gpt":
        df_test = pd.read_csv('/home/arman/lima_class_data/gpt_arg_binary.csv')
        df_test = df_test.rename(columns={"sentence":"text","label":"target"})
        df_test = df_test[['text','target','type']]
    if test_data == "gpt_pro":
        df_test = pd.read_csv('/home/arman/lima_class_data/gpt_pro_test.csv')
        df_test = df_test.rename(columns={"class":"target"})
        print(df_test.head())
        df_test = df_test[['text','target','type']]
    if test_data =="gpt_pro_all":
        df_test = pd.read_csv('/home/arman/lima_class_data/gpt_pro.csv',names=["text","class","topic","type"])
        df_test = df_test.rename(columns={"class":"target"})
        df_test = df_test[['text','target','topic','type']]

    if test_data == "argqual_stance_human":
        df_test = pd.read_csv('/home/arman/lima_class_data/arg_qual_for_stance_human_CTE.csv')
        df_test = df_test[['argument','topic','human_eval','pred_topic']]
        df_test["pred_topic"] = df_test["pred_topic"].str.strip()
        df_test["topic"] = df_test.apply(lambda x: x['pred_topic'] if x['pred_topic'] != "No Topic" else x['topic'],axis=1)
        df_test = df_test.rename(columns={"argument":"text","human_eval":"target"})
        df_test = df_test[['text','target','topic']]
        # df_test = df_test.rename(columns={"target":"target"})

    if test_data == "gpt_pro_all_stance":
        df_test = pd.read_csv('/home/arman/lima_class_data/gpt_pro_all_stance.csv')
        df_test = df_test[['text','target','topic','type','pred_topic']]
        # drop target column
        df_test = df_test.drop(columns=["target"])
        df_test = df_test.rename(columns={"pred_topic":"target"})
        df_test = df_test[['text','target','topic']]


    test_data_text_only = Dataset.from_pandas(df_test)
    
    if add_system_message:
        if task == "stance-classification":
            test_data_text_only = test_data_text_only.map(lambda examples: {"text": [f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n" + "Target: '" + topic +"' Text: '" +text  + "' [/INST] " for topic,text in zip(examples['topic'],examples['text'])]}, batched=True)
        if task == "argument-classification" and add_topic ==False:
            test_data_text_only = test_data_text_only.map(lambda examples: {"text": [f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n" + "Text: '" + prompt +"' [/INST] " for prompt in examples['text']]}, batched=True)
        if task == "argument-classification" and add_topic == True:
            test_data_text_only = test_data_text_only.map(lambda examples: {"text": [f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n" + "Target: '" + topic +"' Text: '" +text  + "' [/INST] " for topic,text in zip(examples['topic'],examples['text'])]}, batched=True)
    else:
        if task == "stance-classification":
            test_data_text_only = test_data_text_only.map(lambda examples: {"text": [f"Target: '" + topic +" Text:'" + text+"'" for topic,text in zip(examples['topic'],examples['text'])]}, batched=True)
        else:
            test_data_text_only = test_data_text_only.map(lambda examples: {"text": [f"Text: '" + text+"'" for text in examples['text']]}, batched=True)
    test_data_text_only.set_format('torch')


    if not add_system_message:
        args = test_data_text_only['text']
    else:
        args = test_data_text_only['text']
    results = []


# # Loop through the validation dataset in batches
    for batch in tqdm.tqdm(args):
        with torch.no_grad():
            input_text = tokenizer(batch, padding=True, truncation=True,max_length=2048,return_tensors="pt").to(device)
            output = model(**input_text)
            logits = output.logits
            predicted_class = torch.argmax(logits, dim=1)
            # Convert logits to a list of predicted labels
            predictions.extend(predicted_class.cpu().tolist())

        # Get the ground truth labels
    df_test["pred_topic"] = predictions

    df_test.to_csv("/home/arman/llama_seq_no_form_arg.csv",index=False)

    
# Evaluate the model
if DO_EVAL:
    id2labels = {
            "NoArgument":0,
            "Argument_for":1,
            "Argument_against":2
    }
    val = pd.read_csv("/home/arman/llama_seq_no_form_arg.csv")

    print(val.head())
    if test_data=="gpt_pro":
        val = val[["text","target","type","pred_topic"]]
    if test_data=="ukp":
        val = val[["text","target","pred_topic"]]

    if task == "stance-classification" and test_data == "ukp":
        val["target"] = val["target"].map(id2labels)



    if task == "stance-classification" and test_data == "ukp":
        val["target"] = val["target"].astype(str)
        # val["target"] = val["target"].map(id2labels)
    if task == "argument-classification" and test_data == "gpt_pro_all":
        val["target"] = val["target"].astype(str)
        val["target"] = val["target"].apply(lambda x: 1 if x != "NoArgument" else 0)



    val["pred_topic"] = val["pred_topic"].astype(int)

    if test_data == "gpt":
        labels = val['target'].tolist()
    if test_data =="gpt_pro":
        labels = val['target'].tolist()

    if test_data =="ukp" and task == "stance-classification":
        labels = val['target'].tolist()

    if test_data =="gpt_pro_all":
        labels = val['target'].tolist()

    if test_data == "ibm" or test_data == "ibm_train" or test_data=="ibm_spoken" or test_data =="ibm_coling" or test_data =="ibm_argqual":
        labels = val['target'].astype(int)
        labels = val['target'].tolist()

    if test_data == "argqual_stance_human":
        labels = val['target'].tolist()

    if test_data == "ukp_human" or test_data == "arg_spoken_human":
        labels = val['target'].tolist()

    if test_data == "gpt_pro_all_stance":
        labels = val['target'].tolist()
    predictions = val["pred_topic"].tolist()


    print("Completed Predictions")
    # Compute the evaluation metric (e.g., accuracy)

    print(labels)
    print(predictions)
    from datasets import load_metric
    import evaluate
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    acc_metric_result = accuracy_metric.compute(predictions=predictions, references=labels)
    f1_macro = f1_metric.compute(predictions=predictions,references=labels,average='macro')
    f1_micro = f1_metric.compute(predictions=predictions,references=labels,average='micro')
    f1_weighted = f1_metric.compute(predictions=predictions,references=labels,average='weighted')
    recall_metric = evaluate.load("recall")
    precision_metric = evaluate.load("precision")
    recall = recall_metric.compute(predictions=predictions,references=labels,average='macro')
    precision = precision_metric.compute(predictions=predictions,references=labels,average='macro')


    # Print evaluation statistics
    print(f"Accuracy: {acc_metric_result['accuracy']:.2%}")
    print(f"F1 Macro: {f1_macro['f1']:.2%}")
    print(f"F1_micro: {f1_micro['f1']:.2%}")
    print(f"F1 Weighted: {f1_weighted['f1']:.2%}")
    print(f"Recall: {recall['recall']:.2%}")
    print(f"Precision: {precision['precision']:.2%}")