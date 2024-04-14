# $\text{WIBA}$
 $\text{An LLM-based Approach For Comprehensive Argument Mining}$
 
 <img src="images/ATN.PNG" width="400 height =400">
 
Overview
=============
This repository contains the scripts necessary to create and reproduce the results from our paper, WIBA: An LLM-based Approach for Comprehensive Argument Mining.


### Out-the-box-use
--------------
```
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("armaniii/llama-argument-classification")
tokenizer = AutoTokenizer.from_pretrained("armaniii/lllama-argument-classification")

model.to(device)
model.eval()

for batch in tqdm.tqdm(data):
    with torch.no_grad():
        input_text = tokenizer(batch, padding=True, truncation=True,max_length=2048,return_tensors="pt").to(device)
        output = model(**input_text)
        logits = output.logits
        predicted_class = torch.argmax(logits, dim=1)
        # Convert logits to a list of predicted labels
        predictions.extend(predicted_class.cpu().tolist())

    # Get the ground truth labels
df["predictions"] = predictions

num2label = {
0:"NoArgument",
1:"Argument"
}
```


### Fine Tuning
-------------
Reference finetune.py

### Blind Eval
-------------
Reference blineval.py and data in /data/ directory.


