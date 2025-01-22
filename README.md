# MiniCSM

This project implements a knowledge distillation framework based on Hugging Face, extracting knowledge from the teacher model to train a smaller and more deployable student model. It is suitable for scenarios where deploying large-scale language models under limited resources is required. 

The base model is a fine-tuned T5-xxl designed for soft contextual adjustment on social media platforms.

---

## **1.Environment**
```bash
pip install torch
pip install transformers
pip install peft
pip install pandas
pip install scikit-learn
pip install tqdm
pip install wandb
```
**or** 
```bash
pip install requirements.txt
```

## **2.Dataset**
### **2.1Source**
- You can download these datasets to test the model **https://huggingface.co/collections/ppaudel**
### **2.2Data-processing**
- To train and test the model, we split the dataset to the "claim" and "stance".
- When training the model, we use **Contrastive Textual Deviation** : we formulating this triplet of i) a consensus statement, ii) a refuting evidence, and iii) a supporting evidence

## **3.Models**
- You can find our teacher model  **https://huggingface.co/collections/ppaudel**
