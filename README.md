# 1. DR3E-Net  
Official repository for DR3E-Net (**D**rug **R**epurposing **Net**work framework based on deep **E**mbedding, differentially **E**xpressed genes, and drug side **E**ffect)  

# 2. DR3E-Net partially depends on the prior study of deep-embedding model for drug repurposing and the implementation of angular penalty softmax losses
- Since the following original paper did not include the source code, I took it upon myself to implement the suggested algorithm.  
Drug Repurposing Using Deep Embeddings of Gene Expression Profiles  
https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.8b00284  
- The following loss function implementation was partially edited and used for this study  
https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch  

# 3. Environment
ubuntu:20.04  
python 3.7  
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

# 4. The procedure of utilizing DR3E-Net framework  
## 4.1. Train (Step 1)  
1) Download LINCS datasets  
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742  
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138  
I downloaded the following list of files and wrote them into jupyter notebook file `analysis_expression_for_train_and_inference.ipynb`  
![image](https://github.com/ymp432/DR3E-Net/assets/101608528/a929d20c-e977-42f0-b38b-ccea5b243476)
2) Run `analysis_expression_for_train_and_inference.ipynb` with `MODE="train"` option  

## 4.2. Inference (Step 1)  
1) Write the step number for model checkpoint file in `model.load_state_dict(torch.load("./trained_model/ckpt_00078.pt"))` from `/prj_root/run/test_model.py`
2) Run `analysis_expression_for_train_and_inference.ipynb` with `MODE="inferencee"` option to create `./result_files/embedding_vectors.pkl`

## 4.3. Calculate cosine similarities with respect to specific drug  (Step 1)  
1) Run `calculate_cosine_similarity_scores.ipynb` which depends on `./result_files/embedding_vectors.pkl` which is generated in the above inference step  

## 4.4. Find agonistic drugs against DEGs associated with the specific disease and merge them with the outcome from Step 1 (Step 2)  
1) Run `find_drugs_acting_agnostic_to_DEG.ipynb`  

## 4.5. Preprocess SIDER and SAEDR databases and merge them with the outcome from Step 2 (Step 3)  
1) Run `create_SIDER_side_effect_data.ipynb`  

# 5. Loss decrease during 78 epochs  
![image](https://github.com/ymp432/deepEDR/assets/101608528/a83b7c9b-d166-4b0b-b82c-9207aa2c110e)

# 6. Some validation of my implementation  
1) Left is the result from my code and model, Right is top10 drugs which are functionally similar to Metformin, reported in referenced paper  
![image](https://github.com/ymp432/deepEDR/assets/101608528/90745cf8-1f31-4998-8a5a-f57a8402be4e)  
Top10 drugs were all included in the top30 of my result list  
The ranking and cosine similarity score can be different because data processing and model configuration were a bit different  

2) I validated repurposed drug candidates for Methimazole (antithyroid agent) based on literatures.  
The related evidences could be found in some repurposed cadidate drugs.  
![image](https://github.com/ymp432/deepEDR/assets/101608528/b7e04750-91b3-4a45-824e-6e80d7b9f89b)

