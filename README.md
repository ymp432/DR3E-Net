# DR3E-Net  
Official repository for DR3E-Net (Drug Repositioning network based on deep embedding, differentially expressed genes, and drug side effect)  

# DR3E-Net depends on the study of deep embedding based drug repurposing
- Since the following original paper did not include the source code, I took it upon myself to implement the suggested algorithm.  
Drug Repurposing Using Deep Embeddings of Gene Expression Profiles  
https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.8b00284

# Additional parts for full DR3E-Net study will be added
- The study is ongoing (2023-06~)  

# Environment
ubuntu:20.04  
python 3.7  
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

# Train
1) Download LINCS datasets  
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742  
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138  
2) Run `analysis_expression_for_train_and_inference.ipynb` with `MODE="train"` option  

# Inference  
1) Write the step number for model checkpoint file in `model.load_state_dict(torch.load("./trained_model/ckpt_00078.pt"))` from `/prj_root/run/test_model.py`
2) Run `analysis_expression_for_train_and_inference.ipynb` with `MODE="inferencee"` option to create `./result_files/embedding_vectors.pkl`

# Calculate cosine similarities with respect to specific drug  
1) Run `calculate_cosine_similarity_scores.ipynb` which depends on `./result_files/embedding_vectors.pkl` which is generated in the above inference step  

# Loss decrease during 78 epochs  
![image](https://github.com/ymp432/deepEDR/assets/101608528/a83b7c9b-d166-4b0b-b82c-9207aa2c110e)

# Some validation of my implementation  
1) Left is the result from my code and model, Right is top10 drugs which are functionally similar to Metformin, reported in referenced paper  
![image](https://github.com/ymp432/deepEDR/assets/101608528/90745cf8-1f31-4998-8a5a-f57a8402be4e)  
Top10 drugs were all included in the top30 of my result list  
The ranking and cosine similarity score can be different because data processing and model configuration were a bit different  

2) I validated repurposed drug candidates for Methimazole (antithyroid agent) based on literatures.  
The related evidences could be found in some repurposed cadidate drugs.  
![image](https://github.com/ymp432/deepEDR/assets/101608528/b7e04750-91b3-4a45-824e-6e80d7b9f89b)

