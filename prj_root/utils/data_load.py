import pandas as pd
import pickle
from cmapPy.pandasGEXpress.parse import parse

from super_classes import param_class as param_class_module

class Utils_Data_Load(param_class_module.Param_Super_Class):

  def __init__(self,configurable_parameters_dict):
    super().__init__(configurable_parameters_dict)

  @staticmethod
  def load_gctx(BASE_DIR,one_path_gctx):
    gctx_file = parse(BASE_DIR+'/'+one_path_gctx)
    data_df = gctx_file.data_df
    return data_df
  @staticmethod
  def load_txt(BASE_DIR,one_path_gctx):
    data_df=pd.read_csv(BASE_DIR+'/'+one_path_gctx,encoding='utf-8',sep='\t')
    return data_df
  @staticmethod
  def load_embedding(BASE_DIR,one_embedding):
    with open(BASE_DIR+one_embedding,'rb') as f:
      mynewlist=pickle.load(f)
    return mynewlist

    


