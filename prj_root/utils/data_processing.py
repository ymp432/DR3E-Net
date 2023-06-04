import pandas as pd

from super_classes import param_class as param_class_module

class Utils_Data_Processing(param_class_module.Param_Super_Class):

  def __init__(self,configurable_parameters_dict):
    super().__init__(configurable_parameters_dict)

  def concat_gct_txt(self,gct_data,txt_data):
    return pd.concat([gct_data,txt_data],axis=1)

  @staticmethod
  def clean_column_row_names(loaded_gctx):
    loaded_gctx.columns.name = None
    loaded_gctx.index.name = None
    return loaded_gctx
