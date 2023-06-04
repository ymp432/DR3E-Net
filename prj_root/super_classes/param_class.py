class Param_Super_Class():

  def __init__(self,configurable_parameters_dict):
    
    # ================================================================================
    # Configurable hyperparameters

    self.BASE_DIR=configurable_parameters_dict["BASE_DIR"]
    self.GCTX_PATH_GSE70138=configurable_parameters_dict["GCTX_PATH_GSE70138"]
    self.GCTX_PATH_GSE92742=configurable_parameters_dict["GCTX_PATH_GSE92742"]
    self.TXT_META_PATH_GSE70138=configurable_parameters_dict["TXT_META_PATH_GSE70138"]
    self.TXT_META_PATH_GSE92742=configurable_parameters_dict["TXT_META_PATH_GSE92742"]
    self.number_of_perturbagens=configurable_parameters_dict["number_of_perturbagens"]
    self.class_weight_dict=configurable_parameters_dict["class_weight_dict"]
