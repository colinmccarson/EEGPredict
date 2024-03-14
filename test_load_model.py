from keras.models import load_model

filepath = './weights/model_vgg_insp_depth2_fc2_bnTrue_f32f_k_11__1_k_d0_5d_convdrop_False_v0_5318396091461182.keras'

loaded_model = load_model(filepath)