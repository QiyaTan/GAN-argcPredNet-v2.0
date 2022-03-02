import hickle as hkl
from Model import  build_STPIN_model
from function import test_predictImage

img_rows = 128
img_cols = 128
channels = 1
input_img_num = 5
img_shape = (img_rows, img_cols, channels)
latent_dim = (6,) + (img_rows, img_cols, channels)
BATCH_SIZE = 4
nt = 6
X_test = hkl.load('/run/media/root/test.hkl')/255.
FRAME = 7
predict_save_path4 = '/run/media/root/predict.hkl'
model = build_STPIN_model(nt, img_rows, img_cols)
model.load_weights("Weight dir")
preimage4 = test_predictImage(X_test, model, BATCH_SIZE, FRAME, img_rows, img_cols)
hkl.dump(preimage4, predict_save_path4)



