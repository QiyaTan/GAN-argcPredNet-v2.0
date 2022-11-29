# GAN-argcPredNet v2.0
This is a radar echo extrapolation model, which can improve the accuracy of rainfall prediction.

The generator is stored in the ArgcPredNet.py file, and the completion process of building GAN-argcPredNet v2.0 is in the Train.py file.

This network is trained to be a prediction model that extrapolates the next 7 frames from the first 5 frames.

This model references the depth coding structure of the prednetmodel proposed by [bill-lotter](https://github.com/coxlab/prednet), and is based on the CBAM design idea of [Sanghyun Woo](https://arxiv.org/pdf/1807.06521.pdf). A Spatiotemporal Information Changes Prediction (STIC-Prediction) network is designed as the generator. The generator focuses on the spatiotemporal variation of radar echo feature sequence. The more accurate images are generated by intensifying the spatiotemporal evolution of previous inputs. Furthermore, discriminator is a Channel-Spatial Convolution (CS-Convolution) network. By focusing on radar echo features from spatial and channel dimensions, the discriminator enhances the ability to identify echoes information.

# Radar data
The experimental data is the radar mosaic of Huanan area provided by Guangdong Meteorological Bureau. It does not support the open sharing.For data access, please contact Kun Zheng (ZhengK@cug.edu.cn) and Qiya Tan (ses_tqy@cug.edu.cn).

# Train
The files of the training model are stored in the GAN-argcPredNet_Train.py file. By inputing your own data into X_train, you can start training.

<pre><code>X_train = hkl.load(data_dir+data_name+'.hkl') / 255.</code></pre>

Save the weight files of the generator and the discriminator respectively:

<pre><code>g.save_weights(weights_dir+'gen_weight'+epoch + '.h5')</code></pre>
<pre><code>d.save_weights(weights_dir+'dis_weight' + epoch + '.h5')</code></pre>

# Prediction
The prediction code is stored in the Predict.py file. X_test = hkl.load(TEST_DIR) loads the test set file, model.load_weights(WEIGHTS_DIR) loads the trained weight file. Then through test_predictImage() functions respectively generate prediction data.Finally save the prediction data by:

<pre><code>hkl.dump(Preimage, PREDICT_DIR)</code></pre>

# Note
You can cite the GAN-argcPredNet v2.0 model repository as follows:
https://github.com/QiyaTan/GAN-argcPredNet-v2.0

# Reference
@article{lotter2016deep,

title={Deep predictive coding networks for video prediction and unsupervised learning

author={Lotter, William and Kreiman, Gabriel and Cox, David},

journal={arXiv preprint arXiv:1605.08104},

year={2016}

And
@inproceedings{woo2018cbam,

title={Cbam: Convolutional block attention module},

author={Woo, Sanghyun and Park, Jongchan and Lee, Joon-Young and Kweon, In So},

booktitle={Proceedings of the European conference on computer vision (ECCV)},

pages={3--19},

year={2018}


 



