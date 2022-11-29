# GAN-argcPredNet v2.0
This is a radar echo extrapolation model, which can improve the accuracy of rainfall prediction.

The generator is stored in the ArgcPredNet.py file, and the completion process of building GAN-argcPredNet v2.0 is in the Train.py file.

This network is trained to be a prediction model that extrapolates the next 7 frames from the first 5 frames.

In GAN-argcPredNet v2.0, a Spatiotemporal Information Changes Prediction (STIC-Prediction) network is designed as the generator. The generator focuses on the spatiotemporal variation of radar echo feature sequence. The more accurate images are generated by intensifying the spatiotemporal evolution of previous inputs. Furthermore, discriminator is a Channel-Spatial Convolution (CS-Convolution) network. By focusing on radar echo features from spatial and channel dimensions, the 
discriminator enhances the ability to identify echoes information.

