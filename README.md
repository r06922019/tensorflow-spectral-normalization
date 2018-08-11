# tensorflow-spectral-normalization
An implementation of spectral normalization in tensorflow

_Tested on tf 1.4_

## References
* [1802.05957 Spectral Normalization for Generative Adversarial Networks
](https://arxiv.org/abs/1802.05957)
* [github.com/minhnhat93/tf-SNDCGAN](https://github.com/minhnhat93/tf-SNDCGAN)
* [github.com/pfnet-research/sngan_projection/issues/15#issuecomment-406939990](https://github.com/pfnet-research/sngan_projection/issues/15#issuecomment-406939990)

## Usage
```
import sn_layers

inputs = ....
output = sn_layers.conv2d(inputs, 
            out_dim=64, k_size=3, strides=1,
            padding='SAME',
            w_init=tf.glorot_uniform_initializer(),
            use_bias=True, 
            spectral_normed=True, name='sn_conv1'
            )
```
