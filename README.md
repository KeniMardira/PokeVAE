# PokeVAE

An application of VAE (Variational Auto Encoder) to encode and generate pictures of Pokemon.

The dataset contains 851 pictures of Pokemon with a size of 256x256.

The Encoder and Decoder architecture is based on this paper: https://arxiv.org/pdf/1610.00291.pdf

Beta value for KL Divergence can be set to >1 for Beta-VAE application.

Below are examples made using Beta = 0.5, LATENT_DIM = 256, and linear interpolation between the latent representation of 2 images.

<p pad="10" align="center"> 
<img src="https://github.com/keniMawson/PokeVAE/blob/master/interpolate/gif/_generate_animation.gif" width="200" height="200" hspace="20"/>
<img src="https://github.com/keniMawson/PokeVAE/blob/master/interpolate/gif/2_generate_animation.gif" width="200" height="200"/> 
<img src="https://github.com/keniMawson/PokeVAE/blob/master/interpolate/gif/3_generate_animation.gif" width="200" height="200"/> 
<img src="https://github.com/keniMawson/PokeVAE/blob/master/interpolate/gif/4_generate_animation.gif" width="200" height="200"/> 
<img src="https://github.com/keniMawson/PokeVAE/blob/master/interpolate/gif/5_generate_animation.gif" width="200" height="200"/>
<img src="https://github.com/keniMawson/PokeVAE/blob/master/interpolate/gif/6_generate_animation.gif" width="200" height="200"/>
</p>

<p align="center">
 <img src="https://github.com/keniMawson/PokeVAE/blob/master/output/epoch_12300.png" width="1200" height="800"/>
</p>
  
