pytorch code base test.

# Matconv_to_pytorch.py
matconv_to_pytorch.py is used to convert matconvnet model to pytorch. I implement this because I would like to reuse the two-stream model from http://www.robots.ox.ac.uk/~vgg/software/two_stream_action/
Only VGG model is supported now.
Usage:
<pre><code>from matconv_to_pytorch import matconv_convertor
convertor = matconv_convertor('/model_dir/model.mat')
net = convertor.get_layer_seq()
</code></pre>