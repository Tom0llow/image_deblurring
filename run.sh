# non-blind
# nohup python3 -u celebahq_256_deconvolution.py > outputs/celebahq_256_deconvolution.log &
# nohup python3 -u mnist_deconvolution.py > outputs/mnist_deconvolution.log &
# nohup python3 -u cifar10_deconvolution.py > outputs/cifar10_deconvolution.log &

# nohup python3  celebahq_256_deconvolution.py > /dev/null &
nohup python3 mnist_deconvolution.py > /dev/null &
nohup python3 cifar10_deconvolution.py > /dev/null &


# blind
# nohup python3 -u mnist_blind_deconvolution.py > outputs/mnist_blind_deconvolution.log &
# nohup python3 -u celebahq_256_blind_deconvolution.py > outputs/celebahq_256_blind_deconvolution.log &

# nohup python3  celebahq_256_blind_deconvolution.py > /dev/null &
# nohup python3  mnist_blind_deconvolution.py > /dev/null &
