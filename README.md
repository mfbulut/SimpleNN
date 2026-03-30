## Neural network experiments

### Network Parameters:
* Samples: 50.000 for training and 10.000 for testing
* Layers: input(784) -> hidden(128) -> output(10)
* Learning rate 0.0005
* Momentum: 0.9

### Datasets:

**Note:** datasets have their headers removed

```
/mnist/ (default)
/fashion/
```

### Build SDG + Momentum with Raylib visualization
```
odin build . -o:speed -microarch:native -no-bounds-check
```

### Build SDG + Momentum

```
odin build sdg_momentum -o:speed -microarch:native -no-bounds-check
```

**Benchmark (MNIST):**
```
Epoch 1 - Accuracy: 95.18% - 12.11s
Epoch 2 - Accuracy: 96.25% - 11.43s
Epoch 3 - Accuracy: 96.79% - 11.21s
Epoch 4 - Accuracy: 97.20% - 11.06s
Epoch 5 - Accuracy: 97.31% - 10.99s
Epoch 6 - Accuracy: 97.34% - 10.97s
Epoch 7 - Accuracy: 97.46% - 11.06s
Epoch 8 - Accuracy: 97.48% - 11.02s
Epoch 9 - Accuracy: 97.55% - 11.07s
Epoch 10 - Accuracy: 97.75% - 11.07s
```
### Build SDG

```
odin build sdg -o:speed -microarch:native -no-bounds-check
```

**Benchmark (MNIST):**
```
Epoch 1 - Accuracy: 91.17% - 3.25s
Epoch 2 - Accuracy: 92.58% - 3.24s
Epoch 3 - Accuracy: 93.34% - 3.26s
Epoch 4 - Accuracy: 94.03% - 3.25s
Epoch 5 - Accuracy: 94.63% - 3.26s
Epoch 6 - Accuracy: 94.95% - 3.24s
Epoch 7 - Accuracy: 95.29% - 3.26s
Epoch 8 - Accuracy: 95.64% - 3.23s
Epoch 9 - Accuracy: 95.89% - 3.22s
Epoch 10 - Accuracy: 96.03% - 3.26s
```
