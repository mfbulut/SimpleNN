package main

import "core:os"
import "core:fmt"
import "core:slice"
import "core:time"
import "core:math"
import "core:math/rand"

Layer :: struct {
    input_size       : int,
    output_size      : int,
    weights          : []f32,
    biases           : []f32,
    activation       : []f32,
    gradient         : []f32,
}

main :: proc() {
    images, _ := os.read_entire_file("mnist/images.bin", context.allocator)
    labels, _ := os.read_entire_file("mnist/labels.bin", context.allocator)

    // images, _ := os.read_entire_file("fashion/images.bin", context.allocator)
    // labels, _ := os.read_entire_file("fashion/labels.bin", context.allocator)

    input  := init_layer(0, 784)
    hidden := init_layer(784, 128)
    output := init_layer(128, 10)

    for epoch in 1..=10 {
        epoch_start := time.now()

        for i in 0..<50000 {
            for j in 0..<784 {
                input.activation[j] = f32(images[i * 784 + j]) / 255.0
            }

            // Forward
            forward(hidden, input.activation)
            forward(output, hidden.activation)
            softmax(output.activation)

            // Backward
            for j in 0..<output.output_size {
                output.gradient[j] = output.activation[j] - (j == int(labels[i]) ? 1 : 0)
            }

            slice.zero(hidden.gradient)
            backward(output, hidden)
            backward(hidden, input)
        }

        correct := 0
        for i in 50000..<60000 {
            for j in 0..<784 {
                input.activation[j] = f32(images[i * 784 + j]) / 255.0
            }

            forward(hidden, input.activation)
            forward(output, hidden.activation)
            softmax(output.activation)

            if slice.max_index(output.activation) == int(labels[i]) {
                correct += 1
            }
        }

        duration := time.duration_seconds(time.since(epoch_start))
        fmt.printf("Epoch %d - Accuracy: %.2f%% - %.2fs\n", epoch, f32(correct) / (10000.0) * 100.0, duration)
    }
}

init_layer :: proc(input_size: int, output_size: int) -> Layer {
    layer := Layer{
        input_size      = input_size,
        output_size     = output_size,
        weights         = make([]f32, input_size * output_size),
        biases          = make([]f32, output_size),
        activation      = make([]f32, output_size),
        gradient        = make([]f32, output_size),
    }

    scale := math.sqrt(f32(2.0) / f32(input_size))
    for &weight in layer.weights {
        weight = rand.float32_normal(0, scale)
    }

    return layer
}

forward :: proc(layer: Layer, input: []f32) {
    for i in 0..<layer.output_size {
        sum := layer.biases[i]
        for j in 0..<layer.input_size {
            sum += layer.weights[i * layer.input_size + j] * input[j]
        }

        layer.activation[i] = max(sum, 0)
    }
}

LEARNING_RATE :: 0.0005

backward :: proc(curr: Layer, prev: Layer) {
    for i in 0..<curr.output_size {
        if curr.activation[i] <= 0 {
            continue
        }

        for j in 0..<curr.input_size {
            idx := i * curr.input_size + j

            prev.gradient[j] += curr.gradient[i] * curr.weights[idx]
            curr.weights[idx] -= curr.gradient[i] * prev.activation[j] * LEARNING_RATE
        }

        curr.biases[i] -= curr.gradient[i] * LEARNING_RATE
    }
}

softmax :: proc(output: []f32) {
    max_val := slice.max(output)

    sum : f32 = 0
    for &v in output {
        v = math.exp(v - max_val)
        sum += v
    }

    for &v in output {
        v /= sum
    }
}