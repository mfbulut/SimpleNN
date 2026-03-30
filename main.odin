package main

import "core:os"
import "core:fmt"
import "core:slice"
import "core:time"
import "core:math"
import "core:math/rand"

import rl "vendor:raylib"

Layer :: struct {
    input_size       : int,
    output_size      : int,
    weights          : []f32,
    biases           : []f32,
    weight_momentum  : []f32,
    bias_momentum    : []f32,
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

    CANVAS_SIZE :: 560
    CELL_SIZE :: 20

    rl.InitWindow(1280, 720, "Neural Networks")

    for !rl.WindowShouldClose() {
        mx := rl.GetMouseX() / CELL_SIZE
        my := rl.GetMouseY() / CELL_SIZE

        if rl.IsMouseButtonDown(.LEFT) {
            input.activation[my*28 + mx] = 1
        }

        if rl.IsMouseButtonDown(.RIGHT) {
            slice.zero(input.activation)
        }

        forward(hidden, input.activation)
        forward(output, hidden.activation)
        softmax(output.activation)

        rl.BeginDrawing()
        rl.ClearBackground({18, 18, 18, 255})

        rl.DrawRectangle(0, 0, CANVAS_SIZE, CANVAS_SIZE, rl.BLACK)

        for y in 0..<28 {
            for x in 0..<28 {
                v := u8(input.activation[y*28+x] * 255)
                rl.DrawRectangle(i32(x * CELL_SIZE), i32(y * CELL_SIZE), CELL_SIZE, CELL_SIZE, {v, v, v, 255})
            }
        }

        best := slice.max_index(output.activation)

        for i in 0..<10 {
            bx := CANVAS_SIZE + i32(i) * (CELL_SIZE + CELL_SIZE)

            rl.DrawRectangle(bx, 0, CELL_SIZE, CANVAS_SIZE, rl.DARKGRAY)
            fill := i32(f32(CANVAS_SIZE) * output.activation[i])

            if i == best {
                rl.DrawRectangle(bx, CANVAS_SIZE - fill, CELL_SIZE, fill, rl.DARKGREEN)
            } else {
                rl.DrawRectangle(bx, CANVAS_SIZE - fill, CELL_SIZE, fill, rl.GRAY)
            }

            rl.DrawText(fmt.ctprintf("%d", i), bx + CELL_SIZE/2 - 5, CANVAS_SIZE + 8, 18, rl.WHITE)
        }

        rl.EndDrawing()
    }
}

init_layer :: proc(input_size: int, output_size: int) -> Layer {
    layer := Layer{
        input_size      = input_size,
        output_size     = output_size,
        weights         = make([]f32, input_size * output_size),
        biases          = make([]f32, output_size),
        weight_momentum = make([]f32, input_size * output_size),
        bias_momentum   = make([]f32, output_size),
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
MOMENTUM :: 0.9

backward :: proc(curr: Layer, prev: Layer) {
    for i in 0..<curr.output_size {
        // if ReLU is zero then derivative is also zero
        if curr.activation[i] <= 0 {
            continue
        }

        for j in 0..<curr.input_size {
            idx := i * curr.input_size + j

            prev.gradient[j] += curr.gradient[i] * curr.weights[idx]

            curr.weight_momentum[idx] = MOMENTUM * curr.weight_momentum[idx] + curr.gradient[i] * prev.activation[j]
            curr.weights[idx] -= LEARNING_RATE * curr.weight_momentum[idx]
        }

        curr.bias_momentum[i] = MOMENTUM * curr.bias_momentum[i] + curr.gradient[i]
        curr.biases[i] -= LEARNING_RATE * curr.bias_momentum[i]
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