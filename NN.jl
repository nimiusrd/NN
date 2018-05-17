const ε = 0.05 # learning late
const number_of_input_node = 3
const number_of_output_node = 1
const number_of_layers = 3
const number_of_middle_layer_node = 4

sigmoid(s) = 1 / (1 + e^-s)
update(x, y, t, w) = w - 2ε * x * y * (y - t) * (1 - y)

# データの読み込み
train = open("train", "r") do io
    [
        [
            parse(Float64, str)
            for str in split(l, " ")
        ]
        for l in readlines(io)
    ]
end

test = open("test", "r") do io
    [
        parse(Float64, l)
        for l in readlines(io)
    ]
end

# 各層のノード数の配列
node = vcat([number_of_input_node], fill(number_of_middle_layer_node, number_of_layers - 2), [number_of_output_node])

# 重みのタプル
ws = ntuple(i -> rand(node[i], node[i + 1]), length(node) - 1)

us = let
    local prev
    ntuple(
        i ->
            if i == 1
                prev = eye(number_of_input_node)
            else
                prev = ws[i - 1]' * prev
            end,
        number_of_layers
    )
end
