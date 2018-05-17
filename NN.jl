const ε = 0.05 # learning late
const number_of_input_node = 3
const number_of_output_node = 1
const number_of_layers = 3
const number_of_middle_layer_node = 4

sigmoid(s) = 1 / (1 + e^-s)
update_out_layer(x, y, w, t) = w - 2ε * x * y * (1 - y) * (y - t) 
update_mid_layer(x, y, w, Y) = w - ε * x * Y * (1 - Y) * y 

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

ys = let
    local prev
    ntuple(
        i ->
            if i === 1
                prev = eye(number_of_input_node)
                [
                    sigmoid(sum(prev[i, :]))
                    for i=1:size(prev)[1]
                ]
            else
                prev = ws[i - 1]' * prev
                [
                    sigmoid(sum(prev[i, :]))
                    for i=1:size(prev)[1]
                ]
            end,
        number_of_layers
    )
end

ws = ntuple(i ->
    reshape([
        i === length(node) - 1 ? update_out_layer(ys[i][x], ys[i + 1][y], ws[i][x, y], 1) : update_mid_layer(ys[i][x], ys[i + 1][y], ws[i][x, y], ys[length(node)][Y])
        for x=1:node[i]
        for y=1:node[i + 1]
        for Y=1:node[length(node)]
    ], (node[i], node[i + 1])),
    length(node) - 1
)
