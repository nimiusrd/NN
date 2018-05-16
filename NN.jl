const ε = 0.05 # learning late
const number_of_input_node = 3
const number_of_output_node = 1
const number_of_layers = 3
const number_of_middle_layer_node = 4

sigmoid(s) = 1 / (1 + e^-s)
update(x, y, t, w) = w - 2ε * x * y * (y - t) * (1 - y)

# 各層のノードの数の配列
node = vcat([number_of_input_node], fill(number_of_middle_layer_node, number_of_layers - 2), [number_of_output_node])

# 重みのタプル
ws = ntuple(i -> rand(node[i], node[i + 1]), length(node) - 1)

