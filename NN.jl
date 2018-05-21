const ε = 0.75 # learning late
const θ = 0.01 # error limit
const number_of_input_node = 3
const number_of_output_node = 1
const number_of_layers = 3
const number_of_middle_layer_node = 4
const learning_limit = 500

sigmoid(s) = 1 / (1 + e^-s)
calc_out_layer_delta(y, t) = ε * (1 - y) * y * 2 * (y - t) 
function calc_mid_layer_delta(y, ws, delta)
    @show ws
    @show delta
    ε * (1 - y) * y * dot(ws, delta)
end
loss(y, t) = (y - t).^2
function get_ys(ws, input)
    local prev
    ntuple(
        i ->
            if i === 1
                prev = eye(number_of_input_node) .* input
                [input; -θ]
            elseif i === number_of_layers
                prev = ws[i - 1]' * prev
                [
                    sigmoid(sum(prev[i, :]))
                    for i=1:size(prev)[1]
                ]
            else
                prev = ws[i - 1]' * prev
                [
                    [
                        sigmoid(sum(prev[i, :]))
                        for i=1:size(prev)[1]
                    ];
                    [-θ]
                ]
            end,
        number_of_layers
    )
end

# データの読み込み
x_train = open("train", "r") do io
    [
        vec([
            parse(Float64, str)
            for str in split(l, " ")
        ])
        for l in readlines(io)
    ]
end

y_train = open("test", "r") do io
    [
        vec([
            parse(Float64, str)
            for str in split(l, " ")
        ])
        for l in readlines(io)
    ]
end

# 各層のノード数の配列
node = [[number_of_input_node]; fill(number_of_middle_layer_node, number_of_layers - 2); [number_of_output_node]]

# 重みの初期化
ws = ntuple(
    i -> rand(node[i], node[i + 1]),
    length(node) - 1
)
function train(ws, input, test)
    ys = get_ys(ws, input)
    ws = let
        local prev_delta
        new_ws = ()
        for i = length(node):-1:2
            if i === length(node)
                delta = [
                    calc_out_layer_delta(ys[i][y], test[y])
                    for y = 1:node[i]
                ]
                new_ws = (
                    reshape(
                        [
                            ws[i - 1][x, y] - delta[y] * ys[i - 1][x]
                            for x = 1:node[i - 1] 
                            for y = 1:node[i]
                        ],
                        (node[i - 1], node[i])
                    ),
                    new_ws...
                )
                prev_delta = delta
            else
                delta = [
                    calc_mid_layer_delta(ys[i][y], ws[i][x, :], prev_delta)
                    for x = 1:node[i]
                    for y = 1:node[i + 1]
                ]
                new_ws = (
                    reshape(
                        [
                            ws[i - 1][x, y] - delta[y] * ys[i - 1][x]
                            for x = 1:node[i - 1] 
                            for y = 1:node[i]
                        ],
                        (node[i - 1], node[i])
                    ),
                    new_ws...
                )
                prev_delta = delta
            end
        end
        new_ws
    end
    ws, ys[number_of_layers]
end

function test(ws, input)
    ys = get_ys(ws, input)
    ys[number_of_layers]
end

let
    c = 0
    err = θ * length(x_train) + 1
    while sum(err) / length(x_train) > θ && learning_limit > c
        err = zeros(number_of_output_node)
        ws = let
            for i=1:length(x_train)
                ws, y = train(ws, x_train[i], y_train[i])
                err += loss(y, y_train[i])
            end
            ws
        end
        c += 1
        @show c
        @show err
    end
end

let
    input = open("input", "r") do io
        [
            vec([
                parse(Float64, str)
                for str in split(l, " ")
            ])
            for l in readlines(io)
        ]
    end
    println("input output")
    for x in input
        result = test(ws, x)
        println(x, result)
    end
end