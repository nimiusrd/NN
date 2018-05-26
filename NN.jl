const NUMBER_OF_INPUT_NODE = 3 # 入力層のノード数
const NUMBER_OF_OUTPUT_NODE = 1 # 出力層のノード数
const NUMBER_OF_MIDDLE_LAYER_NODE = 6 # 中間層のノード数
const NUMBER_OF_LAYERS = 3 # 層数
const ε = 0.2 # 学習率
const LEARNING_LIMIT = 100000 # 最大学習数
const ERROR_LIMIT = 0.01 # 許容する誤差の上限

const node = [[NUMBER_OF_INPUT_NODE + 1]; fill(NUMBER_OF_MIDDLE_LAYER_NODE + 1, NUMBER_OF_LAYERS - 2); [NUMBER_OF_OUTPUT_NODE]]
sigmoid = s -> 1 / (1 + e^-s)
calc_out_layer_delta(y, t) = (1 - y) * y * 2 * (y - t) 
function calc_mid_layer_delta(y, w, delta)
    (1 - y) * y * w * delta
end
loss(y, t) = (y - t).^2
function read_data(filename)
    open(filename, "r") do io
        [
            vec([
                parse(Float64, str)
                for str in split(l, " ")
            ])
            for l in readlines(io)
        ]
    end
end
function get_ys(ws, input)
    local prev
    ntuple(
        i ->
            if i === 1
                prev = input
                [prev; 1]
            elseif i === NUMBER_OF_LAYERS
                prev = map(sigmoid, ws[i - 1]' * [prev; 1])
            else
                prev = map(sigmoid, ws[i - 1]' * [prev; 1])
                [prev; 1]
            end,
        NUMBER_OF_LAYERS
    )
end
function train(ws, input, test)
    ys = get_ys(ws, input)
    ws = let
        local prev_delta
        new_ws = ()
        for i = NUMBER_OF_LAYERS:-1:2
            if i === NUMBER_OF_LAYERS
                delta = [
                    calc_out_layer_delta(ys[i][y], test[y])
                    for y = 1:node[i]
                ]
                new_ws = (
                    reshape(
                        [
                            ws[i - 1][x, y] - ε * delta[y] * ys[i - 1][x]
                            for x = 1:node[i - 1]
                            for y = 1:node[i]
                        ],
                        (node[i], node[i - 1])
                    )',
                    new_ws...
                )
                prev_delta = delta .* ws[i - 1][1:length(ws[i - 1]) - 1]
            else
                delta = reshape(
                    [
                        calc_mid_layer_delta(ys[i][y], ws[i - 1][x, y], prev_delta[y])
                        for x = 1:node[i - 1]
                        for y = 1:node[i] - 1
                    ],
                    (node[i] - 1, node[i - 1])
                )'
                new_ws = (
                    reshape(
                        [
                            ws[i - 1][x, y] - ε * delta[x, y] * ys[i - 1][x]
                            for x = 1:node[i - 1] 
                            for y = 1:node[i] - 1
                        ],
                        (node[i] - 1, node[i - 1])
                    )',
                    new_ws...
                )
                prev_delta = delta
            end
        end
        new_ws
    end
    ws
end
function test(ws, input)
    ys = get_ys(ws, input)
    ys[NUMBER_OF_LAYERS]
end

let
    # データの読み込み
    x_train = read_data("train")
    y_train = read_data("test")

    # 重みの初期化
    ws = ntuple(
        i ->
            if i === NUMBER_OF_LAYERS - 1
                rand(node[i], node[i + 1])
            else
                rand(node[i], node[i + 1] - 1)
            end,
        NUMBER_OF_LAYERS - 1
    )

    # 学習
    c = 0
    err = ERROR_LIMIT * length(x_train) + 1
    while sum(err) / length(x_train) > ERROR_LIMIT && LEARNING_LIMIT > c
        err = zeros(NUMBER_OF_OUTPUT_NODE)
        for i=1:length(x_train)
            ws = train(ws, x_train[i], y_train[i])
        end
        for i=1:length(x_train)
            ys = get_ys(ws, x_train[i])
            err += loss(ys[NUMBER_OF_LAYERS], y_train[i])
        end
        c += 1
        @show c
        @show err / length(x_train)
    end

    # テスト
    input = read_data("input")
    println("input output")
    for x in input
        result = test(ws, x)
        println(x, result)
    end
end