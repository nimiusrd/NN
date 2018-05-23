const ε = 0.75 # 学習率
const NUMBER_OF_INPUT_NODE = 3 # 入力層のノード数
const NUMBER_OF_OUTPUT_NODE = 1 # 出力層のノード数
const NUMBER_OF_LAYERS = 3 # 層数
const NUMBER_OF_MIDDLE_LAYER_NODE = 4 # 中間層のノード数
const LEARNING_LIMIT = 10000 # 最大学習数
const ERROR_LIMIT = 0.001 # 許容する誤差の上限

const node = [[NUMBER_OF_INPUT_NODE + 1]; fill(NUMBER_OF_MIDDLE_LAYER_NODE + 1, NUMBER_OF_LAYERS - 2); [NUMBER_OF_OUTPUT_NODE]]
sigmoid = θ -> s -> 1 / (1 + e^-(s - θ))
calc_out_layer_delta(y, t) = ε * (1 - y) * y * 2 * (y - t) 
calc_mid_layer_delta(y, ws, delta) = ε * (1 - y) * y * dot(ws, delta)
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
                prev = ws[i - 1]' * [prev; 1]
                prev = [
                    sigmoid(ws[i - 1][node[i - 1], s])(prev[s])
                    for s = 1:node[i]
                ]
            else
                prev = ws[i - 1]' * [prev; 1]
                prev = [
                    sigmoid(ws[i - 1][node[i - 1], s])(prev[s])
                    for s = 1:node[i] - 1
                ]
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
                            for y = 1:node[i] - 1
                        ],
                        (node[i - 1], node[i] - 1)
                    ),
                    new_ws...
                )
                prev_delta = delta
            end
        end
        new_ws
    end
    ws, ys[NUMBER_OF_LAYERS]
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

    let
        c = 0
        err = ERROR_LIMIT * length(x_train) + 1
        while sum(err) / length(x_train) > ERROR_LIMIT && LEARNING_LIMIT > c
            err = zeros(NUMBER_OF_OUTPUT_NODE)
            ws = let
                for i=1:length(x_train)
                    ws, y = train(ws, x_train[i], y_train[i])
                    err += loss(y, y_train[i])
                end
                ws
            end
            c += 1
            @show c
            @show err / length(x_train)
        end
    end

    input = read_data("input")
    println("input output")
    for x in input
        result = test(ws, x)
        println(x, result)
    end
end