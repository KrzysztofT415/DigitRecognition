import { MnistData } from './mnist.js'
const canvas = document.getElementById('canvas')
const ctx = canvas.getContext('2d')

const mini = document.getElementById('mini')
const mini_ctx = mini.getContext('2d')

const info = document.getElementById('info')
info.innerText = 'Info log :\n\n'


// FETCHING DATA


let training_data, training_labels, test_data, test_labels, dataLoaded = false, model = null
const loadMnist = async(event) => {
    info.innerText += '-> Loading MNIST data\n'
    for (let data of document.getElementById('data').children)
        data.classList.remove('chosen')
    event.target.classList.add('chosen')

    const data = new MnistData()
    await data.load(60000, 10000)

    info.innerText += '-> Data loaded\n'

    let trainData = data.getTrainData()
    training_data = trainData[0]
    training_labels = trainData[1]
    let testData = data.getTestData()
    test_data = testData[0]
    test_labels = testData[1]
    dataLoaded = true
    let buttons = document.getElementsByClassName('model')
    for (const button of buttons) button.disabled = (model === null)
}
const loadImages = folder_path => {
    return new Promise((resolve, _) => {
        const SIZE = 28**2
        const datasetBytesBuffer = new ArrayBuffer(30 * SIZE * 4)
        const datasetLabels = new Uint8Array(30 * 10)
        let letters = ['a', 'b', 'c']
        for (let number = 0; number < 10; ++number) {
            for (let i = 0; i < letters.length; ++i) {
                let offset = (number * letters.length + i)
                datasetLabels[offset * 10 + number] = 1

                let datasetBytesView = new Float32Array(datasetBytesBuffer, offset * SIZE * 4, SIZE)
                let imageToGuess = new Image()
                imageToGuess.onload = () => {
                    mini_ctx.clearRect(0, 0, mini.width, mini.height)
                    mini_ctx.drawImage(imageToGuess, 0 , 0, 28, 28)
                    let imageData = mini_ctx.getImageData(0, 0, 28, 28)
                    for (let j = 0; j < imageData.data.length / 4; ++j)
                        datasetBytesView[j] = imageData.data[j * 4] / 255
                }
                imageToGuess.id = 'imageToPredict'
                imageToGuess.width = 28
                imageToGuess.height = 28
                imageToGuess.src = './' + folder_path + '/' + number + '' + letters[i] + '.png'
            }
        }
        let datasetBytes = new Float32Array(datasetBytesBuffer)
        resolve([datasetBytes, datasetLabels])
    })
}
const loadData = async(target, folder_path) => {
    info.innerText += '-> Loading ' + folder_path + '\n'
    for (let data of document.getElementById('data').children)
        data.classList.remove('chosen')
    target.classList.add('chosen')

    let imageData = await loadImages(folder_path)

    let trainImages = imageData[0]
    let trainLabels = imageData[1]

    info.innerText += '-> Data loaded\n'

    training_data = test_data = tf.tensor4d(trainImages, [trainImages.length / 28**2, 28, 28, 1])
    training_labels = test_labels = tf.tensor2d(trainLabels, [trainLabels.length / 10, 10])
    dataLoaded = true
    let buttons = document.getElementsByClassName('model')
    for (const button of buttons) button.disabled = (model === null)
}


// MODEL OPERATIONS


const createModel = _ => {
    model = tf.sequential({
        layers: [
            tf.layers.flatten({inputShape: [28, 28, 1]}),
            tf.layers.dense({units: 128, activation: 'relu'}),
            tf.layers.dense({units: 10, activation: 'softmax'})
        ]
    })
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })
    info.innerText += '-> Model created\n'

    let buttons = document.getElementsByClassName('model')
    for (const button of buttons) button.disabled = !dataLoaded
    document.getElementById('createModel').classList.add('chosen')
}
const loadModel = async() => {
    model = await tf.loadLayersModel('localstorage://my-model-1')
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })
    info.innerText += '-> Model loaded from local storage\n'
    let buttons = document.getElementsByClassName('model')
    for (const button of buttons) button.disabled = !dataLoaded
    document.getElementById('createModel').classList.add('chosen')
}
const saveModel = async() => {
    await model.save('localstorage://my-model-1')
    info.innerText += '-> Model saved on local storage\n'
}


// TRAINING AND TESTING MODEL


const train = async() => {
    info.innerText += '-> Beginning training\n: Accuracy\n'
    let epochs = document.getElementById('epochs').value
    if (!epochs) epochs = 1
    await model.fit(training_data, training_labels, {
        epochs: epochs,
        callbacks: { onEpochEnd: async(epoch, logs) => info.innerText += logs.acc + '\n' }
    })
    info.innerText += '-> Model trained\n'
}
const test = async() => {
    info.innerText += '-> Running tests\n: Accuracy '
    const result = await model.evaluate(test_data, test_labels)
    info.innerText += result[1] + '\n'
    info.innerText += '-> Tests ended\n'
}


// PREDICTION


const prediction = document.getElementById('prediction')
const getDataBytesView = async(imageData) => {
    let bytesArray = []
    for (let i = 0; i < imageData.data.length / 4; ++i)
        bytesArray[i] = imageData.data[i * 4] / 255
    return new Float32Array(bytesArray)
}
const predictModel = async(dataBytesView) => {
    let output = await model.predict(dataBytesView)
    let outputData = JSON.parse(output.toString().slice(13,150).split(',]')[0])
    info.innerText += '* Made prediction\n'

    let maximum = 0
    for (let i = 0; i < 10; ++i) {
        document.getElementById('predictions' + i).innerText = i + ' : ' + outputData[i] + '\n'
        if (outputData[i] > outputData[maximum]) maximum = i
    }

    prediction.innerText = maximum.toString()
}
const makePrediction = async() => {
    let imageToGuess = new Image()
    imageToGuess.onload = () => {
        mini_ctx.clearRect(0, 0, mini.width, mini.height)
        mini_ctx.drawImage(imageToGuess,0 , 0, 28, 28)
        let imageData = mini_ctx.getImageData(0, 0, 28, 28)
        getDataBytesView(imageData).then(dataBytesView => predictModel(tf.tensor4d(dataBytesView, [1, 28, 28, 1])))
    }
    imageToGuess.id = 'imageToPredict'
    imageToGuess.width = 28
    imageToGuess.height = 28
    imageToGuess.src = canvas.toDataURL()
}


// CANVAS OPERATIONS


const offset_x = canvas.offsetLeft
const offset_y = canvas.offsetTop
let last_mouse_x, last_mouse_y, mouse_x, mouse_y, mousedown, scale = 50
canvas.addEventListener('mousedown', e => {
    last_mouse_x = mouse_x = e.clientX - offset_x
    last_mouse_y = mouse_y = e.clientY - offset_y
    mousedown = true
})
canvas.addEventListener('mouseup', () => mousedown = false)
canvas.addEventListener('mousemove', e => {
    mouse_x = e.clientX - offset_x
    mouse_y = e.clientY - offset_y

    if(mousedown) {
        ctx.beginPath()
        ctx.globalCompositeOperation = 'source-over'
        ctx.strokeStyle = 'white'
        ctx.lineWidth = scale
        ctx.moveTo(last_mouse_x,last_mouse_y)
        ctx.lineTo(mouse_x,mouse_y)
        ctx.lineJoin = ctx.lineCap = 'round'
        ctx.stroke()
    }

    last_mouse_x = mouse_x
    last_mouse_y = mouse_y
})
canvas.addEventListener('wheel', e => scale += e.deltaY * -0.1)

const saveImg = _ => {
    let imageToGuess = new Image()
    imageToGuess.onload = () => {
        mini_ctx.clearRect(0, 0, mini.width, mini.height)
        mini_ctx.drawImage(imageToGuess,0 , 0, 28, 28)
        window.location.href = mini.toDataURL('image/png')
            .replace('image/png', 'image/octet-stream')
    }
    imageToGuess.width = 28
    imageToGuess.height = 28
    imageToGuess.src = canvas.toDataURL()
}


// BUTTONS AND WINDOW FUNCTIONALITY


document.getElementById('clear').onclick = _ => ctx.clearRect(0, 0, canvas.width, canvas.height)
document.getElementById('saveImg').onclick = saveImg

document.getElementById('createModel').onclick = createModel
document.getElementById('loadModel').onclick = loadModel
document.getElementById('saveModel').onclick = saveModel

document.getElementById('loadMnist').onclick = loadMnist
document.getElementById('loadMy').onclick = event => loadData(event.target, 'test_data_mine')
document.getElementById('loadFriend').onclick = event => loadData(event.target, 'test_data_friend')

document.getElementById('train').onclick = train
document.getElementById('test').onclick = test
document.getElementById('predict').onclick = makePrediction

const getInnermostHovered = _ => {
    let n = document.querySelector(":hover"), nn
    while (n) {
        nn = n
        n = nn.querySelector(":hover")
    }
    return nn
}
window.setInterval(_ => {
    if (getInnermostHovered() !== info)
        info.scrollTop = info.scrollHeight
}, 1000)