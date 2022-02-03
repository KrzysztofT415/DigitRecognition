const IMAGE_H = 28
const IMAGE_W = 28
const IMAGE_SIZE = IMAGE_H * IMAGE_W
const N_CLASSES = 10
const N_DATA = 65000

const MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png'
const MNIST_LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8'

export class MnistData {
    async load(nTrain = 60000, nTest = 5000) {
        const img = new Image()
        const canvas = document.createElement('canvas')
        const ctx = canvas.getContext('2d')
        const imgRequest = new Promise((resolve, reject) => {
            img.crossOrigin = ''
            img.onload = () => {
                img.width = img.naturalWidth
                img.height = img.naturalHeight

                const datasetBytesBuffer = new ArrayBuffer(N_DATA * IMAGE_SIZE * 4)

                const chunkSize = 5000
                canvas.width = img.width
                canvas.height = chunkSize

                for (let i = 0; i < N_DATA / chunkSize; ++i) {
                    const datasetBytesView = new Float32Array(datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4, IMAGE_SIZE * chunkSize)
                    ctx.drawImage(img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width, chunkSize)

                    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)

                    for (let j = 0; j < imageData.data.length / 4; ++j) {
                        // All channels hold an equal value since the image is grayscale, so just read the red channel.
                        datasetBytesView[j] = imageData.data[j * 4] / 255
                    }
                }
                this.datasetImages = new Float32Array(datasetBytesBuffer)
                resolve()
            }
            img.src = MNIST_IMAGES_SPRITE_PATH
        })

        const labelsRequest = fetch(MNIST_LABELS_PATH)
        const [imgResponse, labelsResponse] = await Promise.all([imgRequest, labelsRequest])

        this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer())

        // Slice the the images and labels into train and test sets.
        this.trainImages = this.datasetImages.slice(0, IMAGE_SIZE * nTrain)
        this.testImages = this.datasetImages.slice(IMAGE_SIZE * nTrain, IMAGE_SIZE * (nTrain + nTest))
        this.trainLabels = this.datasetLabels.slice(0, N_CLASSES * nTrain)
        this.testLabels = this.datasetLabels.slice(N_CLASSES * nTrain, N_CLASSES * (nTrain + nTest))
        console.log(this.trainImages, this.trainLabels)
    }

    getTrainData() {
        const x_train = tf.tensor4d(this.trainImages, [this.trainImages.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1])
        const y_train = tf.tensor2d(this.trainLabels, [this.trainLabels.length / N_CLASSES, N_CLASSES])
        return [x_train, y_train]
    }

    getTestData(numExamples) {
        let x_test = tf.tensor4d(this.testImages, [this.testImages.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1])
        let y_test = tf.tensor2d(this.testLabels, [this.testLabels.length / N_CLASSES, N_CLASSES])

        if (numExamples != null) {
            x_test = x_test.slice([0, 0, 0, 0], [numExamples, IMAGE_H, IMAGE_W, 1])
            y_test = y_test.slice([0, 0], [numExamples, N_CLASSES])
        }
        return [x_test, y_test]
    }
}
