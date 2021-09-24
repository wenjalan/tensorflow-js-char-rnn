const tf = require('@tensorflow/tfjs-node');
const fs = require("fs/promises");

// model parameters
const SEQUENCE_LENGTH = 50;
const LSTM_UNITS = 128;

// file parameters
const CORPUS = './corpi/foxinsocks.txt';
const MODEL_DIRECTORY = 'file://./model/';
const BUNDLE_DIRECTORY = './model/bundle.json';

// training parameters
const TRAIN_EPOCHS = 50;
const TRAIN_BATCHES = 20;

// main
async function start() {
    // load training data
    const [trainXs, trainYs, metadata] = await getTrainingData(CORPUS);

    // debug: print out the entirety of both
    // trainXs.print();
    // trainYs.print();

    // create a model
    const model = createModel(metadata.numExamples, metadata.charSetSize);
    model.summary();

    // train the model
    await model.fit(trainXs, trainYs, {
        epochs: TRAIN_EPOCHS,
        batchSize: TRAIN_BATCHES,
    });

    // save the model and bundle to disk
    await model.save(MODEL_DIRECTORY);
    await fs.writeFile(BUNDLE_DIRECTORY, JSON.stringify(metadata));
    console.log('Saved model');
}

// returns a model to fit data to
function createModel(nExamples, charSetSize) {
    // sequential
    const model = tf.sequential();

    // lstm layer
    model.add(tf.layers.lstm({
        inputShape: [SEQUENCE_LENGTH, charSetSize],
        units: LSTM_UNITS,
        returnSequences: true,
    }));

    // dropout
    model.add(tf.layers.dropout({
        rate: 0.2,
    }));

    // flatten
    model.add(tf.layers.flatten());

    // dense
    model.add(tf.layers.dense({
        units: charSetSize,
        activation: 'softmax',
    }));

    // compile
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    // return
    return model;
}

// returns training tensors given a corpus
// corpus: the directory of a file to generate training data from
async function getTrainingData(corpus) {
    // log to console
    console.log('Generating training data from corpus', corpus);

    // read in the file, clean it and turn it into an array
    let data = await fs.readFile(corpus, 'utf-8');
    data = cleanData(data);
    data = [...data];
    console.log('> length =', data.length);

    // get a set of characters found in the data
    const charSet = new Set(data);
    console.log('> unique chars =', charSet.size);

    // encode each character to an id
    const [idToChar, charToId] = getEncodings(charSet);

    // create training examples of SEQUENCE_LENGTH length
    // x: a series of characters, SEQUENCE_LENGTH in length starting from i
    // y: a character that proceeds the characters in x
    const examples = [];
    for (let i = 0; i < data.length - (SEQUENCE_LENGTH - 1); i++) {
        // create input sequence of SEQUENCE_LENGTH length
        const inputSequence = [];
        for (let j = 0; j < SEQUENCE_LENGTH; j++) {
            inputSequence.push(data[i + j]);
        }

        // create output
        const output = [data[i + SEQUENCE_LENGTH]];

        // add both to examples
        examples.push([inputSequence, output]);
    }
    console.log('> n examples =', examples.length);

    // encode all examples to their ids
    const encodedExamples = [];
    examples.forEach(([inputSequence, output]) => {
        const encodedInput = encode(inputSequence, charToId);
        const encodedOutput = encode(output, charToId);
        encodedExamples.push([encodedInput, encodedOutput]);
    });

    // from encoded examples, create 1-hot training tensors
    // trainX: [n examples, sequence length, char set size]
    // trainY: [n examples, char set size]
    const trainXBuffer = tf.buffer([encodedExamples.length, SEQUENCE_LENGTH, charSet.size]);
    const trainYBuffer = tf.buffer([encodedExamples.length, charSet.size]);
    let exampleNumber = 0;
    encodedExamples.forEach(([input, output]) => {
        // 1-hot encode input sequence
        for (let i = 0; i < SEQUENCE_LENGTH; i++) {
            const index = input[i];
            trainXBuffer.set(1, exampleNumber, i, index);
        }

        // 1-hot encode output sequence
        trainYBuffer.set(1, exampleNumber, output[0]);

        // increment example n
        exampleNumber++;
    });
    const trainX = trainXBuffer.toTensor();
    const trainY = trainYBuffer.toTensor();

    // embed some metadata
    const metadata = {
        numExamples: examples.length,
        charSetSize: charSet.size,
        sequenceLength: SEQUENCE_LENGTH,
        charToId: charToId,
        idToChar: idToChar,
    };

    // return
    return [trainX, trainY, metadata];
}

// cleans data for processing
// 1. all lowercase
// 2. no punctuation
// str: a string of data to clean
function cleanData(str) {
    // lowercase
    let clean = str.toLowerCase();

    // remove !.,
    clean = clean.replace(/[!,.]/g, ' ');

    // remove line returns
    clean = clean.replace(/\r\n/g, ' ');

    // regularize spaces
    clean = clean.replace(/\s\s+/g, ' ');

    // return
    return clean;
}

// returns the encoding objects for a given set of characters
// chars: a set of characters
// returns: two objects, mapping an id to a char and vice versa
function getEncodings(chars) {
    const idToChar = {};
    const charToId = {};
    let i = 0;
    chars.forEach((c) => {
        idToChar[i] = c;
        charToId[c] = i;
        i++;
    });
    return [idToChar, charToId];
}

// encodes an array of characters to their ids, or vice versa
// arr: an array of characters
// map: the mapping to use (ex. char to id, id to char)
function encode(arr, map) {
    const encoded = [];
    arr.forEach(e => {
        encoded.push(map[e]);
    });
    return encoded;
}

// starts the program
start();
