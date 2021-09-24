const tf = require('@tensorflow/tfjs-node');
const fs = require('fs/promises');

const MODEL_DIRECTORY = 'file://./model/model.json';
const BUNDLE_DIRECTORY = './model/bundle.json';

const SAMPLE_SENTENCE = 'fox socks box knox knox in box fox in socks knox on fox in socks in box socks on knox and knox in box fox in socks on box on knox';
const SAMPLE_OFFSET = 0;
const N_SAMPLES = 100;
const TEMPERATURE = 0.0;

async function test() {
    // load model from disk
    const model = await tf.loadLayersModel(MODEL_DIRECTORY);
    model.summary();

    // load metadata from disk
    const metadata = JSON.parse(await fs.readFile(BUNDLE_DIRECTORY, 'utf-8'));

    // make predictions
    const seed = SAMPLE_SENTENCE.substr(SAMPLE_OFFSET, metadata.sequenceLength);
    let sentence = seed;
    const predictedChars = [];
    let lastPrediction = null;
    for (let i = 0; i < N_SAMPLES; i++) {
        // encode the sentence and create a tensor
        const sampleEncoded = encode([...sentence], metadata.charToId);
        const input = toTensor(sampleEncoded, metadata.charSetSize);
        // console.log(await toChars(input, metadata.idToChar));

        // make a prediction
        const prediction = await model.predict(input);

        // // check the differences
        // if (lastPrediction != null) {
        //     const sqDiff = await tf.squaredDifference(prediction, lastPrediction).data();
        //     console.log('sqDiff:', [...sqDiff]);
        // }
        // lastPrediction = prediction;

        // sample an answer and record it
        const sampleId = sample(tf.squeeze(prediction), TEMPERATURE);
        const char = metadata.idToChar[sampleId];
        predictedChars.push(char);

        // add the predicted char to the sentence, and chop off a letter at the start
        sentence = sentence.substr(1) + char;
    }

    // print out the starting sentence and predicted next chars
    console.log('>' + seed + '\n' + predictedChars.join(''));
}

// creates a Tensor of shape [1, 1, ids.length, charSetSize] from a list of ids
function toTensor(ids, charSetSize) {
    const buffer = tf.buffer([1, ids.length, charSetSize]);
    for (let i = 0; i < ids.length; i++) {
        const id = ids[i];
        buffer.set(1, 0, i, id);
    }
    return buffer.toTensor();
}

async function toChars(output, idToChar) {
    const chars = [];
    const elements = (await output.array())[0];
    elements.forEach((row) => {
        for (let i = 0; i < row.length; i++) {
            if (row[i] === 1) {
                chars.push(idToChar[i]);
                break;
            }
        }
    })
    return chars.join();
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

// samples an output tensor for an id
// src: https://github.com/tensorflow/tfjs-examples/blob/master/lstm-text-generation/model.js
function sample(probs, temperature) {
    return tf.tidy(() => {
        const logits = tf.div(tf.log(probs), Math.max(temperature, 1e-6));
        const isNormalized = false;
        // `logits` is for a multinomial distribution, scaled by the temperature.
        // We randomly draw a sample from the distribution.
        return tf.multinomial(logits, 1, null, isNormalized).dataSync()[0];
    });
}

test();
