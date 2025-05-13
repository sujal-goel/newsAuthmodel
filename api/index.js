import * as tf from '@tensorflow/tfjs-node'
import fs from 'fs'
import path from 'path'
import express from 'express';
import bodyParser from 'body-parser';

const app = express();
app.use(bodyParser.json());

let model = null;
let wordIndex = {};

async function loadModelAndTokenizer() {
  if (!model) {
    const modelePath = path.join(process.cwd(), 'model_package', 'tfjs_model', 'model.json');
    model = await tf.loadLayersModel(`file://${modelePath}`);
    const wordIndexPath = path.join(process.cwd(), 'model_package', 'tokenizer_word_index.json');
    wordIndex = JSON.parse(fs.readFileSync(wordIndexPath, 'utf-8'));
  }
}

function tokenizeText(text, maxLen = 500) {
  text = text.toLowerCase().replace(/[^a-z\s]/g, '').split(' ');
  let tokens = text.map(word => wordIndex[word] || 0);
  if (tokens.length > maxLen) tokens = tokens.slice(0, maxLen);
  while (tokens.length < maxLen) tokens.unshift(0);
  return tf.tensor2d([tokens]);
}

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }
  await loadModelAndTokenizer();
  const { text } = req.body;
  const input = tokenizeText(text);
  const predictionTensor = model.predict(input);
  const predictionData = Array.isArray(predictionTensor)
    ? await predictionTensor[0].data()
    : await predictionTensor.data();
  const label = predictionData[0] > 0.5 ? 'REAL' : 'FAKE';
  res.status(200).json({ prediction: label, confidence: predictionData[0].toFixed(2) });
}
app.get("/",(res)=>{res.send("apiRunningSuccessfully")})
// Use the handler for POST requests
app.post('/predict', handler);

// Use the port from environment variable or default to 3000
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});