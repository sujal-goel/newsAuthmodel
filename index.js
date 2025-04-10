import express from 'express'
import * as tf from '@tensorflow/tfjs'
import bodyParser from 'body-parser'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url' // Add this import

// Get current directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.json());

let model;
let wordIndex = {};

// Simple tokenizer simulation
function tokenizeText(text, maxLen = 500) {
    text = text.toLowerCase().replace(/[^a-z\s]/g, '').split(' ');
    let tokens = text.map(word => wordIndex[word] || 0);
    if (tokens.length > maxLen) tokens = tokens.slice(0, maxLen);
    while (tokens.length < maxLen) tokens.unshift(0);
    return tf.tensor2d([tokens]);
  }

  app.post('/predict',async (req, res)=>{
    const {text }= req.body();
    const input = tokenizeText(text);
    const prediction =  await model?.predict(input).data();
    const label = prediction[0] > 0.5 ? 'REAL' : 'FAKE';
    res.json({prediction:label, confidence: prediction[0].toFixed(2)});

  });



app.listen(3000,async ()=>{
    try{
        const modelePath = `file:///C:/Users/kanak/Desktop/dbms/Crowd-sourced-news-and-fact-checking-platform/model/model_package/tfjs_model/model.json`;
        // const modelePath = path.join(__dirname, 'model_package','tfjs_model','model.json');
    model = await tf.loadLayersModel(modelePath);
    const wordIndexPath = path.join(__dirname, 'model_package', 'tokenizer_word_index.json');
    wordIndex = JSON.parse(fs.readFileSync(wordIndexPath,'UTF-8'));
    console.log(`app started at : http://localhost:3000/`);
    }
    catch (error) {
        console.error("Error loading model:", error);
    }
})

