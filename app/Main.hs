module Main where

import DecoderTransformer

main :: IO ()
main = do
  let epochs = 28
  let trainFile = "rpg-training-tokenized.txt"
  let evalFile = "rpg-evaluation-tokenized.txt"
  let tokensPerBatch = 260 -- (SeqLen + 1) * BatchSize
  let trainTokens = 192 * tokensPerBatch -- "I want 100 batches"
  let evalTokens = 192 * tokensPerBatch

-- --- SCENARIO 1: Train Fresh (Overwrite old model) ---
  -- putStrLn "--- Starting Fresh Training ---"
  -- program epochs trainFile trainTokens evalFile evalTokens Nothing (Just "haskell_model.pt")

-- --- SCENARIO 2: Resume Training (Load existing, train more, save back) ---
-- putStrLn "--- Resuming Training ---"
-- program 100 trainFile trainBatches evalFile evalBatches (Just "haskell_model.pt") (Just "haskell_untyped_model.pt")

-- --- SCENARIO 3: Generate Only (Skip training) ---
-- Epochs = 0 skips the training loop entirely
  -- putStrLn "--- Generating Text from Saved Model ---"
  -- program 0 trainFile trainTokens evalFile evalTokens (Just "tiny_shakespeare_untyped_model.pt") Nothing

  -- --- SCENARIO 4: Train Fresh (No save, no read) ---
  putStrLn "--- Starting Fresh Training ---"
  program epochs trainFile trainTokens evalFile evalTokens Nothing Nothing
