{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module DecoderTransformer (program) where

import qualified Control.Foldl as L
import Control.Monad (forever, replicateM, when)
import qualified Data.Maybe as Maybe
import qualified Data.Set.Ordered as OSet
import qualified Data.Text as T
import qualified Data.Text.IO as T
import GHC.Generics
import Pipes
import qualified Pipes.Prelude as P
import qualified Pipes.Safe as Safe
import System.IO (hFlush, stdout)
import Torch (DType, Device, Parameter, Tensor, asTensor, defaultOpts, makeIndependent, ones, toDependent, zeros)
import qualified Torch as Th
import qualified Torch.Autograd as TA
import qualified Torch.Functional as F
import qualified Torch.Functional.Internal as FI
import qualified Torch.NN as NN
import qualified Torch.Optim as Optim
import qualified Torch.Serialize as Serialize

--------------------------------------------------------------------------------
-- 1. Custom Layer Definitions
--------------------------------------------------------------------------------

-- | Custom LayerNorm Struct
data LayerNorm = LayerNorm
  { lnWeight :: Parameter,
    lnBias :: Parameter,
    lnShape :: [Int],
    lnEps :: Double
  }
  deriving (Generic, Show)

initLayerNorm :: Int -> Double -> IO LayerNorm
initLayerNorm dim eps = do
  w <- makeIndependent (Th.ones [dim] defaultOpts)
  b <- makeIndependent (zeros [dim] defaultOpts)
  return $ LayerNorm w b [dim] eps

forwardLayerNorm :: LayerNorm -> Tensor -> Tensor
forwardLayerNorm LayerNorm {..} input =
  FI.layer_norm input lnShape (toDependent lnWeight) (toDependent lnBias) lnEps True

instance NN.Parameterized LayerNorm

--------------------------------------------------------------------------------
-- 2. Model Definition
--------------------------------------------------------------------------------

batchSize :: Int
batchSize = 4

seqLen :: Int
seqLen = 64

numAttnLayers :: Int
numAttnLayers = 2

numHeads :: Int
numHeads = 4

ffnDim :: Int
ffnDim = 256

embedDim :: Int
embedDim = 64

paddingIdx :: Int
paddingIdx = 0

data TransformerModel = TransformerModel
  { embedWeights :: Parameter,
    posEncoding :: Parameter, -- <--- CHANGED: Now a Parameter
    layers :: [TransformerBlock],
    finalLayer :: NN.Linear
  }
  deriving (Generic, Show)

data TransformerBlock = TransformerBlock
  { attention :: MultiHeadAttention,
    norm1 :: LayerNorm,
    feedForward :: FeedForward,
    norm2 :: LayerNorm
  }
  deriving (Generic, Show)

data MultiHeadAttention = MultiHeadAttention
  { mhaLinearQ :: NN.Linear,
    mhaLinearK :: NN.Linear,
    mhaLinearV :: NN.Linear,
    mhaLinearOut :: NN.Linear,
    mhaHeads :: Int,
    mhaEmbedDim :: Int
  }
  deriving (Generic, Show)

data FeedForward = FeedForward
  { ffLinear1 :: NN.Linear,
    ffLinear2 :: NN.Linear
  }
  deriving (Generic, Show)

instance NN.Parameterized TransformerModel

instance NN.Parameterized TransformerBlock

instance NN.Parameterized MultiHeadAttention

instance NN.Parameterized FeedForward

initModel :: Int -> IO TransformerModel
initModel vocabSize = do
  -- 1. Word Embeddings (Scaled down)
  rawWeight <- Th.randnIO' [vocabSize, embedDim]
  let initWeight = (0.005 :: Float) `Th.mulScalar` rawWeight
  embedParam <- makeIndependent initWeight

  -- 2. Layers
  blockSpecs <- mapM (const sampleBlock) [1 .. numAttnLayers]
  outProj <- NN.sample $ NN.LinearSpec embedDim vocabSize

  -- 3. FIX: Initialize Positional Encoding as a Parameter
  -- We use the same scaling (0.02) to prevent exploding gradients
  rawPos <- Th.randnIO' [1, seqLen, embedDim]
  let scaledPos = (0.005 :: Float) `Th.mulScalar` rawPos
  posParam <- makeIndependent scaledPos -- <--- Wrap in Parameter
  return $ TransformerModel embedParam posParam blockSpecs outProj
  where
    sampleBlock = do
      attn <- sampleMHA
      n1 <- initLayerNorm embedDim 1e-5
      n2 <- initLayerNorm embedDim 1e-5
      ff <- sampleFF
      return $ TransformerBlock attn n1 ff n2

    sampleMHA = do
      q <- NN.sample $ NN.LinearSpec embedDim embedDim
      k <- NN.sample $ NN.LinearSpec embedDim embedDim
      v <- NN.sample $ NN.LinearSpec embedDim embedDim
      o <- NN.sample $ NN.LinearSpec embedDim embedDim
      return $ MultiHeadAttention q k v o numHeads embedDim

    sampleFF = do
      l1 <- NN.sample $ NN.LinearSpec embedDim ffnDim
      l2 <- NN.sample $ NN.LinearSpec ffnDim embedDim
      return $ FeedForward l1 l2

forward :: TransformerModel -> Tensor -> Tensor
forward TransformerModel {..} input =
  let w = toDependent embedWeights
      emb = F.embedding False False w paddingIdx input

      -- FIX: Unwrap the learned Positional Parameter
      pos = toDependent posEncoding

      -- Add Position to Embedding
      x = emb + pos

      x' = foldl forwardBlock x layers
      logits = NN.forward finalLayer x'
   in logits

forwardBlock :: Tensor -> TransformerBlock -> Tensor
forwardBlock x TransformerBlock {..} =
  let attnOut = forwardMHA attention x
      x2 = forwardLayerNorm norm1 (x + attnOut)
      ffOut = forwardFF feedForward x2
      x3 = forwardLayerNorm norm2 (x2 + ffOut)
   in x3

forwardFF :: FeedForward -> Tensor -> Tensor
forwardFF FeedForward {..} x =
  let hidden = F.relu (NN.forward ffLinear1 x)
   in NN.forward ffLinear2 hidden

-- | NEW: The "Real" Multi-Head Attention Implementation
forwardMHA :: MultiHeadAttention -> Tensor -> Tensor
forwardMHA MultiHeadAttention {..} x =
  let -- 1. Linear Projections
      -- Input x: [Batch, SeqLen, EmbedDim]
      q = NN.forward mhaLinearQ x
      k = NN.forward mhaLinearK x
      v = NN.forward mhaLinearV x

      -- Calculate dimension per head
      headDim = mhaEmbedDim `Prelude.div` mhaHeads
      batch = head (Th.shape x)
      seqLength = Th.shape x !! 1

      -- 2. Reshape and Transpose for Heads
      -- Target: [Batch, Heads, SeqLen, HeadDim]
      -- We first reshape to [Batch, SeqLen, Heads, HeadDim]
      viewShape = [batch, seqLength, mhaHeads, headDim]

      q' = F.transpose (F.Dim 1) (F.Dim 2) $ Th.reshape viewShape q
      k' = F.transpose (F.Dim 1) (F.Dim 2) $ Th.reshape viewShape k
      v' = F.transpose (F.Dim 1) (F.Dim 2) $ Th.reshape viewShape v

      -- 3. Scaled Dot-Product Attention
      -- Scores = Q * K^T / sqrt(d_k)
      -- Q: [B, H, S, D]
      -- K^T: [B, H, D, S] (Transpose last two dims of k')
      kT = F.transpose (F.Dim 2) (F.Dim 3) k'

      scoresRaw = F.matmul q' kT -- Result: [B, H, S, S]

      -- Scale
      dk = asTensor (fromIntegral headDim :: Float)
      scoresScaled = scoresRaw / F.sqrt dk

      -- 4. Causal Masking (Autoregressive)
      -- We want to mask positions where j > i (future tokens)
      -- Create a mask of shape [Seq, Seq] with -inf in upper triangle
      mask = makeCausalMask seqLength (Th.device x) (Th.dtype x)

      -- Apply mask (Add -inf to masked positions)
      -- scoresScaled is [B, H, S, S], mask is [S, S]. Broadcasting handles this.
      scoresMasked = scoresScaled + mask

      -- 5. Softmax
      attnWeights = F.softmax (F.Dim 3) scoresMasked

      -- 6. Context = Attn * V
      -- Attn: [B, H, S, S] * V': [B, H, S, D] -> [B, H, S, D]
      context = F.matmul attnWeights v'

      -- 7. Recombine Heads
      -- Transpose back to [B, S, H, D]
      contextT = F.transpose (F.Dim 1) (F.Dim 2) context
      -- Reshape to [B, S, EmbedDim]
      contextReshaped = Th.reshape [batch, seqLength, mhaEmbedDim] contextT
   in NN.forward mhaLinearOut contextReshaped

-- | Helper: Create a Causal Mask (Upper Triangular = -inf)
makeCausalMask :: Int -> Device -> DType -> Tensor
makeCausalMask sz dev dtype =
  let -- Create a matrix of ones [sz, sz]
      opts = Th.withDevice dev (Th.withDType dtype defaultOpts)
      onesMat = ones [sz, sz] opts
      -- Create Upper Triangular mask (1s in upper triangle, 0s elsewhere)
      -- We assume 'triu' exists or simulate it.
      -- Since Haskell bindings might lack 'triu', we can construct it manually via comparisons if needed.
      -- Fortunately, Hasktorch F.triu exists.
      upperTri = F.triu (F.Diag 1) onesMat

      -- Convert 1s to -inf, 0s to 0.0
      -- We multiply by a very large negative number
      negInf = -1e9 :: Float
   in upperTri * asTensor negInf

--------------------------------------------------------------------------------
-- 3. Data Pipeline (Unchanged)
--------------------------------------------------------------------------------

data TransformerData = TransformerData
  { dataLength :: Int,
    filePath :: FilePath,
    vocab :: OSet.OSet T.Text
  }

program :: Int -> FilePath -> Int -> FilePath -> Int -> Maybe FilePath -> Maybe FilePath -> IO ()
program numEpochs trainingFile trainingLen evaluationFile evaluationLen loadPath savePath = do
  vocab <-
    L.fold (L.Fold (OSet.|<>) (OSet.singleton "[PAD]") id)
      <$> traverse buildVocabFromFile [trainingFile, evaluationFile]
  print $ OSet.size vocab

  let vocabSize = OSet.size vocab

  model <- case loadPath of
    Just path -> do
      putStrLn $ "Loading model from: " ++ path
      -- 1. Create the structure (initialized with random weights for now)
      freshModel <- initModel vocabSize

      -- 2. Load the raw tensors from disk
      loadedTensors <- Serialize.load path

      -- 3. Convert raw Tensors into learnable Parameters
      loadedParams <- mapM makeIndependent loadedTensors

      -- 4. Swap the weights into the model
      return $ Th.replaceParameters freshModel loadedParams
    Nothing -> do
      putStrLn "Initializing fresh model..."
      initModel vocabSize

  -- [DEBUG START] Check Init Magnitude
  let initParams = NN.flattenParameters model
  -- Get the first parameter (likely embedding weights)
  let firstP = head initParams
  -- Calculate max value in that tensor
  let initMax = Th.toDouble $ F.max (toDependent firstP)
  putStrLn $ "DEBUG Init Magnitude: " ++ show initMax
  -- [DEBUG END]

  let optimizer = Optim.mkAdam 0 0.9 0.999 (NN.flattenParameters model)

  let trainData = TransformerData trainingLen trainingFile vocab
  let evalData = TransformerData evaluationLen evaluationFile vocab

  (finalModel, _) <-
    if numEpochs > 0
      then runTraining numEpochs model optimizer trainData evalData
      else return (model, optimizer)

  case savePath of
    Just path -> do
      putStrLn $ "Saving model to: " ++ path
      let finalParams = NN.flattenParameters finalModel
          rawTensors = map toDependent finalParams

      -- 3. Save the raw tensors
      Serialize.save rawTensors path
    Nothing -> return ()

  -- generate finalModel vocab "accounted"
  -- generateSequence finalModel vocab 50 "Would you proceed"
  generate finalModel vocab "callp"
  generateSequence finalModel vocab 50 "c eval"

--------------------------------------------------------------------------------
-- 4. Training Loop
--------------------------------------------------------------------------------

runTraining :: Int -> TransformerModel -> Optim.Adam -> TransformerData -> TransformerData -> IO (TransformerModel, Optim.Adam)
runTraining epochs model optim trainData evalData =
  Safe.runSafeT $
    P.foldM step (return (model, optim)) return (each [1 .. epochs])
  where
    step (currModel, currOptim) epoch = do
      liftIO $ putStrLn $ "--- Epoch " ++ show epoch ++ " ---"

      -- 1. Calculate Actual Batches
      -- len / ((seqLen + 1) * batchSize)
      let tokensPerBatch = (seqLen + 1) * batchSize
      let totalBatches = dataLength trainData `div` tokensPerBatch

      -- 2. Create Stream
      let trainingStream =
            readData (filePath trainData) (dataLength trainData) (vocab trainData)
              >-> batcher batchSize seqLen

      -- Add iteration to the stream so that it can be passed to trainStep
      let numberedStream = P.zip (each [1 ..]) trainingStream

      (trainedModel, newOptim) <-
        P.foldM
          (\acc (iter, batch) -> liftIO (trainStep epoch iter totalBatches acc batch)) -- Adapter
          (return (currModel, currOptim))
          return
          numberedStream

      -- 2. EVALUATION LOOP (The Fix)
      liftIO $ putStrLn "Running Evaluation..."
      evalLoss <- liftIO $ runEvaluation trainedModel evalData
      liftIO $ putStrLn $ ">>> Epoch " ++ show epoch ++ " Validation Loss: " ++ show evalLoss

      return (trainedModel, newOptim)

runEvaluation :: TransformerModel -> TransformerData -> IO Float
runEvaluation model evalData = do
  -- Create a stream for evaluation data
  let stream =
        readData (filePath evalData) (dataLength evalData) (vocab evalData)
          >-> batcher batchSize seqLen

  -- Fold over the stream to calculate total loss
  -- We start with (0.0, 0) -> (Total Loss, Batch Count)
  (totalLoss, count) <- Safe.runSafeT $ P.foldM step (return (0.0, 0 :: Int)) return stream

  return (totalLoss / fromIntegral count)
  where
    step (accLoss, accCount) (input, target) = do
      liftIO $ do
        -- 1. Forward (No Gradient Needed)
        -- In PyTorch we'd use 'with torch.no_grad():', here we just don't call 'grad'
        let logits = forward model input

        -- 2. Loss
        let vocabSize = Th.shape logits !! 2
        let flatLogits = Th.reshape [-1, vocabSize] logits
            flatTarget = Th.reshape [-1] target
            weight = ones [vocabSize] defaultOpts
            loss = FI.cross_entropy_loss flatLogits flatTarget weight 1 paddingIdx 0.0

        lossVal <- realToFrac . Th.toDouble <$> F.detach loss
        return (accLoss + lossVal, accCount + 1)

-- We remove the 'Optim.Optimizer o' constraint because we are doing Manual SGD.
-- We keep 'o' as a dummy argument just to avoid breaking your 'runTraining' signature,
-- or you can remove it entirely.
trainStep ::
  Int ->
  Int ->
  Int -> -- Total Batches
  (TransformerModel, Optim.Adam) ->
  (Tensor, Tensor) ->
  IO (TransformerModel, Optim.Adam)
trainStep epoch iter totalBatches (model, optim) (input, target) = do
  -- 1. Forward
  let logits = forward model input

  -- 2. Loss
  let vocabSize = Th.shape logits !! 2
      flatLogits = Th.reshape [-1, vocabSize] logits
      flatTarget = Th.reshape [-1] target
      weight = ones [vocabSize] defaultOpts
      loss = FI.cross_entropy_loss flatLogits flatTarget weight 1 paddingIdx 0.0

  -- 3. Calculate Gradients
  -- We extract the independent tensors to track history
  let params = Th.flattenParameters model
  let gradients = TA.grad loss params

  -- 4. Clip the gradients
  (totalNorm, clippedGradients) <- clipGradNorm 0.5 gradients

  -- 5. OPTIMIZER STEP
  -- Use the passed-in totalBatches instead of hardcoding 100
  let globalStep = (epoch * totalBatches) + iter

  let warmupSteps = 200.0 :: Float
  let currentStep = fromIntegral globalStep :: Float
  let warmupFactor =
        if currentStep < warmupSteps
          then (currentStep + 1) / warmupSteps
          else 1.0

  let baseLR = 0.0003 :: Float
  let learningRateFloat = baseLR * warmupFactor
  let learningRate = asTensor learningRateFloat

  -- Update the weights
  let currentParamTensors = map toDependent params
  let (newParamTensors, newOptimState) =
        Optim.adam learningRate (Optim.Gradients clippedGradients) currentParamTensors optim

  -- 7. Reconstruct Model
  newParams <- mapM makeIndependent newParamTensors
  let newModel = Th.replaceParameters model newParams

  -- 8. Logging
  when (iter `mod` 50 == 0 || iter == 1 || iter == 192) $ do
    lossVal <- Th.toDouble <$> F.detach loss
    putStrLn $ "Ep " ++ show epoch ++ " | It " ++ show iter ++ " | Loss: " ++ show lossVal ++ " | GradNorm: " ++ show totalNorm

  return (newModel, newOptimState)

-- Helper to calculate norm and return SCALED gradients (Purely)
clipGradNorm :: Float -> [Tensor] -> IO (Float, [Tensor])
clipGradNorm maxNorm gradients = do
  -- 1. Calculate global norm (same as before)
  let squaredNorms = map (\g -> Th.toDouble (F.sumAll (g * g))) gradients
  let totalNorm = realToFrac $ sqrt (sum squaredNorms) :: Float

  -- 2. Calculate scaling coefficient
  let clipCoef =
        if totalNorm > maxNorm
          then maxNorm / (totalNorm + 1e-6)
          else 1.0

  -- 3. Return original or scaled gradients (Functional)
  if clipCoef < 1.0
    then do
      let coefTensor = Th.asTensor clipCoef
      -- Create NEW tensors for the gradients
      let scaledGrads = map (* coefTensor) gradients
      return (totalNorm, scaledGrads)
    else return (totalNorm, gradients)

--------------------------------------------------------------------------------
-- 5. Utilities (Unchanged)
--------------------------------------------------------------------------------

readData ::
  (MonadIO m) =>
  FilePath ->
  Int ->
  OSet.OSet T.Text ->
  Producer Int m ()
readData file len vocab = do
  txt <- liftIO $ T.readFile file
  -- Inject <nl> token here too so it matches the vocab and concatenate
  -- multiple newlines into one
  let fileLines = T.lines txt
  let nonEmptyLines = filter (not . T.null . T.strip) fileLines
  let txtWithNewlines = T.intercalate " <nl> " nonEmptyLines

  let cycledWords = cycle (T.words txtWithNewlines)
  each cycledWords
    >-> P.map (\w -> Maybe.fromMaybe 0 (OSet.findIndex w vocab))
    >-> P.take len

batcher :: (MonadIO m) => Int -> Int -> Pipe Int (Tensor, Tensor) m ()
batcher bSize sLen = forever $ do
  let chunkLen = (sLen + 1) * bSize

  -- FIX: Use replicateM + await to consume from upstream
  -- P.toListM (P.take ...) is INVALID inside a Pipe.
  chunks <- replicateM chunkLen await

  -- Create Tensors
  -- (Logic remains the same)
  let tensorData = asTensor chunks
      reshaped = Th.reshape [bSize, sLen + 1] tensorData
      input = Th.sliceDim 1 0 sLen 1 reshaped
      target = Th.sliceDim 1 1 (sLen + 1) 1 reshaped

  yield (input, target)

generate :: TransformerModel -> OSet.OSet T.Text -> String -> IO ()
generate model vocab prompt = do
  let tokens = T.words (T.pack prompt)
      indices = map (\w -> Maybe.fromMaybe 0 (OSet.findIndex w vocab)) tokens
      padded =
        if length indices < seqLen
          then replicate (seqLen - length indices) 0 ++ indices
          else drop (length indices - seqLen) indices
      input = asTensor [padded]
      logits = forward model input
      lastLogit = Th.select 1 (seqLen - 1) logits
      bestIdx = Th.argmax (F.Dim 1) F.RemoveDim lastLogit
      idx = Th.toInt bestIdx
  case OSet.elemAt vocab idx of
    Just w -> putStrLn $ "Prompt: " ++ prompt ++ " -> " ++ T.unpack w
    Nothing -> putStrLn "Unknown"

generateSequence ::
  TransformerModel ->
  -- | vocab
  OSet.OSet T.Text ->
  -- | steps
  Int ->
  -- | prompt
  String ->
  IO ()
generateSequence model vocab steps prompt = do
  putStrLn $ "Prompt: " ++ prompt ++ " -> "
  go steps prompt []
  where
    vocabSize = OSet.size vocab
    nlIdx = Maybe.fromMaybe (-1) (OSet.findIndex "<nl>" vocab)

    -- We pass 'history' (list of recent indices) to checking for loops
    go 0 _ _ = putStrLn "\nDone."
    go n p history = do
      let tokens = T.words (T.pack p)
          indices = map (\w -> Maybe.fromMaybe 0 (OSet.findIndex w vocab)) tokens

          -- Update history with the latest indices (keep last 5)
          newHistory = take 5 (reverse indices ++ history)

          padded =
            if length indices < seqLen
              then replicate (seqLen - length indices) 0 ++ indices
              else drop (length indices - seqLen) indices
          input = asTensor [padded]

          logits = forward model input
          lastLogit = Th.select 1 (seqLen - 1) logits
          flatLogits = F.squeezeAll lastLogit

          -- 1. HARD BAN: Consecutive Newlines (Already implemented)
          lastWord = if null tokens then "" else last tokens
          nlPenalty =
            if lastWord == T.pack "<nl>" && nlIdx >= 0
              then
                let idxTensor = asTensor nlIdx
                    hot = F.oneHot vocabSize idxTensor
                    neg = asTensor (-1e9 :: Float)
                 in hot * neg
              else zeros [vocabSize] defaultOpts

          -- 2. REPEAT PENALTY (The New Fix)
          -- Reduce prob of ANY token seen recently by a small amount
          -- This discourages A -> B -> A -> B loops
          repPenalty =
            if null newHistory
              then zeros [vocabSize] defaultOpts
              else
                let histTensors = map asTensor newHistory
                    -- Create a tensor where indices in history are 1.0, else 0.0
                    -- We sum up one-hot vectors for all history items
                    histMask = sum (map (F.oneHot vocabSize) histTensors)
                    -- Apply a penalty of -2.0 to recently seen tokens
                    penaltyVal = asTensor (-2.0 :: Float)
                 in histMask * penaltyVal

          -- Combine logits + newline ban + repetition penalty
          adjustedLogits = flatLogits + nlPenalty + repPenalty

      idx <- sample adjustedLogits 0.5 -- Slightly higher temp helps break loops too
      case OSet.elemAt vocab idx of
        Just w -> do
          let word = T.unpack w
          if word == "<nl>"
            then putStr "\n"
            else putStr (" " ++ word)
          hFlush stdout
          go (n - 1) (p ++ " " ++ word) newHistory
        Nothing -> return ()

buildVocabFromFile :: FilePath -> IO (OSet.OSet T.Text)
buildVocabFromFile f = do
  content <- T.readFile f
  -- FIX: Collapse multiple newlines into single <nl> tokens
  let fileLines = T.lines content
  let nonEmptyLines = filter (not . T.null . T.strip) fileLines
  let contentWithNewlines = T.intercalate " <nl> " nonEmptyLines

  let ws = T.words contentWithNewlines
  return $ OSet.fromList ws

-- | Sample an index from logits with temperature.
-- T < 1.0 = More confident (approaches argmax)
-- T > 1.0 = More random/creative (flattens distribution)
sample :: Tensor -> Float -> IO Int
sample logits temperature = do
  -- 1. Apply Temperature (Scale Logits)
  -- We divide logits by T.
  -- (logits / 0.01) makes differences huge (Argmax-like)
  -- (logits / 100) makes differences tiny (Uniform-like)
  let scaledLogits = logits / asTensor temperature

  -- 2. Softmax to get Probabilities (Sum to 1.0)
  -- Dim 0 because input is 1D [VocabSize]
  let probs = F.softmax (F.Dim 0) scaledLogits

  -- 3. Multinomial Sampling
  -- args: input, num_samples, replacement
  -- Returns a tensor containing the index
  sampleIdxTensor <- Th.multinomialIO probs 1 True

  -- 4. Extract Int
  return $ Th.toInt sampleIdxTensor
