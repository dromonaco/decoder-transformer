-- TokenizeHS.hs
module TokenizeHS where

import Data.Char (isAlphaNum, isSpace)
import System.IO (getContents)

-- | 1. Define character categories
data CharClass
  = IdentChar -- a-z, 0-9, _, ' (Things that make up variables/functions)
  | Space -- ' ', \t, \n
  | Punctuation -- ()[]{},;`" (Always stand alone)
  | Symbol -- +-*&^%$#@!.<> (Group these, e.g., "++", "==")
  deriving (Eq)

-- | 2. Classify every character
classify :: Char -> CharClass
classify c
  | isAlphaNum c || c == '_' || c == '\'' = IdentChar
  | isSpace c = Space
  | c `elem` "()[]{},;`\"" = Punctuation
  | otherwise = Symbol

main :: IO ()
main = do
  input <- getContents
  let tokens = tokenize input
  -- Join with spaces so "words" can read it later
  putStrLn (unwords tokens)

-- | 3. The Logic Loop
tokenize :: String -> [String]
tokenize [] = []
tokenize ('-':'-':cs) = tokenize (dropWhile (/= '\n') cs)
tokenize ('{':'-':cs) = tokenize (skipBlock cs)
tokenize (c : cs) =
  let cType = classify c
   in case cType of
        -- Punctuation always becomes a single distinct token
        -- e.g., "[Integer]" -> "[", "Integer", "]"
        Punctuation -> [c] : tokenize cs
        -- Spaces are skipped (we add our own single spaces later)
        Space -> tokenize cs
        -- Identifiers and Symbols grab their neighbors
        -- e.g., "fibs" stays "fibs". "++" stays "++".
        _ ->
          let (token, rest) = span (\x -> classify x == cType) (c : cs)
           in token : tokenize rest

-- | Skips characters until it finds the closing "-}"
skipBlock :: String -> String
skipBlock [] = []
-- If we find "-}", return the rest of the string
skipBlock ('-' : '}' : rest) = rest
-- Otherwise, ignore the character and keep looking
skipBlock (_ : rest) = skipBlock rest
