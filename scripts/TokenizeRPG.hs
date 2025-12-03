-- Tokenize.hs
module Tokenize where

import Data.Char (isAlphaNum, isSpace)
import System.IO (getContents)

-- | 1. Define character categories
data CharClass
  = IdentChar -- a-z, 0-9, _, ' (Things that make up variables/functions)
  | Space -- ' ', \t
  | Newline -- \n
  | Punctuation -- ()[]{},;`" (Always stand alone)
  | Symbol -- +-*&^%$#@!.<> (Group these, e.g., "++", "==")
  deriving (Eq)

-- | 2. Classify every character
classify :: Char -> CharClass
classify c
  | isAlphaNum c || c == '_' || c == '\'' = IdentChar
  | c == '\n' = Newline
  | isSpace c = Space
  | c `elem` "()[]{},;`\"" = Punctuation
  | otherwise = Symbol

main :: IO ()
main = do
  input <- getContents
  -- 1. Split into lines
  -- 2. Filter out lines where isStarComment is True
  -- 3. Join back into a single string
  let cleanInput = unlines $ filter (not . isStarComment) (lines input)

  let tokens = tokenize cleanInput
  putStrLn (unwords tokens)

-- | 3. The Logic Loop
tokenize :: String -> [String]
tokenize [] = []
tokenize ('/' : '*' : cs) = tokenize (skipBlockComment cs)
tokenize (c : cs) =
  let cType = classify c
   in case cType of
        -- Punctuation always becomes a single distinct token
        -- e.g., "[Integer]" -> "[", "Integer", "]"
        Punctuation -> [c] : tokenize cs
        Newline -> [c] : tokenize cs
        -- Spaces are skipped (we add our own single spaces later)
        Space -> tokenize cs
        -- Identifiers and Symbols grab their neighbors
        -- e.g., "fibs" stays "fibs". "++" stays "++".
        _ ->
          let (token, rest) = span (\x -> classify x == cType) (c : cs)
           in token : tokenize rest

-- | Skips characters until it finds the closing "*/"
skipBlockComment :: String -> String
skipBlockComment [] = []
-- If we find "*/", return the rest of the string
skipBlockComment ('*' : '/' : rest) = rest
-- Otherwise, ignore the character and keep looking
skipBlockComment (_ : rest) = skipBlockComment rest

-- | Checks if '*' is the first non-space character on the line
isStarComment :: String -> Bool
isStarComment line =
  case dropWhile isSpace line of
    ('*' : _) -> True -- Found * after skipping spaces
    (_ : '*' : _) -> True
    _ -> False
