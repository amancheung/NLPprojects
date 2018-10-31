#Lap Yan Cheung (lyc286)

import nltk
from nltk.stem import PorterStemmer
import sys

#nltk module that parses word stem
ps = PorterStemmer();

f = open(sys.argv[1]);
input = f.read();

f.close();

eachLine = input.split('\n');

allWords = [];
wordsCount = 0;

#Create class that stores features for each word
class Word:
    def __init__(self, word, pos):
        self.word = word;
        self.capitalized = False;
        self.pos = pos;
        self.bio = '';
        self.word_stem = ps.stem(word);
        self.prev_bio = '';
        self.prev_word = '';
        self.prev_pos = '';
        self.prev_twoWord = '';
        self.prev_twoWordPos = '';
        self.next_word = '';
        self.next_pos = '';
        #Retrieve word stem by nltk stem
        self.sentence_begin = False;
        self.sentence_end = False;


'''
TO-DO:
1) Add more features

Current features:
1) Curr word
2) Curr POS
3) Prev word
4) Prev POS
5) Stem
6) word capitalized
7) If is sentence beginning
8) If is sentence end
9) Prev BIO
10) BIO - training

'''
for l in eachLine:
    if (l != ''):
        lineTokens = l.split('\t');
        currWord = Word(lineTokens[0],lineTokens[1]);
        if (len(lineTokens)>2):
            currWord.bio = lineTokens[2];
        #Tag word as beginning of sentence of last word is end of sentence
        if (lineTokens[0][0].isupper()):
            currWord.capitalized = True;
        if (wordsCount==0):
            currWord.sentence_begin = True;
        if (wordsCount>0):
            #If last word is end of sentence - tag curr word as beginning
            if (allWords[wordsCount-1].word == "line-BREAK"):
                currWord.sentence_begin = True;
            #If word is middle of sentence - add tags of prev word&bio, prev bio, next word&pos
            else:
                currWord.prev_word = allWords[wordsCount-1].word;
                currWord.prev_pos = allWords[wordsCount-1].pos;
                currWord.prev_bio = allWords[wordsCount-1].bio;
                allWords[wordsCount-1].next_word = currWord.word;
                allWords[wordsCount-1].next_pos = currWord.pos;
                #Tag info of 2 words before
                if (wordsCount>1 and allWords[wordsCount-2].word!="line-BREAK"):
                    currWord.prev_twoWord = allWords[wordsCount-2].word;
                    currWord.prev_twoWordPos = allWords[wordsCount-2].pos;
        allWords.append(currWord);
        wordsCount+=1;
    else:
        #if line break is reached - tag previous word as End of Sentence
        if (wordsCount>0):
            allWords[wordsCount-1].sentence_end=True;
        #Mark word entry as line-break
        lineBreak = Word("line-BREAK", "");
        allWords.append(lineBreak);
        wordsCount+=1;

nf = open(sys.argv[2], 'w');

for w in allWords:
    if (w.word!="line-BREAK"):
        #Line for writing to training.features
        if (sys.argv[3]=="training"):
            #Code to write output to comply with gradescope format
            nf.write('{} \t POS={} \t STEM={} \t prev_WORD={} \t prev_POS={} \t prev_TwoWORD={} \t prev_TwoWordPOS={} \t next_WORD={} \t next_POS={} \t word_CAPITALIZED={} \t sentence_BEGIN={} \t sentence_END={} \t prev_BIO={} \t {}'.format(w.word, w.pos, w.word_stem, w.prev_word, w.prev_pos, w.prev_twoWord, w.prev_twoWordPos, w.next_word, w.next_pos, w.capitalized, w.sentence_begin, w.sentence_end, w.prev_bio, w.bio));
            #Code to write output with tabs between features
            #nf.write('{}\tPOS={}\tSTEM={}\tprev_WORD={}\tprev_POS={}\tprev_TwoWORD={}\tprev_TwoWordPOS={}\tnext_WORD={}\tnext_POS={}\tword_CAPITALIZED={}\tsentence_BEGIN={}\tsentence_END={}\tprev_BIO={}\t{}'.format(w.word, w.pos, w.word_stem, w.prev_word, w.prev_pos, w.prev_twoWord, w.prev_twoWordPos, w.next_word, w.next_pos, w.capitalized, w.sentence_begin, w.sentence_end, w.prev_bio, w.bio));

        #Line for writing to test.features/WSJ_23.features
        elif (sys.argv[3]=="test"):
            #Code to write output to comply with gradescope format
            nf.write('{} \t POS={} \t STEM={} \t prev_WORD={} \t prev_POS={} \t prev_TwoWORD={} \t prev_TwoWordPOS={} \t next_WORD={} \t next_POS={} \t word_CAPITALIZED={} \t sentence_BEGIN={} \t sentence_END={} \t prev_BIO=@@'.format(w.word, w.pos, w.word_stem, w.prev_word, w.prev_pos, w.prev_twoWord, w.prev_twoWordPos, w.next_word, w.next_pos, w.capitalized, w.sentence_begin, w.sentence_end));
            #Code to write output with tabs between features
            #nf.write('{}\tPOS={}\tSTEM={}\tprev_WORD={}\tprev_POS={}\tprev_TwoWORD={}\tprev_TwoWordPOS={}\tnext_WORD={}\tnext_POS={}\tword_CAPITALIZED={}\tsentence_BEGIN={}\t sentence_END={}\tprev_BIO=@@'.format(w.word, w.pos, w.word_stem, w.prev_word, w.prev_pos, w.prev_twoWord, w.prev_twoWordPos, w.next_word, w.next_pos, w.capitalized, w.sentence_begin, w.sentence_end));

#Code to write output with tabs between features
    else:
        nf.write(' \t ');
    nf.write('\n');
nf.close();
