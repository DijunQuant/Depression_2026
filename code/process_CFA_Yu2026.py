
from tqdm import tqdm
from transformers import BertTokenizer,BertModel,AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import numpy as np
import torch
import re
import spacy
from spellchecker import SpellChecker

from collections import Counter

##########################
# Set up
###########################

spell = SpellChecker()
nlp = spacy.load("en_core_web_lg")

hf_token='YOUR_HF_TOKEN_HERE'
model_name = "meta-llama/Meta-Llama-3-8B" # Or "mistralai/Mistral-7B-v0.1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer_llama = AutoTokenizer.from_pretrained(model_name,token=hf_token)
model_llama = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    #device_map="auto", # Automatically handles the GPU placement
    token=hf_token
).to(device)
model_llama.eval()

#bert is used for semantic distances
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)


project_to_run=['Study1','Study3','Study2']

rootfolder='ProjectRoot'
outputfolder=rootfolder+'project/thinkaloudLLM/tmp/'

lexicon_loc=rootfolder+'nltk_data/NRC-VAD-Lexicon-v2.1/'
vad_map = pd.read_csv(lexicon_loc+'NRC-VAD-Lexicon-v2.1.txt', sep='\t', index_col='term')


################################
# Functions
################################
def correct_word(word):
    # Returns the most likely correct spelling
    return spell.correction(word)
def get_word_valence(word,verbose=False):
    word = str(word).lower().strip()
    if word in vad_map.index:
        return vad_map.loc[word, 'valence']
    else:
        spellcorrected=spell.correction(word)
        if spellcorrected in vad_map.index:
            if verbose: print('spell corrected: ',word,spellcorrected)
            return vad_map.loc[spellcorrected, 'valence']
        else:
            doc = nlp(word)
            lemma = doc[0].lemma_
            if lemma in vad_map.index:
                if verbose: print('Lemmatized: ',word,lemma)
                return vad_map.loc[lemma, 'valence']
    return np.nan
def get_phrase_valence(phrase,verbose=False):
    words = str(phrase).split(' ')
    valence = [get_word_valence(w,verbose=verbose) for w in words if len(w)>0]
    if verbose:
        print(valence)
    return np.mean(valence)
def collapseRepetitions(wordlist,key=None):
    if key==None:
        #handle a simple list
        words_mod = [wordlist[0]]
        lastword = wordlist[0]
        # collapse repititions
        for i in range(1, len(wordlist)):
            if wordlist[i] != lastword and len(wordlist[i]) > 0:
                lastword = wordlist[i]
                words_mod.append(lastword)

    else:
        words_mod = pd.DataFrame(columns=wordlist.columns)
        words_mod.loc[0] = wordlist.iloc[0]
        lastword=wordlist.iloc[0][key]
        for i in range(1, len(wordlist)):
            currentword=wordlist.iloc[i][key]
            if currentword!= lastword and type(currentword)==str and len(currentword) > 0:
                lastword = currentword
                words_mod.loc[len(words_mod)]=wordlist.iloc[i]
    return words_mod
def phrase_count(phrase_list):
    # Count the occurrences of each phrase
    phrase_counter = Counter(phrase_list)
    return phrase_counter

# Function to get word embeddings
def get_phrase_embeddings(phrases, target_layers=[6, 7]):
    embeddings = {}
    for phrase in phrases:
        try:
            # Tokenize and get the model output for each phrase
            inputs = tokenizer_bert(phrase, return_tensors='pt', truncation=True, padding=True, max_length=128)
            outputs = model_bert(**inputs)
            # Get the embedding of the first token (CLS token)
            # embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
            # embeddings[phrase] = embedding
            # Get the embedding from middle layer
            avg_layers = torch.mean(torch.stack([outputs.hidden_states[i] for i in target_layers]),
                                    dim=0)  # [1, seq_len, 768]
            # Remove special tokens [CLS] and [SEP] (indices 1 to -1)
            subword_embeddings = avg_layers[0, 1:-1, :]
            embeddings[phrase] = torch.mean(subword_embeddings, dim=0).detach().numpy().reshape(1, -1)
        except:
            # Skip unrecognized phrases
            print(f"Warning: '{phrase}' not recognized, skipping.")
    return embeddings

def calculate_loss_in_context(context, phrase, model, tokenizer, device):
    # Tokenize the context and the phrase separately
    context_ids = tokenizer(context, return_tensors='pt')['input_ids'].to(device)
    #phrase_ids = tokenizer(phrase, return_tensors='pt')['input_ids'].to(device)

    phrase_with_space = " " + phrase.strip()
    phrase_ids = tokenizer(phrase_with_space, return_tensors='pt', add_special_tokens=False)['input_ids'].to(
        device)

    # Concatenate the context and the phrase to form the full input
    input_ids = torch.cat([context_ids, phrase_ids], dim=1)

    # Only compute the loss for the tokens corresponding to the last phrase
    labels = input_ids.clone()
    # Set context tokens to -100 so their loss is not computed
    labels[:, :-phrase_ids.size(1)] = -100

    # Get the model outputs (logits)
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
    return loss.item()

def perplexity_with_modcontext(phrasesinput, model, tokenizer, device, verbose=False,ret_loss=False):
    # phrases = filter_unique_phrases(phrasesinput)
    phrases = phrasesinput  # input should already get repititions collapsed
    lossvalues = []
    def chainPreviousText(phrases):
        text="A human subject is performing a chain free association task. They are instructed not to overthink "+\
        "and to voice the very first intuitive association that comes to mind, which may be influenced by the entire sequence of previous words. "+\
            "The entire chain starts from "+phrases[0]+". The following is the response: "
        return text+", ".join(phrases) + ", "

    count = 0
    for i, phrase in enumerate(phrases):
        if i == 0: continue  # the first phrase is cue, skip
        try:
            # Compute perplexity for the last phrase given the previous context
            # Use all previous phrases as context
            previous_text = chainPreviousText(phrases[:i])
            loss = calculate_loss_in_context(previous_text, phrase, model, tokenizer, device)
            lossvalues.append(loss)
            count += 1
            if verbose: print(f"Loss of phrase '{phrase}' given context: {loss}")
        except Exception as e:
            print(f"Error calculating perplexity for '{phrase}': {e}")
    if ret_loss:
        if count == 0:
            return 0, np.nan, np.nan, np.nan, [] # Handle case where no valid phrases were processed
        else: return count, np.average(lossvalues), np.std(lossvalues), pd.Series(lossvalues).autocorr(),lossvalues
    else:
        if count == 0:
            return 0, np.nan, np.nan, np.nan  # Handle case where no valid phrases were processed
        else: return count, np.average(lossvalues), np.std(lossvalues), pd.Series(lossvalues).autocorr()

def compute_semantic_distance_all(embeddings):
    words = list(embeddings.keys())
    alldistances = []
    for i in range(len(words) - 1):
        for j in range(i + 1, len(words)):
            word1 = embeddings[words[i]]
            word2 = embeddings[words[j]]
            # Compute cosine similarity between consecutive words
            cosine_sim = cosine_similarity(word1, word2)[0][0]
            # Convert similarity to distance (1 - similarity)
            distance = 1 - cosine_sim
            alldistances.append(distance)
    return np.mean(alldistances)

#############################
# Script
#############################


if 'Study1' in project_to_run:
    datafolder=rootfolder+'project/thinkaloudLLM/DT_DFA/'
    cueType={}
    cueType['pos']=['complete','finish','improved','memories','imagine','aspire','opportunity','admired']
    cueType['neg']=['failure','shame','mistake','threat','remorse','uncertain','worry','death']

    df=pd.read_csv(datafolder+'DT_DFA_SubjectData.csv')
    cuelist=df.columns.tolist()[2:]
    sublist=df['ID'].unique()
    print(sublist)
    MIN_COUNT_FOR_TRAJECTORY = 3

    cfaResult = pd.DataFrame()
    datacols = ['subid', 'cue', 'ppx_m', 'semdis', 'valence_m','valence_vol', 'fluency']

    # column head is the cue word
    for subid in tqdm(sublist):

        subDF = pd.DataFrame(columns=datacols)
        allembeddings_pos = []
        allembeddings_neg = []
        subdata = df[df['ID'] == subid].set_index('association')[cuelist]
        if subdata.isna().all(axis=None): continue
        for i in cuelist:
            words_list = [i] + list(subdata[i])  # add the first cue word
            words = [word.strip().lower() for word in words_list if type(word) == str]
            words_mod = collapseRepetitions(words)
            fluency=len(words_mod)

            words_mod = [w for w in words_mod if 'xxx' not in w]

            _, ppx_m, ppx_s, ppx_c, ppxlist = perplexity_with_modcontext(words_mod, model_llama, tokenizer_llama, device, ret_loss=True)
            embeddings = get_phrase_embeddings(words_mod)
            brt_mall = compute_semantic_distance_all(embeddings)
            valence = [get_phrase_valence(w) for w in words_mod]
            valence_vol = np.sqrt(np.nanmean(pd.Series(valence).dropna().diff()**2))
            valence_m = np.nanmean(valence[1:])  # not include the seed word
            subDF.loc[len(subDF)] = [subid, i, ppx_m, brt_mall, valence_m,valence_vol,fluency]
            if len(words_mod) < MIN_COUNT_FOR_TRAJECTORY:
                print('skip trial: ', subid, ',cue: ', i)
                continue
            currentEmbedding=[embeddings[x] for x in words_mod]
        cfaResult = pd.concat([cfaResult, subDF])
    cfaResult.to_csv(outputfolder + 'Study1.csv')

if 'Study2' in project_to_run:


    datafolder=rootfolder+'project/thinkaloudLLM/BMRK5/'
    cueType={}
    cueType['pos'] = ['FUTURE', 'TRIP', 'WIN', 'BEAUTIFUL', 'SPOUSE', 'FATHER', 'HEART']
    cueType['neg'] = ['ALONE', 'JEALOUSY', 'FAILURE', 'WORRY', 'SHAME', 'PAIN', 'CRISIS']
    cueType['neu'] = ['NOISE', 'WANT', 'TEACH', 'BREATH', 'ADVICE', 'BEGINNING', 'MEMORY']

    cfaData = pd.read_excel(datafolder + 'FAST_data_Andrews-Hanna.xlsx')
    sublist = cfaData.columns.tolist()[2:]
    cfaData['seed'] = cfaData.apply(lambda r: re.split(r'(\d+)', r['question']), axis=1)
    cfaData['order'] = [int(x[1]) for x in cfaData['seed']]
    cfaData['seed'] = [x[0] for x in cfaData['seed']]
    cfaData.drop(columns='question', inplace=True)
    print(sublist)
    valenceData = pd.read_csv(datafolder + 'total_summary_ordered.csv', index_col=None)
    valenceData['response'] = valenceData['response'].str.lower()

    MIN_COUNT_FOR_TRAJECTORY = 3

    use_rating=True
    cfaResult = pd.DataFrame()

    datacols = ['subid', 'cue', 'ppx_m', 'semdis', 'valence_m','valence_vol', 'fluency']
    # column head is the cue word
    for subid in tqdm(sublist):
        subDF = pd.DataFrame(columns=datacols)
        valence_sub = valenceData[valenceData['subject'] == subid]

        for i,g in cfaData.groupby('seed'):
            words_list = g[subid].values  # add the first cue word
            words = [word.strip().lower() for word in words_list if type(word) == str]
            words_mod = collapseRepetitions(words)
            count = len(phrase_count(words_mod).keys())  # number of unique words
            fluency = len(words_mod)
            iwords_mod = [w for w in words_mod if 'xxx' not in w]
            _, ppx_m, ppx_s, ppx_c, ppxlist = perplexity_with_modcontext(words_mod, model_llama, tokenizer_llama, device, ret_loss=True)
            embeddings = get_phrase_embeddings(words_mod)
            brt_mall = compute_semantic_distance_all(embeddings)
            if use_rating:
                rated_valence = valence_sub.loc[valence_sub['seed'] == i, ['response', 'valence_mean']]
                collapseed_rated_valence = collapseRepetitions(rated_valence, key='response')
                collapseed_rated_valence = collapseed_rated_valence[~collapseed_rated_valence['response'].str.contains('xxx')]
                valence = collapseed_rated_valence['valence_mean'].values
                if len(valence) != len(words_mod):
                    print('Error: ', subid, i)
                    print(words, words_mod)
                    print(collapseed_rated_valence)
            else:
                valence = [get_phrase_valence(w) for w in words_mod]
            valence_vol = np.sqrt(np.nanmean(pd.Series(valence).dropna().diff() ** 2))
            valence_m = np.nanmean(valence[1:])  # not include the seed word

            if len(words_mod) < MIN_COUNT_FOR_TRAJECTORY:
                print('skip trial: ', subid, ',cue: ', i)
                continue
            subDF.loc[len(subDF)] = [subid, i, ppx_m, brt_mall, valence_m, valence_vol, fluency]
        cfaResult = pd.concat([cfaResult, subDF])
    cfaResult.to_csv(outputfolder + 'Study2.csv')

if 'Study3' in project_to_run:
    datafolder=rootfolder+'project/thinkaloudLLM/WWS/'
    cueType={}
    cueType['pos'] = ['future', 'win', 'beautiful', 'kindness', 'mother', 'strength', 'joy']
    cueType['neg'] = ['alone', 'failure', 'rejection', 'worry', 'guilt', 'pain', 'anger']
    cfaData=pd.read_csv(datafolder+'/WWS_FATask_Master.csv')
    TRIAL_LENGTH = 11  # include seed word
    MIN_COUNT_FOR_TRAJECTORY = 3
    cfaResult = pd.DataFrame()
    datacols = ['subid','group', 'cue', 'ppx_m', 'semdis', 'valence_m','valence_vol', 'fluency']
    # column head is the cue word
    for j, g in cfaData.groupby(['StudyID', 'Session', 'Group']):
        subid = j[0]
        session = j[1]
        if session not in ['session 1']: continue #only take baseline
        group = j[2]
        subDF = pd.DataFrame(columns=datacols)
        print(subid,session,group)

        for i in range(14):  # 14 trials
            words_list = g.set_index('Word').loc[(11 * i + 1):(11 * i + 11), 'Word_Actual']
            words = [word.strip().lower() for word in words_list if type(word) == str]
            cue = words[0]
            words_mod = collapseRepetitions(words)
            words_mod = [w for w in words_mod if 'xxx' not in w]

            fluency=len(words_mod)
            count = len(phrase_count(words_mod).keys())  # number of unique words

            _, ppx_m, ppx_s, ppx_c, ppxlist = perplexity_with_modcontext(words_mod, model_llama, tokenizer_llama, device, ret_loss=True)
            embeddings = get_phrase_embeddings(words_mod)
            brt_mall = compute_semantic_distance_all(embeddings)

            valence = [get_phrase_valence(w) for w in words_mod]
            valence_m = np.nanmean(valence[1:])  # not include the seed word
            valence_vol = np.sqrt(np.nanmean(pd.Series(valence).dropna().diff() ** 2))

            if len(words_mod) < MIN_COUNT_FOR_TRAJECTORY:
                print('skip trial: ', subid, ',cue: ', i)
                continue
            subDF.loc[len(subDF)] = [subid, group,'cue', ppx_m, brt_mall, valence_m, valence_vol, fluency]

        cfaResult = pd.concat([cfaResult, subDF])

    cfaResult.to_csv(outputfolder + 'Study3.csv')
