# %% [markdown]
# # Lemmatize PTA data with CLTK (+ POS, + morphology)
# 
# for Greek and Latin texts

# %% [markdown]
# ## Functions

# %%
import os,glob,re
import os.path
import json
import collections
from copy import deepcopy
from dataclasses import dataclass
from boltons.cacheutils import cachedproperty
from cltk import NLP
from cltk.alphabet import lat
from cltk.core.data_types import Doc, Pipeline, Process
from cltk.core.exceptions import CLTKException
from cltk.stops.processes import StopsProcess
#from cltk.dependency import GreekStanzaProcess
from cltk.dependency import GreekSpacyProcess
from cltk.tokenizers import GreekTokenizationProcess

# %%
@dataclass
class NormalisationProcess(Process):
    """
    
    """

    language: str = None

    @cachedproperty
    def algorithm(self):
        if self.language == "grc":
            nor_grc_class = GRCNormalisationProcess()
        else:
            raise CLTKException(f"No normalisation algorithm for language '{self.language}'.")
        return nor_grc_class

    def run(self, input_doc: Doc) -> Doc:
        normalisation_algo = self.algorithm
        output_doc = deepcopy(input_doc)
        for index, word_obj in enumerate(output_doc.words):
            if self.language == "grc":
                word_obj.raw_string = word_obj.string
                word_obj.string = normalisation_algo.normalise(word_obj.string)
                output_doc.words[index] = word_obj
                
            else:
                raise CLTKException(
                    f"``NormalisationProcess()`` not available for language '{self.language}' This should never happen."
                )
        return output_doc


class GRCNormalisationProcess(NormalisationProcess):

    hard_written_dictionary = {
            "ἀλλ’": "ἀλλά",
            "ἀνθ’": "ἀντί",
            "ἀπ’": "ἀπό",
            "ἀφ’": "ἀπό",
            "γ’": "γε",
            "δ’": "δέ",
            "δεῦρ’": "δεῦρο",
            "δι’": "διά",
            "εἶτ’": "εἶτα",
            "ἐπ’": "ἐπί",
            "ἔτ’": "ἔτι",
            "ἐφ’": "ἐπί",
            "ἵν’": "ἵνα",
            "καθ’": "κατά",
            "κατ’": "κατά",
            "μ’": "με",
            "μεθ’": "μετά",
            "μετ’": "μετά",
            "μηδ’": "μηδέ",
            "μήδ’": "μηδέ",  # @@@
            "ὅτ’": "ὅτε",
            "οὐδ’": "οὐδέ",
            "πάνθ’": "πάντα",
            "πάντ’": "πάντα",
            "παρ’": "παρά",
            "ποτ’": "ποτε",
            "σ’": "σε",
            "τ’": "τε",
            "ταῦθ’": "ταῦτα",
            "ταῦτ’": "ταῦτα",
            "τοῦτ’": "τοῦτο",
            "ὑπ’": "ὑπό",
            "ὑφ’": "ὑπό",
        }

    def normalise(self, token: str):
        if token in self.hard_written_dictionary:
            return self.hard_written_dictionary[token]
        return token

# %%
#deelision = NormalisationProcess
#text = Doc(raw="ἠγαπημένῳ ὑπ’ αὐτοῦ· μετὰ ταῦτα ἐπὶ τῆς γῆς ὤφθη, καὶ τοῖς ἀνθρώποις συνανεστράφη. Ἐὰν ")
#tokenizes = GreekTokenizationProcess().run(input_doc=text)
#print(tokenizes.words[1])
#example = deelision(language="grc").run(input_doc=tokenizes)
#print(example[1])

# %%
def files_to_dict(files_path):
    """converts all files in files_path to a list of tuples
    for further processing by spacy.
    The tuples contain the contents of the file and the id
    of the fle (as dict with key id). Contents of files 
    are cleaned from numbers, double spaces and spaces
    on line beginnings"""
    files_dir = os.path.expanduser(files_path)
    all_paths = glob.glob(files_dir)
    texts = []
    for file_path in all_paths:
        data = {}
        with open(file_path, "r", encoding="utf-8") as source:
            contents = source.read()
            # clean unwanted interpunction and spaces
            contents = re.sub(r'[\{\}\n\d]+', r'', contents) 
            contents = re.sub(r'\s{2,}',r' ', contents)
            contents = re.sub(r'^\s+',r'', contents)
            workid = "".join(file_path.split("/")[-1:]).split(".txt")[0] # Windows: "\\"
            data["id"] = workid
            data["contents"] = contents
        texts.append(data)
    return texts
# %%
def tokenize(inputText):
    return [token for token in re.findall(r'\w+', inputText)]

# %%
def remove_interpunction(inputText):
    return re.sub(r'[\?()›»«‹⁘—><\[\]\+\-\n]+', r'', inputText) # .,:··;


# %%
def remove_latin(inputText):
    return re.sub(r'[a-zA-Z]+',r'', inputText)

# %%
def clean(text):
    """Remove superfluous spaces and linebreaks from extracted text"""
    cleaned = re.sub(r"\n",r"",text)
    cleaned = re.sub(r"\s{2,}",r" ",cleaned)
    cleaned = re.sub(r"\s([.,·:;?]+)",r"\1",cleaned)
    return cleaned

# %%
def analyze_grc_files(files_path):
    '''
    Load all files from files_path and analyze with CLTK,
    finally write out 
    - pta_dict: per id info on words (counted, stopwords removed) and lemmata (counted, with stopwords)
    - plaintext lemmatized text is written to pta_data repo. 
    '''
    grc_paths = files_to_dict(files_path)
    pta_grc_dict = []
    wordlemma_grc = []
    grc_pipeline_custom_1 = Pipeline(language="grc", description="", processes=[GreekTokenizationProcess, NormalisationProcess, GreekSpacyProcess, StopsProcess])
    cltk_nlp_grc = NLP(language="grc", custom_pipeline=grc_pipeline_custom_1, suppress_banner=True)
    print("Analysing grc...")
    for path in grc_paths:
        file_dict = {}
        tlgid = path["id"]
        print(tlgid)
        text = path["contents"]
        text_lowered = text.lower() # Remove capitals
        #text_ana = grc.filter_non_greek(text_lowered) # leave only Greek letters, removes also Apostrophe -> not good
        text_ana = remove_latin(remove_interpunction(text_lowered))
        file_dict["urn"] = path["id"]
        nlp = cltk_nlp_grc.analyze(text=text_ana)
        words = [x.string for x in nlp.words if x.upos != "PUNCT"]
        lemmata = [x.lemma for x in nlp.words if x.upos != "PUNCT"]
        pos = [x.upos for x in nlp.words if x.upos != "PUNCT"]
        features = [', '.join(f'{k}: {v}' for k, v in x.features.items()) for x in nlp.words if x.features and x.upos != "PUNCT"]
        dependency = [x.dependency_relation for x in nlp.words if x.upos != "PUNCT"]
        wordlemma_file = list(zip(words,lemmata,pos,features))
        wordlemma_grc.extend(wordlemma_file)
        tokens_filtered = [x.string for x in nlp.words if x.upos != "PUNCT"]
        tokens_counted = collections.Counter(tokens_filtered).most_common() #without stopwords
        lemmata_filtered = [x.lemma for x in nlp.words if x.upos != "PUNCT"] # without stopwords
        lemmata_counted = collections.Counter(lemmata_filtered).most_common()
        pos_counted = collections.Counter(pos).most_common()
        features_counted = collections.Counter(features).most_common()
        dependency_counted = collections.Counter(dependency).most_common()
        #with open("/home/stockhausen/Dokumente/projekte/pta_data/plaintext/"+ptaid+".txt", "w") as text_file:
        #    text_file.write(remove_interpunction(text))
        with open(os.path.expanduser("~/Workshop-PAG/data/lemmatized/")+tlgid+".txt", "w", encoding="utf-8") as text_file:
            text_file.write(" ".join(nlp.lemmata))
        file_dict["tokens"] = tokens_counted
        file_dict["lemmata"] = lemmata_counted
        file_dict["pos"] = pos_counted
        file_dict["morphology"] = features_counted
        file_dict["syntax"] = dependency_counted
        pta_grc_dict.append(file_dict)
    return pta_grc_dict, wordlemma_grc 


# %%
pta_grc_dict, wordlemma_grc = analyze_grc_files(os.path.expanduser('~/cltk_data/grc/works-clean/*'))

# %%
# Write analytical data to file
print("Saving temp Statistics for Greek")
with open(os.path.expanduser('~/Workshop-PAG/data/analyses/letters_grc_statistics.json'), 'w', encoding="utf-8") as fout:
# Ergebnisse werden in eine json-Datei geschrieben
    json.dump(pta_grc_dict, fout, indent=4, ensure_ascii=False)
