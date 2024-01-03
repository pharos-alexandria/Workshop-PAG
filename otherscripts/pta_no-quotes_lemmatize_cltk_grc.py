# %% [markdown]
# # Lemmatize PTA data with CLTK (+ POS, + morphology)
# 
# for Greek and Latin texts

# %% [markdown]
# ## Functions

# %%
import os,glob,re,gc
import os.path
from MyCapytain.resources.texts.local.capitains.cts import CapitainsCtsText
from MyCapytain.common.constants import Mimetypes, XPATH_NAMESPACES
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
from cltk.dependency import GreekSpacyProcess
from cltk.tokenizers import GreekTokenizationProcess
import xml.dom.minidom as minidom

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
def tokenize(inputText):
    return [token for token in re.findall(r'\w+', inputText)]

# %%
def remove_interpunction(inputText):
    return re.sub(r'[\?()›»«‹⁘—><\[\]\+\-\n]+', r'', inputText) # .,:··;

# %%
def remove_numbering(inputText):
    return re.sub(r'[0-9]+', r' ', inputText) 

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
def check_if_quote(file):
    """check whether quote element is extant in file
    if returned value > 0, quotes are extant"""
    doc = minidom.parse(file)
    check = len(doc.getElementsByTagName('quote'))
    return check

# %%
def tei_xml_to_plaintext(file_path):
    converted = []
    with open(file_path, encoding="utf-8") as file_open:
        text = CapitainsCtsText(resource=file_open)
        for ref in text.getReffs(level=len(text.citation)):
            psg = text.getTextualNode(subreference=ref, simple=True)
            textpart = psg.export(Mimetypes.PLAINTEXT, exclude=["tei:quote","tei:note","tei:rdg","tei:witDetail","tei:pb"])
            converted.append(clean(textpart))
    plaintext = " ".join(converted)
    return plaintext

# %%
def analyze_grc_files(files_path):
    '''
    Load all files from files_path and analyze with CLTK,
    finally write out pta_dict: per URN info on words (counted, stopwords removed) and lemmata (counted, stopwords removed)
    - also plaintext files of lemmatized text are written to pta_data repo. 
    '''
    xml_dir = os.path.expanduser(files_path)
    xml_paths = glob.glob(xml_dir)
    grc_paths = [path for path in sorted(xml_paths) if 'grc' in path]
    pta_grc_dict = []
    grc_pipeline_custom_1 = Pipeline(language="grc", description="", processes=[GreekTokenizationProcess, NormalisationProcess, GreekSpacyProcess, StopsProcess])
    cltk_nlp_grc = NLP(language="grc", custom_pipeline=grc_pipeline_custom_1, suppress_banner=True)
    print("Analysing grc...")
    for xml_path in grc_paths:
        if check_if_quote(xml_path) > 0:
            file_dict = {}
            short_path = "/".join(xml_path.split("\\")[5:]) # adjusted for OS Windows (only works with */*/*.xml)
            print(short_path)
            ptaid = "".join(short_path).split(".xml")[0]
            text = tei_xml_to_plaintext(xml_path)
            text_lowered = text.lower() # Remove capitals
            #text_ana = grc.filter_non_greek(text_lowered) # leave only Greek letters, removes also Apostrophe -> not good
            text_ana = remove_latin(remove_interpunction(remove_numbering(text_lowered)))
            file_dict["urn"] = "urn:cts:pta:"+ptaid
            nlp = cltk_nlp_grc.analyze(text=text_ana)
            pos = [x.upos for x in nlp.words if x.upos != "PUNCT"]
            features = [', '.join(f'{k}: {v}' for k, v in x.features.items()) for x in nlp.words if x.features and x.upos != "PUNCT"]
            dependency = [x.dependency_relation for x in nlp.words if x.upos != "PUNCT"]
            tokens_filtered = [x.string for x in nlp.words if x.upos != "PUNCT"]
            tokens_counted = collections.Counter(tokens_filtered).most_common() #without stopwords
            lemmata_filtered = [x.lemma for x in nlp.words if x.upos != "PUNCT"] # without stopwords
            lemmata_counted = collections.Counter(lemmata_filtered).most_common()
            pos_counted = collections.Counter(pos).most_common()
            features_counted = collections.Counter(features).most_common()
            dependency_counted = collections.Counter(dependency).most_common()
            #with open("/home/stockhausen/Dokumente/projekte/pta_data/plaintext/"+ptaid+".txt", "w") as text_file:
            #    text_file.write(remove_interpunction(text))
            with open(os.path.expanduser("~/Documents/projekte/pta_data_plaintext/lemmatized/")+ptaid+"_noquotes.txt", "w", encoding="utf-8") as text_file:
                text_file.write(clean(" ".join(nlp.lemmata)))
            file_dict["tokens"] = tokens_counted
            file_dict["lemmata"] = lemmata_counted
            file_dict["pos"] = pos_counted
            file_dict["morphology"] = features_counted
            file_dict["syntax"] = dependency_counted
            pta_grc_dict.append(file_dict)
        else:
            pass
    return pta_grc_dict


# %%
pta_grc_dict = analyze_grc_files(os.path.expanduser("~/Downloads/pta_data/data/*/*/*.xml"))

# %%
# Write analytical data to file
print("Saving temp Statistics for Greek")
with open(os.path.expanduser('~/Documents/projekte/pta_metadata/pta_noquotes_grc_statistics.json'), 'w', encoding="utf-8") as fout:
# Ergebnisse werden in eine json-Datei geschrieben
    json.dump(pta_grc_dict, fout, indent=4, ensure_ascii=False)

# %%
