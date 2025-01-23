ARG_COMP_TAGS = ['NA', 'B-claim', 'I-claim', 'B-premise', 'I-premise']

SEMANTIC_TYPE_TAGS = [
    'NA', 
    'ethos', 
    'logos', 
    'pathos', 
    'ethos-logos',
    'logos-pathos', 
    'ethos-pathos',
    'ethos-logos-pathos',
    'interpretation', 
    'evaluation-emotional', 
    'evaluation-rational', 
    'disagreement', 
    'agreement']

COMBINED_TAGS = [
    'NA',
    'B-premise-ethos', 
    'I-premise-ethos', 
    'B-premise-logos', 
    'I-premise-logos', 
    'B-premise-pathos',
    'I-premise-pathos',  
    'B-premise-ethos-logos',
    'I-premise-ethos-logos',
    'B-premise-logos-pathos', 
    'I-premise-logos-pathos', 
    'B-premise-ethos-pathos',
    'I-premise-ethos-pathos',
    'B-premise-ethos-logos-pathos',
    'I-premise-ethos-logos-pathos',
    'B-claim-interpretation', 
    'I-claim-interpretation', 
    'B-claim-evaluation-emotional', 
    'I-claim-evaluation-emotional', 
    'B-claim-evaluation-rational',
    'I-claim-evaluation-rational',  
    'B-claim-disagreement', 
    'I-claim-disagreement', 
    'B-claim-agreement',
    'I-claim-agreement'
]

SEMANTIC_TYPE_DICT = {
    "ethos": SEMANTIC_TYPE_TAGS[1],
    "logos": SEMANTIC_TYPE_TAGS[2],
    "pathos": SEMANTIC_TYPE_TAGS[3],
    "ethos_logos": SEMANTIC_TYPE_TAGS[4],
    "logos_pathos": SEMANTIC_TYPE_TAGS[5],
    "ethos_pathos": SEMANTIC_TYPE_TAGS[6],
    "ethos_logos_pathos": SEMANTIC_TYPE_TAGS[7],
    "interpretation": SEMANTIC_TYPE_TAGS[8],
    "evaluation_emotional": SEMANTIC_TYPE_TAGS[9],
    "evaluation_rational": SEMANTIC_TYPE_TAGS[10],
    "disagreement": SEMANTIC_TYPE_TAGS[11],
    "agreement": SEMANTIC_TYPE_TAGS[12],
}

COMBINED_DICT = {
    "ethos": [COMBINED_TAGS[1], COMBINED_TAGS[2]],
    "logos": [COMBINED_TAGS[3], COMBINED_TAGS[4]],
    "pathos": [COMBINED_TAGS[5], COMBINED_TAGS[6]],
    "ethos_logos": [COMBINED_TAGS[7], COMBINED_TAGS[8]],
    "logos_pathos": [COMBINED_TAGS[9], COMBINED_TAGS[10]],
    "ethos_pathos": [COMBINED_TAGS[11], COMBINED_TAGS[12]],
    "ethos_logos_pathos": [COMBINED_TAGS[13], COMBINED_TAGS[14]],
    "interpretation": [COMBINED_TAGS[15], COMBINED_TAGS[16]],
    "evaluation_emotional": [COMBINED_TAGS[17], COMBINED_TAGS[18]],
    "evaluation_rational": [COMBINED_TAGS[19], COMBINED_TAGS[20]],
    "disagreement": [COMBINED_TAGS[21], COMBINED_TAGS[22]],
    "agreement": [COMBINED_TAGS[23], COMBINED_TAGS[24]],
}

