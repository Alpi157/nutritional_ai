import os
import re
import math
import json
import pickle
import logging
import warnings
from fractions import Fraction
from difflib import SequenceMatcher
from logging.handlers import RotatingFileHandler
from ast import literal_eval

import torch
import pandas as pd
from PIL import Image
from flask import Flask, render_template, request
from torchvision import transforms, models

T5_MODEL_DIR = "t5-ingredient-parser"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", message="Failed to load image Python extension", category=UserWarning)

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, 'app.log')
logger = logging.getLogger("nutrition_logger")
logger.setLevel(logging.DEBUG)
fh = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
if not logger.handlers:
    logger.addHandler(fh)
    logger.addHandler(ch)

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, 101)
model.load_state_dict(torch.load('resnet50_food101.pth', map_location='cpu'))
model.eval()
logger.info("ResNet50 model loaded.")

with open('food-101/meta/classes.txt', 'r') as f:
    class_names = [line.strip().replace('_', ' ') for line in f]
logger.info("Food-101 class names loaded.")

recipe_df = pd.read_csv("recipe_matches_df.csv")
recipe_df['ner'] = recipe_df['ner'].apply(lambda x: literal_eval(x) if isinstance(x, str) else [])
recipe_df['instructions'] = recipe_df['instructions'].apply(lambda x: literal_eval(x) if isinstance(x, str) else ["No instructions found."])
recipe_df['ingredients_list'] = recipe_df['ingredients'].apply(lambda x: literal_eval(x) if isinstance(x, str) else [])
logger.info("Recipe matches DataFrame loaded.")

with open("ingredient_to_nutrition_map.pkl", "rb") as f:
    nutrition_map = pickle.load(f)
logger.info("Nutrition map loaded.")

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

KNOWN_UNITS = {"g","gram","grams","kg","mg","oz","ounce","lb","pound","tbsp","tablespoon","tbs","tblsp","tsp","teaspoon","cup","ml","l","clove","slice","packet","pinch","dash","can","box","pkg","package","stick","handful","handfuls","drop","drops","segment","segments","ear","ears","smidgeon","fluid","fluid ounces","litre","liter","liters","litres"}
unit_to_grams = {"g":1,"gram":1,"grams":1,"kg":1000,"mg":0.001,"oz":28.35,"ounce":28.35,"lb":453.6,"pound":453.6,"tbsp":15,"tablespoon":15,"tbs":15,"tblsp":15,"tsp":5,"teaspoon":5,"ml":1,"l":1000,"can":400,"box":432,"pkg":200,"package":200,"stick":113,"packet":100,"pinch":0.36,"dash":0.36,"handful":30,"handfuls":30,"drop":0.05,"drops":0.05,"segment":5,"segments":5,"ear":90,"ears":90,"smidgeon":0.2,"fluid ounces":29.57,"fluid":1,"litre":1000,"liter":1000,"liters":1000,"litres":1000}
cup_specific = {"butter":227,"sugar":200,"flour":120,"all-purpose flour":120,"milk":240,"water":240,"oil":218,"ketchup":240,"peas":160,"green peas":160,"frozen green peas":160,"rice":195,"carrots":128,"default":240}
piece_weights = {"onion":110,"onions":110,"garlic":3,"egg":50,"eggs":50,"potato":213,"potatoes":213,"tomato":120,"tomatoes":120,"carrot":61,"carrots":61,"lemon":58,"lime":67,"mushroom":18,"mushrooms":18,"celery":40,"pepper":120,"chili":45}
OPTIONAL_PHRASES = ["if desired","to taste","optional"]
FRYING_PHRASE = "for deep frying"
STRIP_WORDS = ["finely","chopped","cut","pieces","peeled","grated","warm","frozen","fresh","naturally","even","size","piece","inch","diced","sliced","minced","cooked","level","heaped"]
STOPWORDS_FALLBACK = {"and","or","in","of","without","with","mix","jiffy","pudding"}
ALIAS = {"flour sifted":"flour","potatoes in":"potatoes","ginger root and":"ginger","mild sausage cooked":"sausage","sweet and sour sauce":"sweet and sour sauce","cake mix":"cake mix","white cake mix":"cake mix","chocolate fudge cake mix":"cake mix","wesson oil":"oil"}
VEG_FRUIT_SPICE_WORDS = ["onion","garlic","pea","carrot","celery","lemon","lime","ginger","turmeric","masala","cumin","spice","herb","pepper","chili","tomato"]
MAX_KCAL_VEG = 150
SIMILARITY_MAIN = 0.92
SIMILARITY_FALLBACK = 0.88

SERVING_DEFAULTS = {
    "apple pie":8,"baby back ribs":4,"baklava":16,"beef carpaccio":2,"beef tartare":2,"beet salad":3,"beignets":6,
    "bibimbap":2,"bread pudding":6,"breakfast burrito":1,"bruschetta":4,"caesar salad":2,"cannoli":6,"caprese salad":2,
    "carrot cake":12,"ceviche":4,"cheese plate":4,"cheesecake":12,"chicken curry":3,"chicken quesadilla":1,"chicken wings":4,
    "chocolate cake":12,"chocolate mousse":4,"churros":6,"clam chowder":4,"club sandwich":1,"crab cakes":2,"creme brulee":4,
    "croque madame":1,"cup cakes":6,"cupcakes":6,"deviled eggs":6,"donuts":6,"dumplings":4,"edamame":2,"eggs benedict":1,
    "escargots":2,"falafel":3,"filet mignon":1,"fish and chips":2,"foie gras":2,"french fries":2,"french onion soup":4,
    "french toast":2,"fried calamari":3,"fried rice":3,"frozen yogurt":2,"garlic bread":4,"gnocchi":2,"greek salad":2,
    "grilled cheese sandwich":1,"grilled salmon":2,"guacamole":4,"gyoza":3,"hamburger":1,"hot and sour soup":4,"hot dog":1,
    "huevos rancheros":1,"hummus":4,"ice cream":4,"lasagna":6,"lobster bisque":4,"lobster roll sandwich":1,"macaroni and cheese":4,
    "macarons":8,"miso soup":4,"mussels":2,"nachos":4,"omelette":1,"onion rings":3,"oysters":2,"pad thai":2,"paella":4,
    "pancakes":2,"panna cotta":4,"peking duck":4,"pho":1,"pizza":8,"pork chop":1,"poutine":2,"prime rib":4,"pulled pork sandwich":1,
    "ramen":1,"ravioli":2,"red velvet cake":12,"risotto":3,"samosa":4,"sashimi":2,"scallops":2,"seaweed salad":2,"shrimp and grits":2,
    "spaghetti bolognese":2,"spaghetti carbonara":2,"spring rolls":4,"steak":1,"strawberry shortcake":8,"sushi":8,"tacos":3,
    "takoyaki":4,"tiramisu":8,"tuna tartare":2,"waffles":2
}
SERVING_DEFAULTS = {k.lower(): v for k,v in SERVING_DEFAULTS.items()}

def to_float_qty(q):
    try:
        q=q.strip()
        if ' ' in q:
            a,b=q.split(' ',1)
            return float(a)+float(Fraction(b))
        if '/' in q:
            return float(Fraction(q))
        return float(q)
    except:
        return 1.0

def normalize_unit(u):
    if not u: return ""
    u=u.lower().strip().rstrip('.')
    if u.endswith('s') and u not in {"glass","glass."}:
        if u[:-1] in KNOWN_UNITS: u=u[:-1]
    return 'cup' if u in {'c','c.'} else u

def clean_name(n):
    n=n.lower()
    n=re.sub(r'\([^)]*\)','',n)
    n=n.replace(",", " ")
    tokens=[t for t in n.split() if t not in STRIP_WORDS]
    n=" ".join(tokens).strip()
    return ALIAS.get(n,n)

def simplify_for_fallback(n):
    toks=[t for t in n.split() if t not in STOPWORDS_FALLBACK]
    return " ".join(toks) if toks else n

def similarity(a,b):
    return SequenceMatcher(None,a,b).ratio()

def best_match(name,keys,cutoff=SIMILARITY_MAIN):
    best_key,best_score=None,0.0
    for k in keys:
        s=similarity(name,k)
        if s>best_score:
            best_key,best_score=k,s
    return (best_key,best_score) if best_score>=cutoff else (None,best_score)

def token_fallback_match(name):
    toks=name.split()
    for size in [2,1]:
        for i in range(len(toks)-size+1):
            chunk=" ".join(toks[i:i+size])
            if chunk in nutrition_map: return chunk
            ks=re.sub(r'[^a-z ]','',chunk).strip()
            if ks in nutrition_map: return ks
            cand,_=best_match(ks,nutrition_map.keys(),cutoff=SIMILARITY_FALLBACK)
            if cand: return cand
    return None

def match_nutrition_key(name):
    name=ALIAS.get(name,name)
    if name in nutrition_map: return name
    ks=re.sub(r'[^a-z ]','',name).strip()
    if ks in nutrition_map: return ks
    cand,_=best_match(ks,nutrition_map.keys(),cutoff=SIMILARITY_MAIN)
    if cand: return cand
    return token_fallback_match(simplify_for_fallback(ks))

def safe_nut(v):
    if v is None or (isinstance(v,float) and math.isnan(v)): return 0.0
    return float(v)

def is_low_density(name):
    return any(w in name for w in VEG_FRUIT_SPICE_WORDS)

def sanity_fix(name,nut):
    kcal=safe_nut(nut.get('calories_kcal'))
    if is_low_density(name) and kcal>MAX_KCAL_VEG:
        factor=MAX_KCAL_VEG/kcal if kcal>0 else 0
        return {'calories_kcal':MAX_KCAL_VEG,
                'protein_g':safe_nut(nut.get('protein_g'))*factor,
                'carbs_g':safe_nut(nut.get('carbs_g'))*factor,
                'fat_g':safe_nut(nut.get('fat_g'))*factor}
    return {'calories_kcal':kcal,
            'protein_g':safe_nut(nut.get('protein_g')),
            'carbs_g':safe_nut(nut.get('carbs_g')),
            'fat_g':safe_nut(nut.get('fat_g'))}

def parse_ingredient_rule(s):
    raw=s.strip()
    low=raw.lower()
    m=re.match(r'^\s*(\d+\s+\d+\/\d+|\d+\/\d+|\d+\.\d+|\d+)',low)
    qty=1.0
    rest=low
    if m:
        qty=to_float_qty(m.group(1))
        rest=low[m.end():].strip()
    toks=rest.split()
    unit=""
    name_tokens=toks
    if toks:
        cand=normalize_unit(toks[0])
        if cand in KNOWN_UNITS:
            unit=cand
            name_tokens=toks[1:]
    name_raw=" ".join(name_tokens).strip(",. ")
    opt=any(p in low for p in OPTIONAL_PHRASES)
    fry=FRYING_PHRASE in low
    cleaned=clean_name(name_raw)
    logger.debug(f"RULE PARSED -> raw: '{raw}' | qty: {qty}, unit: '{unit}', name: '{cleaned}'")
    return {"quantity":qty,"unit":unit,"name":cleaned,"raw":raw,"optional":opt,"frying":fry}

def estimate_grams(p):
    qty=p["quantity"]; unit=p["unit"]; name=p["name"]
    if p["optional"]: return 0.0
    if unit=="cup":
        grams_per_cup=cup_specific.get(name,cup_specific["default"])
        grams=qty*grams_per_cup
    elif unit in unit_to_grams:
        grams=qty*unit_to_grams[unit]
    else:
        base=piece_weights.get(name,piece_weights.get(name.rstrip('s'),None))
        grams=qty*base if base is not None else qty*100.0
    if p["frying"]: grams*=0.15
    logger.debug(f"GRAMS -> {qty} {unit} of {name} => {grams:.2f} g")
    return grams

def _strip_t5_tokens(t):
    t=re.sub(r'<extra_id_\d+>','',t)
    t=t.replace('<pad>','').replace('</s>','').strip()
    return t

def _kv_regex_parse(txt):
    fields={}
    for key in ["quantity","unit","ingredient","name"]:
        m=re.search(rf'{key}\s*[:=]\s*"?([^",}}]+)"?',txt,flags=re.IGNORECASE)
        if m: fields[key]=m.group(1).strip()
    return fields if fields else None

def _first_json_block(txt):
    m=re.search(r'\{[^{}]*\}',txt,flags=re.DOTALL)
    return m.group(0) if m else None

def _loose_json_parse(txt):
    try:
        return json.loads(txt)
    except:
        pass
    block=_first_json_block(txt)
    if block:
        blk=block.replace("'",'"')
        blk=re.sub(r'([{,]\s*)(\w+)\s*:',r'\1"\2":',blk)
        blk=re.sub(r',\s*([}\]])',r'\1',blk)
        try:
            return json.loads(blk)
        except:
            pass
    kv=_kv_regex_parse(txt)
    if kv: return kv
    return None

def _heuristic_from_original(orig):
    low=orig.lower()
    m=re.match(r'^\s*(\d+\s+\d+\/\d+|\d+\/\d+|\d+\.\d+|\d+)',low)
    qty="1"
    rest=low
    if m:
        qty=m.group(1)
        rest=low[m.end():].strip()
    toks=rest.split()
    unit=""
    name_tokens=toks
    if toks:
        cand=normalize_unit(toks[0])
        if cand in KNOWN_UNITS:
            unit=cand
            name_tokens=toks[1:]
    name_raw=" ".join(name_tokens).strip(",. ")
    return {"quantity":qty,"unit":unit,"ingredient":name_raw}

USE_T5=False
try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    t5_tokenizer=T5Tokenizer.from_pretrained(T5_MODEL_DIR)
    t5_model=T5ForConditionalGeneration.from_pretrained(T5_MODEL_DIR).to(DEVICE)
    t5_model.eval()
    USE_T5=True
    logger.info("T5 ingredient parser loaded.")
except Exception:
    logger.exception("Failed to load T5 parser. Will use rules only.")

def parse_ingredient_ai(text):
    if not USE_T5: return None
    try:
        inp="parse: "+re.sub(r'\b(oz|tbsp|tsp|c|ml|kg|lb)\.\b',r'\1',text,flags=re.IGNORECASE)
        inputs=t5_tokenizer([inp],return_tensors="pt",truncation=True,max_length=96).to(DEVICE)
        with torch.no_grad():
            out_ids=t5_model.generate(**inputs,max_length=64,do_sample=False,num_beams=4)
        decoded=t5_tokenizer.decode(out_ids[0],skip_special_tokens=True)
        decoded=_strip_t5_tokens(decoded)
        logger.debug(f"T5 RAW OUTPUT: {decoded}")
        data=_loose_json_parse(decoded)
        if not data:
            data=_heuristic_from_original(text)
        q=data.get("quantity","1")
        u=data.get("unit","")
        ing=data.get("ingredient") or data.get("name")
        if not ing: raise ValueError("No ingredient field")
        parsed={"quantity":to_float_qty(str(q)),
                "unit":normalize_unit(str(u)),
                "name":clean_name(ing),
                "raw":text,
                "optional":any(p in text.lower() for p in OPTIONAL_PHRASES),
                "frying":FRYING_PHRASE in text.lower()}
        logger.debug(f"T5 PARSED -> {parsed} (raw: '{text}')")
        return parsed
    except Exception as e:
        logger.warning(f"T5 parse failed for '{text}': {e}")
        return None

def estimate_nutrition(ingredient_strings,fallback_ner=None):
    totals={'calories_kcal':0.0,'protein_g':0.0,'carbs_g':0.0,'fat_g':0.0}
    total_grams=0.0
    details=[]
    logger.info("---- Nutrient Estimation Start ----")
    for raw in ingredient_strings:
        logger.debug(f"RAW INGREDIENT: {raw}")
        parsed=parse_ingredient_ai(raw) or parse_ingredient_rule(raw)
        grams=estimate_grams(parsed)
        if grams<=0 or not parsed['name']:
            logger.debug("Skipped due to zero grams or empty name.")
            continue
        total_grams+=grams
        key=match_nutrition_key(parsed['name'])
        used_fallback=False
        if key is None and fallback_ner:
            for ner_tok in fallback_ner:
                cand=match_nutrition_key(ner_tok)
                if cand:
                    key=cand
                    used_fallback=True
                    break
        if key is None:
            logger.warning(f"No nutrition match for: '{parsed['name']}' (raw: '{raw}')")
            continue
        raw_nut=nutrition_map[key]['nutrition']
        nut=sanity_fix(parsed['name'],raw_nut)
        factor=grams/100.0
        cal=nut['calories_kcal']*factor
        prot=nut['protein_g']*factor
        carb=nut['carbs_g']*factor
        fat =nut['fat_g']*factor
        totals['calories_kcal']+=cal
        totals['protein_g']+=prot
        totals['carbs_g']+=carb
        totals['fat_g']+=fat
        details.append({
            "raw": raw,
            "parsed": parsed,
            "grams": grams,
            "nutr_per_ing": {"kcal": cal, "protein": prot, "carbs": carb, "fat": fat},
            "match_key": key,
            "used_fallback": used_fallback
        })
        logger.debug(f"MATCH -> '{parsed['name']}' -> key='{key}' {'(fallback)' if used_fallback else ''} | grams={grams:.2f}, factor={factor:.2f}")
    logger.info(f"TOTALS -> kcal={totals['calories_kcal']:.2f}, protein={totals['protein_g']:.2f}, carbs={totals['carbs_g']:.2f}, fat={totals['fat_g']:.2f}")
    logger.info("---- Nutrient Estimation End ----")
    return totals,total_grams,details

def extract_servings_from_text(text_blob):
    m=re.search(r'(serves|serve|makes|yield(?:s)?)\s+(\d+)',text_blob,flags=re.IGNORECASE)
    if m: return int(m.group(2))
    return None

def guess_servings(dish_class, ingredients_raw, instructions, total_kcal, total_g):
    blob=" ".join(ingredients_raw+instructions).lower()
    s=extract_servings_from_text(blob)
    if s and s>0: return s
    cls=dish_class.lower()
    if cls in SERVING_DEFAULTS: return SERVING_DEFAULTS[cls]
    for k,v in SERVING_DEFAULTS.items():
        if k in cls: return v
    target_kcal=600
    kcal_based=max(1,round(total_kcal/target_kcal))
    target_g=350
    weight_based=max(1,round(total_g/target_g)) if total_g>0 else kcal_based
    return max(1,min(kcal_based,weight_based))

def predict_image(img_path):
    image=Image.open(img_path).convert('RGB')
    image=test_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs=model(image)
        _,predicted=outputs.max(1)
    pred=class_names[predicted.item()]
    logger.info(f"Predicted dish: {pred}")
    return pred

def _instr_len(steps):
    if not isinstance(steps,list): return 0
    return sum(len(str(s).split()) for s in steps)

def _ing_len(lst):
    if not isinstance(lst,list): return 0
    return len(lst)

def get_recipe_info(dish_class):
    dish_class=dish_class.lower()
    rows=recipe_df[recipe_df['food101_class']==dish_class]
    if rows.empty:
        logger.warning(f"No recipe info found for class: {dish_class}")
        return {'ingredients_raw':[],'ingredients_clean':[],'instructions':["No instructions found."]}
    df=rows.copy()
    df["instr_score"]=df["instructions"].apply(_instr_len)
    df["ing_score"]=df["ingredients_list"].apply(_ing_len)
    df["score"]=df["instr_score"]*0.8+df["ing_score"]*0.2
    r=df.sort_values("score",ascending=False).iloc[0]
    return {'ingredients_raw':r['ingredients_list'],
            'ingredients_clean':r['ner'],
            'instructions':r['instructions']}

@app.route("/",methods=["GET","POST"])
def index():
    if request.method=="POST":
        file=request.files.get('image')
        if file:
            img_path=os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
            file.save(img_path)
            logger.info(f"Image saved to {img_path}")
            dish=predict_image(img_path)
            info=get_recipe_info(dish)
            totals,total_g,details=estimate_nutrition(info['ingredients_raw'],fallback_ner=info['ingredients_clean'])
            servings=guess_servings(dish,info['ingredients_raw'],info['instructions'],totals['calories_kcal'],total_g)
            per_serv={k:v/servings for k,v in totals.items()}
            nutrition={"totals":totals,"per_serving":per_serv,"servings":servings,"total_grams":total_g}
            return render_template("index.html",
                                   image_path=img_path,
                                   dish=dish,
                                   ingredients_raw=info['ingredients_raw'],
                                   parsed_ingredients=details,
                                   instructions=info['instructions'],
                                   nutrition=nutrition)
    return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True)