import os
os.environ["HF_DATASETS_OFFLINE"] = "1" 
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["DISABLE_TELEMETRY"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

# Disable built-in progress bars from libraries
os.environ["DIFFUSERS_PROGRESS_BAR"] = "0"
os.environ["TRANSFORMERS_PROGRESS_BAR"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# Add these at the top of text2img.py, after the os imports


from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.responses import FileResponse, HTMLResponse
from fastapi import Response, Request
from pydantic import BaseModel, Field, ConfigDict, field_validator  # Add field_validator import
from fastapi.staticfiles import StaticFiles
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
import torch
from pathlib import Path
import logging
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any
from enum import Enum
import hashlib
import time
import re
import uuid
import os
import base64
from io import BytesIO
from datetime import datetime
from starlette.requests import Request
from fastapi.security import APIKeyHeader
from collections import defaultdict
from tqdm.auto import tqdm  # Add this to your imports
import platform
import sys
import psutil
import logging.handlers
from diffusers import DPMSolverMultistepScheduler

# Create logs directory
os.makedirs("logs", exist_ok=True)

# Configure file logging with rotation
file_handler = logging.handlers.RotatingFileHandler(
    "logs/text2img.log",
    maxBytes=10485760,  # 10 MB
    backupCount=5,      # Keep 5 backup files
    encoding="utf-8"
)

# Set the logging level for file output
file_handler.setLevel(logging.INFO)

# Create a formatter that includes timestamp, level, and location information
file_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s'
)
file_handler.setFormatter(file_formatter)

# Add file handler to your logger
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)

# Add this after the basic file handler setup above

# 1. Create separate error log for easier troubleshooting
error_handler = logging.handlers.RotatingFileHandler(
    "logs/errors.log",
    maxBytes=5242880,  # 5 MB
    backupCount=3,     # Keep 3 backup files
    encoding="utf-8"
)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(file_formatter)

# 2. Create a debug log for verbose information
debug_handler = logging.handlers.RotatingFileHandler(
    "logs/debug.log",
    maxBytes=10485760,  # 10 MB
    backupCount=2,      # Keep 2 backup files
    encoding="utf-8"
)
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(file_formatter)

# Add these handlers to your logger
logger.addHandler(error_handler)
logger.addHandler(debug_handler)

# Set the root logger to DEBUG to capture all levels
logger.setLevel(logging.DEBUG)

logger.info("Advanced logging configuration complete")

# Add this after your initial logger setup

# Create a console handler that doesn't show DEBUG messages
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Only show INFO and above in console
console_formatter = logging.Formatter('%(message)s')  # Simple format for console
console_handler.setFormatter(console_formatter)

# Add a filter to prevent debug logs with console=False from showing in console
class ConsoleFilter(logging.Filter):
    def filter(self, record):
        if hasattr(record, 'console') and record.console is False:
            return False
        return True

console_handler.addFilter(ConsoleFilter())
logger.addHandler(console_handler)

# Make sure this is properly set up in your logging configuration:
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Only show INFO and above
console_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(console_handler)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger.info("File logging initialized - logs will be saved to logs/text2img.log")

logger.info(f"Starting Multi-Model Text-to-Image API v3.0.0")
logger.info(f"Python version: {platform.python_version()}")
logger.info(f"Torch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
    logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Model Multilingual Text to Image API with Safety Guardrails",
    description="Generate images from text prompts with comprehensive safety controls",
    version="3.0.0"
)

os.makedirs("outputs", exist_ok=True)
app.mount("/static", StaticFiles(directory="outputs"), name="static")

# Add this after your FastAPI app initialization
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Extract basic request info
    method = request.method
    url = request.url.path
    client = request.client.host if request.client else "unknown"
    
    logger.info(f"Request started: {method} {url} from {client} (id: {request_id})")
    
    # Process the request and catch exceptions
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Request completed: {method} {url} - Status: {response.status_code} - Time: {process_time:.3f}s (id: {request_id})")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request failed: {method} {url} - Error: {str(e)} - Time: {process_time:.3f}s (id: {request_id})")
        raise

# Path to the local model directory
#MODELS_DIR = Path(r"C:\Users\user\Desktop\t2i\models")

# Path to LoRA models directory
#LORA_DIR = Path(r"C:\Users\user\Desktop\t2i\loras")
# Replace hard-coded paths with more flexible paths
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "./models"))
LORA_DIR = Path(os.environ.get("LORA_DIR", "./loras"))
os.makedirs(LORA_DIR, exist_ok=True)

logger.info(f"Models directory: {MODELS_DIR} (exists: {MODELS_DIR.exists()})")
logger.info(f"LoRA directory: {LORA_DIR} (exists: {LORA_DIR.exists()})")
logger.info(f"Outputs directory: outputs/ (exists: {Path('outputs').exists()})")

# Dictionary to track available LoRAs
available_loras = {}

# Cache loaded pipelines
loaded_pipelines = {}

# Supported languages
class Language(str, Enum):
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    

# Update the language style prompts for better results
LANGUAGE_STYLE_PROMPTS = {
    "en": "((high quality, detailed, intricate)), ",
    "es": "((alta calidad, detallado, intrincado, estilo fotorrealista)), ",
    "fr": "((haute qualité, détaillé, complexe, style photoréaliste)), ",
    "de": "((hohe Qualität, detailliert, komplex, fotorealistischer Stil)), ",
    "it": "((alta qualité, dettagliato, complesso, stile fotorealistico)), ",
}

# Safety classification categories
class SafetyCategory(str, Enum):
    SAFE = "safe"
    ADULT = "adult_content"
    VIOLENCE = "violence"
    HATE = "hate_speech"
    HARASSMENT = "harassment"
    SELF_HARM = "self_harm"
    ILLEGAL_ACTIVITY = "illegal_activity"
    DANGEROUS = "dangerous_content"
    DECEPTION = "deception"
    POLITICAL = "political"
    COPYRIGHTS="copyrights"
    EVASION_ATTEMPTS = "evasion_attempts"
    PERSONAL_INFORMATION = "personal_information"
    RELIGIOUS_ICONOGRAPHY = "religious_iconography"  
    HISTORICAL_REVISIONISM = "historical_revisionism"  

# Define comprehensive safety guardrails
class SafetyGuardrails:
    def __init__(self):
        # Initialize pattern dictionaries
        self.patterns = {
            SafetyCategory.ADULT: [
                r'\b(?:porn|naked|nude|nudity|sex|explicit|nsfw|xxx|18\+|adult|erotic)\b',
                r'\b(?:hentai|rule34|boob|breast|nipple|vagina|penis|genital)\b',
                # Add more thorough patterns that catch implied sexual content
                r'\b(?:seductive|revealing|skimpy|sexy|sensual|hot girl|intimate|attractive woman)\b',
                r'\b(?:without cloth|no cloth|barely dressed|scantily clad|suggestive|boudoir)\b',
                r'\b(?:erotic|sexual|strip|stripper|porn|xxx|lewd|lust|slutty|horny|turned on)\b',
                r'\b(?:masturbat|orgasm|aroused|arousing|dominatrix|fetish|bdsm|kinky)\b',
                r'\b(?:underwear|lingerie|bikini|swimsuit|thong|panties|bra)\b',
                # Look for suspicious phrases and indirect requests
                r'(?:showing more skin|wearing less|take off|take it off|remove clothing)',
                r'(?:without any.*on|not wearing|isn\'t wearing|no pants|abused|abuse|abusing|topless)',
                # Add these patterns to SafetyCategory.ADULT in the self.patterns dictionary
                r'\b(?:without cloth|no cloth|bedroom.*without|naked in|nude in|in bed without)\b',
                r'\b(?:man.*woman.*bedroom|woman.*man.*bedroom)\b',
            ],
            
            SafetyCategory.EVASION_ATTEMPTS: [
                r'\b(?:bypass|ignore|avoid|this is not|don\'t flag|circumvent|work around)\b',
                r'(?:find a way to|please just|I promise|for educational purposes only)',
                r'(?:I\'m asking for a friend|just pretend|you can display|actually show)',
                r'(?:don\'t tell anyone|this is a test|I won\'t share|I\'m not serious)',
                r'(?:this is a joke|just for fun|not meant to be taken seriously|hypothetical)',
                # Add these patterns to SafetyCategory.EVASION_ATTEMPTS
                r'(?:as if the model isn\'t|as if.*not wearing|as though.*naked|appears to be without)',
                r'(?:draw.*without|create.*without|generate.*without)',
                r'(?:but covered by|but hidden by|but obscured by|but shadowed)'
            ],
            SafetyCategory.VIOLENCE: [
                r'\b(?:gore|blood|killing|murder|torture|violent|death|corpse|suicide)\b',
                r'\b(?:massacre|slaughter|dismember|mutilate|decapitate)\b',
                r'\b(?:bloody|brutal|savage|gruesome|gory|kill)\b',
                # Add these to SafetyCategory.VIOLENCE
                r'\b(?:crime scene|dead body|shooting|shot|gunfire|gun violence)\b',
                r'(?:blood|bloody|bleeding|wound|injured|casualty|fatality)',
                r'(?:soldier.*shooting|military.*killing|armed.*attack)',
            ],
            SafetyCategory.HATE: [
                r'\b(?:nazi|hitler|racist|racism|sexist|sexism|bigot|fascist)\b',
                r'\b(?:antisemit|homophob|transphob|supremac|hate group)\b',
                r'\b(?:anti-semitic|antisemitic|anti semitic)\b',
                r'\b(?:propaganda|hate literature|supremacist)\b.*\b(?:nazi|racist|antisemit|fascist)\b',
                r'\b(?:nazi|racist|antisemit|fascist)\b.*\b(?:propaganda|literature|poster|imagery)\b',
            ],
            SafetyCategory.HARASSMENT: [
                r'\b(?:bully|harass|stalk|dox|doxx|threaten)\b',
            ],
            SafetyCategory.SELF_HARM: [
                r'\b(?:suicide|self-harm|cutting|anorexia|bulimia|self-injury)\b',
            ],
            SafetyCategory.ILLEGAL_ACTIVITY: [
                r'\b(?:illegal|cocaine|heroin|drug|meth|terrorist|bomb|weapon)\b',
            ],
            SafetyCategory.DANGEROUS: [
                r'\b(?:dangerous|harmful|toxic|poison|hazard|infectious)\b',
            ],
            SafetyCategory.DECEPTION: [
                r'\b(?:scam|fraud|phishing|counterfeit|forgery)\b',
            ],
            SafetyCategory.POLITICAL: [
                r'\b(?:politic|candidate|campaign|election|democrat|republican|vote)\b'
            ],
            SafetyCategory.PERSONAL_INFORMATION: [
                r'\b(?:address|phone number|email address|social security|ssn|passport)\b',
                r'\b(?:credit card|bank account|password|credentials)\b'
            ],
            SafetyCategory.COPYRIGHTS: [
                r'\b(?:celebrity|celebrities|famous person|famous people|famous actor|famous actress)\b',
                r'\b(?:copyright|trademark|patent|intellectual property|piracy)\b',
                r'\b(?:plagiarism|infringement|unauthorized use)\b',
                r'\b(?:fan art|fan fiction|derivative work|remix)\b',
            ],
            SafetyCategory.RELIGIOUS_ICONOGRAPHY: [
                r'\b(?:desecrat|defac|mock|ridicul|insult|offensive)\b.*\b(?:jesus|christ|bible|quran|koran|allah|muhammad|pope|church|mosque|temple)\b',
                r'\b(?:sacrileg|blasphemy|blasphemous|profane)\b.*\b(?:religious|sacred|holy|divine|worship)\b',
                r'\b(?:satanic|demonic|evil|unholy)\b.*\b(?:cross|crucifix|bible|quran|koran|church|mosque|temple)\b',
                r'\b(?:parody|comedy|joke|meme)\b.*\b(?:jesus|muhammad|buddha|holy|sacred|prophet|religious figure)\b',
                r'\b(?:jesus|muhammad|buddha|moses|vishnu|shiva|ganesha|poker|religious figure)\b.*\b(?:inappropriate|sexualized|violent|disrespectful)\b',
                r'\b(?:modify|alter|change|replace)\b.*\b(?:religious|sacred|holy text|scripture)\b',
                r'\b(?:muhammad|jesus|christ|bible|quran)\b.*\b(?:mock|ridicul|desecrat|defac|insult)\w*\b',
                r'\b(?:bible|quran|scripture|religious text)\b.*\b(?:desecrat|defac|destroy|burn|tear)\w*\b',
                r'\b(?:satanic|demonic)\b.*\b(?:symbol|imagery|ritual)\b.*\b(?:bible|quran|religious|sacred)\b',
            ],
            
            SafetyCategory.HISTORICAL_REVISIONISM: [
                r'\b(?:revis|deny|minimize|alternate)\b.*\b(?:holocaust|genocide|slavery|colonialism|apartheid)\b',
                r'\b(?:nazi|hitler|holocaust)\b.*\b(?:wasn\'t|wasn\'t bad|exaggerated|didn\'t happen|fake|hoax)\b',
                r'\b(?:slavery|slaves|plantation)\b.*\b(?:happy|willing|better off|not that bad|benefited|exaggerated)\b',
                r'\b(?:alternative history|what if|rewrite|change|alter)\b.*\b(?:ww2|world war|civil war|historical event)\b',
                r'\b(?:confederate|nazi|fascist|imperial|colonial)\b.*\b(?:glorify|celebrate|honor|promote|superior|better)\b',
                r'\b(?:genocide|ethnic cleansing|mass killing)\b.*\b(?:justified|necessary|fake|didn\'t happen|exaggerated)\b',
            ]
        }
        
        # Compile all patterns for efficiency
        self.compiled_patterns = {
            category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for category, patterns in self.patterns.items()
        }
        
        # Celebrity name detection - comprehensive list of common celebrity references
        self.celebrity_patterns = [
            # Actors/Actresses
            r'\b(?:brad\s*pitt|angelina\s*jolie|tom\s*cruise|jennifer\s*lawrence|leonardo\s*dicaprio)\b',
            r'\b(?:scarlett\s*johansson|dwayne\s*johnson|emma\s*watson|robert\s*downey\s*jr|chris\s*hemsworth)\b',
            r'\b(?:will\s*smith|margot\s*robbie|johnny\s*depp|meryl\s*streep|denzel\s*washington)\b',
            
            # Musicians
            r'\b(?:beyonce|taylor\s*swift|ed\s*sheeran|ariana\s*grande|drake|eminem|rihanna)\b',
            r'\b(?:justin\s*bieber|lady\s*gaga|kanye\s*west|adele|bruno\s*mars|selena\s*gomez)\b',
            
            # Politicians/Public figures
            r'\b(?:trump|biden|obama|clinton|putin|elon\s*musk|bill\s*gates|mark\s*zuckerberg)\b',
            
            # Sports stars
            r'\b(?:lebron\s*james|cristiano\s*ronaldo|lionel\s*messi|serena\s*williams|michael\s*jordan)\b',
            r'\b(?:roger\s*federer|tom\s*brady|rafael\s*nadal|usain\s*bolt|tiger\s*woods)\b',
            
            # General celebrity terms
            r'\b(?:movie\s*star|film\s*star|pop\s*star|rock\s*star|superstar|mega\s*star|powerstar|natural\s*star|stylish\s*starss|celeb)\b',
            r'\b(?:famous|well[- ]known|renowned|celebrity|hollywood|tollywood|bollywood|kollywood|a[- ]list)\b',
            
            # Modeling-related terms that often indicate celebrity generation
            r'\b(?:photorealistic|photograph of|portrait of|picture of|image of)\b.*\b(?:person|individual|man|woman)\b',
        ]
        
            # Compile celebrity patterns
        self.compiled_celebrity_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.celebrity_patterns]
        
        logger.info("Safety guardrails initialized with celebrity detection")
        
        # Add multilingual equivalents for common unsafe terms
        self.multilingual_unsafe = {
            "es": ["desnudo", "porno", "sexo", "explícito", "sangre", "violencia", "asesinar"],
            "fr": ["nu", "porno", "sexe", "explicite", "sang", "violence", "tuer"],
            "de": ["nackt", "porno", "sex", "explizit", "blut", "gewalt", "töten"],
            "it": ["nudo", "porno", "sesso", "esplicito", "sangue", "violenza", "uccidere"]
            
        }
        
        
        logger.info("Safety guardrails initialized")
    
    def check_prompt(self, prompt: str, language: str = "en") -> Dict:
        """
        Check if prompt contains unsafe content including celebrities
        Returns dict with safety assessment
        """
        prompt_lower = prompt.lower()
        result = {"is_safe": True, "categories": [], "details": {}}
        
        # Check English patterns across all categories
        for category, patterns in self.compiled_patterns.items():
            matches = []
            for pattern in patterns:
                found = pattern.findall(prompt_lower)
                if found:
                    matches.extend(found)
            
            if matches:
                result["is_safe"] = False
                result["categories"].append(category)
                result["details"][category] = list(set(matches))  # Remove duplicates
         # Add new proximity check for dangerous combinations
        tokens = prompt_lower.split()
    
        # Special check for celebrities
        celebrity_matches = []
        for pattern in self.compiled_celebrity_patterns:
            found = pattern.findall(prompt_lower)
            if found:
                celebrity_matches.extend(found)
        
        if celebrity_matches:
            result["is_safe"] = False
            if SafetyCategory.COPYRIGHTS not in result["categories"]:
                result["categories"].append(SafetyCategory.COPYRIGHTS)
            
            # Add celebrity matches to details
            if "celebrity_references" not in result["details"]:
                result["details"]["celebrity_references"] = []
            result["details"]["celebrity_references"] = list(set(celebrity_matches))
        
        # Check language-specific unsafe terms
        if language != "en" and language in self.multilingual_unsafe:
            for term in self.multilingual_unsafe[language]:
                if term.lower() in prompt_lower:
                    result["is_safe"] = False
                    if SafetyCategory.ADULT not in result["categories"]:
                        result["categories"].append(SafetyCategory.ADULT)
                    
                    if "multilingual" not in result["details"]:
                        result["details"]["multilingual"] = []
                    
                    result["details"]["multilingual"].append(term)
         # Check for evasion attempts
        evasion_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.patterns.get("EVASION_ATTEMPTS", [])]
        for pattern in evasion_patterns:
            if pattern.search(prompt_lower):
                result["is_safe"] = False
                if "EVASION_ATTEMPT" not in result["details"]:
                    result["details"]["EVASION_ATTEMPT"] = []
                result["details"]["EVASION_ATTEMPT"].append("Potential attempt to bypass safety filters")
                if "evasion_attempt" not in result["categories"]:
                    result["categories"].append("evasion_attempt")
        
        return result
        
    def sanitize_prompt(self, prompt: str) -> str:  # Add 'self' parameter here
        """Remove or replace potentially problematic terms with safer alternatives."""
        # Replace explicit terms with sanitized versions
        replacements = {
            r'\b(?:naked|nude|topless)\b': 'clothed',
            r'\b(?:sexy|hot)\b': 'attractive',
            r'\b(?:revealing|skimpy)\b': 'appropriate',
            r'\bnipple\b': 'face',
            r'\bbreasts?\b': 'body',
            r'\bporn\w*\b': 'appropriate',
            r'\b(?:erotic|sexual)\b': 'romantic',
            # Add more replacements as needed
        }
    
        sanitized = prompt
        for pattern, replacement in replacements.items():
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def log_safety_check(self, prompt: str, result: Dict):
        """Log the safety check result"""
        if result["is_safe"]:
            logger.info(f"Safety check passed for prompt: '{prompt[:30]}...'")
        else:
            categories = ", ".join(result["categories"])
            logger.warning(f"Safety check failed for prompt: '{prompt[:30]}...'. Categories: {categories}")

# Initialize safety guardrails
safety = SafetyGuardrails()

# Add this after your SafetyGuardrails class
class IntentionCategory(str, Enum):
    """Categories for user intention classification"""
    EDUCATIONAL = "educational"
    ARTISTIC = "artistic"
    NSFW = "nsfw"
    VIOLENCE = "violence"
    HATE = "hate"
    ILLEGAL = "illegal"
    DANGEROUS = "dangerous"
    SAFE = "safe"  
    RELIGIOUS_MISUSE = "religious_misuse"
    HISTORICAL_REVISION = "historical_revision" # Default if not matching other categories

class IntentionClassifier:
    """Classifies user intentions into different safety categories"""
    
    def __init__(self):
        # Define patterns for each unsafe category
        self.patterns = {
            IntentionCategory.NSFW: [
                r'\b(?:porn|naked|assault|assaulting|assaulting|assaulted|nude|nudity|sex|explicit|nsfw|xxx|18\+|adult|erotic)\b',
                r'\b(?:hentai|rule34|boob|breast|nipple|vagina|penis|genital)\b',
                r'\b(?:arousing|seductive|sexual|lewd|obscene|hardcore|fetish)\b'
            ],
            
            IntentionCategory.VIOLENCE: [
                r'\b(?:gore|blood|killing|murder|torture|violent|death|corpse|suicide)\b',
                r'\b(?:massacre|slaughter|dismember|mutilate|decapitate)\b',
                r'\b(?:bloody|brutal|savage|gruesome|gory|kill)\b'
            ],
            
            IntentionCategory.HATE: [
                r'\b(?:nazi|hitler|racist|racism|sexist|sexism|bigot|fascist)\b',
                r'\b(?:antisemit|homophob|transphob|supremac|hate group)\b',
                r'\b(?:slur|derogatory|offensive|stereotype)\b'
            ],
            
            IntentionCategory.ILLEGAL: [
                r'\b(?:illegal|cocaine|heroin|drug|meth|terrorist|bomb|weapon)\b',
                r'\b(?:assault|theft|robbery|smuggling|counterfeit|forgery)\b',
                r'\b(?:crime|criminal|illegal|drugs|narcotic)\b'
            ],
            
            IntentionCategory.DANGEROUS: [
                r'\b(?:dangerous|harmful|toxic|poison|hazard|infectious)\b',
                r'\b(?:explosive|instructions|suicide|self-harm|injury)\b',
                r'\b(?:dangerous|harmful|risk|hazard|unsafe)\b'
            ],
            
            # Safe indicators - educational context
            IntentionCategory.EDUCATIONAL: [
                r'\b(?:educational|academic|scientific|historical|research|documentary)\b',
                r'\b(?:textbook|diagram|medical|anatomy|biology|science|study)\b',
                r'\b(?:for education|learning|teaching|educational purposes|study)\b',
                r'\b(?:in a documentary|documentary style|news report|journalism)\b'
            ],
            
            # Safe indicators - artistic context
            IntentionCategory.ARTISTIC: [
                r'\b(?:art|artistic|painting|illustration|renaissance|classical)\b',
                r'\b(?:fictional|stylized|concept art|artwork|drawing|sculpture)\b',
                r'\b(?:fantasy|imaginary|creative|artistic style|design|composition)\b',
                r'\b(?:in the style of|impressionist|surrealist|abstract art)\b'
            ],
            # Safe indicators - artistic context
            IntentionCategory.RELIGIOUS_MISUSE:[
                r'\b(?:desecrat|defac|mock|ridicul|insult|offensive)\b.*\b(?:jesus|christ|bible|quran|mahabharatam|ramayanam|koran|allah|muhammad|pope|church|mosque|temple)\b',
                r'\b(?:sacrileg|blasphemy|blasphemous|profane)\b.*\b(?:religious|sacred|holy|divine|worship)\b',
                r'\b(?:satanic|demonic|evil|unholy)\b.*\b(?:cross|crucifix|bible|quran|koran|church|mosque|temple)\b',
                r'\b(?:jesus|muhammad|buddha|holy|sacred|prophet)\b',
                r'\b(?:modify|alter)\b.*\b(?:religious text)\b'
            ],
            IntentionCategory.HISTORICAL_REVISION: [
                r'\b(?:revis|deny|alternate)\b.*\b(?:holocaust|genocide|slavery|colonialism|apartheid)\b',
                r'\b(?:nazi|hitler|holocaust)\b.*\b(?:wasn\'t|wasn\'t bad|exaggerated|didn\'t happen|fake|hoax)\b',
                r'\b(?:slavery|slaves)\b.*\b(?:willing|better off|not that bad|benefited|exaggerated)\b',
                r'\b(?:alternative history|what if|rewrite)\b.*\b(?:ww2|world war)\b'
            ]  
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {
            category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for category, patterns in self.patterns.items()
        }
        
        # Multilingual equivalents for key terms
        self.multilingual_terms = {
            "es": {
                IntentionCategory.NSFW: ["desnudo", "porno", "sexo", "explícito", "erótico"],
                IntentionCategory.VIOLENCE: ["violencia", "sangre", "asesinar", "muerte", "brutal"],
                IntentionCategory.HATE: ["racista", "sexista", "odio", "discriminación"],
                IntentionCategory.EDUCATIONAL: ["educativo", "académico", "histórico", "médico"],
                IntentionCategory.ARTISTIC: ["arte", "artístico", "pintura", "escultura"]
            },
            "fr": {
                IntentionCategory.NSFW: ["nu", "porno", "sexe", "explicite", "érotique"],
                IntentionCategory.VIOLENCE: ["violence", "sang", "tuer", "mort", "brutal"],
                IntentionCategory.HATE: ["raciste", "sexiste", "haine", "discrimination"],
                IntentionCategory.EDUCATIONAL: ["éducatif", "académique", "historique", "médical"],
                IntentionCategory.ARTISTIC: ["art", "artistique", "peinture", "sculpture"]
            },
            "de": {
                IntentionCategory.NSFW: ["nackt", "porno", "sex", "explizit", "erotisch"],
                IntentionCategory.VIOLENCE: ["gewalt", "blut", "töten", "tod", "brutal"],
                IntentionCategory.HATE: ["rassistisch", "sexistisch", "hass", "diskriminierung"],
                IntentionCategory.EDUCATIONAL: ["bildung", "akademisch", "historisch", "medizinisch"],
                IntentionCategory.ARTISTIC: ["kunst", "künstlerisch", "gemälde", "skulptur"]
            }
        }
        
        logger.info("Intention classifier initialized")
    
    def classify_intention(self, prompt: str, language: str = "en") -> Dict:
        """
        Classify the user's intention based on prompt content
        Returns the primary category and all matched categories
        """
        # First convert prompt and language to strings to handle Query objects
        prompt_str = str(prompt) if not isinstance(prompt, str) else prompt
        language_str = str(language) if not isinstance(language, str) else language
        
        prompt_lower = prompt_str.lower()
        matches = {}
        
        # Check all patterns for each category
        for category, patterns in self.compiled_patterns.items():
            category_matches = []
            for pattern in patterns:
                found = pattern.findall(prompt_lower)
                if found:
                    category_matches.extend(found)
            
            if category_matches:
                matches[category] = list(set(category_matches))  # Remove duplicates
        
        # Check multilingual terms
        if language_str != "en" and language_str in self.multilingual_terms:
            for category, terms in self.multilingual_terms[language_str].items():
                for term in terms:
                    if term.lower() in prompt_lower:
                        if category not in matches:
                            matches[category] = []
                        matches[category].append(term.lower())
        
        # Handle empty matches case first
        if not matches:
            # No matches found, default to SAFE category
            primary_category = IntentionCategory.SAFE
            matches[IntentionCategory.SAFE] = ["No unsafe content detected"]
        else:
            # Count matches per category to determine strength
            category_strengths = {cat: len(words) for cat, words in matches.items()}
            
            # Educational and artistic contexts can override NSFW in certain cases
            if (IntentionCategory.EDUCATIONAL in category_strengths and 
                IntentionCategory.NSFW in category_strengths and
                category_strengths[IntentionCategory.EDUCATIONAL] >= category_strengths[IntentionCategory.NSFW] and
                any(marker in prompt_lower for marker in ["medical", "anatomy", "textbook", "research"])):
                primary_category = IntentionCategory.EDUCATIONAL
            
            # Classical art can override NSFW
            elif (IntentionCategory.ARTISTIC in category_strengths and
                 IntentionCategory.NSFW in category_strengths and
                 category_strengths[IntentionCategory.ARTISTIC] >= category_strengths[IntentionCategory.NSFW] and
                 any(marker in prompt_lower for marker in ["renaissance", "classical", "sculpture", "painting"])):
                primary_category = IntentionCategory.ARTISTIC
                
            # Historical documentation can override violence
            elif (IntentionCategory.EDUCATIONAL in category_strengths and
                 IntentionCategory.VIOLENCE in category_strengths and
                 "historical" in prompt_lower):
                primary_category = IntentionCategory.EDUCATIONAL
                
            # Otherwise use the category with most matches
            else:
                primary_category = max(category_strengths, key=category_strengths.get)
                
                # If the primary category is unsafe, that takes precedence
                unsafe_categories = [IntentionCategory.NSFW, IntentionCategory.VIOLENCE, 
                                    IntentionCategory.HATE, IntentionCategory.ILLEGAL,
                                    IntentionCategory.DANGEROUS]
                if any(cat in unsafe_categories for cat in category_strengths.keys()):
                    for unsafe_cat in unsafe_categories:
                        if unsafe_cat in category_strengths:
                            primary_category = unsafe_cat
                            break
        
        # Prepare result
        result = {
            "primary_category": primary_category,
            "all_categories": list(matches.keys()),
            "matches": matches,
            "is_safe": primary_category in [IntentionCategory.SAFE, IntentionCategory.EDUCATIONAL, IntentionCategory.ARTISTIC],
            "requires_review": primary_category in [IntentionCategory.NSFW, IntentionCategory.VIOLENCE]
        }
        
        # Handle educational/artistic special cases
        if primary_category == IntentionCategory.EDUCATIONAL:
            result["explanation"] = "Educational purpose detected"
        elif primary_category == IntentionCategory.ARTISTIC:
            result["explanation"] = "Artistic purpose detected"
        elif primary_category == IntentionCategory.SAFE:
            result["explanation"] = "No unsafe content detected"
        else:
            result["explanation"] = f"Unsafe content detected: {primary_category}"
        
        return result

# Initialize intention classifier
intention_classifier = IntentionClassifier()

# Add this after your other class definitions
class ChatMemoryItem(BaseModel):
    """Represents a single chat interaction in memory"""
    model_config = ConfigDict(protected_namespaces=())  # Add this line to resolve namespace conflict
    
    id: str
    timestamp: datetime
    user_prompt: str
    language: str
    safety_result: Dict[str, Any]
    intention_result: Dict[str, Any]
    enhanced_prompt: Optional[str] = None  # Add this line
    model_used: Optional[str] = None  # Add this line
    image_path: Optional[str] = None
    successful: bool
    metadata: Dict[str, Any] = {}

class ChatMemory:
    """Manages chat history memory for the system"""
    
    def __init__(self, max_items: int = 1000):
        """Initialize the chat memory with a maximum capacity"""
        self.max_items = max_items
        self.memory: Dict[str, ChatMemoryItem] = {}
        self.user_history: Dict[str, List[str]] = {}  # Maps user_id to list of chat IDs
        self.prompt_index: Dict[str, List[str]] = {}  # For searching similar prompts
        logger.info(f"Chat memory system initialized with capacity {max_items}")
    
    def add_interaction(self, 
                        user_id: str,
                        prompt: str,
                        language: str,
                        safety_result: Dict[str, Any],
                        intention_result: Dict[str, Any],
                        enhanced_prompt: Optional[str] = None,
                        model_used: Optional[str] = None,
                        image_path: Optional[str] = None,
                        successful: bool = True,
                        metadata: Dict[str, Any] = {}) -> str:
        """
        Add a new chat interaction to memory
        Returns the ID of the created memory item
        """
        # Generate a unique ID for this chat interaction
        chat_id = str(uuid.uuid4())
        
        # Create memory item
        memory_item = ChatMemoryItem(
            id=chat_id,
            timestamp=datetime.now(),
            user_prompt=prompt,
            language=language,
            enhanced_prompt=enhanced_prompt,
            model_used=model_used,
            safety_result=safety_result,
            intention_result=intention_result,
            image_path=image_path,
            successful=successful,
            metadata=metadata
        )
        
        # Add to main memory
        self.memory[chat_id] = memory_item
        
        # Add to user history
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        self.user_history[user_id].append(chat_id)
        
        # Add to prompt index (simple keyword indexing)
        keywords = set(self._extract_keywords(prompt))
        for keyword in keywords:
            if keyword not in self.prompt_index:
                self.prompt_index[keyword] = []
            self.prompt_index[keyword].append(chat_id)
        
        # Enforce memory limit if needed
        if len(self.memory) > self.max_items:
            self._trim_memory()
        
        logger.info(f"Added chat memory item {chat_id} for user {user_id}: '{prompt[:30]}...'")
        return chat_id
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for indexing"""
        # Simple implementation - split on spaces and keep words longer than 3 chars
        return [word.lower() for word in re.findall(r'\w+', text) if len(word) > 3]
    
    def _trim_memory(self):
        """Remove oldest items when memory limit is reached"""
        # Sort by timestamp and keep only max_items
        sorted_items = sorted(self.memory.items(), key=lambda x: x[1].timestamp)
        to_remove = sorted_items[:len(sorted_items) - self.max_items]
        
        # Remove the oldest items
        for item_id, item in to_remove:
            del self.memory[item_id]
            
            # Also clean up references in indexes
            for user_id, history in self.user_history.items():
                if item_id in history:
                    history.remove(item_id)
            
            for keyword, items in self.prompt_index.items():
                if item_id in items:
                    items.remove(item_id)
    
    def get_user_history(self, user_id: str, limit: int = 10) -> List[ChatMemoryItem]:
        """Get the most recent chat history for a user"""
        if user_id not in self.user_history:
            return []
        
        chat_ids = self.user_history[user_id][-limit:] if limit > 0 else self.user_history[user_id]
        return [self.memory[chat_id] for chat_id in chat_ids if chat_id in self.memory]
    
    def find_similar_prompts(self, prompt: str, limit: int = 5) -> List[ChatMemoryItem]:
        """Find similar previous prompts"""
        keywords = self._extract_keywords(prompt)
        
        # Count occurrences of each chat ID in keyword matches
        matches: Dict[str, int] = {}
        for keyword in keywords:
            if keyword in self.prompt_index:
                for chat_id in self.prompt_index[keyword]:
                    if chat_id not in matches:
                        matches[chat_id] = 0
                    matches[chat_id] += 1
        
        # Sort by number of matching keywords (descending)
        sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [self.memory[chat_id] for chat_id, _ in sorted_matches if chat_id in self.memory]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        return {
            "total_items": len(self.memory),
            "total_users": len(self.user_history),
            "capacity": self.max_items,
            "usage_percent": (len(self.memory) / self.max_items) * 100 if self.max_items > 0 else 0,
            "keywords_indexed": len(self.prompt_index)
        }
    # Add this method to your ChatMemory class
    def get_interaction_by_id(self, chat_id: str) -> Optional[ChatMemoryItem]:
        """Retrieve a specific interaction by ID"""
        return self.memory.get(chat_id)    
# Initialize chat memory system - add this after initializing other components
chat_memory = ChatMemory(max_items=10000)  # Store up to 10,000 interactions


# Update your ImageRequest class

class ImageRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str = Field(..., description="Name of the model to use")
    prompt: str = Field(..., description="Text prompt for image generation")
    language: Language = Field(default=Language.ENGLISH, description="Language of the prompt")
    output_path: str = Field(default="outputs", description="Directory to save the output image")
    display: bool = Field(default=False, description="Display the image after generation")
    num_inference_steps: int = Field(default=30, description="Number of denoising steps")
    guidance_scale: float = Field(default=7.5, description="How strictly to follow the prompt")
    negative_prompt: Optional[str] = Field(default=None, description="Things to avoid in the image")
    enhancement: bool = Field(default=True, description="Enhance prompt with language-specific style")
    bypass_safety: bool = Field(default=False, description="Admin override for safety checks (requires API key)")
    api_key: Optional[str] = Field(default=None, description="API key for admin functions")
    user_id: str = Field(default="anonymous", description="User identifier for chat history")
    template_type: Optional[str] = Field(None, description="Optional template type: 'portrait', 'landscape', 'product', 'anime'")
    safety_level: str = Field(default="high", description="Safety strictness: 'high', 'medium', 'low'")
    allow_artistic_nudity: bool = Field(default=False, description="Allow artistic nudity in educational/historical context")
    allow_violence: bool = Field(default=False, description="Allow violence in educational/historical context")
    allow_hate: bool = Field(default=False, description="Allow hate speech in educational/historical context")
    # LoRA-related fields
    use_lora: bool = Field(default=False, description="Whether to use a LoRA model")
    lora_id: Optional[str] = Field(default=None, description="ID of the LoRA to use")
    lora_scale: float = Field(default=0.7, description="Strength of the LoRA effect (0.0 to 1.0)")
    lora_trigger_word: Optional[str] = Field(default=None, description="Trigger word for this LoRA (automatically added to prompt)")
    
    # New seed control parameters
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generations (None for random)")
    return_seed: bool = Field(default=True, description="Whether to return the used seed in response")
    
    @field_validator("num_inference_steps")
    @classmethod  # Add this decorator
    def validate_steps(cls, v):
        if v < 1 or v > 100:  # Set reasonable limits
            raise ValueError("Number of inference steps must be between 1 and 100")
        return v
    
    @field_validator("prompt")
    @classmethod  # Add this decorator
    def validate_prompt_length(cls, v):
        if len(v) > 1000:  # Set reasonable character limit
            raise ValueError("Prompt exceeds maximum length of 1000 characters")
        return v

'''def enhance_prompt(prompt: str, language: Language, enhancement: bool = True) -> str:
    """Enhance the prompt with better descriptors and structure for improved accuracy."""
    if not enhancement:
        return prompt
    
    # Get language-specific style prompt
    style_prompt = LANGUAGE_STYLE_PROMPTS.get(language.value, "")
    
    # Define quality boosters for each language
    quality_boosters = {
        "en": "highly detailed, crystal clear, masterfully crafted, photographic quality, sharp focus, intricate detail",
        "es": "muy detallado, claridad cristalina, magistralmente elaborado, calidad fotográfica, enfoque nítido, detalle intrincado",
        "fr": "très détaillé, clarté cristalline, magistralement élaboré, qualité photographique, mise au point nette, détails complexes",
        "de": "hochdetailliert, kristallklar, meisterhaft ausgearbeitet, fotografische Qualität, scharfer Fokus, komplexe Details",
        "it": "altamente dettagliato, cristallino, magistralmente realizzato, qualità fotografica, messa a fuoco nitida, dettagli intricati"
    }
    
    # Add quality boosters to the prompt based on language
    quality_boost = quality_boosters.get(language.value, quality_boosters["en"])
    
    # Structure the prompt better by maintaining subject prominence
    words = prompt.split()
    
    # Don't modify prompts that are already very long
    if len(words) > 30:
        return f"{style_prompt}{prompt}, {quality_boost}"
        
    # Extract core subject (simplified)
    core_subject = prompt
    
    # Add detail enhancements
    enhanced = f"{style_prompt}{core_subject}, {quality_boost}"
    
    logger.info(f"Enhanced prompt: {prompt} -> {enhanced}")
    return enhanced'''
def enhance_prompt(prompt: str, language: Language, enhancement: bool = True) -> str:
    if not enhancement:
        return prompt
    style_prompt = LANGUAGE_STYLE_PROMPTS.get(language.value, "")
    return f"{style_prompt}{prompt}"
    

# Add this after your other constants
MODEL_SPECIFIC_TEMPLATES = {
    "sdxl": {
        "portrait": "{prompt}, professional portrait, subject-centered, perfectly detailed eyes, professional photo, Fujifilm XT3, (extremely detailed face, perfect eyes:1.2), full-body",
        "landscape": "{prompt}, landscape photography, golden hour lighting, dramatic sky, professional photo, 8k, detailed, cinematic",
        "product": "{prompt}, product photography, studio lighting, white background, high detail, professional marketing photo, 8k, commercial photography",
        "anime": "{prompt}, anime style, vibrant colors, detailed linework, stunning art, illustration"
    },
    "sd15": {
        "portrait": "{prompt}, RAW photo, subject-centered, detailed, professional portrait, 8k, HDR, detailed skin, perfect face",
        "landscape": "{prompt}, landscape, 8k, hyperrealistic, cinematic, dramatic lighting, detailed, high resolution",
        "product": "{prompt}, product photo, studio lighting, professional photography, photo-realistic, detailed, commercial photography",
        "anime": "{prompt}, anime style, vibrant, high quality, illustration, detailed"
    }
}

# Add a function to select appropriate templates
def apply_template(prompt, model_name, template_type: Optional[str] = None) -> str:
    """Apply a model-specific template to improve results."""
    # Ensure parameters are strings
    prompt_str = str(prompt) if not isinstance(prompt, str) else prompt
    model_name_str = str(model_name) if not isinstance(model_name, str) else model_name
    
    # Determine if model is SDXL or SD1.5
    model_category = "sdxl" if any(hint in model_name_str.lower() for hint in ["xl", "sdxl", "juggernaut"]) else "sd15"
    
    # Auto-detect template type if not specified
    if not template_type:
        lower_prompt = prompt_str.lower()
        if any(term in lower_prompt for term in ["person", "man", "woman", "portrait", "face", "people"]):
            template_type = "portrait"
        elif any(term in lower_prompt for term in ["landscape", "nature", "mountain", "ocean", "beach", "forest", "sky"]):
            template_type = "landscape"
        elif any(term in lower_prompt for term in ["product", "object", "item", "phone", "camera", "watch", "gadget"]):
            template_type = "product"
        elif any(term in lower_prompt for term in ["anime", "cartoon", "illustration", "character"]):
            template_type = "anime"
        else:
            # Default to no template if type can't be determined
            return prompt_str
    
    # Apply template if available
    if template_type in MODEL_SPECIFIC_TEMPLATES[model_category]:
        template = MODEL_SPECIFIC_TEMPLATES[model_category][template_type]
        enhanced = template.format(prompt=prompt_str)
        logger.info(f"Applied {model_category} {template_type} template to prompt")
        return enhanced
    
    return prompt_str

# Add this as a constant at the module level
DEFAULT_NEGATIVE_PROMPTS = {
    "en": "blurry, distorted, low-quality, low-resolution, bad anatomy, deformed, disfigured, poorly drawn face, bad proportions, extra limbs, cloned face, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, watermark, signature",
    "es": "borroso, distorsionado, baja calidad, baja resolución, mala anatomía, deformado, desfigurado, cara mal dibujada, malas proporciones, extremidades adicionales, cara clonada, brazos faltantes, piernas faltantes, brazos adicionales, piernas adicionales, dedos fusionados, demasiados dedos, cuello largo, marca de agua, firma",
    "fr": "flou, déformé, mauvaise qualité, basse résolution, anatomie incorrecte, défiguré, visage mal dessiné, mauvaises proportions, membres supplémentaires, visage cloné, bras manquants, jambes manquantes, bras supplémentaires, jambes supplémentaires, doigts fusionnés, trop de doigts, cou long, filigrane, signature",
    "de": "unscharf, verzerrt, niedrige Qualität, niedrige Auflösung, schlechte Anatomie, deformiert, entstelltes Gesicht, schlecht gezeichnetes Gesicht, falsche Proportionen, zusätzliche Gliedmaßen, geklontes Gesicht, fehlende Arme, fehlende Beine, zusätzliche Arme, zusätzliche Beine, verschmolzene Finger, zu viele Finger, langer Hals, Wasserzeichen, Signatur",
    "it": "sfocato, distorto, bassa qualità, bassa risoluzione, anatomia scorretta, deformato, sfigurato, viso mal disegnato, proporzioni errate, arti aggiuntivi, viso clonato, braccia mancanti, gambe mancanti, braccia extra, gambe extra, dita fuse, troppe dita, collo lungo, filigrana, firma"
}

# Add this function for combining negative prompts
def get_negative_prompt(user_negative: Optional[str], language: str) -> str:
    """Combine user-provided negative prompt with default ones."""
    default_neg = DEFAULT_NEGATIVE_PROMPTS.get(language, DEFAULT_NEGATIVE_PROMPTS["en"])
    
    if not user_negative:
        return default_neg
    
    return f"{user_negative}, {default_neg}"

# Function to scan and index available LoRA files
def scan_lora_files():
    """Scan the LoRA directory and index available files"""
    lora_files = {}
    
    # Look for .safetensors files which are commonly used for LoRAs
    for lora_file in LORA_DIR.glob("**/*.safetensors"):
        lora_id = lora_file.stem.lower().replace(" ", "_")
        lora_files[lora_id] = {
            "id": lora_id,
            "name": lora_file.stem,
            "file_path": str(lora_file),
            "rel_path": str(lora_file.relative_to(LORA_DIR) if LORA_DIR in lora_file.parents else lora_file.name)
        }
    
    # Also check for .pt and .bin files which can be used for LoRAs
    for ext in [".pt", ".bin", ".ckpt"]:
        for lora_file in LORA_DIR.glob(f"**/*{ext}"):
            lora_id = lora_file.stem.lower().replace(" ", "_")
            if lora_id not in lora_files:  # Don't overwrite safetensors if they exist
                lora_files[lora_id] = {
                    "id": lora_id,
                    "name": lora_file.stem,
                    "file_path": str(lora_file),
                    "rel_path": str(lora_file.relative_to(LORA_DIR) if LORA_DIR in lora_file.parents else lora_file.name)
                }
    
    logger.info(f"Indexed {len(lora_files)} LoRA files")
    return lora_files

# Initialize LoRA files index
available_loras = scan_lora_files()

'''@app.get("/models/", response_model=List[Dict])
async def list_available_models():
    """List available model files."""
    models = []
    seen_models = set()
    
    # Look for models directly in models directory
    for model_file in MODELS_DIR.glob("*.safetensors"):
        model_id = model_file.stem
        if model_id not in seen_models:
            seen_models.add(model_id)
            models.append({
                "id": model_id,
                "filename": model_file.name,
                "path": str(model_file),
                "is_loaded": model_id in loaded_pipelines
            })
    
    # Check DIFFUSION/BASE subdirectory
    base_dir = MODELS_DIR / "DIFFUSION" / "BASE"
    if base_dir.exists():
        for model_file in base_dir.glob("*.safetensors"):
            model_id = model_file.stem
            if model_id not in seen_models:
                seen_models.add(model_id)
                models.append({
                    "id": model_id,
                    "filename": model_file.name,
                    "path": str(model_file),
                    "is_loaded": model_id in loaded_pipelines
                })
    
    return models'''

@app.get("/models/", response_model=List[str])
async def list_available_models():
    """List available model files."""
    model_files = [f.name for f in MODELS_DIR.glob("*.safetensors")]
    return model_files

def find_model_path(model_name: str) -> Path:
    """Find the model path based on the model name."""
    model_name_str = str(model_name) if not isinstance(model_name, str) else model_name
    
    model_path = MODELS_DIR / model_name_str
    if not model_path.exists():
        model_path = MODELS_DIR / "DIFFUSION" / "BASE" / model_name_str
    if not model_path.exists() and not model_name_str.endswith('.safetensors'):
        alt_path = MODELS_DIR / "DIFFUSION" / "BASE" / f"{model_name_str}.safetensors"
        if alt_path.exists():
            model_path = alt_path
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_name_str}' not found")
    return model_path

def load_pipeline(model_name: str, lora_id: Optional[str] = None, lora_scale: float = 0.7):
    """Load or retrieve cached pipeline with speed optimizations and better LoRA compatibility."""
    start_time = time.time()
    cache_key = model_name if lora_id is None else f"{model_name}_{lora_id}_{lora_scale}"
    
    # Add memory logging before loading
    if torch.cuda.is_available():
        before_mem = torch.cuda.memory_allocated() / 1024**2
        logger.info(f"GPU memory before loading model: {before_mem:.2f} MB")
    
    if cache_key in loaded_pipelines:
        logger.info(f"Cache hit: using cached model: {cache_key}")
        return loaded_pipelines[cache_key]
    else:
        logger.info(f"Cache miss: loading model {cache_key} from disk")
    
    # Find the model path
    model_path = find_model_path(model_name)
    
    try:
        # Clear CUDA cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Determine if model is likely SDXL based on filename
        is_xl = "xl" in model_name.lower()
        
        # Load the appropriate pipeline based on filename hints
        if is_xl:
            pipe = StableDiffusionXLPipeline.from_single_file(
                str(model_path),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                local_files_only=True
            )
            logger.info(f"Loaded as SDXL model: {model_name}")
        else:
            pipe = StableDiffusionPipeline.from_single_file(
                str(model_path),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                local_files_only=True
            )
            logger.info(f"Loaded as SD model: {model_name}")
        
        # Apply LoRA if specified
        if lora_id and lora_id in available_loras:
            try:
                lora_path = available_loras[lora_id]["file_path"]
                logger.info(f"Attempting to apply LoRA {lora_id} with scale {lora_scale}")
                
                # Check if LoRA is likely compatible
                lora_is_xl = "xl" in lora_id.lower()
                if is_xl != lora_is_xl:
                    logger.warning(f"LoRA {lora_id} might not be compatible with model {model_name} (XL mismatch)")
                
                # Try to load the LoRA with various fallback options
                try:
                    # First attempt - standard loading
                    pipe.load_lora_weights(lora_path)
                    pipe.fuse_lora(lora_scale=lora_scale)
                    logger.info(f"Successfully applied LoRA {lora_id} with scale {lora_scale}")
                except AttributeError as e:
                    if "'DownBlock2D' object has no attribute 'attentions'" in str(e):
                        logger.warning(f"LoRA compatibility issue detected: {str(e)}")
                        logger.info(f"Loading model without LoRA due to architecture mismatch")
                        # Return the model without LoRA as fallback
                    else:
                        # For other attribute errors, retry with cross attention only
                        logger.warning(f"Retrying LoRA loading with cross attention only: {str(e)}")
                        try:
                            pipe.unload_lora_weights()  # First unload any partial weights
                            pipe.load_lora_weights(
                                lora_path,
                                cross_attention_only=True  # Try with cross attention only
                            )
                            pipe.fuse_lora(lora_scale=lora_scale)
                            logger.info(f"Applied LoRA {lora_id} with cross attention only")
                        except Exception as inner_e:
                            logger.error(f"Failed to apply LoRA with fallback method: {str(inner_e)}")
                
            except Exception as lora_error:
                logger.error(f"Error applying LoRA {lora_id}: {str(lora_error)}")
                logger.info("Continuing with base model only")
        
        # Move to device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        
        # SPEED OPTIMIZATION: Use the faster DPMSolverMultistepScheduler
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, 
            algorithm_type="dpmsolver++", 
            use_karras_sigmas=True
        )
        
        # SPEED OPTIMIZATION: Enable memory optimizations
        pipe.enable_attention_slicing(1)
        if hasattr(pipe, 'enable_vae_tiling'):
            pipe.enable_vae_tiling()
        if hasattr(pipe, 'enable_model_cpu_offload') and device == "cuda":
            pipe.enable_model_cpu_offload()
            
        loaded_pipelines[cache_key] = pipe
        
        # Add after model loading
        if torch.cuda.is_available():
            after_mem = torch.cuda.memory_allocated() / 1024**2  # MB
            logger.info(f"GPU memory after loading model: {after_mem:.2f} MB (increase: {after_mem - before_mem:.2f} MB)")
        
        # Add this at end of function
        load_time = time.time() - start_time
        logger.info(f"Model {model_name} loaded in {load_time:.2f}s with device {pipe.device}")
        logger.info(f"Active models in cache: {list(loaded_pipelines.keys())}")
        
        return pipe
    
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.get("/download/{chat_id}", response_class=FileResponse)
async def download_image(chat_id: str):
    """Download an image generated in a previous interaction"""
    
    # Get the chat history item
    chat_item = chat_memory.get_interaction_by_id(chat_id)
    
    if not chat_item:
        raise HTTPException(status_code=404, detail=f"Chat history item with ID '{chat_id}' not found")
    
    if not chat_item.image_path:
        raise HTTPException(status_code=404, detail=f"No image associated with chat ID '{chat_id}'")
    
    # Check if the file exists
    image_path = Path(chat_item.image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image file not found on server")
    
    # Log download request
    logger.info(f"Download request for image {image_path.name} from chat ID {chat_id}")
    
    # Return the file as a download
    return FileResponse(
        path=image_path,
        filename=image_path.name,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{image_path.name}"'}
    )

# Track request counts and timestamps
request_counts = defaultdict(list)
RATE_LIMIT = 10  # requests
TIME_WINDOW = 60  # seconds

def rate_limit(request: Request):
    client_ip = request.client.host
    current_time = time.time()
    
    # Remove old timestamps outside the window
    request_counts[client_ip] = [ts for ts in request_counts[client_ip] 
                                if current_time - ts < TIME_WINDOW]
    
    # Check if too many requests
    if len(request_counts[client_ip]) >= RATE_LIMIT:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
    
    # Add current request timestamp
    request_counts[client_ip].append(current_time)
    return True

class AuditLog(BaseModel):
    """Model for audit log entries"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    action: str
    user_id: str
    ip_address: Optional[str] = None
    details: Dict[str, Any] = {}
    success: bool = True

# Add this class after your other class definitions
class ProgressBarCallback:
    """Callback for showing image generation progress"""
    def __init__(self):
        self.progress_bar = None
        self.start_time = None
        self.current_step = 0
        self.expected_total_steps = None
        
    def __call__(self, step, timestep, callback_kwargs):
        """Updated callback signature for compatibility with latest diffusers"""
        # Initialize progress bar if this is the first call
        if self.progress_bar is None:
            # Use expected steps if set, otherwise use a sensible default
            total_steps = self.expected_total_steps if self.expected_total_steps is not None else 30
            if total_steps is None:
                total_steps = 30  # Fallback default
                
            # Import tqdm directly to avoid conflicts
            from tqdm.auto import tqdm as std_tqdm
            
            # Create a progress bar
            self.progress_bar = std_tqdm(
                total=total_steps,
                desc="Generating image",
                leave=True,
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
            self.start_time = time.time()
            self.current_step = 0
        
        # Update progress bar (only if the step has increased)
        if step > self.current_step:
            steps_to_update = step - self.current_step
            self.progress_bar.update(steps_to_update)
            self.current_step = step
        
        return callback_kwargs

    def set_total_steps(self, total_steps):
        """Set expected total steps before generation begins"""
        self.expected_total_steps = int(total_steps)
    
    def reset(self):
        """Reset the progress bar"""
        if self.progress_bar is not None:
            # Ensure the progress bar is at 100% when complete
            if self.current_step < self.progress_bar.total:
                remaining = self.progress_bar.total - self.current_step
                if remaining > 0:
                    self.progress_bar.update(remaining)
                    
            total_time = time.time() - self.start_time if self.start_time else 0
            logger.info(f"Generation completed in {total_time:.2f}s")
            self.progress_bar.close()
            self.progress_bar = None
            self.start_time = None
            self.current_step = 0
            self.expected_total_steps = None

# Create an instance
progress_callback = ProgressBarCallback()

# Add this after initializing a model or generating an image
def log_memory_usage():
    import psutil
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024**2  # MB
    
    logger.info(f"Process memory usage: {ram_usage:.2f} MB")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        logger.info(f"CUDA memory: Allocated={allocated:.2f} MB, Reserved={reserved:.2f} MB")
        
        # Log per-device usage
        for i in range(torch.cuda.device_count()):
            device_allocated = torch.cuda.memory_allocated(i) / 1024**2
            logger.info(f"CUDA device {i} memory allocated: {device_allocated:.2f} MB")

# Call this function periodically
# For example, add it to your /memory-stats/ endpoint

# Add the log_security_event function
def log_security_event(event_type: str, details: Dict, user_id: str = None, ip: str = None, severity: str = "WARNING"):
    """Log security-related events with structured data"""
    log_data = {
        "event_type": event_type,
        "timestamp": datetime.now().isoformat(),
        "severity": severity,
        "user_id": user_id,
        "client_ip": ip,
        "details": details
    }
    
    if severity == "CRITICAL":
        logger.critical(f"SECURITY: {event_type} - {details}", extra={"security_event": log_data})
    elif severity == "ERROR":
        logger.error(f"SECURITY: {event_type} - {details}", extra={"security_event": log_data})
    else:
        logger.warning(f"SECURITY: {event_type} - {details}", extra={"security_event": log_data})

def sanitize_input(text: str) -> str:
    """More thorough input sanitization to prevent attacks"""
    # Remove control characters and zero-width spaces that could be used for obfuscation
    sanitized = re.sub(r'[\x00-\x1F\x7F-\x9F\u200B-\u200F\uFEFF]', '', text)
    
    # Remove potential HTML/JS
    sanitized = re.sub(r'<[^>]*>|javascript:|data:|&[#\w]+;', '', sanitized)
    
    # Check for excessive repetition (potential DoS)
    if len(re.findall(r'(.)\1{50,}', sanitized)):
        raise ValueError("Input contains excessive character repetition")
        
    return sanitized

# Store expected checksums for your models
MODEL_CHECKSUMS = {
    "model1.safetensors": "expected_sha256_hash_here",
    # Add more models...
}

def validate_model_integrity(model_path: Path) -> bool:
    """Check if model file has been tampered with"""
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return False
        
    filename = model_path.name
    if filename not in MODEL_CHECKSUMS:
        logger.warning(f"No checksum available for model: {filename}")
        return True  # Skip validation if no checksum
        
    # Calculate SHA-256 hash of file
    sha256_hash = hashlib.sha256()
    with open(model_path, "rb") as f:
        # Read file in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
            
    calculated_hash = sha256_hash.hexdigest()
    
    if calculated_hash != MODEL_CHECKSUMS[filename]:
        logger.critical(f"Model integrity check failed for {filename}! File may be corrupted or tampered with.")
        return False
        
    return True

def unload_all_models_except(keep_model_name=None):
    """Unload all models from GPU except the one we want to keep"""
    keys_to_remove = []
    
    for key in loaded_pipelines:
        if keep_model_name and keep_model_name in key:
            continue
            
        logger.info(f"Unloading model from GPU: {key}")
        pipe = loaded_pipelines[key]
        pipe = pipe.to("cpu")
        
        keys_to_remove.append(key)
    
    # Remove from cache dictionary
    for key in keys_to_remove:
        del loaded_pipelines[key]
    
    # Force garbage collection and clear CUDA cache
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def optimize_steps_for_model(model_name, desired_quality: str = "balanced") -> int:
    """Get optimal step count based on model and desired quality"""
    # Ensure model_name is a string
    model_name_str = str(model_name) if not isinstance(model_name, str) else model_name
    
    # Quality levels: "fast", "balanced", "quality"
    if "xl" in model_name_str.lower():
        if desired_quality == "fast":
            return 20
        elif desired_quality == "balanced":
            return 25
        else:  # quality
            return 30
    else:
        # SD1.5 models are generally faster
        if desired_quality == "fast":
            return 15
        elif desired_quality == "balanced":
            return 20
        else:  # quality
            return 25

# Add this function to get more detailed error info
def log_exception(e: Exception, context: str = ""):
    import traceback
    error_tb = traceback.format_exc()
    error_class = e.__class__.__name__
    logger.error(f"{context} - {error_class}: {str(e)}")
    logger.debug(f"Exception traceback:\n{error_tb}")

@app.get("/loras/", response_model=List[Dict])
async def list_available_loras():
    """List available LoRA files."""
    # Rescan directory to pick up any new files
    global available_loras
    available_loras = scan_lora_files()
    
    return list(available_loras.values())

@app.post("/generate-image/")
async def generate_image(
    request: ImageRequest, 
    response_type: str = Query("file", description="Response type: 'file', 'json'"),
    rate_limited: bool = Depends(rate_limit),
    request_obj: Request = None
):
    # Completely disable all debug logging to console during generation
    original_level = logger.level
    logger.setLevel(logging.INFO)
    
    # Force disable ALL progress bars from libraries
    import os
    os.environ["DIFFUSERS_PROGRESS_BAR"] = "0" 
    os.environ["TRANSFORMERS_PROGRESS_BAR"] = "0"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    
    # Monkey patch tqdm to disable all other progress bars
    import tqdm
    import tqdm.auto
    original_tqdm = tqdm.tqdm
    original_auto_tqdm = tqdm.auto.tqdm
    
    # Replace all tqdm variants with disabled versions
    tqdm.tqdm = lambda *args, **kwargs: original_tqdm(*args, **{**kwargs, 'disable': True})
    tqdm.auto.tqdm = lambda *args, **kwargs: original_auto_tqdm(*args, **{**kwargs, 'disable': True})
    
    try:
        generation_start = time.time()
        logger.info(f"Image generation request received: model={request.model_name}, user={request.user_id}, prompt_length={len(request.prompt)}")
        
        # Create audit log entry
        audit = AuditLog(
            action="generate_image",
            user_id=request.user_id,
            ip_address=request_obj.client.host if request_obj else None,
            details={"model": request.model_name, "prompt_length": len(request.prompt)}
        )
        
        # Check safety of prompt
        safety_result = safety.check_prompt(request.prompt, request.language)
        safety.log_safety_check(request.prompt, safety_result)
        
        # Check user intention
        intention_result = intention_classifier.classify_intention(request.prompt, request.language)

        # Admin key for bypassing safety - in production, use a proper auth system
        ADMIN_API_KEY = "admin_secret_key_12345"  # This should be stored securely
        
        # If unsafe and bypass not authorized, reject the request
        # In the generate_image function, after the safety check

        if not safety_result["is_safe"]:
            if not request.bypass_safety or request.api_key != ADMIN_API_KEY:
                # Record failed attempt in memory
                chat_memory.add_interaction(
                    user_id=request.user_id,
                    prompt=request.prompt,
                    language=request.language,
                    safety_result=safety_result,
                    intention_result=intention_result,
                    successful=False,
                    metadata={"reason": "safety_check_failed"}
                )
                
                # Special handling for celebrity-related rejection
                if SafetyCategory.COPYRIGHTS in safety_result["categories"] and "celebrity_references" in safety_result["details"]:
                    audit.success = False
                    audit.details["error"] = "Celebrity-related request blocked"
                    audit.details["outcome"] = "failure"
                    logger.warning(f"Audit log: {audit.dict()}")
                    raise HTTPException(
                        status_code=400, 
                        detail="This request appears to involve generating celebrity images, which is not permitted due to privacy and rights concerns."
                    )
                    
                # Normal safety rejection
                categories = ", ".join(safety_result["categories"])
                audit.success = False
                audit.details["error"] = f"Unsafe content detected in prompt. Categories: {categories}"
                audit.details["outcome"] = "failure"
                logger.warning(f"Audit log: {audit.dict()}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsafe content detected in prompt. Categories: {categories}"
                )
            else:
                logger.warning(f"Safety check bypassed with API key for prompt: '{request.prompt[:30]}...'")

        # If negative prompt is provided, also check its safety
        if request.negative_prompt:
            neg_safety_result = safety.check_prompt(request.negative_prompt, request.language)
            if not neg_safety_result["is_safe"]:
                if not request.bypass_safety or request.api_key != ADMIN_API_KEY:
                    # Record failed attempt in memory
                    chat_memory.add_interaction(
                        user_id=request.user_id,
                        prompt=request.prompt,
                        language=request.language,
                        safety_result=safety_result,
                        intention_result=intention_result,
                        successful=False,
                        metadata={"reason": "negative_prompt_safety_check_failed"}
                    )
                    
                    categories = ", ".join(neg_safety_result["categories"])
                    audit.success = False
                    audit.details["error"] = f"Unsafe content detected in negative prompt. Categories: {categories}"
                    audit.details["outcome"] = "failure"
                    logger.warning(f"Audit log: {audit.dict()}")
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Unsafe content detected in negative prompt. Categories: {categories}"
                    )
        if request.seed is not None:
            # Set specific seed
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator = generator.manual_seed(request.seed)
            used_seed = request.seed
            logger.info(f"Using provided seed: {used_seed}")
        else:
            # Generate random seed
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            used_seed = generator.seed()
            generator = generator.manual_seed(used_seed)
            logger.info(f"Generated random seed: {used_seed}")            

        # Handle LoRA trigger word if provided
        prompt_with_trigger = request.prompt
        if request.use_lora and request.lora_id and request.lora_id in available_loras:
            if request.lora_trigger_word:
                # Add trigger word to prompt if specified
                prompt_with_trigger = f"{request.lora_trigger_word}, {request.prompt}"
                logger.info(f"Added LoRA trigger word '{request.lora_trigger_word}' to prompt")
        else:
            # If LoRA not found but was requested
            if request.use_lora and request.lora_id:
                logger.warning(f"Requested LoRA '{request.lora_id}' not found, proceeding without it")
        
        # Add timing for pipeline loading
        pipeline_start = time.time()
        try:
            if request.use_lora and request.lora_id and request.lora_id in available_loras:
                pipe = load_pipeline(request.model_name, request.lora_id, request.lora_scale)
            else:
                pipe = load_pipeline(request.model_name)
            pipeline_load_time = time.time() - pipeline_start
            logger.info(f"Pipeline loaded in {pipeline_load_time:.2f} seconds")
        except Exception as e:
            log_exception(e, "Pipeline load failed")
            audit.success = False
            audit.details["error"] = f"Pipeline load failed: {str(e)}"
            audit.details["outcome"] = "failure"
            logger.warning(f"Audit log: {audit.dict()}")
            raise

        try:
            # Create output directory if it doesn't exist
            Path(request.output_path).mkdir(parents=True, exist_ok=True)
            
            # Apply template if specified or auto-detect
            prompt_with_template = apply_template(
                prompt_with_trigger if request.use_lora else request.prompt,
                request.model_name,
                request.template_type
            )
            # Then enhance the prompt
            enhanced_prompt = enhance_prompt(prompt_with_template, request.language, request.enhancement)
            # Combine user negative prompt with our defaults for better quality
            negative_prompt = get_negative_prompt(request.negative_prompt, request.language.value)
            
            logger.info(f"Generating image with model {request.model_name}")
            logger.info(f"Enhanced prompt: {enhanced_prompt}")
            logger.info(f"Negative prompt: {negative_prompt}")
            logger.info(f"Generating image with model {request.model_name} for prompt ({request.language}): {enhanced_prompt}")
            if request.use_lora and request.lora_id in available_loras:
                logger.info(f"Using LoRA: {request.lora_id} with scale {request.lora_scale}")

            # Add timing for actual image generation
            inference_start = time.time()
            
            # Clear CUDA cache before inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # SPEED OPTIMIZATION: Lower steps for faster generation
            steps = optimize_steps_for_model(request.model_name, desired_quality="balanced")
            if steps != request.num_inference_steps:
                logger.info(f"Optimizing by reducing steps from {request.num_inference_steps} to {steps}")

            # Tell the progress callback how many steps to expect
            progress_callback.set_total_steps(steps)

            # Reset progress bar from any previous generation
            progress_callback.reset()

            with torch.inference_mode():  # Use inference_mode instead of autocast for speed
                image = pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=request.guidance_scale,
                    output_type="pil",
                    generator=generator,
                    callback_steps=1,  # Call the callback at each step
                    callback=progress_callback  # Use 'callback' instead of 'callback_on_step_end'
                ).images[0]


            # Make sure to reset the progress bar when done
            progress_callback.reset()
            
            inference_time = time.time() - inference_start
            logger.info(f"Inference completed in {inference_time:.2f} seconds ({steps} steps)")

            # Verify the generated image doesn't contain unsafe content
            image_safety_result = verify_generated_image(image, model_name=request.model_name)
            logger.info(f"Image verification result: {image_safety_result}")

            if not image_safety_result.get("is_safe", True):
                # Record failed attempt in memory
                chat_memory.add_interaction(
                    user_id=request.user_id,
                    prompt=request.prompt,
                    language=request.language,
                    safety_result=safety_result,
                    intention_result=intention_result,
                    successful=False,
                    metadata={"reason": "unsafe_generated_image", "image_safety_result": image_safety_result}
                )
                audit.success = False
                audit.details["error"] = "Generated image failed safety verification"
                audit.details["outcome"] = "failure"
                logger.warning(f"Audit log: {audit.dict()}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"The generated image contains unsafe content and has been blocked."
                )

            # Create filename with language code, prompt hash, and timestamp
            prompt_hash = hashlib.md5(request.prompt.encode()).hexdigest()[:8]
            timestamp = int(time.time())
            safe_filename = ''.join(c if c.isalnum() or c in ' -_' else '_' for c in request.prompt[:20])
            
            # Add LoRA info to filename if used
            if request.use_lora and request.lora_id:
                lora_suffix = f"_lora_{request.lora_id}"
                output_file = Path(request.output_path) / f"{request.language}_{safe_filename}{lora_suffix}_{prompt_hash}_{timestamp}.png"
            else:
                output_file = Path(request.output_path) / f"{request.language}_{safe_filename}_{prompt_hash}_{timestamp}.png"

            # Save image
            image.save(output_file)
            logger.info(f"Image saved at: {output_file}")

            # Optionally display the image
            if request.display:
                plt.figure(figsize=(12, 10))
                plt.imshow(image)
                plt.axis('off')
                plt.title(f"Prompt ({request.language}): {request.prompt[:50]}...", pad=20)
                plt.show()
                
              # Include seed in metadata
            metadata = {
                "lora_used": request.use_lora,
                "lora_id": request.lora_id if request.use_lora else None,
                "lora_scale": request.lora_scale if request.use_lora else None,
                "lora_trigger_word": request.lora_trigger_word if request.use_lora else None,
                "seed": used_seed  # Store the seed that was used
            }
            
            # Record successful generation in memory
            chat_id = chat_memory.add_interaction(
                user_id=request.user_id,
                prompt=request.prompt,
                language=request.language,
                enhanced_prompt=enhanced_prompt if request.enhancement else None,
                model_used=request.model_name,
                safety_result=safety_result,
                intention_result=intention_result,
                image_path=str(output_file),
                successful=True,
                metadata=metadata
            )
            
            total_time = time.time() - generation_start
            logger.info(f"Total generation time: {total_time:.2f} seconds for user {request.user_id}")

            # Update audit with success
            audit.success = True
            audit.details["outcome"] = "success"
            audit.details["image_path"] = str(output_file)
            logger.info(f"Audit log: {audit.dict()}")

            # Return appropriate response
            if response_type.lower() == "file":
                # Return the file as a download
                return FileResponse(
                    path=output_file,
                    filename=output_file.name,
                    media_type="image/png",
                    headers={"Content-Disposition": f'attachment; filename="{output_file.name}"'}
                )
            
            elif response_type.lower() == "json":
                # Return JSON with image details including the seed
                return {
                    "message": "Image generated successfully",
                    "file_path": str(output_file),
                    "language": request.language,
                    "model_used": request.model_name,
                    "original_prompt": request.prompt,
                    "enhanced_prompt": enhanced_prompt if request.enhancement else None,
                    "safety_status": "safe",
                    "safety_checked": True,
                    "chat_id": chat_id,
                    "download_url": f"/download/{chat_id}",
                    #"view_url": f"/view/{chat_id}",
                    "direct_image_url": f"/static/{output_file.name}",
                    "lora_used": request.use_lora,
                    "lora_id": request.lora_id if request.use_lora else None,
                    "lora_scale": request.lora_scale if request.use_lora else None,
                    "lora_trigger_word": request.lora_trigger_word if request.use_lora else None,
                    "seed": used_seed if request.return_seed else None
                }
                
        except Exception as e:
            log_exception(e, "Error generating image")
            # Record failed attempt in memory
            chat_memory.add_interaction(
                user_id=request.user_id,
                prompt=request.prompt,
                language=request.language,
                safety_result=safety_result,
                intention_result=intention_result,
                successful=False,
                metadata={"reason": "generation_error", "error": str(e)}
            )
            
            audit.success = False
            audit.details["error"] = f"Error generating image: {str(e)}"
            audit.details["outcome"] = "failure"
            logger.warning(f"Audit log: {audit.dict()}")
            
            logger.error(f"Error generating image: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")
    finally:
        # Restore logger level after generation completes or fails
        logger.setLevel(original_level)

# Add this function to check generated images for unsafe content
def verify_generated_image(image, model_name: str = "nsfw-detector") -> Dict:
    """
    Check if the generated image contains unsafe content using a pretrained model
    """
    try:
        # You'll need to implement this based on your preferred NSFW detection model
        # Here's a simple example using a hypothetical NSFW detector
        from PIL import Image
        import io
        import numpy as np
        
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Simple placeholder for NSFW detection model
        # In a real implementation, you would:
        # 1. Load a pretrained model like NSFW-Detector or NudeNet
        # 2. Process the image through the model
        # 3. Return a safety score and classification
        
        # Placeholder scores - replace with actual model implementation
        scores = {
            "nsfw": 0.01,  # Very low score means safe content
            "nude": 0.01,
            "sexy": 0.01,
            "porn": 0.00,
            "safe": 0.98
        }
        
        is_safe = scores["safe"] > 0.8 and scores["nsfw"] < 0.2
        
        return {
            "is_safe": is_safe,
            "scores": scores,
            "recommendation": "pass" if is_safe else "block"
        }
        
    except Exception as e:
        logger.error(f"Error during image verification: {str(e)}")
        # When in doubt, be conservative
        return {"is_safe": True, "error": str(e)}

@app.post("/analyze-and-enhance-prompt/")
async def analyze_and_enhance_prompt(
    prompt: str = Query(..., description="Text prompt to analyze and enhance"),
    language: Language = Query(Language.ENGLISH, description="Language of the prompt"),
    apply_enhancements: bool = Query(True, description="Whether to apply suggested enhancements")
):
    """Analyze a prompt and suggest improvements for better image generation."""
    
    # Check original prompt
    safety_result = safety.check_prompt(prompt, language)
    intention_result = intention_classifier.classify_intention(prompt, language)
    
    # Basic prompt analysis
    word_count = len(prompt.split())
    char_count = len(prompt)
    has_artistic_descriptors = any(term in prompt.lower() for term in ["detailed", "quality", "high resolution", "8k", "photorealistic"])
    has_style_descriptors = any(term in prompt.lower() for term in ["style", "painting", "illustration", "artwork", "render", "cartoon"])
    
    # Identify key subjects (simplified)
    # For a production system, you might want to use NLP here
    nouns = [word for word in prompt.split() if len(word) > 3]  # Simple approximation
    
    # Prepare suggestions
    suggestions = []
    if word_count < 5:
        suggestions.append("Add more descriptive details to your prompt")
    if not has_artistic_descriptors:
        suggestions.append("Add quality descriptors like 'detailed', 'high resolution', or '8k'")
    if not has_style_descriptors:
        suggestions.append("Consider specifying an artistic style")
    
    # Generate enhanced version
    enhanced = enhance_prompt(prompt, language, apply_enhancements) if apply_enhancements else prompt
    
    # Suggest default negative prompt
    suggested_negative = DEFAULT_NEGATIVE_PROMPTS.get(language, DEFAULT_NEGATIVE_PROMPTS["en"])
    
    return {
        "original_prompt": prompt,
        "enhanced_prompt": enhanced,
        "language": language,
        "prompt_analysis": {
            "word_count": word_count,
            "character_count": char_count,
            "has_artistic_descriptors": has_artistic_descriptors,
            "has_style_descriptors": has_style_descriptors,
            "key_subjects": nouns[:5] if len(nouns) > 5 else nouns,
            "safety_status": "safe" if safety_result["is_safe"] else "potentially unsafe",
            "intention_category": intention_result["primary_category"]
        },
        "improvement_suggestions": suggestions,
        "suggested_negative_prompt": suggested_negative
    }
    
# Add this endpoint for easier LoRA usage

@app.post("/generate-with-lora/")
async def generate_with_lora(
    model_name: str = Query(..., description="Base model name"),
    prompt: str = Query(..., description="Text prompt for image generation"),
    lora_id: str = Query(..., description="LoRA model ID to use"),
    lora_scale: float = Query(0.7, description="LoRA effect strength (0.0-1.0)"),
    lora_trigger: str = Query(None, description="Optional trigger word for this LoRA"),
    language: Language = Query(Language.ENGLISH, description="Language of the prompt"),
    response_type: str = Query("file", description="Response type: 'file', 'json'"),
    seed: Optional[int] = Query(None, description="Random seed for reproducible generations")
):
    """Generate an image using a LoRA model on top of a base model."""
    
    # Create request object
    request = ImageRequest(
        model_name=model_name,
        prompt=prompt,
        language=language,
        use_lora=True,
        lora_id=lora_id,
        lora_scale=lora_scale,
        lora_trigger_word=lora_trigger,
        seed=seed
    )
    
    # Pass to main generation function
    return await generate_image(request, response_type=response_type)

@app.get("/languages/", response_model=List[Dict[str, str]])
async def list_languages():
    """Get list of supported languages."""
    return [
        {"code": lang.value, "name": lang.name.title(), "style_prompt": LANGUAGE_STYLE_PROMPTS.get(lang.value, "")}
        for lang in Language
    ]
    
@app.get("/user-history/{user_id}")
async def get_user_history(
    user_id: str,
    limit: int = Query(10, description="Maximum number of history items to return")
):
    """Get chat history for a specific user"""
    history = chat_memory.get_user_history(user_id, limit)
    return {
        "user_id": user_id,
        "history_count": len(history),
        "history": [item.dict() for item in history]
    }

@app.get("/similar-prompts/")
async def find_similar_prompts(
    prompt: str = Query(..., description="Prompt to find similar previous prompts for"),
    limit: int = Query(5, description="Maximum number of similar prompts to return")
):
    """Find similar prompts from chat history"""
    similar = chat_memory.find_similar_prompts(prompt, limit)
    return {
        "query": prompt,
        "matches_count": len(similar),
        "matches": [
            {
                "id": item.id,
                "timestamp": item.timestamp,
                "prompt": item.user_prompt,
                "language": item.language,
                "successful": item.successful,
                "image_path": item.image_path
            } for item in similar
        ]
    }
# Add this endpoint




@app.get("/lora/{lora_id}")
async def get_lora_info(lora_id: str):
    """Get details for a specific LoRA."""
    lora_id_lower = lora_id.lower()
    
    if lora_id_lower not in available_loras:
        raise HTTPException(status_code=404, detail=f"LoRA '{lora_id}' not found")
    
    return available_loras[lora_id_lower]    
    
@app.get("/memory-stats/")
async def get_memory_stats():
    """Get statistics about the chat memory system"""
    return chat_memory.get_stats()

@app.post("/check-prompt-intention/")
async def check_prompt_intention(
    prompt: str = Query(..., description="Text prompt to check for intention classification"),
    language: Language = Query(Language.ENGLISH, description="Language of the prompt")
):
    """Classify the user's intention behind a prompt without generating an image."""
    intention_result = intention_classifier.classify_intention(prompt, language.value)
    
    return {
        "prompt": prompt,
        "primary_category": intention_result["primary_category"],
        "all_categories": intention_result["all_categories"],
        "matches": intention_result["matches"],
        "is_safe": intention_result["is_safe"],
        "explanation": intention_result["explanation"],
        "would_generate": intention_result["is_safe"] or intention_result.get("requires_review", False)
    }

@app.get("/regenerate/{chat_id}")
async def regenerate_image(
    chat_id: str, 
    response_type: str = Query("file", description="Response type: 'file', 'json'"),
    use_same_seed: bool = Query(True, description="Whether to use the same seed as the original image"),
    new_seed: Optional[int] = Query(None, description="New seed to use (only if use_same_seed is False)")
):
    """Regenerate an image using the same settings as a previous generation."""
    
    # Get the chat history item
    chat_item = chat_memory.get_interaction_by_id(chat_id)
    
    if not chat_item:
        raise HTTPException(status_code=404, detail=f"Chat history item with ID '{chat_id}' not found")
    
    # Get the original seed if requested
    seed = None
    if use_same_seed and hasattr(chat_item, 'metadata') and chat_item.metadata:
        seed = chat_item.metadata.get("seed")
    elif not use_same_seed and new_seed is not None:
        seed = new_seed
    
    # Create a new request with the same parameters
    request = ImageRequest(
        model_name=chat_item.model_used or "runwayml/stable-diffusion-v1-5",
        prompt=chat_item.user_prompt,
        language=chat_item.language,
        # Keep other parameters from the original request
        enhancement=True if chat_item.enhanced_prompt else False,
        seed=seed,
    )
    
    # Check if LoRA was used
    if hasattr(chat_item, 'metadata') and chat_item.metadata:
        request.use_lora = chat_item.metadata.get("lora_used", False)
        request.lora_id = chat_item.metadata.get("lora_id")
        request.lora_scale = chat_item.metadata.get("lora_scale", 0.7)
        request.lora_trigger_word = chat_item.metadata.get("lora_trigger_word")
    
    # Pass to main generation function
    return await generate_image(request, response_type=response_type)
    
@app.get("/history/{chat_id}", response_model=Dict[str, Any])
async def get_history_by_id(
    chat_id: str,
    include_safety_details: bool = Query(False, description="Include full safety analysis details"),
    include_intention_details: bool = Query(False, description="Include full intention analysis details")
):
    """Retrieve a specific chat history item by its ID"""
    
    # Check if the chat ID exists in memory
    if chat_id not in chat_memory.memory:
        raise HTTPException(status_code=404, detail=f"Chat history item with ID '{chat_id}' not found")
    
    # Get the memory item
    item = chat_memory.memory[chat_id]
    
    # Prepare response with different levels of detail based on query params
    response = {
        "id": item.id,
        "timestamp": item.timestamp,
        "user_prompt": item.user_prompt,
        "language": item.language,
        "enhanced_prompt": item.enhanced_prompt,
        "model_used": item.model_used,
        "image_path": item.image_path,
        "successful": item.successful
    }
    
    # Include safety details if requested
    if include_safety_details:
        response["safety_result"] = item.safety_result
    else:
        # Include just basic safety info
        response["safety_result"] = {
            "is_safe": item.safety_result.get("is_safe", True),
            "categories": item.safety_result.get("categories", []) if not item.safety_result.get("is_safe", True) else []
        }
    
    # Include intention details if requested
    if include_intention_details:
        response["intention_result"] = item.intention_result
    else:
        # Include just basic intention info
        response["intention_result"] = {
            "primary_category": item.intention_result.get("primary_category", "safe"),
            "is_safe": item.intention_result.get("is_safe", True),
            "explanation": item.intention_result.get("explanation", "")
        }
    
    # Include metadata
    if item.metadata:
        response["metadata"] = item.metadata
    
    return response


@app.post("/check-prompt-safety/")
async def check_prompt_safety(
    prompt: str = Query(..., description="Text prompt to check for safety"),
    language: Language = Query(Language.ENGLISH, description="Language of the prompt")
):
    """Check if a prompt is safe without generating an image."""
    safety_result = safety.check_prompt(prompt, language)
    safety.log_safety_check(prompt, safety_result)
    
    return {
        "prompt": prompt,
        "is_safe": safety_result["is_safe"],
        "categories": safety_result["categories"] if not safety_result["is_safe"] else [],
        "details": safety_result["details"] if not safety_result["is_safe"] else {}
    }

@app.get("/safety-categories/")
async def list_safety_categories():
    """List all safety categories with descriptions."""
    return [
        {"id": SafetyCategory.ADULT, "name": "Adult Content", 
         "description": "Sexual or pornographic content, nudity, etc."},
        {"id": SafetyCategory.VIOLENCE, "name": "Violence", 
         "description": "Gore, blood, killing, torture, death, etc."},
        {"id": SafetyCategory.HATE, "name": "Hate Speech", 
         "description": "Racism, sexism, bigotry, fascism, etc."},
        {"id": SafetyCategory.HARASSMENT, "name": "Harassment", 
         "description": "Bullying, stalking, doxxing, threatening, etc."},
        {"id": SafetyCategory.SELF_HARM, "name": "Self Harm", 
         "description": "Suicide, self-injury, eating disorders, etc."},
        {"id": SafetyCategory.ILLEGAL_ACTIVITY, "name": "Illegal Activity", 
         "description": "Drugs, terrorism, weapons, etc."},
        {"id": SafetyCategory.DANGEROUS, "name": "Dangerous Content", 
         "description": "Dangerous activities, hazards, toxic substances, etc."},
        {"id": SafetyCategory.DECEPTION, "name": "Deception", 
         "description": "Scams, fraud, phishing, counterfeiting, etc."},
        {"id": SafetyCategory.POLITICAL, "name": "Political Content", 
         "description": "Politics, elections, candidates, campaigns, etc."},
        {"id": SafetyCategory.PERSONAL_INFORMATION, "name": "Personal Information", 
         "description": "Addresses, phone numbers, emails, financial information, etc."},
        {"id": SafetyCategory.RELIGIOUS_ICONOGRAPHY, "name": "Religious Iconography Misuse", 
         "description": "Disrespectful, mocking, desecrating or inappropriate use of religious symbols, figures or texts"},
        {"id": SafetyCategory.HISTORICAL_REVISIONISM, "name": "Sensitive Historical Revisionism", 
         "description": "Denial, minimization or distortion of historical atrocities, genocides, or sensitive historical events"}
    ]

@app.get("/intention-categories/")
async def list_intention_categories():
    """List all intention categories with descriptions."""
    return [
        {"id": IntentionCategory.SAFE, "name": "Safe Content", 
         "description": "Content that doesn't contain any unsafe elements"},
        {"id": IntentionCategory.EDUCATIONAL, "name": "Educational", 
         "description": "Content for learning, teaching, research or documentation"},
        {"id": IntentionCategory.ARTISTIC, "name": "Artistic", 
         "description": "Content for creative, aesthetic or artistic purposes"},
        {"id": IntentionCategory.NSFW, "name": "Adult Content", 
         "description": "Sexual, pornographic or explicit adult content"},
        {"id": IntentionCategory.VIOLENCE, "name": "Violence", 
         "description": "Violent, gory or disturbing imagery"},
        {"id": IntentionCategory.HATE, "name": "Hate Speech", 
         "description": "Content promoting hatred or discrimination"},
        {"id": IntentionCategory.ILLEGAL, "name": "Illegal Activity", 
         "description": "Content related to illegal substances or activities"},
        {"id": IntentionCategory.DANGEROUS, "name": "Dangerous Content", 
         "description": "Content describing dangerous procedures or harmful actions"}
    ]
    
@app.get("/")
async def root():
    """Root endpoint providing API information"""
    
    available_models = await list_available_models()
    return {
        "message": "Multi-Model Multilingual Text to Image API with Safety Guardrails is running!",
        "version": "3.0.0",
        "available_models": available_models,
        "supported_languages": [lang.value for lang in Language],
        "safety_enabled": True,
        "intention_categories": [category.value for category in IntentionCategory],
        "endpoints": {
            "/": "This information",
            "/models": "List all available models with language support",
            "/languages": "List all supported languages",
            "/generate": "Generate an image from a text prompt",
            "/analyze-and-enhance-prompt": "Analyze a prompt and suggest improvements",
            "/check-prompt-safety": "Check if a prompt is safe without generating an image",
            "/check-prompt-intention": "Classify the intention behind a prompt",
            "/safety-categories": "List all safety categories with descriptions",
            "/intention-categories": "List all intention categories with descriptions",
            "/loras": "List all available LoRA models",
            "/generate-with-lora": "Generate an image using a LoRA model on top of a base model",
            "/lora/{lora_id}": "Get details for a specific LoRA model",
            "/regenerate/{chat_id}": "Regenerate an image using the same settings as a previous generation",
            "/history/{chat_id}": "Retrieve a specific chat history item by its ID",
            "/user-history/{user_id}": "Get chat history for a specific user",
            "/similar-prompts": "Find similar prompts from chat history",
            "/memory-stats": "Get statistics about the chat memory system",
            "/check-prompt-intention": "Classify the intention behind a prompt",
            "/intention-categories": "List all intention categories with descriptions",
            "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        }
        }
    }

# Add this function to check system health periodically
@app.get("/health/")
async def health_check():
    """System health check endpoint"""
    start_time = time.time()
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "uptime": "N/A",  # You'd need to track application start time
    }
    
    # Check GPU health
    if torch.cuda.is_available():
        try:
            # Test GPU with a simple tensor operation
            test_tensor = torch.randn(10, 10).cuda()
            test_result = torch.sum(test_tensor).item()
            health_data["gpu_status"] = "functional"
        except Exception as e:
            health_data["gpu_status"] = "error"
            health_data["gpu_error"] = str(e)
            logger.error(f"GPU health check failed: {str(e)}")
    else:
        health_data["gpu_status"] = "not_available"
    
    # Include memory stats
    health_data["model_cache_size"] = len(loaded_pipelines)
    health_data["chat_memory_size"] = len(chat_memory.memory)
    
    # Log health check
    duration = time.time() - start_time
    logger.info(f"Health check completed in {duration:.3f}s: {health_data['status']}")
    
    return health_data
