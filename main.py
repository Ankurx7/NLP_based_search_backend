from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from elasticsearch import AsyncElasticsearch
import re
import os
import spacy
from typing import Dict, List, Optional, Any
import uvicorn
from dotenv import load_dotenv

load_dotenv()

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

app = FastAPI()

mongo_client = None
es_client = None

class SearchRequest(BaseModel):
    query: str

@app.on_event("startup")
async def startup_event():
    global mongo_client, es_client
    mongo_uri = os.getenv("MONGO_URI")
    if mongo_uri and mongo_uri != "your_mongodb_atlas_uri_here":
        try:
            mongo_client = AsyncIOMotorClient(mongo_uri)
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
    es_url = os.getenv("ELASTIC_URL")
    es_api_key = os.getenv("ELASTIC_API_KEY")
    if (es_url and es_url.strip() and es_url != "your_elasticsearch_url_here" and 
        es_api_key and es_api_key.strip() and es_api_key != "your_elasticsearch_api_key_here"):
        try:
            es_client = AsyncElasticsearch([es_url], api_key=es_api_key, verify_certs=True)
            await es_client.info()
            print("✅ Elasticsearch connected successfully")
        except Exception as e:
            print(f"Elasticsearch connection failed: {e}")
            es_client = None
    else:
        print("🔧 Elasticsearch not configured - using MongoDB fallback")

@app.on_event("shutdown")
async def shutdown_event():
    global mongo_client, es_client
    if mongo_client:
        mongo_client.close()
    if es_client:
        await es_client.close()

def parse_query_with_nlp(query: str) -> Dict[str, Any]:
    query_lower = query.lower()
    
    filters = {
        "category": None,
        "brand": None,
        "color": None,
        "price_max": None,
        "price_min": None,
        "gender": None,
        "keywords": [],
        "entities": {},
        "semantic_analysis": {}
    }
    
    brand_entities = {
        "nike": ["nike", "air jordan", "jordan brand", "swoosh"],
        "adidas": ["adidas", "three stripes", "trefoil", "adi"],
        "puma": ["puma", "cat", "fuma"],
        "reebok": ["reebok", "rbk"],
        "converse": ["converse", "chuck taylor", "all star", "chuck"],
        "vans": ["vans", "off the wall"],
        "new balance": ["new balance", "nb", "newbalance"],
        "mizuno": ["mizuno"],
        "asics": ["asics", "tiger"],
        "under armour": ["under armour", "ua", "underarmour"]
    }
    
    color_entities = {
        "black": ["black", "ebony", "charcoal", "dark", "jet black"],
        "white": ["white", "ivory", "cream", "off-white", "snow", "pearl"],
        "red": ["red", "crimson", "burgundy", "scarlet", "cherry", "rose red"],
        "blue": ["blue", "navy", "cobalt", "azure", "royal blue", "sky blue"],
        "green": ["green", "emerald", "olive", "forest", "lime"],
        "yellow": ["yellow", "gold", "golden", "amber", "sunshine"],
        "pink": ["pink", "rose", "magenta", "fuchsia", "blush"],
        "purple": ["purple", "violet", "lavender", "plum", "indigo"],
        "brown": ["brown", "tan", "beige", "khaki", "chocolate", "camel"],
        "grey": ["grey", "gray", "silver", "slate", "charcoal gray"],
        "orange": ["orange", "coral", "peach", "burnt orange"]
    }
    
    category_entities = {
        "sneakers": ["sneakers", "trainers", "athletic shoes", "sports shoes", "running shoes", "runners", "kicks"],
        "flats": ["flats", "ballet flats", "loafers", "slip-ons"],
        "boots": ["boots", "ankle boots", "combat boots", "winter boots", "hiking boots"],
        "sandals": ["sandals", "flip flops", "slippers", "slides"],
        "heels": ["heels", "high heels", "stilettos", "pumps", "wedges"],
        "casual shoes": ["casual", "everyday shoes", "walking shoes"],
        "formal shoes": ["formal", "dress shoes", "office shoes", "oxfords"]
    }
    
    gender_entities = {
        "women": ["women", "female", "girls", "ladies", "womens", "woman"],
        "men": ["men", "male", "boys", "mens", "guys", "man"],
        "unisex": ["unisex", "both", "all", "universal"]
    }
    
    if nlp:
        doc = nlp(query)
        
        entities_found = {}
        for ent in doc.ents:
            entities_found[ent.label_] = ent.text.lower()
        filters["entities"] = entities_found
        
        noun_phrases = []
        adjective_modifiers = []
        
        for token in doc:
            if token.pos_ == "NOUN":
                noun_phrase = token.text.lower()
                for child in token.children:
                    if child.pos_ == "ADJ":
                        adjective_modifiers.append((child.text.lower(), noun_phrase))
                        noun_phrase = f"{child.text.lower()} {noun_phrase}"
                noun_phrases.append(noun_phrase)
            
            if token.pos_ in ["NOUN", "ADJ", "PROPN"] and not token.is_stop and len(token.text) > 2:
                filters["keywords"].append(token.lemma_.lower())
        
        brand_confidence = {}
        for brand, variants in brand_entities.items():
            for variant in variants:
                if variant in query_lower:
                    context_score = 1.0
                    
                    shoe_terms = ["shoes", "sneakers", "boots", "footwear", "kicks"]
                    if any(term in query_lower for term in shoe_terms):
                        context_score += 0.5
                    
                    if variant == query_lower.strip() or f" {variant} " in f" {query_lower} ":
                        context_score += 0.3
                    
                    brand_confidence[brand] = brand_confidence.get(brand, 0) + context_score
        
        if brand_confidence:
            filters["brand"] = max(brand_confidence.items(), key=lambda x: x[1])[0]
        
        color_confidence = {}
        for color, variants in color_entities.items():
            for variant in variants:
                if variant in query_lower:
                    confidence = 1.0
                    for adj, noun in adjective_modifiers:
                        if variant in adj and any(shoe_word in noun for shoe_word in ["shoe", "sneaker", "boot"]):
                            confidence += 0.5
                    color_confidence[color] = color_confidence.get(color, 0) + confidence
        
        if color_confidence:
            filters["color"] = max(color_confidence.items(), key=lambda x: x[1])[0]
        
        category_confidence = {}
        for category, variants in category_entities.items():
            for variant in variants:
                if variant in query_lower:
                    confidence = 1.0
                    if variant in noun_phrases:
                        confidence += 0.3
                    category_confidence[category] = category_confidence.get(category, 0) + confidence
        
        if category_confidence:
            filters["category"] = max(category_confidence.items(), key=lambda x: x[1])[0]
        
        for gender, variants in gender_entities.items():
            if any(variant in query_lower for variant in variants):
                filters["gender"] = gender
                break
        
        if entities_found.get("MONEY"):
            money_text = entities_found["MONEY"]
            price_match = re.search(r'(\d+)', money_text)
            if price_match:
                filters["price_max"] = int(price_match.group(1))
        
        elif entities_found.get("CARDINAL"):
            cardinal_text = entities_found["CARDINAL"]
            print(f"🔢 Found CARDINAL: {cardinal_text}")
            
            under_patterns = [
                r'under\s+' + cardinal_text,
                r'below\s+' + cardinal_text,
                r'less\s+than\s+' + cardinal_text,
                r'<\s*' + cardinal_text,
                r'max\s+' + cardinal_text,
                r'budget\s+' + cardinal_text
            ]
            
            over_patterns = [
                r'over\s+' + cardinal_text,
                r'above\s+' + cardinal_text,
                r'more\s+than\s+' + cardinal_text,
                r'>\s*' + cardinal_text,
                r'min\s+' + cardinal_text,
                r'minimum\s+' + cardinal_text
            ]
            
            found_price = False
            for pattern in under_patterns:
                print(f"🔍 Testing under pattern: {pattern} against {query_lower}")
                if re.search(pattern, query_lower):
                    print(f"✅ Under pattern matched!")
                    try:
                        filters["price_max"] = int(cardinal_text)
                        print(f"💰 Set price_max to: {filters['price_max']}")
                        found_price = True
                        break
                    except ValueError:
                        print(f"❌ Could not convert {cardinal_text} to int")
                        pass
            
            if not found_price:
                for pattern in over_patterns:
                    print(f"🔍 Testing over pattern: {pattern} against {query_lower}")
                    if re.search(pattern, query_lower):
                        print(f"✅ Over pattern matched!")
                        try:
                            filters["price_min"] = int(cardinal_text)
                            print(f"💰 Set price_min to: {filters['price_min']}")
                            break
                        except ValueError:
                            print(f"❌ Could not convert {cardinal_text} to int")
                            pass
        
        if not filters["price_max"] and not filters["price_min"]:
            under_patterns = [
                r'under\s+(\d+)',
                r'below\s+(\d+)',
                r'less\s+than\s+(\d+)',
                r'<\s*(\d+)',
                r'max\s+(\d+)',
                r'budget\s+(\d+)'
            ]
            
            over_patterns = [
                r'over\s+(\d+)',
                r'above\s+(\d+)',
                r'more\s+than\s+(\d+)',
                r'>\s*(\d+)',
                r'min\s+(\d+)',
                r'minimum\s+(\d+)'
            ]
            
            for pattern in under_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    filters["price_max"] = int(match.group(1))
                    print(f"💰 Fallback set price_max to: {filters['price_max']}")
                    break
            
            if not filters["price_max"]:
                for pattern in over_patterns:
                    match = re.search(pattern, query_lower)
                    if match:
                        filters["price_min"] = int(match.group(1))
                        print(f"💰 Fallback set price_min to: {filters['price_min']}")
                        break
        
        filters["semantic_analysis"] = {
            "noun_phrases": noun_phrases,
            "adjective_modifiers": adjective_modifiers,
            "brand_confidence": brand_confidence,
            "color_confidence": color_confidence,
            "category_confidence": category_confidence
        }
    
    else:
        filters = parse_query_fallback(query_lower)
    
    return filters

def parse_query_fallback(query_lower: str) -> Dict[str, Any]:
    filters = {
        "category": None,
        "brand": None,
        "color": None,
        "price_max": None,
        "price_min": None,
        "gender": None,
        "keywords": []
    }
    
    words = [word.strip() for word in query_lower.split() if len(word.strip()) > 2]
    filters["keywords"] = words
    
    category_synonyms = {
        "sneakers": ["sneakers", "trainers", "athletic shoes", "sports shoes", "running shoes"],
        "flats": ["flats", "ballet flats", "loafers"],
        "boots": ["boots", "ankle boots", "combat boots", "winter boots"],
        "sandals": ["sandals", "flip flops", "slippers"],
        "heels": ["heels", "high heels", "stilettos", "pumps"],
        "casual shoes": ["casual", "everyday shoes"],
        "formal shoes": ["formal", "dress shoes", "office shoes"]
    }
    
    shoes_general_terms = ["shoes", "footwear"]
    
    brand_synonyms = {
        "nike": ["nike", "air jordan", "jordan"],
        "adidas": ["adidas", "three stripes"],
        "puma": ["puma"],
        "reebok": ["reebok"],
        "converse": ["converse", "chuck taylor", "all star"],
        "vans": ["vans"],
        "new balance": ["new balance", "nb"],
        "mizuno": ["mizuno"],
        "asics": ["asics"],
        "under armour": ["under armour", "ua"]
    }
    
    color_synonyms = {
        "red": ["red", "crimson", "burgundy", "scarlet"],
        "blue": ["blue", "navy", "cobalt", "azure"],
        "black": ["black", "ebony", "charcoal"],
        "white": ["white", "ivory", "cream", "off-white"],
        "green": ["green", "emerald", "olive"],
        "yellow": ["yellow", "gold", "golden"],
        "pink": ["pink", "rose", "magenta"],
        "purple": ["purple", "violet", "lavender"],
        "brown": ["brown", "tan", "beige", "khaki"],
        "grey": ["grey", "gray", "silver"],
        "orange": ["orange", "coral", "peach"]
    }
    
    gender_synonyms = {
        "women": ["women", "female", "girls", "ladies", "womens"],
        "men": ["men", "male", "boys", "mens", "guys"],
        "unisex": ["unisex", "both", "all"]
    }
    
    for main_category, synonyms in category_synonyms.items():
        for synonym in synonyms:
            if synonym in query_lower:
                filters["category"] = main_category
                break
        if filters["category"]:
            break
    
    for general_term in shoes_general_terms:
        if general_term in query_lower and not filters["category"]:
            pass
    
    for main_brand, synonyms in brand_synonyms.items():
        for synonym in synonyms:
            if synonym in query_lower:
                filters["brand"] = main_brand
                break
        if filters["brand"]:
            break
    
    for main_color, synonyms in color_synonyms.items():
        for synonym in synonyms:
            if synonym in query_lower:
                filters["color"] = main_color
                break
        if filters["color"]:
            break
    
    for main_gender, synonyms in gender_synonyms.items():
        for synonym in synonyms:
            if synonym in query_lower:
                filters["gender"] = main_gender
                break
        if filters["gender"]:
            break
    
    under_patterns = [
        r'under\s+(\d+)',
        r'below\s+(\d+)',
        r'less\s+than\s+(\d+)',
        r'<\s*(\d+)',
        r'price\s+<\s*(\d+)',
        r'max\s+(\d+)',
        r'maximum\s+(\d+)',
        r'budget\s+(\d+)'
    ]
    
    over_patterns = [
        r'over\s+(\d+)',
        r'above\s+(\d+)',
        r'more\s+than\s+(\d+)',
        r'>\s*(\d+)',
        r'price\s+>\s*(\d+)',
        r'min\s+(\d+)',
        r'minimum\s+(\d+)'
    ]
    
    for pattern in under_patterns:
        match = re.search(pattern, query_lower)
        if match:
            filters["price_max"] = int(match.group(1))
            break
    
    if not filters["price_max"]:
        for pattern in over_patterns:
            match = re.search(pattern, query_lower)
            if match:
                filters["price_min"] = int(match.group(1))
                break
    
    return filters

@app.post("/search")
async def search_products(request: SearchRequest):
    try:
        print(f"🔍 Parsing query: {request.query}")
        filters = parse_query_with_nlp(request.query)
        print(f"✅ Filters parsed successfully: {filters}")
        
        should_clauses = []
        filter_clauses = []
        
        multi_match = {
            "multi_match": {
                "query": request.query,
                "fields": ["title^3", "description^2", "category^2", "brand^2", "tags"],
                "fuzziness": "AUTO",
                "type": "best_fields"
            }
        }
        should_clauses.append(multi_match)
        
        for keyword in filters.get("keywords", []):
            if len(keyword) > 2:
                should_clauses.append({
                    "multi_match": {
                        "query": keyword,
                        "fields": ["title^3", "description", "category", "brand", "tags"],
                        "fuzziness": "AUTO"
                    }
                })
        
        if filters.get("semantic_analysis"):
            semantic = filters["semantic_analysis"]
            
            if semantic.get("brand_confidence"):
                for brand, confidence in semantic["brand_confidence"].items():
                    boost_value = min(confidence * 2.0, 5.0)
                    should_clauses.append({
                        "term": {
                            "brand.keyword": {
                                "value": brand,
                                "boost": boost_value
                            }
                        }
                    })
            
            if semantic.get("color_confidence"):
                for color, confidence in semantic["color_confidence"].items():
                    boost_value = min(confidence * 1.8, 4.0)
                    should_clauses.append({
                        "term": {
                            "color.keyword": {
                                "value": color,
                                "boost": boost_value
                            }
                        }
                    })
            
            if semantic.get("category_confidence"):
                for category, confidence in semantic["category_confidence"].items():
                    boost_value = min(confidence * 1.5, 3.0)
                    should_clauses.append({
                        "term": {
                            "category.keyword": {
                                "value": category,
                                "boost": boost_value
                            }
                        }
                    })
            
            if semantic.get("noun_phrases"):
                for phrase in semantic["noun_phrases"]:
                    if len(phrase) > 3:
                        should_clauses.append({
                            "match_phrase": {
                                "title": {
                                    "query": phrase,
                                    "boost": 1.3
                                }
                            }
                        })
        
        else:
            if filters["category"]:
                should_clauses.append({
                    "term": {
                        "category.keyword": {
                            "value": filters["category"],
                            "boost": 2.0
                        }
                    }
                })
            
            if filters["brand"]:
                should_clauses.append({
                    "term": {
                        "brand.keyword": {
                            "value": filters["brand"],
                            "boost": 2.0
                        }
                    }
                })
            
            if filters["color"]:
                should_clauses.append({
                    "term": {
                        "color.keyword": {
                            "value": filters["color"],
                            "boost": 2.0
                        }
                    }
                })
        
        if filters["gender"]:
            gender_should = []
            if filters["gender"] == "women":
                gender_should = ["women", "girls", "female"]
            elif filters["gender"] == "men":
                gender_should = ["men", "boys", "male"]
            else:
                gender_should = [filters["gender"]]
            
            for gender_term in gender_should:
                should_clauses.append({
                    "term": {
                        "gender.keyword": {
                            "value": gender_term,
                            "boost": 1.5
                        }
                    }
                })
        
        if filters["price_max"]:
            filter_clauses.append({"range": {"price": {"lte": filters["price_max"]}}})
        
        if filters["price_min"]:
            filter_clauses.append({"range": {"price": {"gte": filters["price_min"]}}})
        
        if filter_clauses:
            es_query = {
                "query": {
                    "bool": {
                        "should": should_clauses,
                        "filter": filter_clauses,
                        "minimum_should_match": 1
                    }
                },
                "sort": [
                    {"_score": {"order": "desc"}},
                    {"rating": {"order": "desc"}},
                    {"price": {"order": "asc"}}
                ],
                "size": 50
            }
        else:
            es_query = {
                "query": {
                    "bool": {
                        "should": should_clauses,
                        "minimum_should_match": 1
                    }
                },
                "sort": [
                    {"_score": {"order": "desc"}},
                    {"rating": {"order": "desc"}},
                    {"price": {"order": "asc"}}
                ],
                "size": 50
            }
        
        print(f"🔍 Searching for: {request.query}")
        print(f"📊 Filters extracted: {filters}")
        print(f"🔧 Elasticsearch query: {es_query}")
        
        if es_client:
            try:
                print("🔌 Using Elasticsearch")
                response = await es_client.search(index="ecommerce", body=es_query)
                products = []
                
                for hit in response["hits"]["hits"]:
                    product = hit["_source"]
                    product["_score"] = hit["_score"]
                    products.append(product)
                
                print(f"✅ Elasticsearch found {len(products)} products")
                print(f"📊 ES Response total: {response['hits']['total']['value']}")
                
                products = apply_nlp_scoring(products, filters, request.query)
                
                return products
            except Exception as e:
                print(f"❌ Elasticsearch search failed: {e}")
                print("🔄 Falling back to MongoDB")
        
        if mongo_client:
            print("🔌 Using MongoDB")
            db = mongo_client.ecommerce
            collection = db.products
            
            mongo_query = {"$and": []}
            
            if filters["category"]:
                mongo_query["$and"].append({
                    "category": {"$regex": filters["category"], "$options": "i"}
                })
            
            if filters["brand"]:
                mongo_query["$and"].append({
                    "brand": {"$regex": filters["brand"], "$options": "i"}
                })
            
            if filters["color"]:
                mongo_query["$and"].append({
                    "color": {"$regex": filters["color"], "$options": "i"}
                })
            
            if filters["gender"]:
                mongo_query["$and"].append({
                    "gender": {"$regex": filters["gender"], "$options": "i"}
                })
            
            if filters["price_max"]:
                mongo_query["$and"].append({
                    "price": {"$lte": filters["price_max"]}
                })
            
            if filters["price_min"]:
                mongo_query["$and"].append({
                    "price": {"$gte": filters["price_min"]}
                })
            
            if filters.get("keywords") or not any([filters["category"], filters["brand"], filters["color"], filters["gender"]]):
                search_terms = filters.get("keywords", []) or [request.query.lower()]
                text_conditions = []
                
                for term in search_terms:
                    if len(term) > 2:
                        term_condition = {
                            "$or": [
                                {"title": {"$regex": term, "$options": "i"}},
                                {"description": {"$regex": term, "$options": "i"}},
                                {"category": {"$regex": term, "$options": "i"}},
                                {"brand": {"$regex": term, "$options": "i"}},
                                {"color": {"$regex": term, "$options": "i"}},
                                {"tags": {"$regex": term, "$options": "i"}}
                            ]
                        }
                        text_conditions.append(term_condition)
                
                if text_conditions:
                    if len(text_conditions) == 1:
                        mongo_query["$and"].append(text_conditions[0])
                    else:
                        mongo_query["$and"].append({"$and": text_conditions})
            
            if not mongo_query["$and"]:
                mongo_query = {}
            elif len(mongo_query["$and"]) == 1:
                mongo_query = mongo_query["$and"][0]
            
            print(f"📝 MongoDB query: {mongo_query}")
            
            pipeline = [
                {"$match": mongo_query},
                {
                    "$addFields": {
                        "score": {
                            "$add": [
                                {"$multiply": ["$rating", 2]},
                                {"$cond": [{"$gt": ["$stock", 0]}, 5, 0]},
                                {"$divide": [100, {"$add": ["$price", 1]}]}
                            ]
                        }
                    }
                },
                {"$sort": {"score": -1, "price": 1, "rating": -1}},
                {"$limit": 50}
            ]
            
            cursor = collection.aggregate(pipeline)
            products = []
            
            async for product in cursor:
                product["_id"] = str(product["_id"])
                products.append(product)
            
            print(f"✅ MongoDB found {len(products)} products")
            return products
        else:
            raise HTTPException(status_code=500, detail="No search backend configured")
    
    except Exception as e:
        print(f"❌ Search error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def apply_nlp_scoring(products: List[Dict[str, Any]], filters: Dict[str, Any], original_query: str) -> List[Dict[str, Any]]:
    
    if not products:
        return products
    
    semantic = filters.get("semantic_analysis", {})
    
    for product in products:
        nlp_score = product.get("_score", 1.0)
        
        if filters.get("brand") and semantic.get("brand_confidence"):
            product_brand = product.get("brand", "").lower()
            target_brand = filters["brand"].lower()
            
            if product_brand == target_brand:
                brand_confidence = semantic["brand_confidence"].get(target_brand, 1.0)
                nlp_score += brand_confidence * 3.0
                print(f"🎯 Exact brand match: {product_brand} -> +{brand_confidence * 3.0} points")
            else:
                nlp_score *= 0.3
                print(f"❌ Brand mismatch: expected {target_brand}, got {product_brand} -> 70% penalty")
        
        if filters.get("color") and semantic.get("color_confidence"):
            product_color = product.get("color", "").lower()
            target_color = filters["color"].lower()
            
            if product_color == target_color:
                color_confidence = semantic["color_confidence"].get(target_color, 1.0)
                nlp_score += color_confidence * 2.0
                print(f"🎨 Exact color match: {product_color} -> +{color_confidence * 2.0} points")
            else:
                nlp_score *= 0.4
                print(f"❌ Color mismatch: expected {target_color}, got {product_color} -> 60% penalty")
        
        if filters.get("category") and semantic.get("category_confidence"):
            product_category = product.get("category", "").lower()
            target_category = filters["category"].lower()
            
            if product_category == target_category:
                category_confidence = semantic["category_confidence"].get(target_category, 1.0)
                nlp_score += category_confidence * 1.5
                print(f"📂 Exact category match: {product_category} -> +{category_confidence * 1.5} points")
            elif "shoes" in original_query.lower() and product_category in ["sneakers", "flats", "boots", "sandals", "heels"]:
                nlp_score += 0.5
                print(f"👟 General shoes match: {product_category} -> +0.5 points")
            else:
                nlp_score *= 0.7
                print(f"❌ Category mismatch: expected {target_category}, got {product_category} -> 30% penalty")
        
        if filters.get("gender"):
            product_gender = product.get("gender", "").lower()
            target_gender = filters["gender"].lower()
            
            gender_match = False
            if target_gender == "women" and product_gender in ["women", "girls", "female"]:
                gender_match = True
            elif target_gender == "men" and product_gender in ["men", "boys", "male"]:
                gender_match = True
            elif target_gender == product_gender:
                gender_match = True
            
            if gender_match:
                nlp_score += 1.0
                print(f"👤 Gender match: {product_gender} -> +1.0 points")
            else:
                nlp_score *= 0.6
                print(f"❌ Gender mismatch: expected {target_gender}, got {product_gender} -> 40% penalty")
        
        if semantic.get("noun_phrases"):
            product_title = product.get("title", "").lower()
            for phrase in semantic["noun_phrases"]:
                if phrase in product_title:
                    nlp_score += 0.8
                    print(f"📝 Noun phrase match: '{phrase}' in title -> +0.8 points")
        
        product["_nlp_score"] = nlp_score
        product["_original_score"] = product.get("_score", 1.0)
    
    products.sort(key=lambda x: x.get("_nlp_score", 0), reverse=True)
    
    print(f"🧠 NLP Scoring Results:")
    for i, product in enumerate(products[:5]):
        print(f"  {i+1}. {product.get('title', 'Unknown')} - NLP Score: {product.get('_nlp_score', 0):.2f} (Original: {product.get('_original_score', 0):.2f})")
    
    return products

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
